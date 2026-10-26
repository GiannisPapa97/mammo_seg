"""Training a segmentation model on one cross-validation fold.

``train_fold`` builds the data, the model and the optimizer for a fold of the
patient-level splits and calls ``fit``, which alternates an epoch of training with
a pass of validation:

- training draws batches balanced between images with and without a mass, or, with
  a ``BatchComposition``, batches of a fixed make-up of patches
  (``loaders.training_loader``) and minimizes ``DiceBCELoss`` with Adam, with a
  learning rate that follows a cosine curve down to 0 over the epochs. The loss
  counts every pixel, the padding too, as background: no mass can be there, and
  the network sees padding at inference as well, so it learns that flat 0 is not a
  mass. Without it, the flat background was left untrained and validation predicted
  masses all over it
- validation runs on every image of the validation fold once and computes the
  metrics of ``metrics.py`` for images with a mass, without one, and all images. A
  model trained on crops predicts the whole image in sliding windows of the crop
  size, one trained on whole images in one pass (``inference.py``), so it sees at
  validation what it saw in training; ``train_fold`` records which in the
  checkpoint, for ``evaluate``
- mixed precision (float16 on the GPU) is used for the network only; the loss is
  computed in float32 and a gradient scaler guards against underflow. It is off on
  the CPU, where it brings nothing
- one line per epoch goes to the log and, if an output folder is given, one JSON
  line per epoch to ``history.jsonl`` there (an undefined metric is ``null``). On a
  GPU, both record the peak memory of the epoch, which tells whether a larger
  input or batch still fits

The model with the best value of the selection metric is the one to keep; by
default that is the mean Dice over all validation images.

Built for free cloud GPUs, where a session ends after a few hours or without
warning. With an output folder, every epoch writes ``last.pt`` (weights, optimizer,
schedule, history: all that is needed to continue) and, when the selection metric
improved, ``best.pt`` (weights only), both atomically (see ``checkpoint.py``).
``resume=True`` continues from ``last.pt`` and gives the same result as a run that
was never interrupted, because the random draws of an epoch depend only on the seed
and the epoch number (see ``epoch_seed``). A run stops by itself when the selection
metric has not improved for ``patience`` epochs, or, before a session is cut off,
when another epoch would not fit in ``time_limit_minutes``; the last finished epoch
is on disk either way.
"""

import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
import torch
from pydantic import ConfigDict, Field, TypeAdapter, field_validator, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mammo_seg.data.augmentation import Augmentation, AugmentationSettings
from mammo_seg.data.dataset import (
    DEFAULT_CANVAS,
    DEFAULT_MULTIPLE,
    MammogramDataset,
    Sample,
)
from mammo_seg.data.folds import fold_ids
from mammo_seg.data.loaders import BatchComposition, evaluation_loader, training_loader
from mammo_seg.data.preprocessing import PreprocessingSettings
from mammo_seg.data.schema import Manifest
from mammo_seg.data.splits import SplitsFile
from mammo_seg.device import resolve_device
from mammo_seg.engine.checkpoint import load_checkpoint, save_checkpoint
from mammo_seg.engine.inference import Inference, predict
from mammo_seg.engine.losses import DiceBCELoss
from mammo_seg.engine.metrics import GroupSummary, Summary, image_scores, summarize
from mammo_seg.engine.model import ModelSettings, build_model

logger = logging.getLogger(__name__)

HISTORY_FILE = "history.jsonl"
LAST_FILE = "last.pt"  # everything needed to resume
BEST_FILE = "best.pt"  # the weights of the best epoch
GROUPS = ("with_mass", "without_mass", "all_images")
METRICS = ("dice", "iou", "precision", "sensitivity")
# "<group>.<metric>", for example "with_mass.dice"; higher is better for all.
SELECTION_METRICS = tuple(f"{group}.{metric}" for group in GROUPS for metric in METRICS)
DEFAULT_SELECTION_METRIC = "all_images.dice"


@pydantic_dataclass(frozen=True, config=ConfigDict(strict=True, extra="forbid"))
class TrainingSettings:
    """How a model is trained."""

    epochs: Annotated[int, Field(ge=1)] = 60
    batch_size: Annotated[int, Field(ge=1)] = 4
    learning_rate: Annotated[float, Field(gt=0)] = 1e-4  # at the start; cosine decay
    # Share of images with a mass in a training batch; None draws every image once.
    positive_fraction: Annotated[float, Field(gt=0, lt=1)] | None = 0.5
    # Every batch holds this many patches of each kind, instead of the draw above
    # (which is then not used). It needs crops and a batch size equal to its size.
    composition: BatchComposition | None = None
    # Images drawn per epoch; None means as many as there are training images.
    samples_per_epoch: Annotated[int, Field(ge=1)] | None = None
    amp: bool = True  # mixed precision, used on a CUDA GPU only
    # The validation metric that decides which model is the best; see
    # SELECTION_METRICS. All metrics are logged every epoch whatever this is.
    selection_metric: str = DEFAULT_SELECTION_METRIC
    # Stop when the selection metric has not improved for this many epochs; None
    # always trains for all the epochs.
    patience: Annotated[int, Field(ge=1)] | None = 15
    seed: Annotated[int, Field(ge=0)] = 0
    num_workers: Annotated[int, Field(ge=0)] = 0

    @field_validator("selection_metric")
    @classmethod
    def _known_metric(cls, name: str) -> str:
        if name not in SELECTION_METRICS:
            raise ValueError(
                f"unknown selection metric {name!r}; choose one of "
                f"{', '.join(SELECTION_METRICS)}"
            )
        return name

    @model_validator(mode="after")
    def _batch_matches_composition(self) -> Self:
        if self.composition is not None and self.composition.size != self.batch_size:
            raise ValueError(
                f"the batch size is {self.batch_size} but the composition "
                f"{self.composition} has {self.composition.size} patches"
            )
        return self


@dataclass(frozen=True)
class EpochRecord:
    """What one epoch produced; epochs are counted from 1."""

    epoch: int
    learning_rate: float
    train_loss: float
    seconds: float
    validation: Summary
    selection_value: float  # NaN when the selection metric is undefined
    # The most GPU memory PyTorch held during the epoch, training and validation;
    # None off the GPU, and in the history of a run from before it was recorded.
    peak_gpu_memory_gb: float | None = None


# Settings that may differ between the sessions of one run: they do not change what
# is learned (mixed precision only changes the arithmetic). The number of workers
# does: the workers share the samples, so it changes the random draws.
_FREE_ON_RESUME = ("amp",)
_SETTINGS = TypeAdapter(TrainingSettings)

StopReason = Literal["finished", "early_stopping", "time_limit"]


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[EpochRecord, ...]  # every epoch, including those before a resume
    best_epoch: int | None  # None if the selection metric was never defined
    best_value: float
    stop_reason: StopReason


def epoch_seed(seed: int, epoch: int) -> int:
    """The seed of the random draws of an epoch of a run with ``seed``.

    Both numbers are mixed by NumPy's ``SeedSequence``, so every pair gives its own,
    unrelated seed. A sum would not: seed 1 at epoch 1 would repeat the draws of
    seed 0 at epoch 2, and runs with neighbouring seeds would share their batches.
    Raises ValueError for a negative seed or epoch.
    """
    if seed < 0 or epoch < 0:
        raise ValueError(
            f"the seed and the epoch must not be negative: {seed}, {epoch}"
        )
    return int(np.random.SeedSequence([seed, epoch]).generate_state(1, np.uint64)[0])


def selection_value(summary: Summary, name: str) -> float:
    """The value of a selection metric such as ``"all_images.dice"``."""
    if name not in SELECTION_METRICS:
        raise ValueError(
            f"unknown selection metric {name!r}; choose one of "
            f"{', '.join(SELECTION_METRICS)}"
        )
    group, metric = name.split(".")
    value: float = getattr(getattr(summary, group), metric)
    return value


def _to_device(
    batch: Sample, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["image"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
        batch["valid"].to(device, non_blocking=True),
    )


def reset_peak_memory(device: torch.device) -> None:
    """Start measuring the peak GPU memory again; nothing off the GPU."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float | None:
    """The most memory PyTorch held on the GPU since ``reset_peak_memory``, in GB
    (2**30 bytes), or None off the GPU. It is the memory reserved by the caching
    allocator, not only that of live tensors, because that is what runs out."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_reserved(device) / 2**30


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Sample],
    loss_function: DiceBCELoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
) -> float:
    """One pass over the training loader; returns the mean loss per batch.

    The loss counts every pixel of a sample, the padding included, as the mask says
    (background there). Raises ValueError if the loader gives no batch (fewer images
    than a batch).
    """
    model.train()
    total, batches = 0.0, 0
    for batch in loader:
        images, masks, _ = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            logits = model(images)
        # The padding counts, as background; only the metrics leave it out.
        everywhere = torch.ones_like(masks, dtype=torch.bool)
        loss = loss_function(logits.float(), masks, everywhere)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        batches += 1
    if batches == 0:
        raise ValueError("the training loader gave no batch: too few images?")
    return total / batches


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader[Sample],
    device: torch.device,
    amp: bool,
    inference: Inference = "whole",
    window: tuple[int, int] | None = None,
) -> Summary:
    """The metrics of the model on every image of the loader, predicted in one pass
    or in sliding windows of ``window`` (see ``inference.predict``)."""
    model.eval()
    scores = []
    for batch in loader:
        images, masks, valid = _to_device(batch, device)
        probabilities = predict(model, images, inference, window, amp)
        scores.extend(image_scores(probabilities, masks, valid))
    return summarize(scores)


def _json_safe(value: Any) -> Any:
    """Replace NaN by None, which JSON can hold, in nested dicts."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def record_to_json(record: EpochRecord) -> str:
    """One line of ``history.jsonl``."""
    return json.dumps(_json_safe(asdict(record)))


def describe(record: EpochRecord, selection_metric: str) -> str:
    """The one-line log entry of an epoch."""
    everything, positives = record.validation.all_images, record.validation.with_mass
    return (
        f"epoch {record.epoch}: lr {record.learning_rate:.2e}, "
        f"train loss {record.train_loss:.4f}, "
        f"validation Dice {everything.dice:.4f} (with mass {positives.dice:.4f}, "
        f"sensitivity {positives.sensitivity:.4f}), "
        f"false alarms {record.validation.without_mass.false_alarm_rate:.3f}, "
        f"{selection_metric} {record.selection_value:.4f}, {record.seconds:.0f} s"
        + (
            f", peak GPU memory {record.peak_gpu_memory_gb:.1f} GB"
            if record.peak_gpu_memory_gb is not None
            else ""
        )
    )


def record_from_dict(data: Mapping[str, Any]) -> EpochRecord:
    """Rebuild an epoch record from ``dataclasses.asdict`` of it."""
    validation = data["validation"]
    return EpochRecord(
        **{
            **data,
            "validation": Summary(
                **{name: GroupSummary(**validation[name]) for name in GROUPS}
            ),
        }
    )


@dataclass
class _Progress:
    """What the loop remembers between epochs; saved in ``last.pt``."""

    history: list[EpochRecord]
    best_epoch: int | None = None
    best_value: float = -math.inf
    epochs_since_best: int = 0


def _differences(saved: TrainingSettings, given: TrainingSettings) -> list[str]:
    before, after = asdict(saved), asdict(given)
    return [
        f"{key}: {before[key]} -> {after[key]}"
        for key in before
        if key not in _FREE_ON_RESUME and before[key] != after[key]
    ]


def _restore(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    settings: TrainingSettings,
    model_settings: ModelSettings,
    preprocessing_settings: PreprocessingSettings,
    metadata: Mapping[str, Any],
    device: torch.device,
) -> _Progress:
    """Load ``last.pt`` into the model and the optimizer, schedule and scaler.

    Raises ValueError if there is no checkpoint, or if it was made with other
    settings, so a run is never continued with something else than it started with.
    """
    if not path.is_file():
        raise ValueError(f"nothing to resume: {path} does not exist")
    checkpoint = load_checkpoint(path, device)
    state = checkpoint.training_state
    if state is None:
        raise ValueError(f"{path.name} has no training state to resume from")
    if checkpoint.model_settings != model_settings:
        raise ValueError(
            f"{path.name} holds {checkpoint.model_settings}, not {model_settings}"
        )
    if checkpoint.preprocessing_settings != preprocessing_settings:
        raise ValueError(
            f"{path.name} was trained on images preprocessed with "
            f"{checkpoint.preprocessing_settings}, not {preprocessing_settings}"
        )
    for key, value in metadata.items():
        if checkpoint.metadata.get(key) != value:
            raise ValueError(
                f"{path.name} has {key} = {checkpoint.metadata.get(key)}, not {value}"
            )
    saved = _SETTINGS.validate_json(checkpoint.metadata["training_settings"])
    if differences := _differences(saved, settings):
        raise ValueError(
            f"{path.name} was trained with other settings ({'; '.join(differences)})"
        )
    model.load_state_dict(checkpoint.model_state)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    return _Progress(
        history=[record_from_dict(item) for item in state["history"]],
        best_epoch=state["best_epoch"],
        best_value=state["best_value"],
        epochs_since_best=state["epochs_since_best"],
    )


def _save(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    progress: _Progress,
    improved: bool,
    settings: TrainingSettings,
    model_settings: ModelSettings,
    preprocessing_settings: PreprocessingSettings,
    metadata: Mapping[str, Any],
) -> None:
    """Write ``last.pt``, and ``best.pt`` too when this epoch was the best so far."""
    epoch = len(progress.history)
    common = {
        **metadata,
        "training_settings": _SETTINGS.dump_json(settings).decode(),
        "selection_metric": settings.selection_metric,
    }
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_epoch": progress.best_epoch,
        "best_value": progress.best_value,
        "epochs_since_best": progress.epochs_since_best,
        "history": [asdict(record) for record in progress.history],
    }
    save_checkpoint(
        output_dir / LAST_FILE,
        model,
        model_settings,
        preprocessing_settings,
        training_state=state,
        metadata={**common, "epoch": epoch},
    )
    if improved:
        save_checkpoint(
            output_dir / BEST_FILE,
            model,
            model_settings,
            preprocessing_settings,
            metadata={**common, "epoch": epoch, "value": progress.best_value},
        )


def fit(
    model: nn.Module,
    train_loader: DataLoader[Sample],
    validation_loader: DataLoader[Sample],
    settings: TrainingSettings,
    device: torch.device,
    output_dir: Path | None = None,
    *,
    model_settings: ModelSettings,
    preprocessing_settings: PreprocessingSettings,
    metadata: Mapping[str, Any] | None = None,
    resume: bool = False,
    time_limit_minutes: float | None = None,
    inference: Inference = "whole",
    window: tuple[int, int] | None = None,
) -> TrainingResult:
    """Train ``model`` for up to ``settings.epochs`` epochs, validating after each,
    with the ``inference`` and ``window`` of ``validate``.

    The model is moved to ``device``. With an ``output_dir``, ``history.jsonl``,
    ``last.pt`` and ``best.pt`` are written there after every epoch. ``metadata``
    (plain values such as the fold) is stored in the checkpoints, and a resume
    refuses a checkpoint whose values differ.

    ``resume=True`` continues from ``output_dir/last.pt`` and raises ValueError if
    there is none or it was made with other settings (only mixed precision may
    differ). Training stops early after ``settings.patience``
    epochs without improvement, and stops before an epoch that would not end within
    ``time_limit_minutes`` of the start of this call; at least one epoch is always
    run, so repeated sessions make progress.
    """
    if resume and output_dir is None:
        raise ValueError("an output folder is needed to resume")
    metadata = metadata or {}
    model.to(device)
    amp = settings.amp and device.type == "cuda"
    loss_function = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=settings.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=settings.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    progress = _Progress(history=[])
    if resume:
        assert output_dir is not None
        progress = _restore(
            output_dir / LAST_FILE,
            model,
            optimizer,
            scheduler,
            scaler,
            settings,
            model_settings,
            preprocessing_settings,
            metadata,
            device,
        )
        logger.info("resuming after epoch %d", len(progress.history))
    history_file = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        history_file = output_dir / HISTORY_FILE
        history_file.write_text(
            "".join(record_to_json(record) + "\n" for record in progress.history)
        )

    session_start = time.monotonic()
    slowest = 0.0  # seconds of the slowest epoch of this session
    stop_reason: StopReason = "finished"
    for epoch in range(len(progress.history) + 1, settings.epochs + 1):
        if settings.patience is not None and (
            progress.epochs_since_best >= settings.patience
        ):
            logger.info(
                "early stopping: no improvement for %d epochs", settings.patience
            )
            stop_reason = "early_stopping"
            break
        if (
            time_limit_minutes is not None
            and slowest > 0
            and time.monotonic() - session_start + slowest > time_limit_minutes * 60
        ):
            logger.info("stopping before epoch %d: the time limit is near", epoch)
            stop_reason = "time_limit"
            break
        started = time.monotonic()
        # The draws of an epoch depend only on the seed and the epoch, so a resumed
        # run repeats an uninterrupted one.
        seed = epoch_seed(settings.seed, epoch)
        torch.manual_seed(seed)
        if train_loader.generator is not None:
            train_loader.generator.manual_seed(seed)
        learning_rate = optimizer.param_groups[0]["lr"]
        reset_peak_memory(device)
        train_loss = train_one_epoch(
            model, train_loader, loss_function, optimizer, scaler, device, amp
        )
        scheduler.step()
        summary = validate(model, validation_loader, device, amp, inference, window)
        value = selection_value(summary, settings.selection_metric)
        record = EpochRecord(
            epoch=epoch,
            learning_rate=learning_rate,
            train_loss=train_loss,
            seconds=time.monotonic() - started,
            validation=summary,
            selection_value=value,
            peak_gpu_memory_gb=peak_memory_gb(device),
        )
        slowest = max(slowest, record.seconds)
        progress.history.append(record)
        improved = value > progress.best_value  # False for NaN: undefined never wins
        if improved:
            progress.best_epoch, progress.best_value = epoch, value
            progress.epochs_since_best = 0
        else:
            progress.epochs_since_best += 1
        logger.info(describe(record, settings.selection_metric))
        if output_dir is not None:
            _save(
                output_dir,
                model,
                optimizer,
                scheduler,
                scaler,
                progress,
                improved,
                settings,
                model_settings,
                preprocessing_settings,
                metadata,
            )
        if history_file is not None:
            with history_file.open("a") as stream:
                stream.write(record_to_json(record) + "\n")
    return TrainingResult(
        history=tuple(progress.history),
        best_epoch=progress.best_epoch,
        best_value=progress.best_value if progress.best_epoch is not None else math.nan,
        stop_reason=stop_reason,
    )


def _training_canvas(
    augmentation: AugmentationSettings, canvas: tuple[int, int]
) -> tuple[int, int]:
    """The canvas of the training samples: that of the images, or, when they are
    cropped, the crop size rounded up to what the network needs, so a batch of
    crops costs what its pixels cost. Validation keeps the whole-image canvas."""
    if augmentation.crop_size is None:
        return canvas
    rows, cols = (
        -(-side // DEFAULT_MULTIPLE) * DEFAULT_MULTIPLE
        for side in augmentation.crop_size
    )
    return rows, cols


def train_fold(
    manifest: Manifest,
    splits_file: SplitsFile,
    preprocessed_dir: Path,
    fold: int,
    settings: TrainingSettings = TrainingSettings(),  # noqa: B008
    model_settings: ModelSettings = ModelSettings(),  # noqa: B008
    preprocessing_settings: PreprocessingSettings = PreprocessingSettings(),  # noqa: B008
    augmentation_settings: AugmentationSettings = AugmentationSettings(),  # noqa: B008
    device: torch.device | None = None,
    output_dir: Path | None = None,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    resume: bool = False,
    time_limit_minutes: float | None = None,
    inference: Inference | None = None,
) -> TrainingResult:
    """Train on every fold but ``fold`` of the splits and validate on ``fold``.

    ``device`` defaults to ``resolve_device()``. ``resume`` and
    ``time_limit_minutes`` are those of ``fit``. Validation predicts in sliding
    windows of the training canvas when the training uses crops and in one pass
    otherwise, unless ``inference`` says otherwise; the choice and the window are
    stored in the checkpoints, and a run is not resumed with other ones. Raises
    ValueError for a fold outside the splits, splits made from another manifest,
    preprocessed data made with other settings, a composition without crops, or a
    checkpoint that cannot be resumed.
    """
    if settings.composition is not None and augmentation_settings.crop_size is None:
        raise ValueError(
            "batches with a composition are made of patches: set the crop size"
        )
    device = device or resolve_device()
    ids = fold_ids(manifest, splits_file, fold)
    window = _training_canvas(augmentation_settings, canvas)
    if inference is None:
        inference = "whole" if augmentation_settings.crop_size is None else "sliding"
    train_data = MammogramDataset(
        preprocessed_dir,
        ids.train,
        preprocessing_settings,
        window,
        transform=Augmentation(augmentation_settings),
    )
    validation_data = MammogramDataset(
        preprocessed_dir, ids.validation, preprocessing_settings, canvas
    )
    torch.manual_seed(settings.seed)
    model = build_model(model_settings)
    train_loader = training_loader(
        train_data,
        manifest,
        settings.batch_size,
        settings.positive_fraction,
        settings.samples_per_epoch,
        settings.num_workers,
        settings.seed,
        settings.composition,
    )
    validation_loader = evaluation_loader(
        validation_data, settings.batch_size, settings.num_workers
    )
    logger.info(
        "fold %d on %s: %d training and %d validation images, validated %s",
        fold,
        device,
        len(train_data),
        len(validation_data),
        "in one pass" if inference == "whole" else f"in windows of {window}",
    )
    return fit(
        model,
        train_loader,
        validation_loader,
        settings,
        device,
        output_dir,
        model_settings=model_settings,
        preprocessing_settings=preprocessing_settings,
        metadata={
            "fold": fold,
            "augmentation": json.dumps(asdict(augmentation_settings)),
            "inference": inference,
            "window": list(window),
        },
        resume=resume,
        time_limit_minutes=time_limit_minutes,
        inference=inference,
        window=window,
    )
