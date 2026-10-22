"""Training a segmentation model on one cross-validation fold.

``train_fold`` builds the data, the model and the optimizer for a fold of the
patient-level splits and calls ``fit``, which alternates an epoch of training with
a pass of validation:

- training draws batches balanced between images with and without a mass
  (``loaders.training_loader``) and minimizes ``DiceBCELoss`` with Adam, with a
  learning rate that follows a cosine curve down to 0 over the epochs
- validation runs on every image of the validation fold once and computes the
  metrics of ``metrics.py`` for images with a mass, without one, and all images
- mixed precision (float16 on the GPU) is used for the network only; the loss is
  computed in float32 and a gradient scaler guards against underflow. It is off on
  the CPU, where it brings nothing
- one line per epoch goes to the log and, if an output folder is given, one JSON
  line per epoch to ``history.jsonl`` there (an undefined metric is ``null``)

The model with the best value of the selection metric is the one to keep; by
default that is the mean Dice over all validation images.
"""

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import torch
from pydantic import ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mammo_seg.data.augmentation import Augmentation, AugmentationSettings
from mammo_seg.data.dataset import DEFAULT_CANVAS, MammogramDataset, Sample
from mammo_seg.data.folds import fold_ids
from mammo_seg.data.loaders import evaluation_loader, training_loader
from mammo_seg.data.preprocessing import PreprocessingSettings
from mammo_seg.data.schema import Manifest
from mammo_seg.data.splits import SplitsFile
from mammo_seg.device import resolve_device
from mammo_seg.engine.losses import DiceBCELoss
from mammo_seg.engine.metrics import Summary, image_scores, summarize
from mammo_seg.engine.model import ModelSettings, build_model

logger = logging.getLogger(__name__)

HISTORY_FILE = "history.jsonl"
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
    # Images drawn per epoch; None means as many as there are training images.
    samples_per_epoch: Annotated[int, Field(ge=1)] | None = None
    amp: bool = True  # mixed precision, used on a CUDA GPU only
    # The validation metric that decides which model is the best; see
    # SELECTION_METRICS. All metrics are logged every epoch whatever this is.
    selection_metric: str = DEFAULT_SELECTION_METRIC
    seed: int = 0
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


@dataclass(frozen=True)
class EpochRecord:
    """What one epoch produced; epochs are counted from 1."""

    epoch: int
    learning_rate: float
    train_loss: float
    seconds: float
    validation: Summary
    selection_value: float  # NaN when the selection metric is undefined


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[EpochRecord, ...]
    best_epoch: int | None  # None if the selection metric was never defined
    best_value: float


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

    Raises ValueError if the loader gives no batch (fewer images than a batch).
    """
    model.train()
    total, batches = 0.0, 0
    for batch in loader:
        images, masks, valid = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            logits = model(images)
        loss = loss_function(logits.float(), masks, valid)
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
) -> Summary:
    """The metrics of the model on every image of the loader."""
    model.eval()
    scores = []
    for batch in loader:
        images, masks, valid = _to_device(batch, device)
        with torch.autocast(device.type, enabled=amp):
            logits = model(images)
        probabilities = torch.sigmoid(logits.float())
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
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader[Sample],
    validation_loader: DataLoader[Sample],
    settings: TrainingSettings,
    device: torch.device,
    output_dir: Path | None = None,
) -> TrainingResult:
    """Train ``model`` for ``settings.epochs`` epochs, validating after each.

    The model is moved to ``device``. If ``output_dir`` is given, one JSON line per
    epoch is written to ``history.jsonl`` there.
    """
    model.to(device)
    amp = settings.amp and device.type == "cuda"
    loss_function = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=settings.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=settings.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    history_file = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        history_file = output_dir / HISTORY_FILE
        history_file.write_text("")

    history: list[EpochRecord] = []
    best_epoch, best_value = None, -math.inf
    for epoch in range(1, settings.epochs + 1):
        started = time.monotonic()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, loss_function, optimizer, scaler, device, amp
        )
        scheduler.step()
        summary = validate(model, validation_loader, device, amp)
        value = selection_value(summary, settings.selection_metric)
        record = EpochRecord(
            epoch=epoch,
            learning_rate=learning_rate,
            train_loss=train_loss,
            seconds=time.monotonic() - started,
            validation=summary,
            selection_value=value,
        )
        history.append(record)
        logger.info(describe(record, settings.selection_metric))
        if history_file is not None:
            with history_file.open("a") as stream:
                stream.write(record_to_json(record) + "\n")
        if value > best_value:  # False for NaN, so an undefined value never wins
            best_epoch, best_value = epoch, value
    return TrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_value=best_value if best_epoch is not None else math.nan,
    )


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
) -> TrainingResult:
    """Train on every fold but ``fold`` of the splits and validate on ``fold``.

    ``device`` defaults to ``resolve_device()``. Raises ValueError for a fold
    outside the splits, splits made from another manifest, or preprocessed data
    made with other settings.
    """
    device = device or resolve_device()
    ids = fold_ids(manifest, splits_file, fold)
    train_data = MammogramDataset(
        preprocessed_dir,
        ids.train,
        preprocessing_settings,
        canvas,
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
    )
    validation_loader = evaluation_loader(
        validation_data, settings.batch_size, settings.num_workers
    )
    logger.info(
        "fold %d on %s: %d training and %d validation images",
        fold,
        device,
        len(train_data),
        len(validation_data),
    )
    return fit(model, train_loader, validation_loader, settings, device, output_dir)
