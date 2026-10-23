"""Scoring trained models on validation folds or the held-out test set.

``evaluate_folds`` loads the checkpoint of each fold, predicts the images the model
did not train on (its own validation fold, or the test set) and scores them with
the metrics of ``metrics.py`` under a grid of post-processing settings:

- a **threshold** on the predicted probability, and
- a **minimum region size**: connected regions of predicted pixels smaller than this
  are removed (8-connectivity), so a stray spot of a few pixels no longer counts as
  a false alarm.

Scoring a validation fold with the model that did not train on it, for every fold,
gives out-of-fold predictions for all development images; the results of the folds
are pooled by concatenating the per-image scores. Choosing a setting on these
results tunes it on data it was scored on, so it is slightly optimistic: fix the
setting first, then score the test set once.

Every sweep also scores the baseline of predicting nothing, which already gets the
Dice of 1 on every image without a mass.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader

from mammo_seg.data.dataset import DEFAULT_CANVAS, MammogramDataset, Sample
from mammo_seg.data.folds import fold_ids, held_out_ids
from mammo_seg.data.loaders import evaluation_loader
from mammo_seg.data.preprocessing import PreprocessingSettings
from mammo_seg.data.schema import Manifest
from mammo_seg.data.splits import SplitsFile, fold_name
from mammo_seg.device import resolve_device
from mammo_seg.engine.checkpoint import load_checkpoint, model_from_checkpoint
from mammo_seg.engine.metrics import ImageScores, Summary, image_scores, summarize

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_MIN_AREAS = (0, 100, 300, 1000)
_EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


@dataclass(frozen=True)
class PostProcessing:
    """How a probability map becomes a prediction."""

    threshold: float = 0.5
    min_area: int = 0  # pixels; regions with fewer pixels are removed

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError(f"the threshold must be in (0, 1), got {self.threshold}")
        if self.min_area < 0:
            raise ValueError(f"the minimum area cannot be negative: {self.min_area}")


def settings_grid(
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    min_areas: Sequence[int] = DEFAULT_MIN_AREAS,
) -> tuple[PostProcessing, ...]:
    """Every threshold with every minimum region size."""
    if not thresholds or not min_areas:
        raise ValueError("at least one threshold and one minimum area are needed")
    return tuple(PostProcessing(t, a) for t in thresholds for a in min_areas)


def remove_small_regions(prediction: torch.Tensor, min_area: int) -> torch.Tensor:
    """Drop the connected regions (8-connectivity) of a ``[H, W]`` boolean
    prediction that have fewer than ``min_area`` pixels."""
    if min_area <= 1 or not prediction.any():
        return prediction
    labels, count = ndimage.label(prediction.numpy(), structure=_EIGHT_CONNECTED)
    keep = np.bincount(labels.ravel(), minlength=count + 1) >= min_area
    keep[0] = False  # label 0 is the background
    return torch.from_numpy(keep[labels])


@dataclass(frozen=True)
class ImageResult:
    scores: ImageScores
    predicted_pixels: int  # on the valid pixels, after post-processing


def score_batch(
    probabilities: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    setting: PostProcessing,
) -> list[ImageResult]:
    """The scores of every image of a batch of ``[B, 1, H, W]`` CPU tensors.

    Raises ValueError for tensors of different shapes.
    """
    if not probabilities.shape == masks.shape == valid.shape:
        raise ValueError(
            f"shapes differ: probabilities {tuple(probabilities.shape)}, masks "
            f"{tuple(masks.shape)}, valid {tuple(valid.shape)}"
        )
    valid = valid.bool()
    predicted = (probabilities >= setting.threshold) & valid
    cleaned = torch.stack(
        [remove_small_regions(image[0], setting.min_area) for image in predicted]
    ).unsqueeze(1)
    scores = image_scores(cleaned.float(), masks, valid)
    pixels = cleaned.flatten(1).sum(1).tolist()
    return [
        ImageResult(score, int(count))
        for score, count in zip(scores, pixels, strict=True)
    ]


@dataclass(frozen=True)
class SettingResult:
    """The summary of a set of images under one post-processing setting."""

    setting: PostProcessing
    summary: Summary
    # Mean predicted pixels on the images without a mass; NaN if there are none.
    false_positive_pixels: float
    # The mean of the Dice on images with a mass and on images without one, so both
    # kinds count equally; NaN if either kind is missing.
    balanced_dice: float


def summarize_results(
    setting: PostProcessing, results: Sequence[ImageResult]
) -> SettingResult:
    summary = summarize([result.scores for result in results])
    negatives = [r.predicted_pixels for r in results if not r.scores.has_mass]
    return SettingResult(
        setting=setting,
        summary=summary,
        false_positive_pixels=sum(negatives) / len(negatives)
        if negatives
        else math.nan,
        balanced_dice=(summary.with_mass.dice + summary.without_mass.dice) / 2,
    )


@dataclass(frozen=True)
class ModelResults:
    """The per-image results of one model, for every setting and for the baseline."""

    settings: dict[PostProcessing, list[ImageResult]]
    baseline: list[ImageResult]


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[Sample],
    settings: Sequence[PostProcessing],
    device: torch.device,
    amp: bool = False,
) -> ModelResults:
    """Predict every image of the loader once and score it under every setting."""
    model.eval()
    results: dict[PostProcessing, list[ImageResult]] = {s: [] for s in settings}
    baseline: list[ImageResult] = []
    for batch in loader:
        with torch.autocast(device.type, enabled=amp):
            logits = model(batch["image"].to(device, non_blocking=True))
        probabilities = torch.sigmoid(logits.float()).cpu()
        masks, valid = batch["mask"], batch["valid"]
        for setting in settings:
            results[setting] += score_batch(probabilities, masks, valid, setting)
        baseline += score_batch(
            torch.zeros_like(probabilities), masks, valid, PostProcessing()
        )
    return ModelResults(results, baseline)


@dataclass(frozen=True)
class FoldEvaluation:
    """Results over a set of images (one fold, or all folds pooled)."""

    images: int
    settings: tuple[SettingResult, ...]
    baseline: SettingResult  # predicting nothing


@dataclass(frozen=True)
class Evaluation:
    on: str  # "validation" or "test"
    checkpoint: str
    per_fold: dict[int, FoldEvaluation]
    pooled: FoldEvaluation


def _fold_evaluation(model_results: ModelResults) -> FoldEvaluation:
    return FoldEvaluation(
        images=len(model_results.baseline),
        settings=tuple(
            summarize_results(setting, results)
            for setting, results in model_results.settings.items()
        ),
        baseline=summarize_results(PostProcessing(), model_results.baseline),
    )


def evaluate_folds(
    manifest: Manifest,
    splits_file: SplitsFile,
    preprocessed_dir: Path,
    runs_dir: Path,
    folds: Sequence[int],
    settings: Sequence[PostProcessing],
    preprocessing_settings: PreprocessingSettings = PreprocessingSettings(),  # noqa: B008
    checkpoint_name: str = "last.pt",
    test: bool = False,
    batch_size: int = 4,
    num_workers: int = 0,
    device: torch.device | None = None,
    amp: bool = False,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
) -> Evaluation:
    """Score the checkpoint of each fold on its validation fold, or, with ``test``,
    on the held-out test set.

    The checkpoints are ``<runs_dir>/fold_<k>/<checkpoint_name>``. Raises ValueError
    for no folds or a repeated fold, a checkpoint of another fold or made with other
    preprocessing settings, splits without a test set, or splits made from another
    manifest.
    """
    if not folds or len(set(folds)) != len(folds):
        raise ValueError(f"the folds must be given once each, got {list(folds)}")
    device = device or resolve_device()
    per_fold: dict[int, FoldEvaluation] = {}
    pooled = ModelResults({s: [] for s in settings}, [])
    for fold in folds:
        path = runs_dir / fold_name(fold) / checkpoint_name
        checkpoint = load_checkpoint(path, device)
        if checkpoint.metadata.get("fold") != fold:
            raise ValueError(
                f"{path} was trained for fold {checkpoint.metadata.get('fold')}, "
                f"not fold {fold}"
            )
        if checkpoint.preprocessing_settings != preprocessing_settings:
            raise ValueError(
                f"{path} was trained on images preprocessed with "
                f"{checkpoint.preprocessing_settings}, not {preprocessing_settings}"
            )
        ids = (
            held_out_ids(manifest, splits_file)
            if test
            else fold_ids(manifest, splits_file, fold).validation
        )
        dataset = MammogramDataset(
            preprocessed_dir, ids, preprocessing_settings, canvas
        )
        model = model_from_checkpoint(checkpoint).to(device)
        results = evaluate_model(
            model,
            evaluation_loader(dataset, batch_size, num_workers),
            settings,
            device,
            amp,
        )
        per_fold[fold] = _fold_evaluation(results)
        for setting in settings:
            pooled.settings[setting].extend(results.settings[setting])
        pooled.baseline.extend(results.baseline)
    return Evaluation(
        on="test" if test else "validation",
        checkpoint=checkpoint_name,
        per_fold=per_fold,
        pooled=_fold_evaluation(pooled),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def evaluation_to_dict(evaluation: Evaluation) -> dict[str, Any]:
    """The evaluation as plain values for JSON; NaN becomes None."""
    content: dict[str, Any] = _json_safe(asdict(evaluation))
    return content


def best_setting(results: FoldEvaluation) -> SettingResult:
    """The setting with the highest balanced Dice (an undefined one never wins)."""
    defined = [r for r in results.settings if not math.isnan(r.balanced_dice)]
    if not defined:
        raise ValueError("no setting has a defined balanced Dice")
    return max(defined, key=lambda r: r.balanced_dice)


def _show(value: float, digits: int = 3) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


_HEADER = (
    f"{'threshold':>9} {'min area':>8}  | {'mass Dice':>9} {'sens.':>6} {'prec.':>6} | "
    f"{'false alarms':>12} {'FP pixels':>9} | {'all Dice':>8} {'balanced':>8}"
)


def _row(label: str, result: SettingResult) -> str:
    groups = result.summary
    return (
        f"{label} | {_show(groups.with_mass.dice):>9} "
        f"{_show(groups.with_mass.sensitivity):>6} "
        f"{_show(groups.with_mass.precision):>6} | "
        f"{_show(groups.without_mass.false_alarm_rate):>12} "
        f"{_show(result.false_positive_pixels, 0):>9} | "
        f"{_show(groups.all_images.dice):>8} {_show(result.balanced_dice):>8}"
    )


def _label(result: SettingResult, mark: str = " ") -> str:
    return f"{result.setting.threshold:>9.2f} {result.setting.min_area:>8}{mark}"


def format_evaluation(evaluation: Evaluation) -> str:
    """The pooled sweep (best balanced Dice marked ``*``), the baseline, and the
    folds at the best setting, as text."""
    pooled = evaluation.pooled
    best = best_setting(pooled)
    lines = [
        f"{evaluation.checkpoint} on the {evaluation.on} images "
        f"({pooled.images} scored, {len(evaluation.per_fold)} fold(s) pooled)",
        "",
        _HEADER,
    ]
    for result in pooled.settings:
        lines.append(_row(_label(result, "*" if result is best else " "), result))
    lines.append(_row(f"{'predict nothing':>19}", pooled.baseline))
    lines += [
        "",
        f"per fold at threshold {best.setting.threshold}, "
        f"min area {best.setting.min_area}:",
        _HEADER,
    ]
    for fold, evaluated in evaluation.per_fold.items():
        chosen = next(r for r in evaluated.settings if r.setting == best.setting)
        lines.append(_row(f"{'fold ' + str(fold):>18} ", chosen))
    return "\n".join(lines)
