"""Segmentation metrics: Dice, IoU, precision and sensitivity per image.

Every metric is computed per image, only on its valid pixels (not on the padding),
from the predicted probabilities thresholded at 0.5. A validation set is then
summarized by the mean over its images, so a small mass counts as much as a large
one. The summary is reported for images with a mass, images without one, and all
images, so a model cannot hide false alarms behind good scores on the lesions.

Conventions for images without a mass or without a prediction:

- Dice and IoU are 1 when the image has no mass and nothing is predicted (a correct
  "nothing here"), and 0 when something is predicted on an image without a mass.
- Precision is undefined when nothing is predicted, and sensitivity is undefined on
  an image without a mass. Undefined values are NaN and left out of the means.
- On images without a mass, the false-alarm rate is the share of images where
  anything is predicted.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class ImageScores:
    """The scores of one image; NaN where a score is undefined."""

    dice: float
    iou: float
    precision: float
    sensitivity: float
    has_mass: bool
    predicted: bool  # anything predicted on the valid pixels


@dataclass(frozen=True)
class GroupSummary:
    """Means over a group of images; NaN where nothing was defined."""

    images: int
    dice: float
    iou: float
    precision: float
    sensitivity: float
    false_alarm_rate: float  # images without a mass with anything predicted


@dataclass(frozen=True)
class Summary:
    with_mass: GroupSummary
    without_mass: GroupSummary
    all_images: GroupSummary


def _ratio(numerator: int, denominator: int, empty: float) -> float:
    return numerator / denominator if denominator else empty


def image_scores(
    probabilities: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[ImageScores]:
    """The scores of every image of a batch.

    All three tensors are ``[B, 1, H, W]``: predicted probabilities, 0/1 masks and
    the valid pixels. Raises ValueError for tensors of different shapes.
    """
    if not probabilities.shape == masks.shape == valid.shape:
        raise ValueError(
            f"shapes differ: probabilities {tuple(probabilities.shape)}, masks "
            f"{tuple(masks.shape)}, valid {tuple(valid.shape)}"
        )
    valid = valid.bool()
    predicted = (probabilities >= threshold) & valid
    truth = (masks > 0.5) & valid
    true_positives = (predicted & truth).flatten(1).sum(1).tolist()
    false_positives = (predicted & ~truth).flatten(1).sum(1).tolist()
    false_negatives = (~predicted & truth).flatten(1).sum(1).tolist()
    scores = []
    for tp, fp, fn in zip(
        true_positives, false_positives, false_negatives, strict=True
    ):
        has_mass = tp + fn > 0
        scores.append(
            ImageScores(
                dice=_ratio(2 * tp, 2 * tp + fp + fn, empty=1.0),
                iou=_ratio(tp, tp + fp + fn, empty=1.0),
                precision=_ratio(tp, tp + fp, empty=math.nan),
                sensitivity=_ratio(tp, tp + fn, empty=math.nan),
                has_mass=has_mass,
                predicted=tp + fp > 0,
            )
        )
    return scores


def _mean(values: Sequence[float]) -> float:
    defined = [value for value in values if not math.isnan(value)]
    return sum(defined) / len(defined) if defined else math.nan


def _group(scores: Sequence[ImageScores]) -> GroupSummary:
    negatives = [score for score in scores if not score.has_mass]
    return GroupSummary(
        images=len(scores),
        dice=_mean([score.dice for score in scores]),
        iou=_mean([score.iou for score in scores]),
        precision=_mean([score.precision for score in scores]),
        sensitivity=_mean([score.sensitivity for score in scores]),
        false_alarm_rate=_mean([float(score.predicted) for score in negatives]),
    )


def summarize(scores: Sequence[ImageScores]) -> Summary:
    """The per-image means for images with a mass, without one, and all images."""
    return Summary(
        with_mass=_group([score for score in scores if score.has_mass]),
        without_mass=_group([score for score in scores if not score.has_mass]),
        all_images=_group(scores),
    )
