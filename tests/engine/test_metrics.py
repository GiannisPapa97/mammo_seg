import math
from collections.abc import Sequence

import pytest
import torch

from mammo_seg.engine.metrics import (
    DEFAULT_THRESHOLD,
    ImageScores,
    image_scores,
    summarize,
)

Grid = Sequence[Sequence[float]]


def batch(
    *pairs: tuple[Grid, Grid],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Probabilities and masks from nested lists, every pixel valid."""
    probabilities = torch.tensor([[p] for p, _ in pairs], dtype=torch.float32)
    masks = torch.tensor([[m] for _, m in pairs], dtype=torch.float32)
    return probabilities, masks, torch.ones_like(masks, dtype=torch.bool)


def score(probabilities: Grid, mask: Grid) -> ImageScores:
    return image_scores(*batch((probabilities, mask)))[0]


# --- one image ---


def test_a_perfect_prediction_scores_one() -> None:
    result = score([[0.9, 0.1], [0.8, 0.0]], [[1, 0], [1, 0]])
    assert (result.dice, result.iou, result.precision, result.sensitivity) == (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    assert result.has_mass and result.predicted


def test_scores_by_hand() -> None:
    # 4 mass pixels; 2 found, 1 false alarm: tp=2, fp=1, fn=2
    probabilities = [[0.9, 0.9, 0.1, 0.0], [0.1, 0.1, 0.0, 0.7]]
    mask = [[1, 1, 1, 1], [0, 0, 0, 0]]
    result = score(probabilities, mask)
    assert result.dice == pytest.approx(4 / 7)  # 2*2 / (2*2 + 1 + 2)
    assert result.iou == pytest.approx(2 / 5)  # 2 / (2 + 1 + 2)
    assert result.precision == pytest.approx(2 / 3)
    assert result.sensitivity == pytest.approx(2 / 4)


def test_the_threshold_decides_what_is_predicted() -> None:
    probabilities, masks, valid = batch(([[0.6, 0.4]], [[1, 1]]))
    assert image_scores(probabilities, masks, valid)[0].sensitivity == 0.5
    assert image_scores(probabilities, masks, valid, threshold=0.3)[0].sensitivity == 1
    assert image_scores(probabilities, masks, valid, threshold=0.7)[0].sensitivity == 0
    assert DEFAULT_THRESHOLD == 0.5


def test_a_probability_equal_to_the_threshold_counts_as_predicted() -> None:
    assert score([[0.5]], [[1]]).dice == 1.0


# --- the empty-image conventions ---


def test_no_mass_and_no_prediction_is_a_perfect_one() -> None:
    result = score([[0.1, 0.2]], [[0, 0]])
    assert (result.dice, result.iou) == (1.0, 1.0)
    assert math.isnan(result.precision) and math.isnan(result.sensitivity)
    assert not result.has_mass and not result.predicted


def test_a_false_alarm_on_an_image_without_a_mass_scores_zero() -> None:
    result = score([[0.9, 0.2]], [[0, 0]])
    assert (result.dice, result.iou, result.precision) == (0.0, 0.0, 0.0)
    assert math.isnan(result.sensitivity)
    assert result.predicted and not result.has_mass


def test_a_missed_mass_scores_zero_and_has_no_precision() -> None:
    result = score([[0.1, 0.2]], [[1, 0]])
    assert (result.dice, result.iou, result.sensitivity) == (0.0, 0.0, 0.0)
    assert math.isnan(result.precision)
    assert result.has_mass and not result.predicted


# --- the padding ---


def test_pixels_outside_the_valid_region_do_not_count() -> None:
    probabilities = torch.tensor([[[[0.9, 0.9, 0.9]]]])
    masks = torch.tensor([[[[1.0, 0.0, 1.0]]]])
    valid = torch.tensor([[[[True, False, False]]]])
    result = image_scores(probabilities, masks, valid)[0]
    assert (result.dice, result.precision, result.sensitivity) == (1.0, 1.0, 1.0)


def test_an_image_whose_mass_is_only_on_padding_has_no_mass() -> None:
    probabilities = torch.zeros(1, 1, 1, 2)
    masks = torch.tensor([[[[0.0, 1.0]]]])
    valid = torch.tensor([[[[True, False]]]])
    assert not image_scores(probabilities, masks, valid)[0].has_mass


# --- a batch ---


def test_every_image_of_a_batch_is_scored_separately() -> None:
    probabilities, masks, valid = batch(
        ([[0.9, 0.1]], [[1, 0]]),
        ([[0.1, 0.1]], [[1, 0]]),
        ([[0.1, 0.1]], [[0, 0]]),
    )
    results = image_scores(probabilities, masks, valid)
    assert [result.dice for result in results] == [1.0, 0.0, 1.0]


def test_tensors_of_different_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        image_scores(
            torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 3), torch.ones(1, 1, 2, 2)
        )


# --- the summary of a set of images ---


def test_the_summary_averages_per_image_so_a_small_mass_counts() -> None:
    # a small mass mostly missed and a large one found: per-image mean is 0.5ish
    small = torch.zeros(1, 1, 10, 100)
    small_truth = small.clone()
    small_truth[0, 0, 0, :10] = 1
    small[0, 0, 0, :1] = 1  # 1 of 10 found
    large = torch.zeros(1, 1, 10, 100)
    large_truth = large.clone()
    large_truth[0, 0, :, :90] = 1
    large[0, 0, :, :90] = 1  # all 900 found
    probabilities = torch.cat([small, large])
    masks = torch.cat([small_truth, large_truth])
    summary = summarize(
        image_scores(probabilities, masks, torch.ones_like(masks).bool())
    )
    small_dice = 2 * 1 / (2 * 1 + 0 + 9)
    assert summary.with_mass.dice == pytest.approx((small_dice + 1.0) / 2)
    pooled = 2 * 901 / (2 * 901 + 0 + 9)
    assert summary.with_mass.dice < pooled - 0.4


def scored(
    dice: float,
    iou: float,
    precision: float,
    sensitivity: float,
    *,
    has_mass: bool,
    predicted: bool,
) -> ImageScores:
    return ImageScores(dice, iou, precision, sensitivity, has_mass, predicted)


def test_the_summary_has_three_groups() -> None:
    nan = math.nan
    scores = [
        scored(0.8, 0.6, 0.9, 0.7, has_mass=True, predicted=True),
        scored(0.4, 0.2, nan, 0.0, has_mass=True, predicted=False),
        scored(1.0, 1.0, nan, nan, has_mass=False, predicted=False),
        scored(0.0, 0.0, 0.0, nan, has_mass=False, predicted=True),
    ]
    summary = summarize(scores)
    assert summary.with_mass.images == 2
    assert summary.with_mass.dice == pytest.approx(0.6)
    assert summary.with_mass.precision == pytest.approx(0.9)  # NaN left out
    assert summary.with_mass.sensitivity == pytest.approx(0.35)
    assert math.isnan(summary.with_mass.false_alarm_rate)
    assert summary.without_mass.images == 2
    assert summary.without_mass.dice == pytest.approx(0.5)
    assert summary.without_mass.false_alarm_rate == pytest.approx(0.5)
    assert math.isnan(summary.without_mass.sensitivity)
    assert summary.all_images.images == 4
    assert summary.all_images.dice == pytest.approx(0.55)
    assert summary.all_images.false_alarm_rate == pytest.approx(0.5)


def test_an_empty_set_summarizes_to_nan() -> None:
    summary = summarize([])
    assert summary.all_images.images == 0
    assert math.isnan(summary.all_images.dice)
