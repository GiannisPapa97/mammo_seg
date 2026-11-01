import math

import pytest
import torch
from torch.nn import functional

from mammo_seg.engine.losses import DiceBCELoss

BIG = 20.0  # a logit so large that sigmoid(BIG) is 1 to float precision


def ones(*shape: int) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.bool)


def test_a_confident_correct_prediction_has_almost_no_loss() -> None:
    masks = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    logits = (masks * 2 - 1) * BIG
    assert DiceBCELoss()(logits, masks, ones(1, 1, 2, 2)).item() == pytest.approx(
        0, abs=1e-6
    )


def test_a_confident_wrong_prediction_has_a_large_loss() -> None:
    masks = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    logits = (1 - masks * 2) * BIG
    loss = DiceBCELoss()(logits, masks, ones(1, 1, 2, 2)).item()
    assert loss > 10


def test_the_parts_by_hand() -> None:
    logits = torch.tensor([[[[0.0, 0.0]]]])  # probability 0.5 everywhere
    masks = torch.tensor([[[[1.0, 0.0]]]])
    valid = ones(1, 1, 1, 2)
    bce = math.log(2)  # every pixel: -log(0.5)
    dice = (2 * 0.5 + 1) / (1.0 + 1.0 + 1)  # overlap 0.5, sums 1 and 1, smooth 1
    assert DiceBCELoss(dice_weight=0, bce_weight=1)(
        logits, masks, valid
    ).item() == pytest.approx(bce)
    assert DiceBCELoss(dice_weight=1, bce_weight=0)(
        logits, masks, valid
    ).item() == pytest.approx(1 - dice)
    both = DiceBCELoss(dice_weight=2, bce_weight=3)(logits, masks, valid).item()
    assert both == pytest.approx(2 * (1 - dice) + 3 * bce)


def test_the_cross_entropy_matches_torch_on_valid_pixels() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 4, 5)
    masks = (torch.rand(2, 1, 4, 5) > 0.7).float()
    valid = ones(2, 1, 4, 5)
    expected = functional.binary_cross_entropy_with_logits(logits, masks)
    loss = DiceBCELoss(dice_weight=0, bce_weight=1)(logits, masks, valid)
    assert loss.item() == pytest.approx(expected.item())


def test_the_padding_does_not_count() -> None:
    logits = torch.tensor([[[[BIG, -BIG, BIG]]]])
    masks = torch.tensor([[[[1.0, 0.0, 0.0]]]])  # the last pixel is wrong ...
    valid = torch.tensor([[[[True, True, False]]]])  # ... but it is padding
    assert DiceBCELoss()(logits, masks, valid).item() == pytest.approx(0, abs=1e-6)


def test_predicting_nothing_on_an_image_without_a_mass_costs_nothing() -> None:
    logits = torch.full((1, 1, 3, 3), -BIG)
    masks = torch.zeros(1, 1, 3, 3)
    assert DiceBCELoss()(logits, masks, ones(1, 1, 3, 3)).item() == pytest.approx(
        0, abs=1e-6
    )


def test_a_false_alarm_on_an_image_without_a_mass_costs_something() -> None:
    logits = torch.full((1, 1, 3, 3), -BIG)
    logits[0, 0, 1, 1] = BIG
    masks = torch.zeros(1, 1, 3, 3)
    assert DiceBCELoss(bce_weight=0)(
        logits, masks, ones(1, 1, 3, 3)
    ).item() == pytest.approx(0.5)


def test_dice_is_averaged_per_image() -> None:
    # image 1 perfect, image 2 completely missed: the Dice part is the mean of 0 and ~1
    logits = torch.full((2, 1, 1, 10), -BIG)
    masks = torch.zeros(2, 1, 1, 10)
    masks[:, 0, 0, :5] = 1
    logits[0, 0, 0, :5] = BIG
    loss = DiceBCELoss(bce_weight=0)(logits, masks, ones(2, 1, 1, 10)).item()
    missed = 1 - 1 / (5 + 1)  # overlap 0, sums 0 and 5, smooth 1
    assert loss == pytest.approx((0 + missed) / 2)


def test_the_loss_has_gradients() -> None:
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    masks = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    DiceBCELoss()(logits, masks, ones(1, 1, 2, 2)).backward()
    assert logits.grad is not None
    assert logits.grad[0, 0, 0, 0] < 0  # raising the mass pixel lowers the loss
    assert logits.grad[0, 0, 1, 1] > 0


def test_padding_gets_no_gradient() -> None:
    logits = torch.zeros(1, 1, 1, 2, requires_grad=True)
    masks = torch.tensor([[[[1.0, 1.0]]]])
    valid = torch.tensor([[[[True, False]]]])
    DiceBCELoss()(logits, masks, valid).backward()
    assert logits.grad is not None and logits.grad[0, 0, 0, 1] == 0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"dice_weight": -1.0}, "must not be negative"),
        ({"dice_weight": 0.0, "bce_weight": 0.0}, "not both 0"),
        ({"smooth": 0.0}, "must be positive"),
    ],
)
def test_invalid_settings_are_rejected(
    arguments: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DiceBCELoss(**arguments)


def test_tensors_of_different_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        DiceBCELoss()(
            torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 3), ones(1, 1, 2, 2)
        )


def test_a_batch_without_valid_pixels_is_rejected() -> None:
    with pytest.raises(ValueError, match="no valid pixel"):
        DiceBCELoss()(
            torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), ~ones(1, 1, 2, 2)
        )


# --- from probabilities ---


def test_the_loss_from_probabilities_is_the_loss_from_logits() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 1, 8, 8) * 4
    masks = (torch.rand(3, 1, 8, 8) > 0.8).float()
    valid = torch.rand(3, 1, 8, 8) > 0.2
    loss = DiceBCELoss(dice_weight=2, bce_weight=3)
    assert loss.from_probabilities(
        torch.sigmoid(logits), masks, valid
    ).item() == pytest.approx(loss(logits, masks, valid).item(), rel=1e-5)


def test_a_saturated_wrong_probability_costs_about_16_not_100() -> None:
    masks = torch.zeros(1, 1, 1, 4)
    probabilities = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])  # one pixel wrong
    bce = DiceBCELoss(dice_weight=0, bce_weight=1).from_probabilities(
        probabilities, masks, ones(1, 1, 1, 4)
    )
    # 1 - 1e-7 in float32 is 1 - 1.19e-7, so the pixel costs 15.9 instead of 100.
    highest = torch.tensor(1 - 1e-7, dtype=torch.float32).item()
    assert bce.item() == pytest.approx(-math.log(1 - highest) / 4, rel=1e-4)
    assert 15.5 < bce.item() * 4 < 16.5


def test_the_loss_from_probabilities_checks_its_inputs() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        DiceBCELoss().from_probabilities(
            torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 3), ones(1, 1, 2, 2)
        )
    with pytest.raises(ValueError, match="no valid pixel"):
        DiceBCELoss().from_probabilities(
            torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), ~ones(1, 1, 2, 2)
        )


# --- focal Tversky ---

TVERSKY = {"alpha": 0.7, "beta": 0.3, "gamma": 0.75}


def soft_dice_loss(probabilities: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """The Dice part as it was written before the Tversky settings."""
    overlap = (probabilities * masks).flatten(1).sum(1)
    total = probabilities.flatten(1).sum(1) + masks.flatten(1).sum(1)
    return (1 - (2 * overlap + 1) / (total + 1)).mean()


def random_batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.randn(3, 1, 8, 8) * 4
    masks = (torch.rand(3, 1, 8, 8) > 0.8).float()
    return logits, masks


def test_the_default_is_the_soft_dice_to_the_bit() -> None:
    logits, masks = random_batch()
    loss = DiceBCELoss(bce_weight=0)(logits, masks, ones(3, 1, 8, 8))
    assert torch.equal(loss, soft_dice_loss(torch.sigmoid(logits), masks))


def test_the_tversky_index_by_hand() -> None:
    logits = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # probability 0.5 everywhere
    masks = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    # overlap 1, misses 2 - 1 = 1, false positives 1.5 - 1 = 0.5, smooth 1
    index = (1 + 0.5) / (1 + 0.7 * 1 + 0.3 * 0.5 + 0.5)
    loss = DiceBCELoss(bce_weight=0, alpha=0.7, beta=0.3)
    assert loss(logits, masks, ones(1, 1, 1, 3)).item() == pytest.approx(1 - index)
    focal = DiceBCELoss(bce_weight=0, **TVERSKY)
    assert focal(logits, masks, ones(1, 1, 1, 3)).item() == pytest.approx(
        (1 - index) ** 0.75
    )


def test_a_larger_alpha_costs_a_miss_more_and_a_false_positive_less() -> None:
    masks = torch.zeros(1, 1, 1, 10)
    masks[0, 0, 0, :4] = 1
    missed = torch.where(masks.bool(), BIG, -BIG)
    missed[0, 0, 0, 0] = -BIG  # one mass pixel missed
    false_positive = torch.where(masks.bool(), BIG, -BIG)
    false_positive[0, 0, 0, 9] = BIG  # one pixel too many
    dice, tversky = (
        DiceBCELoss(bce_weight=0),
        DiceBCELoss(bce_weight=0, alpha=0.7, beta=0.3),
    )
    valid = ones(1, 1, 1, 10)
    assert tversky(missed, masks, valid) > dice(missed, masks, valid)
    assert tversky(false_positive, masks, valid) < dice(false_positive, masks, valid)


def test_a_gamma_under_1_raises_the_loss_of_nearly_right_images_most() -> None:
    logits, masks = random_batch()
    valid = ones(1, 1, 8, 8)
    plain = DiceBCELoss(bce_weight=0, alpha=0.7, beta=0.3)
    focal = DiceBCELoss(bce_weight=0, **TVERSKY)
    for image in range(3):
        one = logits[image : image + 1], masks[image : image + 1]
        region = plain(*one, valid).item()
        assert focal(*one, valid).item() == pytest.approx(region**0.75)
        assert focal(*one, valid).item() > region


@pytest.mark.parametrize("has_mass", [True, False])
def test_a_perfect_image_gets_a_finite_gradient_with_a_gamma_under_1(
    has_mass: bool,
) -> None:
    masks = torch.zeros(1, 1, 3, 3)
    if has_mass:
        masks[0, 0, 1, 1] = 1
    # Saturated in float32: the probabilities are exactly 0 and 1, so 1 - index is 0.
    logits = torch.where(masks.bool(), 200.0, -200.0).requires_grad_()
    loss = DiceBCELoss(bce_weight=0, **TVERSKY)(logits, masks, ones(1, 1, 3, 3))
    loss.backward()
    assert loss.item() == pytest.approx(0, abs=1e-4)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_the_focal_tversky_loss_from_probabilities_is_the_loss_from_logits() -> None:
    logits, masks = random_batch()
    valid = torch.rand(3, 1, 8, 8) > 0.2
    loss = DiceBCELoss(**TVERSKY)
    assert loss.from_probabilities(
        torch.sigmoid(logits), masks, valid
    ).item() == pytest.approx(loss(logits, masks, valid).item(), rel=1e-5)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"alpha": 0.0}, "alpha and beta"),
        ({"beta": 1.5}, "alpha and beta"),
        ({"gamma": 0.0}, "gamma must be positive"),
    ],
)
def test_invalid_tversky_settings_are_rejected(
    arguments: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DiceBCELoss(**arguments)


def test_the_loss_shows_its_settings() -> None:
    assert "alpha=0.7, beta=0.3, gamma=0.75" in repr(DiceBCELoss(**TVERSKY))
