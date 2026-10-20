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
