"""The training loss: soft Dice plus binary cross-entropy, on valid pixels only.

Masses cover about 1% of a positive image and none of a negative one. The Dice
part measures the overlap of prediction and mask per image, so it is not swamped
by the background; the cross-entropy part gives a smooth signal on every pixel,
which keeps training stable early on. Pixels where ``valid`` is false count in
neither. Training passes every pixel as valid, so the padding counts as background
(see ``train.py``); the metrics leave the padding out.

The Dice part is computed per image and averaged, like the metric. A smoothing
term keeps it defined on images without a mass: predicting nothing there gives a
Dice of 1 and no loss, matching the metric's convention.
"""

import torch
from torch import nn
from torch.nn import functional

DEFAULT_SMOOTH = 1.0


class DiceBCELoss(nn.Module):
    """``dice_weight * (1 - soft Dice) + bce_weight * cross-entropy``.

    Takes raw model outputs (logits), 0/1 masks and the valid pixels, all
    ``[B, 1, H, W]``, and returns a scalar.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = DEFAULT_SMOOTH,
    ) -> None:
        super().__init__()
        if dice_weight < 0 or bce_weight < 0 or dice_weight + bce_weight == 0:
            raise ValueError(
                "the weights must not be negative and not both 0, got "
                f"{dice_weight} and {bce_weight}"
            )
        if smooth <= 0:
            raise ValueError(f"the smoothing term must be positive, got {smooth}")
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, masks: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        if not logits.shape == masks.shape == valid.shape:
            raise ValueError(
                f"shapes differ: logits {tuple(logits.shape)}, masks "
                f"{tuple(masks.shape)}, valid {tuple(valid.shape)}"
            )
        weight = valid.to(logits.dtype)
        if weight.sum() == 0:
            raise ValueError("no valid pixel to compute the loss on")
        targets = masks.to(logits.dtype)
        bce = functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        bce_loss = (bce * weight).sum() / weight.sum()
        probabilities = torch.sigmoid(logits) * weight
        targets = targets * weight
        overlap = (probabilities * targets).flatten(1).sum(1)
        total = probabilities.flatten(1).sum(1) + targets.flatten(1).sum(1)
        dice = (2 * overlap + self.smooth) / (total + self.smooth)
        dice_loss = (1 - dice).mean()
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
