"""Predicting whole images: in one pass, or in overlapping windows.

A model trained on patches has only seen inputs of the patch size. Predicting a
whole image in one pass shows it more context, the real image borders and more
padding than it was trained on. **Sliding windows** of the training patch size
avoid that: the image is covered by windows that overlap by half (the last one
touches the far edge), each window is predicted on its own, and the probabilities
are averaged with Gaussian weights that trust the centre of a window more than its
borders, where a convolution sees padding instead of the image (nnU-Net does the
same, with a standard deviation of an eighth of the window).

A model trained on whole images predicts whole images in one pass (``"whole"``).
"""

from typing import Literal

import torch
from torch import nn

Inference = Literal["whole", "sliding"]
INFERENCES: tuple[Inference, ...] = ("whole", "sliding")
# The standard deviation of the Gaussian weights, as a share of the window side.
SIGMA_SCALE = 1 / 8


def window_starts(length: int, window: int) -> list[int]:
    """The starts of windows of ``window`` pixels that cover ``length`` pixels with
    an overlap of half a window; the last window ends at ``length``.

    Raises ValueError for a window or length below 1.
    """
    if window < 1 or length < 1:
        raise ValueError(
            f"the window and the length must be positive: {window}, {length}"
        )
    if length <= window:
        return [0]
    step = max(window // 2, 1)
    starts = list(range(0, length - window, step))
    return [*starts, length - window]


def gaussian_weights(shape: tuple[int, int]) -> torch.Tensor:
    """``[H, W]`` weights that are 1 at the centre and fall off towards the borders
    (a Gaussian with a standard deviation of ``SIGMA_SCALE`` of each side), never 0."""
    axes = []
    for side in shape:
        position = torch.arange(side, dtype=torch.float32) - (side - 1) / 2
        sigma = max(side * SIGMA_SCALE, 1e-6)
        axes.append(torch.exp(-0.5 * (position / sigma) ** 2))
    weights = axes[0][:, None] * axes[1][None, :]
    return (weights / weights.max()).clamp(min=1e-4)


def predict(
    model: nn.Module,
    images: torch.Tensor,
    inference: Inference = "whole",
    window: tuple[int, int] | None = None,
    amp: bool = False,
) -> torch.Tensor:
    """The probabilities of ``model`` for a ``[B, 1, H, W]`` batch, as ``[B, 1, H, W]``
    float32 on the device of ``images``.

    ``"whole"`` runs the model once on the batch; ``"sliding"`` runs it on windows of
    ``window`` (rows, columns; a side larger than the image is cut to it) and blends
    them (see the module docstring). Raises ValueError for sliding windows without
    a window, or an unknown inference.
    """
    device_type = images.device.type
    if inference == "whole":
        with torch.autocast(device_type, enabled=amp):
            logits = model(images)
        return torch.sigmoid(logits.float())
    if inference != "sliding":
        raise ValueError(f"unknown inference {inference!r}; choose one of {INFERENCES}")
    if window is None:
        raise ValueError("sliding windows need a window size")
    height, width = images.shape[-2:]
    rows, cols = min(window[0], height), min(window[1], width)
    weights = gaussian_weights((rows, cols)).to(images.device)
    total = torch.zeros(images.shape, dtype=torch.float32, device=images.device)
    norm = torch.zeros((height, width), dtype=torch.float32, device=images.device)
    for top in window_starts(height, rows):
        for left in window_starts(width, cols):
            area = (..., slice(top, top + rows), slice(left, left + cols))
            with torch.autocast(device_type, enabled=amp):
                logits = model(images[area])
            total[area] += torch.sigmoid(logits.float()) * weights
            norm[area[1:]] += weights
    return total / norm
