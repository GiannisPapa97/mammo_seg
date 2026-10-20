"""Random augmentation of training samples, applied to image and mask together.

Built on torchvision ``transforms.v2``: geometric changes (rotation, scale, shift
and an optional random crop) move the image and its mask the same way, with the
mask resampled by nearest neighbour so it stays 0/1. Intensity changes
(brightness, contrast, gamma, noise) only touch the image.

There are no flips: preprocessing puts every chest wall on the left, and flipping
would show the model breasts it never sees at inference.

Augmentation is only for training. It uses PyTorch's random number generator, so
seeding PyTorch (the data loaders do) makes it reproducible.
"""

from typing import Annotated

import torch
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from torchvision import tv_tensors
from torchvision.transforms import v2


@dataclass(frozen=True, config=ConfigDict(strict=True, extra="forbid"))
class AugmentationSettings:
    """How strong every random change is; 0 switches a change off."""

    rotation: Annotated[float, Field(ge=0, le=180)] = 10.0  # degrees, either way
    scale: Annotated[float, Field(ge=0, lt=1)] = 0.1  # zoom by 1 +- scale
    shift: Annotated[float, Field(ge=0, le=1)] = 0.1  # of the width and height
    brightness: Annotated[float, Field(ge=0, lt=1)] = 0.1  # multiply by 1 +- this
    contrast: Annotated[float, Field(ge=0, lt=1)] = 0.1
    gamma: Annotated[float, Field(ge=0, lt=1)] = 0.2  # exponent in 1 +- gamma
    noise: Annotated[float, Field(ge=0)] = 0.01  # standard deviation
    # A random patch of (height, width) instead of the whole image; off by default.
    crop_size: (
        tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]] | None
    ) = None


class Augmentation:
    """A transform for ``MammogramDataset``: ``(image, mask) -> (image, mask)``.

    Both are ``[1, H, W]`` float tensors, the image in 0..1 and the mask 0/1. The
    result has the same shape, or ``crop_size`` when it is set.
    """

    def __init__(self, settings: AugmentationSettings = AugmentationSettings()) -> None:  # noqa: B008
        self.settings = settings
        geometric: list[v2.Transform] = []
        if settings.rotation or settings.scale or settings.shift:
            geometric.append(
                v2.RandomAffine(
                    degrees=settings.rotation,
                    translate=(settings.shift, settings.shift),
                    scale=(1 - settings.scale, 1 + settings.scale),
                    interpolation=v2.InterpolationMode.BILINEAR,
                )
            )
        if settings.crop_size is not None:
            geometric.append(v2.RandomCrop(settings.crop_size, pad_if_needed=True))
        intensity: list[v2.Transform] = []
        if settings.brightness or settings.contrast:
            intensity.append(
                v2.ColorJitter(
                    brightness=settings.brightness, contrast=settings.contrast
                )
            )
        if settings.noise:
            intensity.append(
                v2.GaussianNoise(mean=0.0, sigma=settings.noise, clip=True)
            )
        # Compose needs at least one transform; Identity stands for none.
        self.geometric = v2.Compose(geometric or [v2.Identity()])
        self.intensity = v2.Compose(intensity or [v2.Identity()])

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        moved_image, moved_mask = self.geometric(
            tv_tensors.Image(image), tv_tensors.Mask(mask)
        )
        changed = self.intensity(moved_image)
        if self.settings.gamma:
            low, high = 1 - self.settings.gamma, 1 + self.settings.gamma
            exponent = low + (high - low) * torch.rand(()).item()
            changed = changed.clamp(min=0) ** exponent
        result = changed.as_subclass(torch.Tensor).clamp(0.0, 1.0)
        return result, moved_mask.as_subclass(torch.Tensor)
