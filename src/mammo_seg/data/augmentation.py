"""Random augmentation of training samples, applied to image and mask together.

Built on torchvision ``transforms.v2``: geometric changes (rotation, scale and
shift) move the image and its mask the same way, with the mask resampled by
nearest neighbour so it stays 0/1. Intensity changes (brightness, contrast, gamma,
noise) only touch the image. An optional crop then cuts a patch out of both. The
masses cover well under 1% of an image, so a uniformly random patch seldom holds
one: with probability ``crop_mass_fraction`` the patch is placed around a randomly
chosen mass pixel (when the image still has one after the geometric changes). An
image smaller than the patch gives a smaller patch on that side; the dataset pads
it to the canvas and marks the padding as not valid.

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
    # A patch of (height, width) instead of the whole image; off by default.
    crop_size: (
        tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]] | None
    ) = None
    # Chance that the patch contains a mass pixel, for an image that has a mass;
    # otherwise the patch is placed at random. Used with ``crop_size`` only.
    crop_mass_fraction: Annotated[float, Field(ge=0, le=1)] = 0.5


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
        mask_result = moved_mask.as_subclass(torch.Tensor)
        if self.settings.crop_size is not None:
            return self._crop(result, mask_result)
        return result, mask_result

    def _crop(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A patch of ``crop_size``, or the whole image on a side that is smaller."""
        assert self.settings.crop_size is not None
        height, width = image.shape[-2:]
        rows = min(self.settings.crop_size[0], height)
        cols = min(self.settings.crop_size[1], width)
        top = int(torch.randint(0, height - rows + 1, ()))
        left = int(torch.randint(0, width - cols + 1, ()))
        masses = torch.nonzero(mask[0] > 0.5)
        if len(masses) and torch.rand(()).item() < self.settings.crop_mass_fraction:
            row, col = masses[int(torch.randint(0, len(masses), ()))].tolist()
            # The chosen pixel lands at a random place of the patch.
            top = min(max(row - int(torch.randint(0, rows, ())), 0), height - rows)
            left = min(max(col - int(torch.randint(0, cols, ())), 0), width - cols)
        window = (slice(None), slice(top, top + rows), slice(left, left + cols))
        return image[window], mask[window]
