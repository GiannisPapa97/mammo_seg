"""Random augmentation of training samples, applied to image and mask together.

Built on torchvision ``transforms.v2``: geometric changes (rotation, scale and
shift, then an optional elastic deformation) move the image and its mask the same
way, with the mask resampled by nearest neighbour so it stays 0/1. Intensity
changes (brightness, contrast, gamma, noise) only touch the image. An optional crop
then cuts a patch out of both. The
masses cover well under 1% of an image, so a uniformly random patch seldom holds
one: with probability ``crop_mass_fraction`` the patch is placed around a randomly
chosen mass pixel (when the image still has one after the geometric changes). An
image smaller than the patch gives a smaller patch on that side; the dataset pads
it to the canvas and marks the padding as not valid.

The background of the images is exactly 0 (the scanner sets the air around the
breast to 0), and so is the area that a rotation or shift brings in. Validation and
inference see that background flat, so it stays exactly 0 here too: pixels that are
0 after the geometric changes are set back to 0 after the intensity changes, which
therefore only change the breast. Noise or a contrast shift on the background would
give training an input that validation never has, and leave the network untrained
on the flat background it meets there.

The elastic deformation bends the image and its masses smoothly, so the model sees
other shapes of the same masses. Random displacements are drawn on a coarse grid of
control points, ``elastic_spacing`` pixels apart, and interpolated smoothly
(bicubic) to every pixel. torchvision's ``ElasticTransform`` blurs a random field
of the full size instead, which takes about 6 s per 1024 x 800 image on a CPU core,
too slow for a data loader; the coarse grid takes a small fraction of that.

There are no flips: preprocessing puts every chest wall on the left, and flipping
would show the model breasts it never sees at inference. That holds for a flip from
top to bottom too, which would put the pectoral muscle of the MLO views at the
bottom.

A crop can also be asked for by kind, to build batches of a known composition: a
``"mass"`` patch contains at least one mass pixel (when the image still has one) and
a ``"negative"`` patch contains none, when a patch can avoid the mass, and is centred
on the breast, so it is not just background. Without a kind the patch is placed as
described above.

Augmentation is only for training. It uses PyTorch's random number generator, so
seeding PyTorch (the data loaders do) makes it reproducible.
"""

import math
from typing import Annotated, Literal

import torch
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from torch.nn import functional
from torchvision import tv_tensors
from torchvision.transforms import v2

PatchKind = Literal["mass", "negative"]

# Pixels brighter than this (on the 0..1 scale) are breast: the background of the
# preprocessed images is close to 0 and the breast is far above it. This is the
# threshold of the preprocessing, 100 of 4095, rounded up; on the INbreast images
# the two differ on 0.1% of the breast pixels.
BREAST_THRESHOLD = 0.05
# Tries to place a negative patch where it holds no mass pixel; if all fail, the try
# with the fewest mass pixels is used. All can fail whenever the patch is large
# against the image, for example a mass in the middle rows of a narrow image seen
# through a patch half as high as the image, so such a patch can hold a sliver of
# the mass.
NEGATIVE_TRIES = 20


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
    # Elastic deformation: the standard deviation of the displacement at the control
    # points, in pixels (0 is off; over all pixels it is about 0.85 of this), and
    # the distance between the control points.
    elastic: Annotated[float, Field(ge=0)] = 0.0
    elastic_spacing: Annotated[int, Field(ge=2)] = 64
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
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        kind: PatchKind | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment, and cut a patch of ``crop_size`` if it is set.

        ``kind`` asks for a patch with or without a mass (see the module docstring)
        and needs ``crop_size``; ValueError otherwise.
        """
        if kind is not None and self.settings.crop_size is None:
            raise ValueError(f"a {kind!r} patch needs crop_size to be set")
        moved_image, moved_mask = self.geometric(
            tv_tensors.Image(image), tv_tensors.Mask(mask)
        )
        if self.settings.elastic:
            bent_image, bent_mask = elastic(
                moved_image.as_subclass(torch.Tensor),
                moved_mask.as_subclass(torch.Tensor),
                self.settings.elastic,
                self.settings.elastic_spacing,
            )
            moved_image = tv_tensors.Image(bent_image)
            moved_mask = tv_tensors.Mask(bent_mask)
        background = moved_image.as_subclass(torch.Tensor) == 0
        changed = self.intensity(moved_image)
        if self.settings.gamma:
            low, high = 1 - self.settings.gamma, 1 + self.settings.gamma
            exponent = low + (high - low) * torch.rand(()).item()
            changed = changed.clamp(min=0) ** exponent
        # The background stays exactly 0, as validation and inference see it.
        result = (
            changed.as_subclass(torch.Tensor)
            .clamp(0.0, 1.0)
            .masked_fill(background, 0.0)
        )
        mask_result = moved_mask.as_subclass(torch.Tensor)
        if self.settings.crop_size is not None:
            return self._crop(result, mask_result, kind)
        return result, mask_result

    def _crop(
        self, image: torch.Tensor, mask: torch.Tensor, kind: PatchKind | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A patch of ``crop_size``, or the whole image on a side that is smaller."""
        assert self.settings.crop_size is not None
        height, width = image.shape[-2:]
        rows = min(self.settings.crop_size[0], height)
        cols = min(self.settings.crop_size[1], width)
        top = int(torch.randint(0, height - rows + 1, ()))
        left = int(torch.randint(0, width - cols + 1, ()))
        masses = torch.nonzero(mask[0] > 0.5)
        if kind == "negative":
            top, left = self._negative_window(image, mask, rows, cols)
        elif kind == "mass":
            # A mass that the geometric changes pushed out of the image leaves a
            # patch at random.
            if len(masses):
                top, left = _around(masses, rows, cols, height, width)
        elif len(masses) and torch.rand(()).item() < self.settings.crop_mass_fraction:
            top, left = _around(masses, rows, cols, height, width)
        window = (slice(None), slice(top, top + rows), slice(left, left + cols))
        return image[window], mask[window]

    @staticmethod
    def _negative_window(
        image: torch.Tensor, mask: torch.Tensor, rows: int, cols: int
    ) -> tuple[int, int]:
        """The corner of a patch centred on a random breast pixel, without mass.

        A patch with a mass pixel is drawn again, up to ``NEGATIVE_TRIES`` times.
        """
        height, width = image.shape[-2:]
        breast = torch.nonzero(image[0] > BREAST_THRESHOLD)
        best: tuple[float, int, int] | None = None
        for _ in range(NEGATIVE_TRIES):
            if len(breast):
                row, col = breast[int(torch.randint(0, len(breast), ()))].tolist()
            else:  # no breast found (a blank image): anywhere
                row = int(torch.randint(0, height, ()))
                col = int(torch.randint(0, width, ()))
            top = min(max(row - rows // 2, 0), height - rows)
            left = min(max(col - cols // 2, 0), width - cols)
            held = float(mask[0, top : top + rows, left : left + cols].sum())
            if held == 0:
                return top, left
            if best is None or held < best[0]:
                best = (held, top, left)
        assert best is not None
        return best[1], best[2]


def elastic_displacement(
    height: int, width: int, strength: float, spacing: int
) -> torch.Tensor:
    """A smooth random displacement of every pixel, ``[2, height, width]`` (rows,
    then columns), in pixels.

    Drawn from a normal distribution with a standard deviation of ``strength`` at
    control points ``spacing`` pixels apart, which cover the image, and interpolated
    bicubically in between.
    """
    points = (math.ceil(height / spacing) + 1, math.ceil(width / spacing) + 1)
    coarse = torch.randn(1, 2, *points) * strength
    # The outer control points lie on the borders, so they are at most ``spacing``
    # pixels apart.
    return functional.interpolate(
        coarse, size=(height, width), mode="bicubic", align_corners=True
    )[0]


def elastic(
    image: torch.Tensor, mask: torch.Tensor, strength: float, spacing: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bend a ``[1, H, W]`` image and its mask by the same smooth random
    displacement (see ``elastic_displacement``): the image is resampled
    bilinearly, the mask by nearest neighbour, so it stays 0/1. What comes in from
    outside the image is 0."""
    height, width = image.shape[-2:]
    shift = elastic_displacement(height, width, strength, spacing)
    rows = torch.arange(height, dtype=torch.float32)[:, None] + shift[0]
    cols = torch.arange(width, dtype=torch.float32)[None, :] + shift[1]
    # grid_sample takes (x, y) in -1..1, with align_corners the centres of the
    # first and last pixels.
    grid = torch.stack(
        (cols * 2 / max(width - 1, 1) - 1, rows * 2 / max(height - 1, 1) - 1), dim=-1
    )[None]

    def sample(values: torch.Tensor, mode: str) -> torch.Tensor:
        return functional.grid_sample(
            values[None].float(),
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True,
        )[0].to(values.dtype)

    return sample(image, "bilinear"), sample(mask, "nearest")


def _around(
    pixels: torch.Tensor, rows: int, cols: int, height: int, width: int
) -> tuple[int, int]:
    """The corner of a patch that holds a random one of the ``pixels``, at a random
    place of the patch (the patch stays inside the image)."""
    row, col = pixels[int(torch.randint(0, len(pixels), ()))].tolist()
    top = min(max(row - int(torch.randint(0, rows, ())), 0), height - rows)
    left = min(max(col - int(torch.randint(0, cols, ())), 0), width - cols)
    return top, left
