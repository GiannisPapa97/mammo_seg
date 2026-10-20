"""Preprocessing of mammograms: from a raw DICOM image to the model input.

The chain is: read the DICOM, crop to the breast, rescale so the longer side has a
fixed length, flip right breasts so every chest wall is on the same side, and
normalize the intensities. The same code runs for training (where the first steps
are cached on disk) and for inference, so both see identical input.

This module grows step by step; it starts with finding the breast.
"""

from typing import Any, Self, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass
from scipy import ndimage

# The raw INbreast images are 12-bit values in a 16-bit container. The background
# is close to 0 and breast tissue is far above it.
DEFAULT_THRESHOLD = 100
DEFAULT_MARGIN = 0.02  # of the breast box size, added on every side
DEFAULT_DOWNSCALE = 8  # the breast mask is made on every 8th pixel
DEFAULT_OPENING = 1  # removes bright strips thinner than about 3 mask pixels

ScalarT = TypeVar("ScalarT", bound=np.generic)


@dataclass(frozen=True, config=ConfigDict(strict=True, extra="forbid"))
class BoundingBox:
    """A rectangle of pixels as slice bounds: rows top:bottom, columns left:right."""

    top: int
    bottom: int
    left: int
    right: int

    @model_validator(mode="after")
    def _check_valid(self) -> Self:
        if self.top < 0 or self.left < 0:
            raise ValueError(f"a box cannot start before the image: {self}")
        if self.bottom <= self.top or self.right <= self.left:
            raise ValueError(f"a box must have a positive size: {self}")
        return self

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left

    def crop(self, pixels: npt.NDArray[ScalarT]) -> npt.NDArray[ScalarT]:
        """The part of a 2-D array inside the box (a view, not a copy)."""
        rows, cols = pixels.shape
        if self.bottom > rows or self.right > cols:
            raise ValueError(f"the box {self} does not fit an image of {pixels.shape}")
        return pixels[self.top : self.bottom, self.left : self.right]


def breast_bbox(
    pixels: npt.NDArray[np.number[Any]],
    threshold: int = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
    downscale: int = DEFAULT_DOWNSCALE,
    opening: int = DEFAULT_OPENING,
) -> BoundingBox:
    """The smallest box around the breast, plus a margin.

    The breast is the largest connected region (8-connected) of pixels brighter
    than ``threshold``. Taking the largest region keeps stray bright pixels,
    labels and markers in the background from widening the box. The mask is made
    on every ``downscale``-th pixel, which is fast and good enough for a box (its
    edges are accurate to ``downscale`` pixels, far less than the margin), and is
    ``opening`` times morphologically opened to cut off thin bright strips along
    the image border that touch the breast. ``margin`` is a fraction of the box
    height and width added on every side, limited to the image.

    Raises ValueError for an image that is not 2-D, invalid arguments, or an image
    with no breast.
    """
    if pixels.ndim != 2:
        raise ValueError(f"expected a 2-D image, got {pixels.ndim} dimensions")
    if margin < 0:
        raise ValueError(f"the margin cannot be negative, got {margin}")
    if downscale < 1 or opening < 0:
        raise ValueError("downscale must be at least 1 and opening at least 0")
    mask = pixels[::downscale, ::downscale] > threshold
    if opening:
        mask = ndimage.binary_opening(
            mask, structure=np.ones((3, 3)), iterations=opening
        )
    regions, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        raise ValueError(f"no breast found: no region of pixels above {threshold}")
    sizes = np.bincount(regions.ravel())[1:]
    rows, cols = ndimage.find_objects(regions)[int(sizes.argmax())]
    rows_max, cols_max = pixels.shape
    top, bottom = rows.start * downscale, min(rows_max, rows.stop * downscale)
    left, right = cols.start * downscale, min(cols_max, cols.stop * downscale)
    pad_rows = round(margin * (bottom - top))
    pad_cols = round(margin * (right - left))
    return BoundingBox(
        top=max(0, top - pad_rows),
        bottom=min(rows_max, bottom + pad_rows),
        left=max(0, left - pad_cols),
        right=min(cols_max, right + pad_cols),
    )
