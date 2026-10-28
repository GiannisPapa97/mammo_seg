"""Preprocessing of mammograms: from a raw image to the model input.

The chain is: read the image (DICOM, or JPEG for CBIS-DDSM), crop to the breast,
rescale so the longer side has a fixed length, flip so every chest wall is on the
left, and normalize the intensities. The same code runs for training (where the
first steps are cached on disk) and for inference, so both see identical input.

``preprocess`` runs the first four steps on an image and ``preprocess_mask`` applies
the same geometry to its mask (``preprocess_points`` to points, such as the
polygons of the annotations). Both return or use a ``PreprocessParams`` record
(crop box, output size, flip) that ``restore_mask`` uses to map a predicted mask
back onto the original image.
"""

from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self, TypeVar

import numpy as np
import numpy.typing as npt
import pydicom
from PIL import Image
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from scipy import ndimage

# The raw INbreast images are 12-bit values in a 16-bit container. The background
# is close to 0 and breast tissue is far above it.
DEFAULT_THRESHOLD = 100
DEFAULT_MARGIN = 0.02  # of the breast box size, added on every side
DEFAULT_DOWNSCALE = 8  # the breast mask is made on every 8th pixel
DEFAULT_OPENING = 1  # removes bright strips thinner than about 3 mask pixels
DEFAULT_SIZE = 1024  # length of the longer side after rescaling
DEFAULT_MAX_INTENSITY = 4095  # the 12-bit range of the INbreast images
IMAGE_SUFFIXES: Final = (".jpg", ".jpeg", ".png")  # read as images, the rest as DICOM

# Right breasts are flipped so the chest wall is always on the left, where it
# already is in the left breasts (``orientation="laterality"``, INbreast).
FLIPPED_LATERALITY: Final = "R"

ScalarT = TypeVar("ScalarT", bound=np.generic)
Pixels = npt.NDArray[np.uint16]
Mask = npt.NDArray[np.uint8]
Shape = tuple[int, int]  # (rows, columns)

_CONFIG = ConfigDict(strict=True, extra="forbid")


@dataclass(frozen=True, config=_CONFIG)
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


def breast_region(
    pixels: npt.NDArray[np.number[Any]],
    threshold: int = DEFAULT_THRESHOLD,
    downscale: int = DEFAULT_DOWNSCALE,
    opening: int = DEFAULT_OPENING,
    border: int = 0,
) -> npt.NDArray[np.bool_]:
    """The breast on every ``downscale``-th pixel of the image, as a boolean mask.

    The breast is the largest connected region (8-connected) of pixels brighter
    than ``threshold``. Taking the largest region keeps stray bright pixels,
    labels and markers in the background out of it. The mask is ``opening`` times
    morphologically opened to cut off thin bright strips along the image border
    that touch the breast, and the outer ``border`` mask pixels on every side are
    left out first, for the bright film edges of scanned mammograms.

    Raises ValueError for an image that is not 2-D, invalid arguments, or an image
    with no breast.
    """
    if pixels.ndim != 2:
        raise ValueError(f"expected a 2-D image, got {pixels.ndim} dimensions")
    if downscale < 1 or opening < 0 or border < 0:
        raise ValueError(
            "downscale must be at least 1, opening at least 0 and border at least 0"
        )
    mask = pixels[::downscale, ::downscale] > threshold
    if border:
        mask[:border] = mask[-border:] = False
        mask[:, :border] = mask[:, -border:] = False
    if opening:
        mask = ndimage.binary_opening(
            mask, structure=np.ones((3, 3)), iterations=opening
        )
    regions, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        raise ValueError(f"no breast found: no region of pixels above {threshold}")
    sizes = np.bincount(regions.ravel())[1:]
    region: npt.NDArray[np.bool_] = regions == int(sizes.argmax()) + 1
    return region


def region_bbox(
    region: npt.NDArray[np.bool_],
    shape: Shape,
    downscale: int = DEFAULT_DOWNSCALE,
    margin: float = DEFAULT_MARGIN,
) -> BoundingBox:
    """The smallest box around a ``breast_region`` in an image of ``shape``, plus a
    margin.

    The edges are accurate to ``downscale`` pixels, far less than the margin.
    ``margin`` is a fraction of the box height and width added on every side,
    limited to the image. Raises ValueError for a negative margin.
    """
    if margin < 0:
        raise ValueError(f"the margin cannot be negative, got {margin}")
    rows, cols = ndimage.find_objects(region.astype(np.uint8))[0]
    rows_max, cols_max = shape
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


def breast_bbox(
    pixels: npt.NDArray[np.number[Any]],
    threshold: int = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
    downscale: int = DEFAULT_DOWNSCALE,
    opening: int = DEFAULT_OPENING,
    border: int = 0,
) -> BoundingBox:
    """The smallest box around the breast (``breast_region``), plus a margin.

    Raises ValueError for an image that is not 2-D, invalid arguments, or an image
    with no breast.
    """
    if margin < 0:
        raise ValueError(f"the margin cannot be negative, got {margin}")
    region = breast_region(pixels, threshold, downscale, opening, border)
    return region_bbox(region, (pixels.shape[0], pixels.shape[1]), downscale, margin)


def chest_wall_on_right(
    pixels: npt.NDArray[np.number[Any]],
    region: npt.NDArray[np.bool_],
    downscale: int = DEFAULT_DOWNSCALE,
) -> bool:
    """Whether the chest wall is on the right of the image, from the image itself.

    The breast is thickest, so brightest, at the chest wall: the wall is on the
    side whose half of the image holds more of the intensity of the breast region.
    For datasets whose orientation does not follow the laterality (CBIS-DDSM).
    """
    breast = np.where(region, pixels[::downscale, ::downscale], 0).astype(np.float64)
    half = breast.shape[1] // 2
    return bool(breast[:, breast.shape[1] - half :].sum() > breast[:, :half].sum())


@dataclass(frozen=True, config=_CONFIG)
class PreprocessingSettings:
    """Everything that decides what the preprocessing does.

    Training and inference must use the same settings, so they are stored with the
    preprocessed data and with a trained model.
    """

    threshold: Annotated[int, Field(ge=0)] = DEFAULT_THRESHOLD
    margin: Annotated[float, Field(ge=0)] = DEFAULT_MARGIN
    downscale: Annotated[int, Field(ge=1)] = DEFAULT_DOWNSCALE
    opening: Annotated[int, Field(ge=0)] = DEFAULT_OPENING
    size: Annotated[int, Field(ge=1)] = DEFAULT_SIZE
    max_intensity: Annotated[int, Field(ge=1)] = DEFAULT_MAX_INTENSITY
    # Per-image scaling by this percentile of the tissue instead of max_intensity.
    percentile: Annotated[float, Field(gt=0, le=100)] | None = None
    # Mask pixels left out on every side when finding the breast.
    border: Annotated[int, Field(ge=0)] = 0
    # How the side of the chest wall is known: from the laterality (right breasts
    # are flipped) or from the image (``chest_wall_on_right``).
    orientation: Literal["laterality", "image"] = "laterality"


@dataclass(frozen=True, config=_CONFIG)
class PreprocessParams:
    """What was done to one image: enough to map a mask back onto the original."""

    source_shape: tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]]
    box: BoundingBox  # the crop, in the original image
    shape: tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]]  # the result
    flipped: bool

    @model_validator(mode="after")
    def _check_box_fits(self) -> Self:
        rows, cols = self.source_shape
        if self.box.bottom > rows or self.box.right > cols:
            raise ValueError(f"the box {self.box} does not fit {self.source_shape}")
        return self


def read_pixels(path: Path) -> Pixels:
    """The raw pixel values of a mammogram as a 2-D ``uint16`` array.

    A file with one of the ``IMAGE_SUFFIXES`` is read as an image, which must be
    8-bit grey (the JPEGs of CBIS-DDSM); its values stay 0 to 255. Any other file is
    read as DICOM, and only what INbreast contains is supported: one channel,
    unsigned integers up to 16 bits, bright tissue on a dark background
    (``MONOCHROME2``) and no rescale slope or intercept. Anything else raises
    ValueError instead of being guessed.
    """
    if path.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError(
                    f"{path.name}: expected an 8-bit grey image, got mode {image.mode}"
                )
            return np.asarray(image).astype(np.uint16)
    dataset = pydicom.dcmread(path)
    if dataset.get("SamplesPerPixel", 1) != 1:
        raise ValueError(f"{path.name}: expected one channel")
    if dataset.get("PhotometricInterpretation") != "MONOCHROME2":
        raise ValueError(
            f"{path.name}: expected MONOCHROME2, got "
            f"{dataset.get('PhotometricInterpretation')!r}"
        )
    if float(dataset.get("RescaleSlope", 1)) != 1 or float(
        dataset.get("RescaleIntercept", 0)
    ):
        raise ValueError(f"{path.name}: a rescale slope or intercept is not supported")
    pixels = dataset.pixel_array
    if pixels.ndim != 2 or pixels.dtype.kind != "u" or pixels.dtype.itemsize > 2:
        raise ValueError(
            f"{path.name}: expected a 2-D unsigned image of at most 16 bits, "
            f"got {pixels.dtype} with shape {pixels.shape}"
        )
    return pixels.astype(np.uint16, copy=False)


def target_shape(height: int, width: int, size: int) -> Shape:
    """The shape of a ``height`` x ``width`` image whose longer side is ``size``."""
    scale = size / max(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def _resize(
    array: npt.NDArray[np.generic],
    shape: Shape,
    resample: Image.Resampling = Image.Resampling.BOX,
) -> npt.NDArray[np.float32]:
    """Resize a 2-D array to ``shape`` (rows, columns) in floating point.

    The default filter averages the source pixels that fall in each target pixel,
    which is the right choice for shrinking.
    """
    image = Image.fromarray(array.astype(np.float32), mode="F")
    resized = image.resize((shape[1], shape[0]), resample=resample)
    return np.asarray(resized, dtype=np.float32)


def preprocess(
    pixels: npt.NDArray[np.number[Any]],
    laterality: Literal["L", "R"],
    settings: PreprocessingSettings = PreprocessingSettings(),  # noqa: B008
) -> tuple[Pixels, PreprocessParams]:
    """Crop to the breast, rescale so the longer side is ``settings.size`` and flip
    so the chest wall is on the left: right breasts, or the images whose chest wall
    is on the right (``settings.orientation``). Returns the image (``uint16``,
    intensities unchanged apart from the averaging of the rescale) and what was
    done.

    Raises ValueError for an image with no breast or an unknown laterality.
    """
    if laterality not in ("L", "R"):
        raise ValueError(f"laterality must be L or R, got {laterality!r}")
    region = breast_region(
        pixels,
        settings.threshold,
        settings.downscale,
        settings.opening,
        settings.border,
    )
    box = region_bbox(
        region, (pixels.shape[0], pixels.shape[1]), settings.downscale, settings.margin
    )
    shape = target_shape(box.height, box.width, settings.size)
    resized = _resize(box.crop(pixels), shape)
    result = np.clip(np.rint(resized), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if settings.orientation == "laterality":
        flipped = laterality == FLIPPED_LATERALITY
    else:
        flipped = chest_wall_on_right(pixels, region, settings.downscale)
    if flipped:
        result = np.ascontiguousarray(result[:, ::-1])
    params = PreprocessParams(
        source_shape=(pixels.shape[0], pixels.shape[1]),
        box=box,
        shape=shape,
        flipped=flipped,
    )
    return result, params


def preprocess_mask(mask: Mask, params: PreprocessParams) -> Mask:
    """Apply the geometry of ``preprocess`` to the 0/1 mask of the same image.

    The mask is averaged down like the image and set where at least half of it is
    covered, so it stays binary. Raises ValueError if the mask does not have the
    shape of the source image, is not 0/1, or has pixels outside the crop box,
    which would cut a lesion.
    """
    if mask.shape != params.source_shape:
        raise ValueError(
            f"mask shape {mask.shape} is not the image {params.source_shape}"
        )
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("the mask must contain only 0 and 1")
    cropped = params.box.crop(mask)
    if int(cropped.sum()) != int(mask.sum()):
        raise ValueError("the crop box cuts the mask: part of a lesion is outside it")
    result = (_resize(cropped, params.shape) >= 0.5).astype(np.uint8)
    return np.ascontiguousarray(result[:, ::-1]) if params.flipped else result


def preprocess_points(
    points: npt.ArrayLike, params: PreprocessParams
) -> npt.NDArray[np.float64]:
    """Apply the geometry of ``preprocess`` to ``[n, 2]`` points ``(x, y)`` of the
    source image, such as the polygon of an ROI.

    Coordinates are continuous: pixel ``i`` spans ``[i, i + 1)``, as for the
    polygons. Points outside the crop box map outside the result. Raises ValueError
    for points that are not ``[n, 2]``.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"expected [n, 2] points (x, y), got shape {array.shape}")
    box = params.box
    rows, cols = params.shape
    x = (array[:, 0] - box.left) * cols / box.width
    y = (array[:, 1] - box.top) * rows / box.height
    if params.flipped:
        x = cols - x
    return np.column_stack([x, y])


def restore_mask(mask: Mask, params: PreprocessParams) -> Mask:
    """Map a 0/1 mask of a preprocessed image back onto the original image.

    The reverse of ``preprocess_mask``: undo the flip, scale up to the crop box and
    place it in an empty mask of the original shape.
    """
    if mask.shape != params.shape:
        raise ValueError(f"mask shape {mask.shape} is not the result {params.shape}")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("the mask must contain only 0 and 1")
    unflipped = mask[:, ::-1] if params.flipped else mask
    box = params.box
    scaled = _resize(unflipped, (box.height, box.width), Image.Resampling.BILINEAR)
    restored = np.zeros(params.source_shape, dtype=np.uint8)
    restored[box.top : box.bottom, box.left : box.right] = scaled >= 0.5
    return restored


def normalize_intensity(
    pixels: npt.NDArray[np.number[Any]],
    settings: PreprocessingSettings = PreprocessingSettings(),  # noqa: B008
) -> npt.NDArray[np.float32]:
    """Scale an image to 0..1 and clip at 1.

    By default every pixel is divided by ``settings.max_intensity`` (4095, the
    12-bit range of the INbreast images), which keeps absolute intensities. On the
    dataset this gives tissue of more similar brightness across images than scaling
    each image by its own bright percentile, because the brightest pixels are
    outliers such as calcifications and bright masses.

    With ``settings.percentile`` set, each image is instead divided by that
    percentile of its own tissue (the pixels above ``settings.threshold``), for
    data from other scanners. Nothing is learned from other images either way, so
    this is the same for training and inference and cannot leak between splits.

    This runs when an image is loaded, after ``preprocess``, and is not applied to
    the stored preprocessed images.

    Raises ValueError for an image that is not 2-D, or, in percentile mode, has no
    pixel above the threshold.
    """
    if pixels.ndim != 2:
        raise ValueError(f"expected a 2-D image, got {pixels.ndim} dimensions")
    if settings.percentile is None:
        high = float(settings.max_intensity)
    else:
        tissue = pixels[pixels > settings.threshold]
        if tissue.size == 0:
            raise ValueError(f"no tissue found: no pixel is above {settings.threshold}")
        high = float(np.percentile(tissue, settings.percentile))
    scaled: npt.NDArray[np.float32] = np.clip(
        pixels.astype(np.float32) / high, 0.0, 1.0
    )
    return scaled
