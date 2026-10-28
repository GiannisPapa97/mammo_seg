"""The manifest schema: what is stored for every image and its lesions.

One schema for every dataset. A manifest names its dataset and lists its images;
each image carries its metadata and every annotated lesion (ROI) with a type and
an outline, either a polygon or a mask image. Images left out of the dataset,
with the reason, are listed apart. It is stored as JSON::

    {"version": 2, "dataset": "inbreast", "images": [
      {"file_id": "20586908", "patient_id": "6c613a14b80a8591",
       "laterality": "R", "view": "CC",
       "image_path": "AllDICOMs/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm",
       "rows": 4084, "cols": 3328, "pixel_size_mm": 0.07,
       "acquisition_date": "200902", "acr": 2, "birads": "2",
       "official_split": null, "has_annotation": true,
       "rois": [{"lesion_type": "mass", "points": [[2964.6, 3428.0], ...]}, ...]},
      ...
    ], "dropped": []}

Every field of an image is always written, ``null`` where the dataset has no
value. The fields of an ROI that a dataset does not have (a mask for a polygon,
the pathology for INbreast) are left out of the file instead, so that the polygons
of INbreast stay as compact as they were.

The classes are frozen pydantic dataclasses in strict mode: an object cannot be
built, or a file read, with a wrong type, an out-of-range value or an unknown
field value. Pydantic
raises ``pydantic.ValidationError`` (a ``ValueError``) saying which value is
wrong.

Coordinates are pixels, ``(x, y)`` = ``(column, row)``. Facts that follow from
the ROIs (such as the number of masses) are computed, not stored.
"""

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_serializer,
    model_validator,
)
from pydantic.dataclasses import dataclass

SCHEMA_VERSION: Final = 2


class ManifestError(ValueError):
    """The raw files of a dataset are inconsistent. ``problems`` lists every issue.

    Raised by the manifest builder of every dataset.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("invalid dataset:\n- " + "\n- ".join(problems))


def describe(error: ValidationError) -> str:
    """The invalid fields of a validation error as one line."""
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors()
    )


Point = tuple[FiniteFloat, FiniteFloat]

# Constraints on the image fields. Ids name files, so they keep to letters,
# digits, ``_``, ``-`` and ``.``: INbreast has 8-digit file ids and 16-character
# hex patient ids, CBIS-DDSM folder names such as ``Mass-Training_P_00001_LEFT_CC``
# and patient ids such as ``P_00001``. Dates are a year and a semester.
FileId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
PatientId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_]*$")]
RelativePath = Annotated[str, StringConstraints(min_length=1)]
Pixels = Annotated[int, Field(gt=0)]
PixelSize = Annotated[float, Field(gt=0)]  # in mm
AcquisitionDate = Annotated[str, StringConstraints(pattern=r"^\d{6}$")]
Density = Annotated[int, Field(ge=1, le=4)]  # ACR breast density category
BiRads = Literal["1", "2", "3", "4a", "4b", "4c", "5", "6"]
Score = Annotated[int, Field(ge=0, le=5)]
Dataset = Literal["inbreast", "cbis-ddsm"]
Pathology = Literal["MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK"]

_CONFIG = ConfigDict(strict=True, extra="forbid")


class View(StrEnum):
    """The mammographic views present in the dataset."""

    CC = "CC"  # craniocaudal
    MLO = "MLO"  # mediolateral oblique
    FB = "FB"  # from below (caudocranial), a reversed CC; one right-breast image


class LesionType(StrEnum):
    """The closed set of lesion types an ROI can have."""

    MASS = "mass"
    CALCIFICATION = "calcification"
    CLUSTER = "cluster"
    SPICULATED_REGION = "spiculated_region"
    DISTORTION = "distortion"
    ASYMMETRY = "asymmetry"
    UNKNOWN = "unknown"


def polygon_area(points: Sequence[Point]) -> float:
    """Area of a polygon in pixels (shoelace formula); 0 for fewer than 3 points."""
    if len(points) < 3:
        return 0.0
    twice_area = 0.0
    for (x1, y1), (x2, y2) in zip(points, [*points[1:], points[0]], strict=True):
        twice_area += x1 * y2 - x2 * y1
    return abs(twice_area) / 2


@dataclass(frozen=True, config=_CONFIG)
class Roi:
    """One annotated lesion: its type, its outline and what is known about it.

    The outline is a polygon (``points``, INbreast) or a mask image of the size of
    the image (``mask_path`` with ``mask_area_px``, its pixels, CBIS-DDSM), never
    both. The other fields are what a dataset records per lesion; None where it
    records nothing.
    """

    lesion_type: LesionType
    points: tuple[Point, ...] = ()
    mask_path: RelativePath | None = None  # relative to the dataset folder
    mask_area_px: Pixels | None = None
    abnormality_id: Annotated[int, Field(ge=1)] | None = None  # its number in the image
    pathology: Pathology | None = None
    assessment: Score | None = None  # BI-RADS assessment of the lesion
    subtlety: Score | None = None  # 0 to 5, 5 the most obvious
    shape: str | None = None
    margins: str | None = None

    @model_validator(mode="after")
    def _one_outline(self) -> Self:
        if (self.mask_path is None) != (self.mask_area_px is None):
            raise ValueError("a mask needs both its path and its area")
        if self.mask_path is not None and self.points:
            raise ValueError("an ROI has a polygon or a mask, not both")
        return self

    @model_serializer(mode="wrap")
    def _leave_out_empty(
        self, serialize: Callable[[Any], dict[str, Any]]
    ) -> dict[str, Any]:
        # Leaves out what the dataset has no value for: None, or no polygon.
        return {
            key: value
            for key, value in serialize(self).items()
            if value is not None and value != []
        }

    @property
    def area_px(self) -> float:
        """The area in pixels: that of the mask, or of the polygon."""
        if self.mask_area_px is not None:
            return float(self.mask_area_px)
        return polygon_area(self.points)


@dataclass(frozen=True, config=_CONFIG)
class ImageRecord:
    """One mammogram: its metadata and all its annotated lesions."""

    file_id: FileId
    patient_id: PatientId
    laterality: Literal["L", "R"]
    view: View
    image_path: RelativePath  # the DICOM or JPEG file, relative to the dataset folder
    rows: Pixels
    cols: Pixels
    pixel_size_mm: PixelSize | None  # the side of a native pixel; None if unknown
    acquisition_date: AcquisitionDate | None  # year and semester, e.g. "200902"
    acr: Density | None  # None where the dataset has no value
    birads: BiRads | None  # of the whole image; None where only lesions have one
    official_split: Literal["train", "test"] | None  # the dataset's own split
    has_annotation: bool  # an annotation exists for the image
    rois: tuple[Roi, ...]

    @property
    def n_masses(self) -> int:
        return sum(roi.lesion_type is LesionType.MASS for roi in self.rois)

    @property
    def has_mass(self) -> bool:
        return self.n_masses > 0

    def area_mm2(self, roi: Roi) -> float:
        """The area of an ROI of this image in mm², at the native pixel size.

        Raises ValueError when the pixel size of the image is unknown.
        """
        if self.pixel_size_mm is None:
            raise ValueError(
                f"image {self.file_id}: the pixel size is unknown, so areas in mm² "
                "cannot be computed"
            )
        return roi.area_px * self.pixel_size_mm**2


@dataclass(frozen=True, config=_CONFIG)
class DroppedImage:
    """An image of the dataset left out of the manifest, and why."""

    file_id: FileId
    reason: Annotated[str, StringConstraints(min_length=1)]


@dataclass(frozen=True, config=_CONFIG)
class Manifest:
    """All images of a dataset, with the version of this schema, and the images
    left out of it."""

    version: Literal[2]
    dataset: Dataset
    images: tuple[ImageRecord, ...]
    dropped: tuple[DroppedImage, ...] = ()


_IMAGE = TypeAdapter(ImageRecord)
_DROPPED = TypeAdapter(DroppedImage)
_MANIFEST = TypeAdapter(Manifest)


def image_to_json(image: ImageRecord) -> str:
    """One image as compact JSON."""
    return _IMAGE.dump_json(image).decode()


def dropped_to_json(dropped: DroppedImage) -> str:
    """One left-out image as compact JSON."""
    return _DROPPED.dump_json(dropped).decode()


def manifest_from_json(text: str) -> Manifest:
    """Parse and validate a manifest; raises ``pydantic.ValidationError``."""
    return _MANIFEST.validate_json(text)
