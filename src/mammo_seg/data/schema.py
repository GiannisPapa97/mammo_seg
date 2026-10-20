"""The manifest schema: what is stored for every image and its lesions.

A manifest is a list of images. Each image carries its metadata and every
annotated lesion (ROI) with a type and a polygon. It is stored as JSON::

    {"version": 1, "images": [
      {"file_id": "20586908", "patient_id": "6c613a14b80a8591",
       "laterality": "R", "view": "CC",
       "dicom_path": "AllDICOMs/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm",
       "rows": 4084, "cols": 3328, "acquisition_date": "200902",
       "acr": 2, "birads": "2", "has_annotation": true,
       "rois": [{"lesion_type": "mass", "points": [[2964.6, 3428.0], ...]}, ...]},
      ...
    ]}

The classes are frozen pydantic dataclasses in strict mode: an object cannot be
built, or a file read, with a wrong type or an unknown field value. Pydantic
raises ``pydantic.ValidationError`` (a ``ValueError``) saying which value is
wrong.

Coordinates are pixels, ``(x, y)`` = ``(column, row)``. Facts that follow from
the ROIs (such as the number of masses) are computed, not stored.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Final, Literal

from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

SCHEMA_VERSION: Final = 1

Point = tuple[float, float]

_CONFIG = ConfigDict(strict=True, extra="forbid")


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
    """One annotated lesion: its type and its polygon outline."""

    lesion_type: LesionType
    points: tuple[Point, ...]

    @property
    def area_px(self) -> float:
        return polygon_area(self.points)


@dataclass(frozen=True, config=_CONFIG)
class ImageRecord:
    """One mammogram: its metadata and all its annotated lesions."""

    file_id: str
    patient_id: str
    laterality: Literal["L", "R"]
    view: str  # CC, MLO or FB
    dicom_path: str  # relative to the dataset folder
    rows: int
    cols: int
    acquisition_date: str  # year and semester, e.g. "200902"
    acr: int | None  # breast density 1 to 4; missing for one image
    birads: str  # e.g. "4a"
    has_annotation: bool  # an annotation file exists for the image
    rois: tuple[Roi, ...]

    @property
    def n_masses(self) -> int:
        return sum(roi.lesion_type is LesionType.MASS for roi in self.rois)

    @property
    def has_mass(self) -> bool:
        return self.n_masses > 0


@dataclass(frozen=True, config=_CONFIG)
class Manifest:
    """All images of the dataset, with the version of this schema."""

    version: Literal[1]
    images: tuple[ImageRecord, ...]


_IMAGE = TypeAdapter(ImageRecord)
_MANIFEST = TypeAdapter(Manifest)


def image_to_json(image: ImageRecord) -> str:
    """One image as compact JSON."""
    return _IMAGE.dump_json(image).decode()


def manifest_from_json(text: str) -> Manifest:
    """Parse and validate a manifest; raises ``pydantic.ValidationError``."""
    return _MANIFEST.validate_json(text)
