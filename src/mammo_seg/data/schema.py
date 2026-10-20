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

Coordinates are pixels, ``(x, y)`` = ``(column, row)``. Facts that follow from
the ROIs (such as the number of masses) are computed, not stored.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Self

SCHEMA_VERSION = 1

Point = tuple[float, float]


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


@dataclass(frozen=True)
class Roi:
    """One annotated lesion: its type and its polygon outline."""

    lesion_type: LesionType
    points: tuple[Point, ...]

    @property
    def area_px(self) -> float:
        return polygon_area(self.points)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesion_type": self.lesion_type.value,
            "points": [list(point) for point in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        points = []
        for point in data["points"]:
            if len(point) != 2:
                raise ValueError(f"expected a point [x, y], got {point!r}")
            points.append((float(point[0]), float(point[1])))
        return cls(lesion_type=LesionType(data["lesion_type"]), points=tuple(points))


@dataclass(frozen=True)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "patient_id": self.patient_id,
            "laterality": self.laterality,
            "view": self.view,
            "dicom_path": self.dicom_path,
            "rows": self.rows,
            "cols": self.cols,
            "acquisition_date": self.acquisition_date,
            "acr": self.acr,
            "birads": self.birads,
            "has_annotation": self.has_annotation,
            "rois": [roi.to_dict() for roi in self.rois],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        laterality = data["laterality"]
        if laterality not in ("L", "R"):
            raise ValueError(f"laterality must be L or R, got {laterality!r}")
        acr = data["acr"]
        return cls(
            file_id=str(data["file_id"]),
            patient_id=str(data["patient_id"]),
            laterality=laterality,
            view=str(data["view"]),
            dicom_path=str(data["dicom_path"]),
            rows=int(data["rows"]),
            cols=int(data["cols"]),
            acquisition_date=str(data["acquisition_date"]),
            acr=None if acr is None else int(acr),
            birads=str(data["birads"]),
            has_annotation=bool(data["has_annotation"]),
            rois=tuple(Roi.from_dict(roi) for roi in data["rois"]),
        )


@dataclass(frozen=True)
class Manifest:
    """All images of the dataset."""

    images: tuple[ImageRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "images": [image.to_dict() for image in self.images],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        version = data.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest version {version!r}, expected {SCHEMA_VERSION}"
            )
        return cls(
            images=tuple(ImageRecord.from_dict(image) for image in data["images"])
        )
