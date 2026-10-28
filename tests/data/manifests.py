"""Helpers that build small synthetic manifests directly, without raw files."""

import math

from mammo_seg.data.manifest import INBREAST_PIXEL_SIZE_MM
from mammo_seg.data.schema import ImageRecord, LesionType, Manifest, Point, Roi, View

SQUARE = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))


def square_of(mm2: float) -> tuple[Point, ...]:
    """A square polygon of ``mm2`` square millimetres at the INbreast pixel size."""
    side = math.sqrt(mm2) / INBREAST_PIXEL_SIZE_MM
    return ((0.0, 0.0), (side, 0.0), (side, side), (0.0, side))


def make_image(
    file_id: int,
    patient: int,
    *,
    view: View = View.CC,
    masses: int = 0,
    acr: int | None = 2,
    mass_mm2: float | None = None,
) -> ImageRecord:
    """``masses`` masses of ``mass_mm2`` each, or of the 12-pixel ``SQUARE``."""
    outline = SQUARE if mass_mm2 is None else square_of(mass_mm2)
    rois = tuple(Roi(LesionType.MASS, outline) for _ in range(masses))
    return ImageRecord(
        file_id=f"{file_id:08d}",
        patient_id=f"{patient:016x}",
        laterality="L",
        view=view,
        image_path=f"AllDICOMs/{file_id}.dcm",
        rows=10,
        cols=10,
        pixel_size_mm=INBREAST_PIXEL_SIZE_MM,
        acquisition_date="200902",
        acr=acr,
        birads="2",
        official_split=None,
        has_annotation=bool(rois),
        rois=rois,
    )


def make_manifest(with_mass: int, without_mass: int, per_patient: int = 4) -> Manifest:
    """Patients with a mass have it in their first image only; the rest are negative."""
    images = []
    for patient in range(with_mass + without_mass):
        for position in range(per_patient):
            images.append(
                make_image(
                    patient * per_patient + position,
                    patient,
                    view=View.CC if position % 2 == 0 else View.MLO,
                    masses=1 if patient < with_mass and position == 0 else 0,
                )
            )
    return Manifest(version=2, dataset="inbreast", images=tuple(images))


def pid(patient: int) -> str:
    return f"{patient:016x}"
