"""Helpers that build small synthetic manifests directly, without raw files."""

from mammo_seg.data.schema import ImageRecord, LesionType, Manifest, Roi, View

SQUARE = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))


def make_image(
    file_id: int, patient: int, *, view: View = View.CC, masses: int = 0
) -> ImageRecord:
    rois = tuple(Roi(LesionType.MASS, SQUARE) for _ in range(masses))
    return ImageRecord(
        file_id=f"{file_id:08d}",
        patient_id=f"{patient:016x}",
        laterality="L",
        view=view,
        dicom_path=f"AllDICOMs/{file_id}.dcm",
        rows=10,
        cols=10,
        acquisition_date="200902",
        acr=2,
        birads="2",
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
    return Manifest(version=1, images=tuple(images))


def pid(patient: int) -> str:
    return f"{patient:016x}"
