from typing import Any

import pytest

from mammo_seg.data.schema import (
    SCHEMA_VERSION,
    ImageRecord,
    LesionType,
    Manifest,
    Roi,
    polygon_area,
)

SQUARE = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0))


def make_image(**changes: Any) -> ImageRecord:
    fields: dict[str, Any] = {
        "file_id": "1",
        "patient_id": "a" * 16,
        "laterality": "L",
        "view": "CC",
        "dicom_path": "AllDICOMs/x.dcm",
        "rows": 6,
        "cols": 8,
        "acquisition_date": "200902",
        "acr": 2,
        "birads": "4a",
        "has_annotation": True,
        "rois": (
            Roi(LesionType.MASS, SQUARE),
            Roi(LesionType.CALCIFICATION, ((1.0, 1.0),)),
            Roi(LesionType.MASS, SQUARE),
        ),
    }
    return ImageRecord(**{**fields, **changes})


def test_polygon_area_of_square() -> None:
    assert polygon_area(SQUARE) == 12.0


def test_polygon_area_of_triangle_in_either_direction() -> None:
    triangle = ((0.0, 0.0), (4.0, 0.0), (0.0, 3.0))
    assert polygon_area(triangle) == 6.0
    assert polygon_area(triangle[::-1]) == 6.0


@pytest.mark.parametrize("n_points", [0, 1, 2])
def test_polygon_area_needs_three_points(n_points: int) -> None:
    assert polygon_area(((0.0, 0.0), (1.0, 1.0))[:n_points]) == 0.0


def test_roi_area_uses_its_points() -> None:
    assert Roi(LesionType.MASS, SQUARE).area_px == 12.0


def test_roi_round_trips_through_a_dict() -> None:
    roi = Roi(LesionType.SPICULATED_REGION, SQUARE)
    data = roi.to_dict()
    assert data == {
        "lesion_type": "spiculated_region",
        "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
    }
    assert Roi.from_dict(data) == roi


def test_roi_rejects_a_point_with_three_values() -> None:
    with pytest.raises(ValueError, match="expected a point"):
        Roi.from_dict({"lesion_type": "mass", "points": [[1, 2, 3]]})


def test_roi_rejects_an_unknown_lesion_type() -> None:
    with pytest.raises(ValueError, match="cyst"):
        Roi.from_dict({"lesion_type": "cyst", "points": []})


def test_image_counts_masses_among_all_rois() -> None:
    image = make_image()
    assert image.n_masses == 2
    assert image.has_mass


def test_image_without_masses_is_negative() -> None:
    image = make_image(rois=(Roi(LesionType.CLUSTER, SQUARE),))
    assert image.n_masses == 0
    assert not image.has_mass
    assert not make_image(rois=(), has_annotation=False).has_mass


@pytest.mark.parametrize("acr", [2, None])
def test_image_round_trips_through_a_dict(acr: int | None) -> None:
    image = make_image(acr=acr)
    assert ImageRecord.from_dict(image.to_dict()) == image


def test_image_rejects_an_invalid_laterality() -> None:
    data = make_image().to_dict()
    data["laterality"] = "X"
    with pytest.raises(ValueError, match="laterality must be L or R"):
        ImageRecord.from_dict(data)


def test_image_rejects_a_missing_field() -> None:
    data = make_image().to_dict()
    del data["rows"]
    with pytest.raises(KeyError, match="rows"):
        ImageRecord.from_dict(data)


def test_manifest_round_trips_through_a_dict() -> None:
    manifest = Manifest(images=(make_image(), make_image(file_id="2", acr=None)))
    data = manifest.to_dict()
    assert data["version"] == SCHEMA_VERSION
    assert Manifest.from_dict(data) == manifest


def test_manifest_rejects_another_version() -> None:
    with pytest.raises(ValueError, match="unsupported manifest version 99"):
        Manifest.from_dict({"version": 99, "images": []})
    with pytest.raises(ValueError, match="unsupported manifest version None"):
        Manifest.from_dict({"images": []})
