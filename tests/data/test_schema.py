import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from pydantic import ValidationError

from mammo_seg.data.schema import (
    SCHEMA_VERSION,
    ImageRecord,
    LesionType,
    Manifest,
    Roi,
    image_to_json,
    manifest_from_json,
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


def manifest_json(*images: ImageRecord, version: int = SCHEMA_VERSION) -> str:
    items = ",".join(image_to_json(image) for image in images)
    return f'{{"version": {version}, "images": [{items}]}}'


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


def test_image_counts_masses_among_all_rois() -> None:
    image = make_image()
    assert image.n_masses == 2
    assert image.has_mass


def test_image_without_masses_is_negative() -> None:
    image = make_image(rois=(Roi(LesionType.CLUSTER, SQUARE),))
    assert image.n_masses == 0
    assert not image.has_mass
    assert not make_image(rois=(), has_annotation=False).has_mass


def test_records_are_immutable() -> None:
    image = make_image()
    with pytest.raises(FrozenInstanceError):
        image.rows = 7  # type: ignore[misc]


def test_image_json_has_the_documented_shape() -> None:
    data = json.loads(image_to_json(make_image(rois=(Roi(LesionType.MASS, SQUARE),))))
    assert list(data) == [
        "file_id",
        "patient_id",
        "laterality",
        "view",
        "dicom_path",
        "rows",
        "cols",
        "acquisition_date",
        "acr",
        "birads",
        "has_annotation",
        "rois",
    ]
    assert data["rois"] == [
        {
            "lesion_type": "mass",
            "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
        }
    ]


@pytest.mark.parametrize("acr", [2, None])
def test_manifest_round_trips_through_json(acr: int | None) -> None:
    images = (make_image(acr=acr), make_image(file_id="2", rois=()))
    assert manifest_from_json(manifest_json(*images)) == Manifest(
        version=SCHEMA_VERSION, images=images
    )


def test_manifest_rejects_another_version() -> None:
    with pytest.raises(ValidationError, match="version"):
        manifest_from_json(manifest_json(version=99))


def test_manifest_rejects_a_missing_version() -> None:
    with pytest.raises(ValidationError, match="version"):
        manifest_from_json('{"images": []}')


def test_invalid_laterality_is_rejected() -> None:
    with pytest.raises(ValidationError, match="laterality"):
        make_image(laterality="X")


def test_unknown_lesion_type_in_a_file_is_rejected() -> None:
    text = manifest_json(make_image()).replace('"mass"', '"cyst"')
    with pytest.raises(ValidationError, match="lesion_type"):
        manifest_from_json(text)


def test_point_with_three_values_in_a_file_is_rejected() -> None:
    text = manifest_json(make_image()).replace("[1.0,1.0]", "[1.0,1.0,1.0]")
    with pytest.raises(ValidationError, match="points"):
        manifest_from_json(text)


def test_missing_field_in_a_file_is_rejected() -> None:
    text = manifest_json(make_image()).replace('"rows":6,', "")
    with pytest.raises(ValidationError, match="rows"):
        manifest_from_json(text)


def test_extra_field_in_a_file_is_rejected() -> None:
    text = manifest_json(make_image()).replace('"rows":6,', '"rows":6,"extra":1,')
    with pytest.raises(ValidationError, match="extra"):
        manifest_from_json(text)


def test_strict_mode_does_not_convert_types() -> None:
    with pytest.raises(ValidationError, match="rows"):
        make_image(rows="6")
    text = manifest_json(make_image()).replace('"rows":6', '"rows":"6"')
    with pytest.raises(ValidationError, match="rows"):
        manifest_from_json(text)
