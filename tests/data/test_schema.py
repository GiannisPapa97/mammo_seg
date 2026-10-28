import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from pydantic import ValidationError

from mammo_seg.data.schema import (
    SCHEMA_VERSION,
    DroppedImage,
    ImageRecord,
    LesionType,
    Manifest,
    Roi,
    View,
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
        "view": View.CC,
        "image_path": "AllDICOMs/x.dcm",
        "rows": 6,
        "cols": 8,
        "pixel_size_mm": 0.07,
        "acquisition_date": "200902",
        "acr": 2,
        "birads": "4a",
        "official_split": None,
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
    return f'{{"version": {version}, "dataset": "inbreast", "images": [{items}]}}'


def mask_roi(**changes: Any) -> Roi:
    fields: dict[str, Any] = {
        "lesion_type": LesionType.MASS,
        "mask_path": "jpeg/1.2/1-2.jpg",
        "mask_area_px": 20,
        "abnormality_id": 1,
        "pathology": "BENIGN",
        "assessment": 4,
        "subtlety": 3,
        "shape": "OVAL",
        "margins": None,
    }
    return Roi(**{**fields, **changes})


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


def test_the_area_of_a_mask_roi_is_that_of_its_mask() -> None:
    assert mask_roi().area_px == 20.0


def test_an_roi_has_a_polygon_or_a_mask_not_both() -> None:
    with pytest.raises(ValidationError, match="not both"):
        mask_roi(points=SQUARE)


@pytest.mark.parametrize("changes", [{"mask_area_px": None}, {"mask_path": None}])
def test_a_mask_needs_its_path_and_its_area(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="both its path and its area"):
        mask_roi(**changes)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mask_area_px", 0),
        ("abnormality_id", 0),
        ("pathology", "UNSURE"),
        ("assessment", 6),
        ("subtlety", -1),
    ],
)
def test_out_of_range_lesion_field_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        mask_roi(**{field: value})


def test_area_in_mm2_uses_the_pixel_size_of_the_image() -> None:
    image = make_image(pixel_size_mm=0.1)
    assert image.area_mm2(Roi(LesionType.MASS, SQUARE)) == pytest.approx(0.12)
    assert image.area_mm2(mask_roi()) == pytest.approx(0.2)


def test_area_in_mm2_needs_a_known_pixel_size() -> None:
    image = make_image(pixel_size_mm=None)
    with pytest.raises(ValueError, match="image 1: the pixel size is unknown"):
        image.area_mm2(Roi(LesionType.MASS, SQUARE))


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
        "image_path",
        "rows",
        "cols",
        "pixel_size_mm",
        "acquisition_date",
        "acr",
        "birads",
        "official_split",
        "has_annotation",
        "rois",
    ]
    # The lesion fields without a value are left out, as for every INbreast lesion.
    assert data["rois"] == [
        {
            "lesion_type": "mass",
            "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
        }
    ]


def test_an_image_field_without_a_value_is_written_as_null() -> None:
    data = json.loads(image_to_json(make_image(acr=None, pixel_size_mm=None)))
    assert data["acr"] is None and data["pixel_size_mm"] is None
    assert data["official_split"] is None


def test_a_mask_roi_is_written_without_a_polygon() -> None:
    data = json.loads(image_to_json(make_image(rois=(mask_roi(),))))
    assert data["rois"] == [
        {
            "lesion_type": "mass",
            "mask_path": "jpeg/1.2/1-2.jpg",
            "mask_area_px": 20,
            "abnormality_id": 1,
            "pathology": "BENIGN",
            "assessment": 4,
            "subtlety": 3,
            "shape": "OVAL",
        }
    ]


@pytest.mark.parametrize("acr", [2, None])
def test_manifest_round_trips_through_json(acr: int | None) -> None:
    images = (make_image(acr=acr), make_image(file_id="2", rois=()))
    assert manifest_from_json(manifest_json(*images)) == Manifest(
        version=SCHEMA_VERSION, dataset="inbreast", images=images
    )


def test_a_mask_roi_round_trips_through_json() -> None:
    image = make_image(rois=(mask_roi(), mask_roi(shape=None, margins="SPICULATED")))
    assert manifest_from_json(manifest_json(image)).images == (image,)


def test_dropped_images_round_trip_through_json() -> None:
    dropped = '[{"file_id": "Mass-Test_P_00001_LEFT_CC", "reason": "no mask"}]'
    text = manifest_json().replace("]}", f'], "dropped": {dropped}}}')
    assert manifest_from_json(text).dropped == (
        DroppedImage("Mass-Test_P_00001_LEFT_CC", "no mask"),
    )


def test_a_manifest_names_a_known_dataset() -> None:
    assert manifest_from_json(manifest_json()).dataset == "inbreast"
    with pytest.raises(ValidationError, match="dataset"):
        manifest_from_json(manifest_json().replace('"inbreast"', '"mias"'))
    with pytest.raises(ValidationError, match="dataset"):
        manifest_from_json('{"version": 2, "images": []}')


def test_a_dropped_image_needs_a_reason() -> None:
    with pytest.raises(ValidationError, match="reason"):
        DroppedImage(file_id="1", reason="")


def test_manifest_rejects_another_version() -> None:
    with pytest.raises(ValidationError, match="version"):
        manifest_from_json(manifest_json(version=99))


def test_manifest_rejects_a_missing_version() -> None:
    with pytest.raises(ValidationError, match="version"):
        manifest_from_json('{"images": []}')


def test_invalid_laterality_is_rejected() -> None:
    with pytest.raises(ValidationError, match="laterality"):
        make_image(laterality="X")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("file_id", ""),
        ("file_id", "12/a"),  # ids name files
        ("file_id", "-1"),
        ("patient_id", ""),
        ("patient_id", "P 1"),
        ("patient_id", "P-1"),
        ("image_path", ""),
        ("pixel_size_mm", 0.0),
        ("pixel_size_mm", -0.07),
        ("official_split", "validation"),
        ("rows", 0),
        ("rows", -5),
        ("cols", 0),
        ("cols", -1),
        ("acquisition_date", ""),
        ("acquisition_date", "20090"),
        ("acquisition_date", "2009021"),
        ("acquisition_date", "20090a"),
        ("acr", 0),
        ("acr", 5),
        ("birads", "0"),
        ("birads", "7"),
        ("birads", "4d"),
        ("birads", ""),
        ("birads", 4),
    ],
)
def test_out_of_range_image_field_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        make_image(**{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"rows": 1, "cols": 1},
        {"acr": 1},
        {"acr": 4},
        {"acr": None},
        {"file_id": "0"},
        {"patient_id": "0123456789abcdef"},
        # the ids of CBIS-DDSM
        {"file_id": "Mass-Training_P_00001_LEFT_CC", "patient_id": "P_00001"},
        {"pixel_size_mm": None},
        {"acquisition_date": None},
        {"birads": None},
        {"official_split": "train"},
        {"official_split": "test"},
        *({"birads": value} for value in ["1", "2", "3", "4a", "4b", "4c", "5", "6"]),
    ],
)
def test_boundary_values_are_accepted(changes: dict[str, Any]) -> None:
    image = make_image(**changes)
    for field, value in changes.items():
        assert getattr(image, field) == value


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_coordinates_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        Roi(LesionType.MASS, ((value, 1.0),))
    with pytest.raises(ValidationError):
        Roi(LesionType.MASS, ((1.0, value),))


def test_non_finite_coordinates_in_a_file_are_rejected() -> None:
    text = manifest_json(make_image()).replace("[1.0,1.0]", "[NaN,1.0]")
    with pytest.raises(ValidationError):
        manifest_from_json(text)


def test_unknown_view_in_a_file_is_rejected() -> None:
    text = manifest_json(make_image()).replace('"view":"CC"', '"view":"XX"')
    with pytest.raises(ValidationError, match="view"):
        manifest_from_json(text)


def test_view_is_an_enum_that_compares_equal_to_its_text() -> None:
    assert make_image(view=View.MLO).view == "MLO"
    with pytest.raises(ValidationError, match="view"):
        make_image(view="CC")


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
