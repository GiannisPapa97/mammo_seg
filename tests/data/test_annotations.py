import plistlib
from pathlib import Path

import pytest
from tests.data.synthetic import square, write_annotation

from mammo_seg.data.annotations import parse_lesion_type, read_rois
from mammo_seg.data.schema import LesionType


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Mass", LesionType.MASS),
        (" mass ", LesionType.MASS),
        ("Calcification", LesionType.CALCIFICATION),
        ("Calcifications", LesionType.CALCIFICATION),
        ("Cluster", LesionType.CLUSTER),
        ("Spiculated Region", LesionType.SPICULATED_REGION),
        ("Spiculated region", LesionType.SPICULATED_REGION),
        ("Espiculated Region", LesionType.SPICULATED_REGION),
        ("Distortion", LesionType.DISTORTION),
        ("Asymmetry", LesionType.ASYMMETRY),
        ("Assymetry", LesionType.ASYMMETRY),
        ("Point 1", LesionType.UNKNOWN),
        ("Point 3", LesionType.UNKNOWN),
        ("Unnamed", LesionType.UNKNOWN),
        ("", LesionType.UNKNOWN),
    ],
)
def test_parse_lesion_type_maps_every_known_spelling(
    name: str, expected: LesionType
) -> None:
    assert parse_lesion_type(name) is expected


def test_parse_lesion_type_rejects_an_unseen_name() -> None:
    with pytest.raises(ValueError, match="unknown lesion name 'Cyst'"):
        parse_lesion_type("Cyst")


def test_read_rois_parses_types_and_points(tmp_path: Path) -> None:
    path = tmp_path / "x.xml"
    write_annotation(
        path, [(" Mass ", square(1, 2, 3)), ("Calcification", ["(5.5, 6.25)"])]
    )
    mass, calcification = read_rois(path)
    assert mass.lesion_type is LesionType.MASS
    assert mass.points[0] == (1.0, 2.0)
    assert len(mass.points) == 4
    assert mass.area_px == 9.0
    assert calcification.lesion_type is LesionType.CALCIFICATION
    assert calcification.points == ((5.5, 6.25),)
    assert calcification.area_px == 0.0


def test_read_rois_of_annotation_without_rois(tmp_path: Path) -> None:
    path = tmp_path / "x.xml"
    write_annotation(path, [])
    assert read_rois(path) == []


def test_read_rois_rejects_malformed_point(tmp_path: Path) -> None:
    path = tmp_path / "x.xml"
    write_annotation(path, [("Mass", ["(1.0, 2.0, 3.0)"])])
    with pytest.raises(ValueError, match="x.xml: expected a point"):
        read_rois(path)


def test_read_rois_rejects_unknown_lesion_name(tmp_path: Path) -> None:
    path = tmp_path / "x.xml"
    write_annotation(path, [("Cyst", square(0, 0, 2))])
    with pytest.raises(ValueError, match="x.xml: unknown lesion name 'Cyst'"):
        read_rois(path)


def test_read_rois_rejects_several_images(tmp_path: Path) -> None:
    path = tmp_path / "x.xml"
    path.write_bytes(plistlib.dumps({"Images": [{"ROIs": []}, {"ROIs": []}]}))
    with pytest.raises(ValueError, match="expected 1 image"):
        read_rois(path)
