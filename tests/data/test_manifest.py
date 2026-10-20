import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.data.synthetic import write_annotation, write_dicom

from mammo_seg.data.manifest import (
    IMAGE_COLUMNS,
    ROI_COLUMNS,
    ManifestError,
    build_manifest,
    images_frame,
    main,
    read_manifest,
    rois_frame,
    write_manifest,
)
from mammo_seg.data.schema import ImageRecord, LesionType, Manifest, View


def by_id(dataset_dir: Path) -> dict[str, ImageRecord]:
    return {image.file_id: image for image in build_manifest(dataset_dir).images}


def test_manifest_has_one_record_per_image_sorted_by_id(dataset_dir: Path) -> None:
    manifest = build_manifest(dataset_dir)
    assert [image.file_id for image in manifest.images] == [
        "11111111",
        "11111112",
        "22222221",
        "22222222",
    ]
    assert {image.patient_id for image in manifest.images} == {"a" * 16, "b" * 16}


def test_manifest_reads_filename_csv_and_header_fields(dataset_dir: Path) -> None:
    first = by_id(dataset_dir)["11111111"]
    assert first.patient_id == "a" * 16
    assert (first.laterality, first.view) == ("L", "CC")
    assert (first.rows, first.cols) == (6, 8)
    assert first.dicom_path == f"AllDICOMs/11111111_{'a' * 16}_MG_L_CC_ANON.dcm"
    assert first.acquisition_date == "200902"
    assert first.birads == "4a"
    assert first.acr == 2
    assert by_id(dataset_dir)["22222221"].rows == 10


def test_view_is_mlo_for_filename_ml_and_csv_mlo(dataset_dir: Path) -> None:
    views = {file_id: image.view for file_id, image in by_id(dataset_dir).items()}
    assert views == {
        "11111111": "CC",
        "11111112": "MLO",
        "22222221": "CC",
        "22222222": "MLO",
    }


def test_unexpected_view_in_a_filename_is_an_error(dataset_dir: Path) -> None:
    write_dicom(
        dataset_dir / "AllDICOMs" / f"33333333_{'c' * 16}_MG_L_XX_ANON.dcm", 4, 4
    )
    with pytest.raises(ManifestError, match="unexpected view 'XX'"):
        build_manifest(dataset_dir)


def test_fb_view_is_kept(dataset_dir: Path) -> None:
    write_dicom(
        dataset_dir / "AllDICOMs" / f"33333333_{'c' * 16}_MG_R_FB_ANON.dcm", 4, 4
    )
    csv = dataset_dir / "INbreast.csv"
    csv.write_text(csv.read_text() + "removed;removed;R;FB;201001;33333333;1;1\n")
    assert by_id(dataset_dir)["33333333"].view is View.FB


def test_blank_acr_becomes_none(dataset_dir: Path) -> None:
    images = by_id(dataset_dir)
    assert images["22222222"].acr is None
    assert images["11111111"].acr == 2


def test_manifest_attaches_all_rois_with_type_and_polygon(dataset_dir: Path) -> None:
    images = by_id(dataset_dir)
    first = images["11111111"]
    assert [roi.lesion_type for roi in first.rois] == [
        LesionType.MASS,
        LesionType.CALCIFICATION,
    ]
    assert first.rois[0].points == ((1.0, 1.0), (11.0, 1.0), (11.0, 11.0), (1.0, 11.0))
    assert first.rois[0].area_px == 100.0
    assert [roi.area_px for roi in images["11111112"].rois] == [100.0, 25.0]
    assert [roi.lesion_type for roi in images["22222221"].rois] == [
        LesionType.CALCIFICATION
    ]


def test_image_without_annotation_has_no_rois(dataset_dir: Path) -> None:
    normal = by_id(dataset_dir)["22222222"]
    assert not normal.has_annotation
    assert normal.rois == ()
    assert by_id(dataset_dir)["22222221"].has_annotation


def test_build_manifest_is_deterministic(dataset_dir: Path) -> None:
    assert build_manifest(dataset_dir) == build_manifest(dataset_dir)


def test_images_frame_has_columns_and_derived_labels(dataset_dir: Path) -> None:
    frame = images_frame(build_manifest(dataset_dir)).set_index("file_id")
    assert list(images_frame(build_manifest(dataset_dir)).columns) == IMAGE_COLUMNS
    assert frame.n_masses.to_dict() == {
        "11111111": 1,
        "11111112": 2,
        "22222221": 0,
        "22222222": 0,
    }
    assert frame.has_mass.to_dict() == {
        "11111111": True,
        "11111112": True,
        "22222221": False,
        "22222222": False,
    }
    assert all(type(view) is str for view in frame.view)
    assert frame.view.to_dict() == {
        "11111111": "CC",
        "11111112": "MLO",
        "22222221": "CC",
        "22222222": "MLO",
    }
    assert frame.acr.isna().sum() == 1
    assert str(frame.acr.dtype) == "Int64"


def test_rois_frame_has_one_row_per_roi(dataset_dir: Path) -> None:
    frame = rois_frame(build_manifest(dataset_dir))
    assert list(frame.columns) == ROI_COLUMNS
    assert len(frame) == 5
    masses = frame[frame.lesion_type == "mass"]
    assert masses.area_px.tolist() == [100.0, 100.0, 25.0]
    assert frame[frame.file_id == "11111111"].roi_index.tolist() == [0, 1]
    assert frame[frame.lesion_type == "calcification"].n_points.tolist() == [1, 4]


def test_frames_of_an_empty_manifest_keep_their_columns() -> None:
    empty = Manifest(version=1, images=())
    assert list(images_frame(empty).columns) == IMAGE_COLUMNS
    assert list(rois_frame(empty).columns) == ROI_COLUMNS


def test_image_without_csv_row_is_an_error(dataset_dir: Path) -> None:
    write_dicom(
        dataset_dir / "AllDICOMs" / f"33333333_{'c' * 16}_MG_L_CC_ANON.dcm", 4, 4
    )
    with pytest.raises(ManifestError, match="no INbreast.csv row for image 33333333"):
        build_manifest(dataset_dir)


def test_csv_row_without_image_is_an_error(dataset_dir: Path) -> None:
    csv = dataset_dir / "INbreast.csv"
    csv.write_text(csv.read_text() + "removed;removed;L;CC;201001;44444444;1;1\n")
    with pytest.raises(ManifestError, match="44444444 has no DICOM image"):
        build_manifest(dataset_dir)


def test_annotation_without_image_is_an_error(dataset_dir: Path) -> None:
    write_annotation(dataset_dir / "AllXML" / "55555555.xml", [])
    with pytest.raises(ManifestError, match="55555555.xml has no DICOM image"):
        build_manifest(dataset_dir)


def test_laterality_and_view_mismatch_is_an_error(dataset_dir: Path) -> None:
    csv = dataset_dir / "INbreast.csv"
    csv.write_text(
        csv.read_text()
        .replace("L;CC;200902;11111111", "R;CC;200902;11111111")
        .replace("R;CC;201001;22222221", "R;MLO;201001;22222221")
    )
    with pytest.raises(ManifestError) as error:
        build_manifest(dataset_dir)
    assert len(error.value.problems) == 2
    assert "laterality" in error.value.problems[0]
    assert "view" in error.value.problems[1]


def test_unexpected_dicom_filename_is_an_error(dataset_dir: Path) -> None:
    write_dicom(dataset_dir / "AllDICOMs" / "not_an_inbreast_name.dcm", 4, 4)
    with pytest.raises(ManifestError, match="unexpected DICOM filename"):
        build_manifest(dataset_dir)


def test_duplicate_file_id_is_an_error(dataset_dir: Path) -> None:
    write_dicom(
        dataset_dir / "AllDICOMs" / f"11111111_{'c' * 16}_MG_L_CC_ANON.dcm", 4, 4
    )
    with pytest.raises(ManifestError, match="11111111 appears in more than one DICOM"):
        build_manifest(dataset_dir)


def test_duplicate_csv_row_is_an_error(dataset_dir: Path) -> None:
    csv = dataset_dir / "INbreast.csv"
    csv.write_text(csv.read_text() + "removed;removed;L;CC;200902;11111111;2;4a\n")
    with pytest.raises(ManifestError, match="11111111 appears.*INbreast.csv"):
        build_manifest(dataset_dir)


def test_csv_missing_a_column_is_an_error(dataset_dir: Path) -> None:
    csv = dataset_dir / "INbreast.csv"
    csv.write_text(csv.read_text().replace("Bi-Rads", "Birads"))
    with pytest.raises(ManifestError, match="missing columns"):
        build_manifest(dataset_dir)


def test_invalid_acr_is_an_error(dataset_dir: Path) -> None:
    csv = dataset_dir / "INbreast.csv"
    csv.write_text(
        csv.read_text().replace("200902;11111111;2;4a", "200902;11111111;x;4a")
    )
    with pytest.raises(ManifestError, match="image 11111111: invalid ACR 'x'"):
        build_manifest(dataset_dir)


def test_malformed_annotation_is_an_error(dataset_dir: Path) -> None:
    write_annotation(dataset_dir / "AllXML" / "22222222.xml", [("Mass", ["(1, 2, 3)"])])
    with pytest.raises(ManifestError, match="expected a point"):
        build_manifest(dataset_dir)


def test_unknown_lesion_name_is_an_error(dataset_dir: Path) -> None:
    write_annotation(dataset_dir / "AllXML" / "22222222.xml", [("Cyst", ["(1, 2)"])])
    with pytest.raises(ManifestError, match="unknown lesion name 'Cyst'"):
        build_manifest(dataset_dir)


def test_all_problems_are_reported_together(dataset_dir: Path) -> None:
    write_annotation(dataset_dir / "AllXML" / "55555555.xml", [])
    write_dicom(
        dataset_dir / "AllDICOMs" / f"33333333_{'c' * 16}_MG_L_CC_ANON.dcm", 4, 4
    )
    with pytest.raises(ManifestError) as error:
        build_manifest(dataset_dir)
    assert len(error.value.problems) == 2


def test_write_and_read_manifest_round_trip(dataset_dir: Path, tmp_path: Path) -> None:
    manifest = build_manifest(dataset_dir)
    path = tmp_path / "processed" / "manifest.json"
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest


def test_written_manifest_is_json_with_one_image_per_line(
    dataset_dir: Path, tmp_path: Path
) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(build_manifest(dataset_dir), path)
    data = json.loads(path.read_text())
    assert data["version"] == 1
    assert [image["file_id"] for image in data["images"]][0] == "11111111"
    assert data["images"][0]["rois"][0]["lesion_type"] == "mass"
    assert len(path.read_text().splitlines()) == 4 + 2


def test_read_manifest_rejects_a_file_with_a_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"version": 1, "images": [{"file_id": "1"}]}')
    with pytest.raises(ValidationError, match="Field required"):
        read_manifest(path)


def test_read_manifest_rejects_another_version(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"version": 2, "images": []}')
    with pytest.raises(ValidationError, match="version"):
        read_manifest(path)


def test_main_writes_the_manifest(
    dataset_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out" / "manifest.json"
    code = main(["--dataset-dir", str(dataset_dir), "--output", str(output)])
    assert code == 0
    assert len(read_manifest(output).images) == 4
    assert "4 images, 2 patients, 2 with a mass, 5 ROIs" in capsys.readouterr().out


def test_main_reports_problems_and_writes_nothing(
    dataset_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_annotation(dataset_dir / "AllXML" / "55555555.xml", [])
    output = tmp_path / "manifest.json"
    code = main(["--dataset-dir", str(dataset_dir), "--output", str(output)])
    assert code == 1
    assert not output.exists()
    assert "55555555.xml has no DICOM image" in capsys.readouterr().err
