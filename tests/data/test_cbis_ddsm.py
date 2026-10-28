"""The CBIS-DDSM manifest, built from a tiny synthetic copy; no real data is used."""

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mammo_seg.data.cbis_ddsm import (
    MASK_THRESHOLD,
    MASS_CSVS,
    build_cbis_ddsm_manifest,
    mask_area,
)
from mammo_seg.data.manifest import main, read_manifest
from mammo_seg.data.schema import DroppedImage, ImageRecord, LesionType, ManifestError

HEADER = (
    "patient_id",
    "breast_density",
    "left or right breast",
    "image view",
    "abnormality id",
    "abnormality type",
    "mass shape",
    "mass margins",
    "assessment",
    "pathology",
    "subtlety",
    "image file path",
    "cropped image file path",
    "ROI mask file path",
)
ROWS, COLS = 32, 24


def write_jpeg(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, quality=100)


def mask_pixels(top: int, left: int, side: int = 8) -> np.ndarray:
    """A mask of the size of the images with a square mass; squares on the 8-pixel
    grid of JPEG stay exactly binary."""
    pixels = np.zeros((ROWS, COLS), dtype=np.uint8)
    pixels[top : top + side, left : left + side] = 255
    return pixels


def mass_row(
    image_id: str, abnormality_id: int, full_series: str, mask_series: str
) -> dict[str, str]:
    """A CSV row whose columns agree with the image folder ``image_id``."""
    _, patient, number, side, view = image_id.split("_")
    mask_folder = f"{image_id}_{abnormality_id}"
    return {
        "patient_id": f"{patient}_{number}",
        "breast_density": "2",
        "left or right breast": side,
        "image view": view,
        "abnormality id": str(abnormality_id),
        "abnormality type": "mass",
        "mass shape": "OVAL",
        "mass margins": "CIRCUMSCRIBED",
        "assessment": "4",
        "pathology": "MALIGNANT",
        "subtlety": "5",
        "image file path": f"{image_id}/1.9/{full_series}/000000.dcm",
        "cropped image file path": f"{mask_folder}/1.9/{mask_series}/000000.dcm",
        # The real mask paths end with a newline.
        "ROI mask file path": f"{mask_folder}/1.9/{mask_series}/000001.dcm\n",
    }


@dataclass
class Cbis:
    """A synthetic copy of the dataset: its rows by split, written by ``write``."""

    dataset_dir: Path
    rows: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def write(self, header: Sequence[str] = HEADER) -> None:
        for split, name in MASS_CSVS.items():
            path = self.dataset_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as file:
                writer = csv.DictWriter(file, header, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.rows.get(split, []))

    def jpeg(self, series: str, name: str, pixels: np.ndarray) -> None:
        write_jpeg(self.dataset_dir / "jpeg" / series / name, pixels)

    def build(self) -> dict[str, ImageRecord]:
        manifest = build_cbis_ddsm_manifest(self.dataset_dir)
        return {image.file_id: image for image in manifest.images}


CC = "Mass-Training_P_00001_LEFT_CC"
MLO = "Mass-Training_P_00001_LEFT_MLO"
TEST = "Mass-Test_P_00002_RIGHT_CC"


@pytest.fixture
def cbis(tmp_path: Path) -> Cbis:
    """Three images: one mass, two masses, and a test image whose mass has only
    its crop (no mask), which is dropped."""
    cbis = Cbis(tmp_path / "cbis-ddsm")
    full = np.full((ROWS, COLS), 90, dtype=np.uint8)
    crop = np.full((8, 8), 200, dtype=np.uint8)
    for series in ("1.1", "2.1", "3.1"):
        cbis.jpeg(series, "1-1.jpg", full)
    # The crop comes first in the folder, so the mask is chosen by its size.
    cbis.jpeg("1.2", "1-1.jpg", crop)
    cbis.jpeg("1.2", "1-2.jpg", mask_pixels(8, 8))
    cbis.jpeg("2.2", "1-1.jpg", mask_pixels(0, 0))
    cbis.jpeg("2.3", "1-1.jpg", crop)
    cbis.jpeg("2.3", "1-2.jpg", mask_pixels(16, 8, side=16))
    cbis.jpeg("3.2", "1-1.jpg", crop)
    second = mass_row(MLO, 2, "2.1", "2.3")
    second |= {"pathology": "BENIGN", "mass shape": "N/A", "mass margins": "N/A"}
    cbis.rows = {
        "train": [
            mass_row(CC, 1, "1.1", "1.2"),
            second,
            mass_row(MLO, 1, "2.1", "2.2"),
        ],
        "test": [mass_row(TEST, 1, "3.1", "3.2")],
    }
    cbis.write()
    return cbis


def test_one_record_per_image_with_a_mask_for_every_mass(cbis: Cbis) -> None:
    manifest = build_cbis_ddsm_manifest(cbis.dataset_dir)
    assert (manifest.version, manifest.dataset) == (2, "cbis-ddsm")
    assert [image.file_id for image in manifest.images] == [CC, MLO]
    assert manifest.dropped == (
        DroppedImage(
            file_id=TEST, reason="no mask of the size of the image for mass 1"
        ),
    )


def test_image_fields_come_from_the_folder_name_the_csv_and_the_jpeg(
    cbis: Cbis,
) -> None:
    image = cbis.build()[CC]
    assert (image.patient_id, image.laterality, image.view) == ("P_00001", "L", "CC")
    assert image.image_path == "jpeg/1.1/1-1.jpg"
    assert (image.rows, image.cols) == (ROWS, COLS)
    assert (image.acr, image.official_split, image.has_annotation) == (2, "train", True)
    assert (image.pixel_size_mm, image.acquisition_date, image.birads) == (
        None,
        None,
        None,
    )


def test_the_official_split_is_that_of_the_csv(cbis: Cbis) -> None:
    cbis.jpeg("3.2", "1-2.jpg", mask_pixels(0, 0))
    images = cbis.build()
    assert {i.file_id: i.official_split for i in images.values()} == {
        CC: "train",
        MLO: "train",
        TEST: "test",
    }
    assert (images[TEST].laterality, images[TEST].patient_id) == ("R", "P_00002")


def test_the_mask_is_the_file_of_the_size_of_the_image(cbis: Cbis) -> None:
    (roi,) = cbis.build()[CC].rois
    assert roi.lesion_type is LesionType.MASS
    assert roi.mask_path == "jpeg/1.2/1-2.jpg"
    assert roi.mask_area_px == 64 and roi.area_px == 64.0
    assert roi.points == ()


def test_the_masses_are_in_the_order_of_their_ids_with_their_csv_values(
    cbis: Cbis,
) -> None:
    first, second = cbis.build()[MLO].rois
    assert (first.abnormality_id, second.abnormality_id) == (1, 2)
    assert (first.mask_path, second.mask_path) == (
        "jpeg/2.2/1-1.jpg",
        "jpeg/2.3/1-2.jpg",
    )
    assert (first.mask_area_px, second.mask_area_px) == (64, 256)
    assert (first.pathology, first.assessment, first.subtlety) == ("MALIGNANT", 4, 5)
    assert (first.shape, first.margins) == ("OVAL", "CIRCUMSCRIBED")
    assert second.pathology == "BENIGN"
    assert (second.shape, second.margins) == (None, None)


def test_an_image_is_dropped_if_any_of_its_masses_has_no_mask(cbis: Cbis) -> None:
    cbis.rows["train"].append(mass_row(MLO, 3, "2.1", "2.4"))
    cbis.jpeg("2.4", "1-1.jpg", np.zeros((16, 16), dtype=np.uint8))
    cbis.write()
    manifest = build_cbis_ddsm_manifest(cbis.dataset_dir)
    assert [image.file_id for image in manifest.images] == [CC]
    assert [d.file_id for d in manifest.dropped] == [TEST, MLO]
    assert manifest.dropped[1].reason.endswith("for mass 3")


def test_the_manifest_command_writes_the_cbis_ddsm_manifest(
    cbis: Cbis, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out" / "manifest.json"
    args = ["--dataset", "cbis-ddsm", "--dataset-dir", str(cbis.dataset_dir)]
    assert main([*args, "--output", str(output)]) == 0
    assert read_manifest(output) == build_cbis_ddsm_manifest(cbis.dataset_dir)
    assert "2 images, 1 patients, 2 with a mass, 3 ROIs, 1 images left out" in (
        capsys.readouterr().out
    )


def test_the_manifest_command_reports_the_problems(
    cbis: Cbis, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cbis.rows["train"][0]["pathology"] = "UNKNOWN"
    cbis.write()
    args = ["--dataset", "cbis-ddsm", "--dataset-dir", str(cbis.dataset_dir)]
    assert main([*args, "--output", str(tmp_path / "manifest.json")]) == 1
    assert "unknown pathology 'UNKNOWN'" in capsys.readouterr().err
    assert not (tmp_path / "manifest.json").exists()


def test_a_missing_column_is_an_error(cbis: Cbis) -> None:
    cbis.write([name for name in HEADER if name != "subtlety"])
    with pytest.raises(ManifestError, match=r"train_set.csv is missing .*subtlety"):
        build_cbis_ddsm_manifest(cbis.dataset_dir)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("abnormality type", "calcification", "abnormality type 'calcification'"),
        ("patient_id", "P_00009", "does not match the split, patient"),
        ("image view", "CC", "does not match the split, patient"),
        ("image file path", "Calc-Training_P_00001_LEFT_CC/1/1.1/0.dcm", "folder"),
        ("image file path", f"{MLO}/1.9/x/0.dcm", "unexpected image path"),
        ("image file path", f"{MLO}/2.1/0.dcm", "unexpected image path"),
        ("ROI mask file path", f"{MLO}_2/1.9/../0.dcm", "unexpected mask path"),
        ("ROI mask file path", f"{MLO}_1/1.9/2.3/0.dcm", "is not that of mass 2"),
        ("breast_density", "x", "invalid breast_density 'x'"),
        ("abnormality id", "", "invalid abnormality id ''"),
        ("assessment", "four", "invalid assessment 'four'"),
        ("subtlety", "1.5", "invalid subtlety '1.5'"),
        ("pathology", "benign", "unknown pathology 'benign'"),
    ],
)
def test_an_invalid_row_is_an_error_naming_its_line(
    cbis: Cbis, column: str, value: str, message: str
) -> None:
    cbis.rows["train"][1][column] = value
    cbis.write()
    with pytest.raises(ManifestError, match="train_set.csv line 3: .*" + message):
        build_cbis_ddsm_manifest(cbis.dataset_dir)


def test_a_value_out_of_range_is_an_error(cbis: Cbis) -> None:
    cbis.rows["train"][0]["assessment"] = "6"
    for row in cbis.rows["train"][1:]:  # both masses of MLO
        row["breast_density"] = "5"
    cbis.write()
    with pytest.raises(ManifestError) as error:
        build_cbis_ddsm_manifest(cbis.dataset_dir)
    assert error.value.problems == [
        f"image {CC}, mass 1: assessment: Input should be less than or equal to 5",
        f"image {MLO}: acr: Input should be less than or equal to 4",
    ]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("breast_density", "3", "more than one density"),
        ("image file path", f"{MLO}/1.9/1.1/000000.dcm", "more than one image series"),
        ("abnormality id", "1", "an abnormality id appears more than once"),
        ("ROI mask file path", f"{MLO}_2/1.9/2.2/0.dcm", "the same mask series"),
    ],
)
def test_the_rows_of_an_image_must_agree(
    cbis: Cbis, column: str, value: str, message: str
) -> None:
    row = cbis.rows["train"][1]  # the second mass of MLO
    row[column] = value
    if column == "abnormality id":
        row["ROI mask file path"] = f"{MLO}_1/1.9/2.3/000001.dcm"
    cbis.write()
    with pytest.raises(ManifestError, match=f"image {MLO}: .*{message}"):
        build_cbis_ddsm_manifest(cbis.dataset_dir)


def test_every_problem_is_reported_at_once(cbis: Cbis) -> None:
    cbis.rows["train"][0]["pathology"] = "x"
    cbis.rows["test"][0]["subtlety"] = "x"
    cbis.write()
    with pytest.raises(ManifestError) as error:
        build_cbis_ddsm_manifest(cbis.dataset_dir)
    assert len(error.value.problems) == 2


def test_the_series_of_a_full_image_must_have_one_jpeg(cbis: Cbis) -> None:
    cbis.jpeg("1.1", "1-2.jpg", np.zeros((ROWS, COLS), dtype=np.uint8))
    (cbis.dataset_dir / "jpeg" / "2.1" / "1-1.jpg").unlink()
    with pytest.raises(ManifestError) as error:
        build_cbis_ddsm_manifest(cbis.dataset_dir)
    assert error.value.problems == [
        f"image {CC}: 2 JPEG files for its series 1.1, not 1",
        f"image {MLO}: 0 JPEG files for its series 2.1, not 1",
    ]


def test_a_full_image_must_be_grey(cbis: Cbis) -> None:
    cbis.jpeg("1.1", "1-1.jpg", np.zeros((ROWS, COLS, 3), dtype=np.uint8))
    with pytest.raises(ManifestError, match=f"image {CC}: mode RGB, not a grey image"):
        build_cbis_ddsm_manifest(cbis.dataset_dir)


def test_a_mass_without_its_series_is_an_error_not_a_dropped_image(
    cbis: Cbis,
) -> None:
    for path in (cbis.dataset_dir / "jpeg" / "1.2").iterdir():
        path.unlink()
    with pytest.raises(ManifestError, match=f"{CC}, mass 1: no JPEG files"):
        build_cbis_ddsm_manifest(cbis.dataset_dir)


def test_two_files_of_the_size_of_the_image_are_an_error(cbis: Cbis) -> None:
    cbis.jpeg("1.2", "1-3.jpg", mask_pixels(0, 0))
    with pytest.raises(ManifestError, match=f"{CC}, mass 1: 2 files of the size"):
        build_cbis_ddsm_manifest(cbis.dataset_dir)


def test_a_mask_that_is_not_binary_is_an_error(cbis: Cbis) -> None:
    cbis.jpeg("1.2", "1-2.jpg", np.full((ROWS, COLS), 128, dtype=np.uint8))
    with pytest.raises(ManifestError, match=f"{CC}, mass 1: 1-2.jpg: .*mid-grey"):
        build_cbis_ddsm_manifest(cbis.dataset_dir)


def save_png(tmp_path: Path, pixels: np.ndarray) -> Path:
    path = tmp_path / "mask.png"
    Image.fromarray(pixels).save(path)
    return path


def test_the_mask_area_counts_the_pixels_above_the_threshold(tmp_path: Path) -> None:
    pixels = np.zeros((40, 50), dtype=np.uint8)
    pixels[0, :3] = [MASK_THRESHOLD, MASK_THRESHOLD + 1, 255]
    pixels[1, 0] = 63  # dark enough not to count as grey
    assert mask_area(save_png(tmp_path, pixels)) == 2


def test_a_few_grey_pixels_are_allowed_in_a_mask(tmp_path: Path) -> None:
    pixels = np.zeros((40, 50), dtype=np.uint8)
    pixels[0, :3] = [255, 100, 150]  # 2 of 2000 grey pixels: 0.1%
    assert mask_area(save_png(tmp_path, pixels)) == 2
    pixels[0, 3] = 64
    with pytest.raises(ValueError, match="0.15% of the pixels are mid-grey"):
        mask_area(save_png(tmp_path, pixels))


def test_a_mask_must_be_grey_and_not_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode RGB"):
        mask_area(save_png(tmp_path, np.full((4, 4, 3), 255, dtype=np.uint8)))
    with pytest.raises(ValueError, match="the mask is empty"):
        mask_area(save_png(tmp_path, np.zeros((4, 4), dtype=np.uint8)))
