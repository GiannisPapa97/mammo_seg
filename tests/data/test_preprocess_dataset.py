from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from pydantic import ValidationError
from tests.data.manifests import make_image
from tests.data.synthetic import write_dicom_pixels

from mammo_seg.data.manifest import write_manifest
from mammo_seg.data.masks import mask_path, read_mask, write_masks
from mammo_seg.data.preprocess_dataset import (
    DATASET_SETTINGS,
    RECORD_FILE,
    PreprocessingError,
    PreprocessingRecord,
    check_mask_kept,
    check_settings,
    image_path,
    main,
    preprocess_dataset,
    processed_mask_path,
    read_image,
    read_record,
    write_image,
    write_record,
)
from mammo_seg.data.preprocessing import (
    BoundingBox,
    PreprocessingSettings,
    PreprocessParams,
    preprocess,
    preprocess_mask,
    read_pixels,
)
from mammo_seg.data.schema import LesionType, Manifest, Roi

SETTINGS = PreprocessingSettings(margin=0.05, downscale=1, opening=0, size=100)
ID1, ID2, ID3 = "00000001", "00000002", "00000003"
IDS = (ID1, ID2, ID3)


def mass(x1: float, y1: float, x2: float, y2: float) -> Roi:
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    return Roi(LesionType.MASS, corners)


Lesion = tuple[int, int, int, int]  # top, bottom, left, right


def breast(lesion: Lesion | None) -> np.ndarray:
    """A 400 x 300 image with a 200 x 100 breast and, optionally, a bright lesion."""
    pixels = np.zeros((400, 300), dtype=np.uint16)
    pixels[100:300, 100:200] = 1500
    if lesion:
        top, bottom, left, right = lesion
        pixels[top:bottom, left:right] = 3500
    return pixels


@dataclass(frozen=True)
class Dataset:
    manifest: Manifest
    dicoms: Path
    masks: Path
    output: Path


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Three images: a left and a right breast with a mass, and a normal one."""
    (tmp_path / "AllDICOMs").mkdir()
    specs: list[tuple[int, Literal["L", "R"], Lesion | None, tuple[Roi, ...]]] = [
        (1, "L", (150, 200, 120, 160), (mass(120, 150, 160, 200),)),
        (2, "R", (220, 270, 140, 180), (mass(140, 220, 180, 270),)),
        (3, "L", None, ()),
    ]
    images = []
    for number, laterality, lesion, rois in specs:
        name = f"{number:08d}.dcm"
        write_dicom_pixels(tmp_path / "AllDICOMs" / name, breast(lesion))
        images.append(
            replace(
                make_image(number, number),
                laterality=laterality,
                image_path=f"AllDICOMs/{name}",
                rows=400,
                cols=300,
                rois=rois,
            )
        )
    manifest = Manifest(version=2, dataset="inbreast", images=tuple(images))
    write_masks(manifest, tmp_path / "masks")
    return Dataset(manifest, tmp_path, tmp_path / "masks", tmp_path / "preprocessed")


def run(dataset: Dataset, settings: PreprocessingSettings = SETTINGS):  # type: ignore[no-untyped-def]
    return preprocess_dataset(
        dataset.manifest, dataset.dicoms, dataset.masks, dataset.output, settings
    )


# --- files ---


def test_a_16_bit_image_round_trips_through_a_png(tmp_path: Path) -> None:
    pixels = np.array([[0, 1, 4095], [4096, 40000, 65535]], dtype=np.uint16)
    write_image(pixels, tmp_path / "sub" / "a.png")
    read = read_image(tmp_path / "sub" / "a.png")
    assert read.dtype == np.uint16
    assert np.array_equal(read, pixels)


def test_an_8_bit_png_is_not_accepted_as_an_image(tmp_path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / "a.png")
    with pytest.raises(ValueError, match="expected a 16-bit image"):
        read_image(tmp_path / "a.png")


def test_the_output_paths_use_the_file_id(tmp_path: Path) -> None:
    assert image_path(tmp_path, "7") == tmp_path / "images" / "7.png"
    assert processed_mask_path(tmp_path, "7") == tmp_path / "masks" / "7.png"


# --- the preprocessed dataset ---


def test_every_image_and_mask_is_written_with_a_record(dataset: Dataset) -> None:
    summary = run(dataset)
    for file_id in IDS:
        assert image_path(dataset.output, file_id).exists()
        assert processed_mask_path(dataset.output, file_id).exists()
    assert (dataset.output / RECORD_FILE).exists()
    assert summary.images == 3
    assert summary.images_with_masks == 2


def test_the_stored_images_are_what_the_on_the_fly_chain_gives(
    dataset: Dataset,
) -> None:
    run(dataset)
    for image in dataset.manifest.images:
        pixels = read_pixels(dataset.dicoms / image.image_path)
        expected, params = preprocess(pixels, image.laterality, SETTINGS)
        stored = read_image(image_path(dataset.output, image.file_id))
        assert np.array_equal(stored, expected)
        mask = read_mask(mask_path(dataset.masks, image.file_id))
        stored_mask = read_mask(processed_mask_path(dataset.output, image.file_id))
        assert np.array_equal(stored_mask, preprocess_mask(mask, params))


def test_the_stored_mask_has_the_size_of_the_stored_image(dataset: Dataset) -> None:
    run(dataset)
    for file_id in IDS:
        image = read_image(image_path(dataset.output, file_id))
        mask = read_mask(processed_mask_path(dataset.output, file_id))
        assert mask.shape == image.shape
        assert max(image.shape) == 100


def test_a_normal_image_has_an_empty_mask(dataset: Dataset) -> None:
    run(dataset)
    assert not read_mask(processed_mask_path(dataset.output, ID3)).any()
    assert read_mask(processed_mask_path(dataset.output, ID1)).any()


def test_the_right_breast_is_flipped_and_the_record_says_so(dataset: Dataset) -> None:
    run(dataset)
    record = read_record(dataset.output / RECORD_FILE)
    assert [record.images[i].flipped for i in IDS] == [False, True, False]


def test_the_record_holds_the_settings_and_the_parameters(dataset: Dataset) -> None:
    run(dataset)
    record = read_record(dataset.output / RECORD_FILE)
    assert record.version == 1
    assert record.settings == SETTINGS
    assert set(record.images) == set(IDS)
    pixels = read_pixels(dataset.dicoms / "AllDICOMs" / f"{ID1}.dcm")
    assert record.images[ID1] == preprocess(pixels, "L", SETTINGS)[1]


def test_the_record_round_trips(dataset: Dataset, tmp_path: Path) -> None:
    run(dataset)
    record = read_record(dataset.output / RECORD_FILE)
    write_record(record, tmp_path / "again" / "record.json")
    assert read_record(tmp_path / "again" / "record.json") == record


def test_a_mask_can_be_mapped_back_with_the_stored_parameters(dataset: Dataset) -> None:
    from mammo_seg.data.preprocessing import restore_mask

    run(dataset)
    record = read_record(dataset.output / RECORD_FILE)
    for file_id in (ID1, ID2):
        original = read_mask(mask_path(dataset.masks, file_id))
        small = read_mask(processed_mask_path(dataset.output, file_id))
        restored = restore_mask(small, record.images[file_id])
        overlap = ((restored == 1) & (original == 1)).sum()
        assert overlap / ((restored == 1) | (original == 1)).sum() > 0.85


def test_the_size_setting_is_used_and_recorded(dataset: Dataset) -> None:
    settings = replace(SETTINGS, size=60)
    run(dataset, settings)
    assert max(read_image(image_path(dataset.output, ID1)).shape) == 60
    assert read_record(dataset.output / RECORD_FILE).settings.size == 60


def test_running_again_gives_the_same_files(dataset: Dataset) -> None:
    run(dataset)
    first = image_path(dataset.output, ID2).read_bytes()
    record = (dataset.output / RECORD_FILE).read_text()
    run(dataset)
    assert image_path(dataset.output, ID2).read_bytes() == first
    assert (dataset.output / RECORD_FILE).read_text() == record


# --- checking the settings ---


def test_data_preprocessed_with_other_settings_is_refused(dataset: Dataset) -> None:
    run(dataset)
    record = read_record(dataset.output / RECORD_FILE)
    check_settings(record, SETTINGS)
    with pytest.raises(ValueError, match="different settings than expected"):
        check_settings(record, PreprocessingSettings())


def test_a_record_with_an_unknown_version_or_field_is_rejected(
    dataset: Dataset,
) -> None:
    run(dataset)
    text = (dataset.output / RECORD_FILE).read_text()
    path = dataset.output / "other.json"
    path.write_text(text.replace('"version": 1', '"version": 2'))
    with pytest.raises(ValidationError, match="version"):
        read_record(path)
    path.write_text(text.replace('"size": 100', '"size": 100, "extra": 1'))
    with pytest.raises(ValidationError, match="extra"):
        read_record(path)


@pytest.mark.parametrize("file_id", ["a/b", "-1", ""])
def test_the_record_rejects_a_key_that_is_not_a_file_id(
    dataset: Dataset, file_id: str
) -> None:
    # File ids name files, so a path separator or a leading dash is refused.
    run(dataset)
    record = read_record(dataset.output / RECORD_FILE)
    with pytest.raises(ValidationError):
        PreprocessingRecord(
            version=1, settings=SETTINGS, images={file_id: record.images[ID1]}
        )


# --- problems ---


def test_every_failing_image_is_reported_and_no_record_is_written(
    dataset: Dataset,
) -> None:
    (dataset.dicoms / "AllDICOMs" / f"{ID2}.dcm").unlink()  # missing DICOM
    mask_path(dataset.masks, ID3).unlink()  # missing mask
    with pytest.raises(PreprocessingError) as error:
        run(dataset)
    assert len(error.value.problems) == 2
    assert error.value.problems[0].startswith(f"image {ID2}:")
    assert error.value.problems[1].startswith(f"image {ID3}:")
    assert image_path(dataset.output, ID1).exists()  # the good image is written
    assert not image_path(dataset.output, ID2).exists()
    assert not (dataset.output / RECORD_FILE).exists()


def test_a_dicom_of_another_size_than_the_manifest_is_a_problem(
    dataset: Dataset,
) -> None:
    write_dicom_pixels(dataset.dicoms / "AllDICOMs" / f"{ID1}.dcm", breast(None)[:300])
    with pytest.raises(PreprocessingError, match="the image is \\(300, 300\\)"):
        run(dataset)


def test_an_unreadable_dicom_is_a_problem(dataset: Dataset) -> None:
    (dataset.dicoms / "AllDICOMs" / f"{ID1}.dcm").write_text("not a dicom")
    with pytest.raises(PreprocessingError, match=f"image {ID1}:"):
        run(dataset)


def test_a_mass_outside_the_breast_box_is_a_problem(dataset: Dataset) -> None:
    pixels = breast((150, 200, 120, 160))
    pixels[10:30, 10:30] = 3000  # a bright patch far from the breast
    write_dicom_pixels(dataset.dicoms / "AllDICOMs" / f"{ID1}.dcm", pixels)
    image = replace(dataset.manifest.images[0], rois=(mass(10, 10, 30, 30),))
    manifest = Manifest(
        version=2, dataset="inbreast", images=(image, *dataset.manifest.images[1:])
    )
    write_masks(manifest, dataset.masks)
    with pytest.raises(
        PreprocessingError, match=f"image {ID1}: the crop box cuts the mask"
    ):
        preprocess_dataset(
            manifest, dataset.dicoms, dataset.masks, dataset.output, SETTINGS
        )


def test_a_mask_that_vanishes_when_rescaled_is_a_problem(dataset: Dataset) -> None:
    image = replace(dataset.manifest.images[0], rois=(mass(150, 200, 151, 201),))
    manifest = Manifest(
        version=2, dataset="inbreast", images=(image, *dataset.manifest.images[1:])
    )
    write_masks(manifest, dataset.masks)
    tiny = PreprocessingSettings(margin=0.05, downscale=1, opening=0, size=10)
    with pytest.raises(PreprocessingError, match="vanishes when rescaled"):
        preprocess_dataset(
            manifest, dataset.dicoms, dataset.masks, dataset.output, tiny
        )


def test_a_mask_of_another_size_than_the_image_is_a_problem(dataset: Dataset) -> None:
    from mammo_seg.data.masks import write_mask

    write_mask(np.zeros((10, 10), dtype=np.uint8), mask_path(dataset.masks, ID3))
    with pytest.raises(PreprocessingError, match=f"image {ID3}: mask shape"):
        run(dataset)


# --- the command ---


def test_main_preprocesses_the_dataset_and_prints_a_summary(
    dataset: Dataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(dataset.manifest, manifest_path)
    code = main(
        [
            "--manifest", str(manifest_path),
            "--dataset-dir", str(dataset.dicoms),
            "--masks-dir", str(dataset.masks),
            "--output-dir", str(dataset.output),
            "--size", "80",
        ]
    )  # fmt: skip
    assert code == 0
    text = capsys.readouterr().out
    assert "Wrote 3 images and masks" in text
    assert "longer side 80" in text
    assert "2 with a mass" in text
    assert read_record(dataset.output / RECORD_FILE).settings.size == 80


def test_main_reports_problems(
    dataset: Dataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(dataset.manifest, manifest_path)
    (dataset.dicoms / "AllDICOMs" / f"{ID2}.dcm").unlink()
    code = main(
        [
            "--manifest", str(manifest_path),
            "--dataset-dir", str(dataset.dicoms),
            "--masks-dir", str(dataset.masks),
            "--output-dir", str(dataset.output),
        ]
    )  # fmt: skip
    assert code == 1
    assert f"image {ID2}:" in capsys.readouterr().err


def test_main_reports_a_missing_manifest_and_an_invalid_size(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--manifest", str(tmp_path / "missing.json")]) == 1
    assert "error:" in capsys.readouterr().err
    assert main(["--manifest", str(tmp_path / "m.json"), "--size", "0"]) == 1
    assert "error:" in capsys.readouterr().err


# --- checking that a mask survives the rescaling ---

# A 100 x 100 crop rescaled to 50 x 50: every rescaled pixel stands for 4 original ones.
HALVING = PreprocessParams(
    source_shape=(200, 200),
    box=BoundingBox(top=0, bottom=100, left=0, right=100),
    shape=(50, 50),
    flipped=False,
)


def square(size: int, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10 : 10 + size, 10 : 10 + size] = 1
    return mask


def test_a_mask_that_keeps_its_area_passes() -> None:
    original = square(40, (200, 200))  # 1600 px, 400 after halving
    check_mask_kept(original, square(20, (50, 50)), HALVING)


def test_a_small_difference_along_the_edge_passes() -> None:
    original = square(40, (200, 200))
    off_by_a_row = square(20, (50, 50))
    off_by_a_row[10, 10:30] = 0  # 20 pixels of 400, about the edge
    check_mask_kept(original, off_by_a_row, HALVING)


def test_two_empty_masks_pass() -> None:
    empty = np.zeros((200, 200), dtype=np.uint8)
    check_mask_kept(empty, np.zeros((50, 50), dtype=np.uint8), HALVING)


def test_a_mask_that_vanishes_is_an_error() -> None:
    with pytest.raises(ValueError, match="vanishes when rescaled"):
        check_mask_kept(
            square(40, (200, 200)), np.zeros((50, 50), dtype=np.uint8), HALVING
        )


def test_a_mask_with_the_wrong_area_is_an_error() -> None:
    too_big = square(30, (50, 50))  # 900 px instead of about 400
    with pytest.raises(ValueError, match="2.25 times its expected area"):
        check_mask_kept(square(40, (200, 200)), too_big, HALVING)


def test_an_empty_mask_that_gains_pixels_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty mask has pixels"):
        check_mask_kept(
            np.zeros((200, 200), dtype=np.uint8), square(20, (50, 50)), HALVING
        )


def test_cbis_ddsm_has_its_settings_and_inbreast_the_defaults() -> None:
    assert DATASET_SETTINGS["inbreast"] == PreprocessingSettings()
    cbis = DATASET_SETTINGS["cbis-ddsm"]
    assert (cbis.max_intensity, cbis.orientation) == (255, "image")
    assert (cbis.threshold, cbis.margin, cbis.opening, cbis.border) == (20, 0.04, 3, 2)


def test_main_preprocesses_cbis_ddsm_with_its_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An 8-bit image of a left breast whose chest wall is on the right is flipped
    by its image, and keeps its values."""
    from PIL import Image

    from mammo_seg.data.masks import write_mask

    pixels = np.zeros((400, 300), dtype=np.uint8)
    pixels[60:340, 120:300] = np.linspace(60, 200, 180, dtype=np.uint8)
    (tmp_path / "jpeg").mkdir()
    Image.fromarray(pixels).save(tmp_path / "jpeg" / "1.png")
    lesion = np.zeros((400, 300), dtype=np.uint8)
    lesion[150:190, 200:240] = 1
    write_mask(lesion, mask_path(tmp_path / "masks", ID1))
    roi = Roi(LesionType.MASS, mask_path="jpeg/1-mask.png", mask_area_px=1600)
    image = replace(
        make_image(1, 1),
        image_path="jpeg/1.png",
        rows=400,
        cols=300,
        pixel_size_mm=None,
        official_split="train",
        rois=(roi,),
    )
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        Manifest(version=2, dataset="cbis-ddsm", images=(image,)), manifest_path
    )
    code = main(
        [
            "--dataset", "cbis-ddsm",
            "--manifest", str(manifest_path),
            "--dataset-dir", str(tmp_path),
            "--masks-dir", str(tmp_path / "masks"),
            "--output-dir", str(tmp_path / "out"),
            "--size", "100",
        ]
    )  # fmt: skip
    assert code == 0
    assert "1 with a mass" in capsys.readouterr().out
    record = read_record(tmp_path / "out" / RECORD_FILE)
    assert record.settings == replace(DATASET_SETTINGS["cbis-ddsm"], size=100)
    assert record.images[ID1].flipped
    result = read_image(image_path(tmp_path / "out", ID1))
    assert 190 <= result.max() <= 200  # the 8-bit values are kept
    assert result[:, :10].mean() > result[:, -10:].mean()  # the wall is on the left
    assert read_mask(processed_mask_path(tmp_path / "out", ID1)).any()
