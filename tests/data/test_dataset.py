from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from tests.data.preprocessed import CANVAS, SETTINGS, WIDTHS, write_preprocessed
from torch.utils.data import DataLoader

from mammo_seg.data.dataset import DEFAULT_CANVAS, MammogramDataset, canvas_of, pad_to
from mammo_seg.data.preprocess_dataset import RECORD_FILE, read_record
from mammo_seg.data.preprocessing import PreprocessingSettings


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    write_preprocessed(tmp_path, SETTINGS)
    return tmp_path


def dataset(directory: Path, **kwargs: object) -> MammogramDataset:
    return MammogramDataset(
        directory,
        kwargs.pop("file_ids", list(WIDTHS)),  # type: ignore[arg-type]
        kwargs.pop("settings", SETTINGS),  # type: ignore[arg-type]
        kwargs.pop("canvas", CANVAS),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


# --- the canvas ---


def test_the_canvas_holds_the_largest_image_with_sides_divisible_by_32(
    directory: Path,
) -> None:
    record = read_record(directory / RECORD_FILE)  # images 64 x (30, 48, 20)
    assert canvas_of(record) == (64, 64)
    assert canvas_of(record, multiple=10) == (70, 50)
    assert canvas_of(record, multiple=1) == (64, 48)


def test_a_record_without_images_has_no_canvas(directory: Path) -> None:
    record = read_record(directory / RECORD_FILE)
    with pytest.raises(ValueError, match="no images"):
        canvas_of(replace(record, images={}))


# --- padding ---


def test_pad_to_adds_zeros_on_the_right_and_at_the_bottom() -> None:
    tensor = torch.ones(1, 2, 3)
    padded = pad_to(tensor, (4, 5))
    assert padded.shape == (1, 4, 5)
    assert padded[0, :2, :3].sum() == 6
    assert padded.sum() == 6


def test_pad_to_keeps_a_tensor_of_the_canvas_size() -> None:
    tensor = torch.rand(1, 4, 5)
    assert torch.equal(pad_to(tensor, (4, 5)), tensor)


def test_pad_to_rejects_a_tensor_larger_than_the_canvas() -> None:
    with pytest.raises(ValueError, match="an image of 4 x 6 does not fit 4 x 5"):
        pad_to(torch.ones(1, 4, 6), (4, 5))


def test_the_default_canvas_fits_every_preprocessed_image() -> None:
    assert DEFAULT_CANVAS == (1024, 800)
    assert DEFAULT_CANVAS[0] % 32 == 0 and DEFAULT_CANVAS[1] % 32 == 0


# --- samples ---


def test_the_dataset_has_one_sample_per_file_id(directory: Path) -> None:
    assert len(dataset(directory)) == 3
    assert len(dataset(directory, file_ids=["00000002"])) == 1


def test_a_sample_has_padded_tensors_of_the_canvas_size(directory: Path) -> None:
    sample = dataset(directory)[0]
    assert sample["file_id"] == "00000001"
    assert sample["image"].shape == (1, 64, 48)
    assert sample["mask"].shape == (1, 64, 48)
    assert sample["valid"].shape == (1, 64, 48)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    assert sample["valid"].dtype == torch.bool


def test_the_image_is_normalized_and_zero_on_the_padding(directory: Path) -> None:
    image = dataset(directory)[0]["image"]
    expected = np.linspace(0, 4095, 30, dtype=np.uint16).astype(np.float32) / 4095
    assert torch.allclose(image[0, 5, :30], torch.from_numpy(expected))
    assert image.max() == 1.0
    assert (image[0, :, 30:] == 0).all()


def test_the_mask_is_on_the_mass_and_zero_elsewhere(directory: Path) -> None:
    mask = dataset(directory)[0]["mask"]
    assert mask[0, 10:20, 5:15].eq(1).all()
    assert mask.sum() == 100
    assert not dataset(directory)[1]["mask"].any()


def test_valid_marks_the_image_and_not_the_padding(directory: Path) -> None:
    samples = dataset(directory)
    first, full = samples[0]["valid"], samples[1]["valid"]
    assert first[0, :, :30].all() and not first[0, :, 30:].any()
    assert full.all()  # a 48 px wide image fills the canvas


def test_the_settings_decide_the_normalization(tmp_path: Path) -> None:
    per_image = PreprocessingSettings(size=64, percentile=50.0)
    write_preprocessed(tmp_path, per_image)
    image = dataset(tmp_path, settings=per_image)[0]["image"]
    assert (image[0, :, :30] == 1.0).float().mean() > 0.4


def test_a_transform_gets_the_unpadded_image_and_mask(directory: Path) -> None:
    seen: list[tuple[torch.Size, torch.Size]] = []

    def flip(
        image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seen.append((image.shape, mask.shape))
        return image.flip(-2), mask.flip(-2)

    sample = dataset(directory, transform=flip)[0]
    assert seen == [(torch.Size([1, 64, 30]), torch.Size([1, 64, 30]))]
    assert sample["mask"][0, 44:54, 5:15].eq(1).all()  # flipped upside down
    assert sample["image"].shape == (1, 64, 48)


def test_samples_batch_with_a_data_loader(directory: Path) -> None:
    batch = next(iter(DataLoader(dataset(directory), batch_size=3)))
    assert batch["image"].shape == (3, 1, 64, 48)
    assert batch["mask"].shape == (3, 1, 64, 48)
    assert batch["valid"].shape == (3, 1, 64, 48)
    assert batch["file_id"] == list(WIDTHS)


# --- errors ---


def test_data_preprocessed_with_other_settings_is_refused(directory: Path) -> None:
    with pytest.raises(ValueError, match="different settings"):
        dataset(directory, settings=PreprocessingSettings())


def test_an_unknown_file_id_is_refused(directory: Path) -> None:
    with pytest.raises(
        ValueError, match="not in the preprocessed data: \\['00000009'\\]"
    ):
        dataset(directory, file_ids=["00000001", "00000009"])


def test_an_empty_list_of_file_ids_is_refused(directory: Path) -> None:
    with pytest.raises(ValueError, match="no images to load"):
        dataset(directory, file_ids=[])


def test_an_image_wider_than_the_canvas_is_an_error(directory: Path) -> None:
    with pytest.raises(ValueError, match="does not fit 64 x 40"):
        dataset(directory, canvas=(64, 40))[1]


def test_a_missing_record_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MammogramDataset(tmp_path, ["00000001"], SETTINGS, CANVAS)
