"""The PyTorch dataset of preprocessed mammograms and their mass masks.

A sample is read from the files written by ``preprocess_dataset``: the 16-bit
image is normalized to 0..1, an optional transform (the training augmentation) is
applied to image and mask together, and both are padded on the right and at the
bottom to a common canvas so samples can be stacked into batches. The chest wall
is on the left in every preprocessed image, so the padding only adds background
next to the breast.

Every sample also carries ``valid``, which is true on the pixels of the image and
false on the padding, so losses and metrics can leave the padding out. A crop of an
image smaller than the crop is padded too, so its padding is not valid either.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict

import torch
from torch.nn import functional
from torch.utils.data import Dataset

from mammo_seg.data.masks import read_mask
from mammo_seg.data.preprocess_dataset import (
    RECORD_FILE,
    PreprocessingRecord,
    check_settings,
    image_path,
    processed_mask_path,
    read_image,
    read_record,
)
from mammo_seg.data.preprocessing import PreprocessingSettings, normalize_intensity

# Every preprocessed image is 1024 px high and at most 784 px wide; 800 is the next
# multiple of 32, which a U-Net needs.
DEFAULT_CANVAS = (1024, 800)

# A U-Net halves the image five times, so its input sides must divide by 32.
DEFAULT_MULTIPLE = 32

Transform = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class Sample(TypedDict):
    image: torch.Tensor  # float32, [1, H, W], 0..1, 0 on the padding
    mask: torch.Tensor  # float32, [1, H, W], 1 on a mass
    valid: torch.Tensor  # bool, [1, H, W], false on the padding
    file_id: str


def canvas_of(
    record: PreprocessingRecord, multiple: int = DEFAULT_MULTIPLE
) -> tuple[int, int]:
    """The smallest canvas, with sides divisible by ``multiple``, that holds every
    preprocessed image of the record: (1024, 800) for INbreast at size 1024."""
    if not record.images:
        raise ValueError("the preprocessed data has no images")
    rows = max(params.shape[0] for params in record.images.values())
    cols = max(params.shape[1] for params in record.images.values())
    return -(-rows // multiple) * multiple, -(-cols // multiple) * multiple


def pad_to(tensor: torch.Tensor, canvas: tuple[int, int]) -> torch.Tensor:
    """Pad a ``[C, H, W]`` tensor with zeros on the right and at the bottom.

    Raises ValueError if the tensor is larger than the canvas: cropping would
    silently lose part of the breast.
    """
    height, width = tensor.shape[-2:]
    rows, cols = canvas
    if height > rows or width > cols:
        raise ValueError(f"an image of {height} x {width} does not fit {rows} x {cols}")
    return functional.pad(tensor, (0, cols - width, 0, rows - height))


class MammogramDataset(Dataset[Sample]):
    """The preprocessed images and masks of the given file ids.

    Raises ValueError if the data was preprocessed with other settings than
    ``settings`` or a file id is not in the preprocessed data.
    """

    def __init__(
        self,
        preprocessed_dir: Path,
        file_ids: Sequence[str],
        settings: PreprocessingSettings = PreprocessingSettings(),  # noqa: B008
        canvas: tuple[int, int] = DEFAULT_CANVAS,
        transform: Transform | None = None,
    ) -> None:
        record = read_record(preprocessed_dir / RECORD_FILE)
        check_settings(record, settings)
        if not file_ids:
            raise ValueError("there are no images to load")
        missing = sorted(set(file_ids) - set(record.images))
        if missing:
            raise ValueError(f"images not in the preprocessed data: {missing}")
        self.directory = preprocessed_dir
        self.file_ids = tuple(file_ids)
        self.settings = settings
        self.canvas = canvas
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, index: int) -> Sample:
        file_id = self.file_ids[index]
        pixels = read_image(image_path(self.directory, file_id))
        normalized = normalize_intensity(pixels, self.settings)
        image = torch.from_numpy(normalized).unsqueeze(0)
        mask_array = read_mask(processed_mask_path(self.directory, file_id))
        mask = torch.from_numpy(mask_array).unsqueeze(0).float()
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        valid = torch.ones_like(image, dtype=torch.bool)
        return Sample(
            image=pad_to(image, self.canvas),
            mask=pad_to(mask, self.canvas),
            valid=pad_to(valid, self.canvas),
            file_id=file_id,
        )
