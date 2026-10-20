"""Helpers that write a small synthetic preprocessed dataset; no real data."""

from pathlib import Path

import numpy as np

from mammo_seg.data.masks import write_mask
from mammo_seg.data.preprocess_dataset import (
    RECORD_FILE,
    PreprocessingRecord,
    image_path,
    processed_mask_path,
    write_image,
    write_record,
)
from mammo_seg.data.preprocessing import (
    BoundingBox,
    PreprocessingSettings,
    PreprocessParams,
)

SETTINGS = PreprocessingSettings(size=64)
CANVAS = (64, 48)
# file id -> width of its preprocessed image (all are 64 px high)
WIDTHS = {"00000001": 30, "00000002": 48, "00000003": 20}


def write_preprocessed(directory: Path, settings: PreprocessingSettings) -> None:
    """Small preprocessed images: a ramp of intensities, a mass mask on image 1."""
    params = {}
    for file_id, width in WIDTHS.items():
        pixels = np.tile(np.linspace(0, 4095, width, dtype=np.uint16), (64, 1))
        mask = np.zeros((64, width), dtype=np.uint8)
        if file_id == "00000001":
            mask[10:20, 5:15] = 1
        write_image(pixels, image_path(directory, file_id))
        write_mask(mask, processed_mask_path(directory, file_id))
        params[file_id] = PreprocessParams(
            source_shape=(400, 300),
            box=BoundingBox(top=0, bottom=400, left=0, right=200),
            shape=(64, width),
            flipped=False,
        )
    record = PreprocessingRecord(version=1, settings=settings, images=params)
    write_record(record, directory / RECORD_FILE)
