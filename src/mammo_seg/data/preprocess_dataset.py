"""Preprocess every image of the manifest once and store the result on disk.

This is the cache of the first steps of the preprocessing chain (crop, rescale,
flip, see ``preprocessing.py``). Training reads these files instead of decoding
the raw images on every epoch; inference runs the same ``preprocess`` function on
the incoming image, so both see identical input. The intensity normalization is
not applied here: the images keep their values and are normalized when loaded.

Each dataset has its settings (``DATASET_SETTINGS``). CBIS-DDSM is scanned film
in 8-bit JPEGs: its values stay 0 to 255 (``max_intensity`` 255), its background
is darker (a lower threshold) with bright film edges and labels (a border left out
and a stronger opening), and its chest wall is not always on the side the
laterality says, so the side is found from the image.

Output, in ``data/processed/<dataset>/preprocessed/`` by default:

- ``images/<file_id>.png``: the preprocessed image, 16-bit (0 to 255 for CBIS-DDSM)
- ``masks/<file_id>.png``: its mass mask, 8-bit with the values 0 and 255
- ``preprocessing.json``: the settings and, for every image, what was done to it
  (crop box, size, flip), which maps a prediction back onto the original image

Run it with ``python -m mammo_seg.data.preprocess_dataset [--dataset cbis-ddsm]``.
"""

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass as std_dataclass
from dataclasses import replace
from pathlib import Path
from typing import Final, Literal

import numpy as np
from PIL import Image
from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass
from pydicom.errors import InvalidDicomError
from scipy import ndimage

from mammo_seg.data.manifest import read_manifest
from mammo_seg.data.masks import mask_path, read_mask, write_mask
from mammo_seg.data.paths import (
    INBREAST_PROCESSED_DIR,
    INBREAST_RAW_DIR,
    PROCESSED_DIRS,
    RAW_DIRS,
)
from mammo_seg.data.preprocessing import (
    DEFAULT_SIZE,
    Mask,
    Pixels,
    PreprocessingSettings,
    PreprocessParams,
    preprocess,
    preprocess_mask,
    read_pixels,
)
from mammo_seg.data.schema import Dataset, FileId, Manifest

DEFAULT_MANIFEST = INBREAST_PROCESSED_DIR / "manifest.json"
DEFAULT_DATASET_DIR = INBREAST_RAW_DIR
DEFAULT_MASKS_DIR = INBREAST_PROCESSED_DIR / "masks"
DEFAULT_OUTPUT_DIR = INBREAST_PROCESSED_DIR / "preprocessed"
RECORD_FILE = "preprocessing.json"
RECORD_VERSION: Final = 1
# The settings of each dataset. On CBIS-DDSM a threshold of 40 cuts masses off the
# crop; with 20, the margin of 2% still cut 3 of them by a few pixels (masks drawn
# past the dim skin line, and a mass at the image edge, behind the border left
# out), which a margin of 3% already keeps.
DATASET_SETTINGS: Final[dict[Dataset, PreprocessingSettings]] = {
    "inbreast": PreprocessingSettings(),
    "cbis-ddsm": PreprocessingSettings(
        threshold=20,
        margin=0.04,
        opening=3,
        border=2,
        max_intensity=255,
        orientation="image",
    ),
}

# A mask averaged down to the new size keeps its area, up to the pixels on its edge:
# up to 5% of the expected area plus half a pixel along the edge. On the INbreast
# masses the rescaled area is within 1.3% of the expected one.
MASK_AREA_TOLERANCE = 0.05
MASK_EDGE_TOLERANCE = 0.5  # per pixel on the edge of the rescaled mask


class PreprocessingError(ValueError):
    """Some images could not be preprocessed. ``problems`` lists every issue."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("preprocessing failed:\n- " + "\n- ".join(problems))


@dataclass(frozen=True, config=ConfigDict(strict=True, extra="forbid"))
class PreprocessingRecord:
    """``preprocessing.json``: the settings and what was done to every image."""

    version: Literal[1]
    settings: PreprocessingSettings
    images: dict[FileId, PreprocessParams]


_RECORD = TypeAdapter(PreprocessingRecord)


@std_dataclass(frozen=True)
class PreprocessingSummary:
    images: int
    images_with_masks: int
    shortest_side: tuple[int, int]  # smallest and largest short side of the results


def image_path(output_dir: Path, file_id: str) -> Path:
    return output_dir / "images" / f"{file_id}.png"


def processed_mask_path(output_dir: Path, file_id: str) -> Path:
    return output_dir / "masks" / f"{file_id}.png"


def write_image(pixels: Pixels, path: Path) -> None:
    """Write a ``uint16`` image as a 16-bit PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def read_image(path: Path) -> Pixels:
    """Read an image written by ``write_image`` as a ``uint16`` array."""
    with Image.open(path) as image:
        pixels = np.asarray(image)
    if pixels.dtype != np.uint16:
        raise ValueError(f"{path.name}: expected a 16-bit image, got {pixels.dtype}")
    return pixels


def write_record(record: PreprocessingRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_RECORD.dump_json(record, indent=2) + b"\n")


def read_record(path: Path) -> PreprocessingRecord:
    """Read ``preprocessing.json``; raises ``pydantic.ValidationError`` if invalid."""
    return _RECORD.validate_json(path.read_text())


def check_settings(
    record: PreprocessingRecord, expected: PreprocessingSettings
) -> None:
    """Raise ValueError unless the data was preprocessed with the ``expected`` settings.

    A model has to be trained and used with the same settings as the data it reads.
    """
    if record.settings != expected:
        raise ValueError(
            "the data was preprocessed with different settings than expected: "
            f"{record.settings} instead of {expected}"
        )


def check_mask_kept(mask: Mask, rescaled: Mask, params: PreprocessParams) -> None:
    """Raise ValueError if a lesion vanished or changed size when rescaling."""
    if not mask.any():
        if rescaled.any():
            raise ValueError("an empty mask has pixels after rescaling")
        return
    if not rescaled.any():
        raise ValueError("the mask vanishes when rescaled")
    scale = (params.shape[0] / params.box.height) * (params.shape[1] / params.box.width)
    expected = mask.sum() * scale
    edge = (rescaled == 1) & ~ndimage.binary_erosion(rescaled == 1)
    tolerance = MASK_AREA_TOLERANCE * expected + MASK_EDGE_TOLERANCE * edge.sum()
    if abs(rescaled.sum() - expected) > tolerance:
        raise ValueError(
            f"the rescaled mask has {rescaled.sum() / expected:.2f} times its "
            "expected area"
        )


def preprocess_dataset(
    manifest: Manifest,
    dataset_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    settings: PreprocessingSettings = PreprocessingSettings(),  # noqa: B008
) -> PreprocessingSummary:
    """Preprocess every image and its mask and write them with ``preprocessing.json``.

    Raises PreprocessingError listing every image that fails. Images that fail are
    not written and the record file is not written.
    """
    problems: list[str] = []
    params_of: dict[str, PreprocessParams] = {}
    with_masks = 0
    short_sides: list[int] = []
    for image in manifest.images:
        try:
            pixels = read_pixels(dataset_dir / image.image_path)
            if pixels.shape != (image.rows, image.cols):
                raise ValueError(
                    f"the image is {pixels.shape} but the manifest says "
                    f"{(image.rows, image.cols)}"
                )
            result, params = preprocess(pixels, image.laterality, settings)
            mask = read_mask(mask_path(masks_dir, image.file_id))
            rescaled = preprocess_mask(mask, params)
            check_mask_kept(mask, rescaled, params)
        except (OSError, ValueError, InvalidDicomError) as error:
            problems.append(f"image {image.file_id}: {error}")
            continue
        write_image(result, image_path(output_dir, image.file_id))
        write_mask(rescaled, processed_mask_path(output_dir, image.file_id))
        params_of[image.file_id] = params
        with_masks += int(rescaled.any())
        short_sides.append(min(result.shape))
    if problems:
        raise PreprocessingError(problems)
    record = PreprocessingRecord(
        version=RECORD_VERSION, settings=settings, images=params_of
    )
    write_record(record, output_dir / RECORD_FILE)
    return PreprocessingSummary(
        images=len(params_of),
        images_with_masks=with_masks,
        shortest_side=(min(short_sides, default=0), max(short_sides, default=0)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Crop, rescale and flip every image of the manifest and store it."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_SETTINGS),
        default="inbreast",
        help="its folders are the defaults of the paths, and its settings are used",
    )
    parser.add_argument(
        "--manifest", type=Path, help="default: data/processed/<dataset>/manifest.json"
    )
    parser.add_argument(
        "--dataset-dir", type=Path, help="the raw dataset (default: that of --dataset)"
    )
    parser.add_argument(
        "--masks-dir", type=Path, help="default: data/processed/<dataset>/masks"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: data/processed/<dataset>/preprocessed",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="length of the longer side of the results, in pixels",
    )
    args = parser.parse_args(argv)
    dataset: Dataset = args.dataset
    processed = PROCESSED_DIRS[dataset]
    output_dir = args.output_dir or processed / "preprocessed"

    started = time.time()
    try:
        settings = replace(DATASET_SETTINGS[dataset], size=args.size)
        summary = preprocess_dataset(
            read_manifest(args.manifest or processed / "manifest.json"),
            args.dataset_dir or RAW_DIRS[dataset],
            args.masks_dir or processed / "masks",
            output_dir,
            settings,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"Wrote {summary.images} images and masks to {output_dir} in "
        f"{time.time() - started:.0f}s: longer side {settings.size}, shorter side "
        f"{summary.shortest_side[0]} to {summary.shortest_side[1]}, "
        f"{summary.images_with_masks} with a mass"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
