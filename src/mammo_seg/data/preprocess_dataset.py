"""Preprocess every image of the manifest once and store the result on disk.

This is the cache of the first steps of the preprocessing chain (crop, rescale,
flip, see ``preprocessing.py``). Training reads these files instead of decoding
the DICOMs on every epoch; inference runs the same ``preprocess`` function on the
incoming image, so both see identical input. The intensity normalization is not
applied here: the images keep their 16-bit values and are normalized when loaded.

Output, in ``data/processed/preprocessed/`` by default:

- ``images/<file_id>.png``: the preprocessed image, 16-bit
- ``masks/<file_id>.png``: its mass mask, 8-bit with the values 0 and 255
- ``preprocessing.json``: the settings and, for every image, what was done to it
  (crop box, size, flip), which maps a prediction back onto the original image

Run it with ``python -m mammo_seg.data.preprocess_dataset``.
"""

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass as std_dataclass
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
from mammo_seg.data.schema import FileId, Manifest

DEFAULT_MANIFEST = Path("data/processed/manifest.json")
DEFAULT_DATASET_DIR = Path("data/raw/INbreast Release 1.0")
DEFAULT_MASKS_DIR = Path("data/processed/masks")
DEFAULT_OUTPUT_DIR = Path("data/processed/preprocessed")
RECORD_FILE = "preprocessing.json"
RECORD_VERSION: Final = 1

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
            pixels = read_pixels(dataset_dir / image.dicom_path)
            if pixels.shape != (image.rows, image.cols):
                raise ValueError(
                    f"the DICOM is {pixels.shape} but the manifest says "
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
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="length of the longer side of the results, in pixels",
    )
    args = parser.parse_args(argv)

    started = time.time()
    try:
        settings = PreprocessingSettings(size=args.size)
        summary = preprocess_dataset(
            read_manifest(args.manifest),
            args.dataset_dir,
            args.masks_dir,
            args.output_dir,
            settings,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"Wrote {summary.images} images and masks to {args.output_dir} in "
        f"{time.time() - started:.0f}s: longer side {settings.size}, shorter side "
        f"{summary.shortest_side[0]} to {summary.shortest_side[1]}, "
        f"{summary.images_with_masks} with a mass"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
