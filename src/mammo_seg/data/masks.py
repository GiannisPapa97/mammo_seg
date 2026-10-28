"""Binary segmentation masks from the lesions of the manifest.

A mask has the size of its image and holds 1 where a lesion of the chosen type
(masses, by default) covers the pixel and 0 elsewhere. Several lesions of one
image are merged into one mask. INbreast has no mask images, only polygons, which
are drawn; CBIS-DDSM has a mask image per lesion, which is read from the dataset
folder (``cbis_ddsm.read_lesion_mask``) and checked against the area of the
manifest.

Polygons are in continuous image coordinates, ``(x, y)`` = ``(column, row)``, with
the origin at the top-left corner, so the centre of pixel ``(row, column)`` is at
``(column + 0.5, row + 0.5)``. A pixel is set when its centre is inside the
polygon (even-odd rule). Polygons that leave the image are clipped to it.

Masks are stored at the native image size as PNG files with the values 0 and 255,
one per image, in ``data/processed/<dataset>/masks/<file_id>.png``. Writing them
for the whole manifest also checks every mask against its lesions and fails
loudly.

Run it with ``python -m mammo_seg.data.masks [--dataset cbis-ddsm]``.
"""

import argparse
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from mammo_seg.data.cbis_ddsm import read_lesion_mask
from mammo_seg.data.manifest import read_manifest
from mammo_seg.data.paths import INBREAST_PROCESSED_DIR, PROCESSED_DIRS, RAW_DIRS
from mammo_seg.data.schema import Dataset, ImageRecord, LesionType, Manifest, Roi

DEFAULT_MANIFEST = INBREAST_PROCESSED_DIR / "manifest.json"
DEFAULT_OUTPUT_DIR = INBREAST_PROCESSED_DIR / "masks"

Mask = npt.NDArray[np.uint8]
Shape = tuple[int, int]  # (rows, columns)
Span = tuple[int, int, int]  # (row, start column, stop column), stop exclusive

# A polygon rasterized with the pixel-centre rule covers its area up to the pixels
# along its edge, so the pixel count and the shoelace area differ slightly. On the
# INbreast masses the difference is at most 0.4% of the area and 0.05 pixels per
# pixel of perimeter; a polygon that crosses itself differs far more.
AREA_TOLERANCE = 0.005  # of the polygon area
AREA_TOLERANCE_PER_EDGE_PIXEL = 0.1  # per pixel of perimeter


class MaskError(ValueError):
    """Masks do not match their polygons. ``problems`` lists every issue."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("invalid masks:\n- " + "\n- ".join(problems))


def _spans(points: npt.NDArray[np.float64], rows: int, cols: int) -> Iterator[Span]:
    """The ``(row, start, stop)`` runs of pixels whose centre is inside the polygon.

    ``stop`` is exclusive. The runs are clipped to the image. Edges are treated as
    half-open in ``y``, so a polygon vertex on a pixel-centre row is counted once.
    """
    x1, y1 = points[:, 0], points[:, 1]
    next_points = np.roll(points, -1, axis=0)
    x2, y2 = next_points[:, 0], next_points[:, 1]
    first = max(0, math.ceil(y1.min() - 0.5))
    last = min(rows, math.ceil(y1.max() - 0.5))
    for row in range(first, last):
        centre = row + 0.5
        crossing = ((y1 <= centre) & (centre < y2)) | ((y2 <= centre) & (centre < y1))
        x_at = x1[crossing] + (centre - y1[crossing]) * (
            (x2[crossing] - x1[crossing]) / (y2[crossing] - y1[crossing])
        )
        x_at.sort()
        starts = np.maximum(np.ceil(x_at[0::2] - 0.5), 0).astype(int)
        stops = np.minimum(np.ceil(x_at[1::2] - 0.5), cols).astype(int)
        for start, stop in zip(starts, stops, strict=True):
            if stop > start:
                yield row, int(start), int(stop)


def _as_array(roi: Roi) -> npt.NDArray[np.float64]:
    return np.array(roi.points, dtype=np.float64).reshape(-1, 2)


def polygon_pixels(roi: Roi, shape: Shape) -> int:
    """The number of pixels of an image of ``shape`` that the polygon covers."""
    if len(roi.points) < 3:
        return 0
    rows, cols = shape
    return sum(stop - start for _, start, stop in _spans(_as_array(roi), rows, cols))


def rasterize(
    rois: Sequence[Roi],
    shape: Shape,
    lesion_type: LesionType = LesionType.MASS,
) -> Mask:
    """The mask of all ROIs of ``lesion_type``, merged, for an image of ``shape``.

    ROIs with fewer than 3 points (single-point calcifications) have no area and
    leave the mask empty.
    """
    rows, cols = shape
    mask = np.zeros(shape, dtype=np.uint8)
    for roi in rois:
        if roi.lesion_type is not lesion_type or len(roi.points) < 3:
            continue
        for row, start, stop in _spans(_as_array(roi), rows, cols):
            mask[row, start:stop] = 1
    return mask


def mask_problems(
    mask: Mask,
    rois: Sequence[Roi],
    shape: Shape,
    lesion_type: LesionType = LesionType.MASS,
) -> list[str]:
    """What is wrong with a mask, compared with the polygons it was made from."""
    problems = []
    if mask.shape != shape:
        problems.append(f"mask shape {mask.shape} is not the image shape {shape}")
        return problems
    if mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
        problems.append("mask is not binary (uint8 with the values 0 and 1)")
    rows, cols = shape
    pixels = []
    for index, roi in enumerate(r for r in rois if r.lesion_type is lesion_type):
        label = f"{lesion_type.value} {index}"
        if len(roi.points) < 3:
            problems.append(f"{label}: fewer than 3 points, so no polygon")
            continue
        points = _as_array(roi)
        covered = polygon_pixels(roi, shape)
        pixels.append(covered)
        if covered == 0:
            problems.append(f"{label}: outside the image or smaller than a pixel")
            continue
        inside = (
            points.min() >= 0
            and points[:, 0].max() <= cols
            and points[:, 1].max() <= rows
        )
        if inside:
            edges = np.hypot(*(np.roll(points, -1, axis=0) - points).T).sum()
            tolerance = (
                AREA_TOLERANCE * roi.area_px + AREA_TOLERANCE_PER_EDGE_PIXEL * edges
            )
            if abs(covered - roi.area_px) > tolerance:
                problems.append(
                    f"{label}: covers {covered} pixels but the polygon area is "
                    f"{roi.area_px:.0f}; the polygon may cross itself"
                )
    total = int(mask.sum())
    if pixels and not max(pixels) <= total <= sum(pixels):
        problems.append(
            f"mask has {total} pixels, outside the {max(pixels)} to {sum(pixels)} "
            f"covered by its polygons"
        )
    if not pixels and total:
        problems.append(f"mask has {total} pixels but there are no polygons")
    return problems


def mask_path(output_dir: Path, file_id: str) -> Path:
    return output_dir / f"{file_id}.png"


def write_mask(mask: Mask, path: Path) -> None:
    """Write a 0/1 mask as an 8-bit PNG with the values 0 and 255."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask * 255, mode="L").save(path)


def read_mask(path: Path) -> Mask:
    """Read a mask written by ``write_mask`` as a 0/1 array."""
    with Image.open(path) as image:
        return (np.asarray(image) > 0).astype(np.uint8)


@dataclass(frozen=True)
class MaskSummary:
    images: int
    images_with_masks: int
    masses: int
    clipped_masses: int  # polygons that leave the image
    mask_pixels: int


def _lesion_masks(
    image: ImageRecord, dataset_dir: Path | None
) -> tuple[Mask, list[str]]:
    """The merged mask images of the masses of an image, and what is wrong."""
    shape = (image.rows, image.cols)
    mask = np.zeros(shape, dtype=np.uint8)
    rois = [r for r in image.rois if r.mask_path is not None]
    if not rois:
        return mask, []
    if dataset_dir is None:
        return mask, ["its lesions are mask images, and no dataset folder is given"]
    problems = []
    for roi in rois:
        if roi.lesion_type is not LesionType.MASS:
            continue
        label = f"mass {roi.abnormality_id}"
        try:
            lesion = read_lesion_mask(dataset_dir / str(roi.mask_path))
        except (OSError, ValueError) as error:
            problems.append(f"{label}: {error}")
            continue
        if lesion.shape != shape:
            problems.append(
                f"{label}: mask shape {lesion.shape} is not the image shape {shape}"
            )
        elif int(lesion.sum()) != roi.mask_area_px:
            problems.append(
                f"{label}: the mask has {int(lesion.sum())} pixels, the manifest "
                f"says {roi.mask_area_px}"
            )
        else:
            mask |= lesion
    return mask, problems


def write_masks(
    manifest: Manifest, output_dir: Path, dataset_dir: Path | None = None
) -> MaskSummary:
    """Write the mass mask of every image; raises MaskError listing all problems.

    Polygons are drawn and mask images (CBIS-DDSM) are read from ``dataset_dir``,
    which they need. A mask with a problem is not written.
    """
    problems: list[str] = []
    images_with_masks = masses = clipped = pixels = 0
    for image in manifest.images:
        shape = (image.rows, image.cols)
        polygons = [roi for roi in image.rois if roi.mask_path is None]
        mask = rasterize(polygons, shape)
        found = mask_problems(mask, polygons, shape)
        lesions, lesion_problems = _lesion_masks(image, dataset_dir)
        found += lesion_problems
        if found:
            problems.extend(f"image {image.file_id}: {problem}" for problem in found)
            continue
        mask |= lesions
        write_mask(mask, mask_path(output_dir, image.file_id))
        images_with_masks += image.has_mass
        masses += image.n_masses
        clipped += sum(
            _leaves_image(roi, shape)
            for roi in polygons
            if roi.lesion_type is LesionType.MASS
        )
        pixels += int(mask.sum())
    if problems:
        raise MaskError(problems)
    return MaskSummary(
        images=len(manifest.images),
        images_with_masks=images_with_masks,
        masses=masses,
        clipped_masses=clipped,
        mask_pixels=pixels,
    )


def _leaves_image(roi: Roi, shape: Shape) -> bool:
    points = _as_array(roi)
    rows, cols = shape
    return bool(
        points.min() < 0 or points[:, 0].max() > cols or points[:, 1].max() > rows
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the mass masks of every image of the manifest."
    )
    parser.add_argument("--dataset", choices=list(RAW_DIRS), default="inbreast")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="default: data/processed/<dataset>/manifest.json",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="the raw dataset, for mask images (default: that of --dataset)",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="default: data/processed/<dataset>/masks"
    )
    args = parser.parse_args(argv)
    dataset: Dataset = args.dataset
    output_dir = args.output_dir or PROCESSED_DIRS[dataset] / "masks"

    try:
        summary = write_masks(
            read_manifest(args.manifest or PROCESSED_DIRS[dataset] / "manifest.json"),
            output_dir,
            args.dataset_dir or RAW_DIRS[dataset],
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"Wrote {summary.images} masks to {output_dir}: "
        f"{summary.images_with_masks} with masses ({summary.masses} masses, "
        f"{summary.clipped_masses} clipped to the image), "
        f"{summary.images - summary.images_with_masks} empty"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
