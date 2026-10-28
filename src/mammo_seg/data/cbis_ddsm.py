"""Build and validate the manifest of CBIS-DDSM: its masses, on the full images.

Reads the Kaggle copy of the dataset: ``csv/`` and ``jpeg/<series>/``. The two
mass CSVs (the official train and test sets) have one row per mass: its image,
its mask and what was recorded about it (pathology, BI-RADS assessment,
subtlety, shape, margins) and the breast density of the image. A path in the CSVs
names a DICOM series; its JPEG files are in ``jpeg/<series>/``. The series of a
full image has one file, that of a mass has its mask and its crop, and the mask is
the file with the size of the full image. The calcification CSVs are not read, so
calcifications on these images are not annotated.

Masks are 8-bit JPEGs, nearly binary: a pixel above ``MASK_THRESHOLD`` is mass.
An image with a mass that has no mask of its size is left out (``dropped``), as a
mass left unannotated would teach the model that it is background. The JPEGs keep
no pixel size and the CSVs no acquisition date (the dates of ``dicom_info.csv``
are those of the conversion) and no BI-RADS of the whole image, so these are None.

Run it with ``python -m mammo_seg.data.manifest --dataset cbis-ddsm``.
"""

import csv
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, NamedTuple, get_args

import numpy as np
from PIL import Image
from pydantic import ValidationError

from mammo_seg.data.schema import (
    SCHEMA_VERSION,
    DroppedImage,
    ImageRecord,
    LesionType,
    Manifest,
    ManifestError,
    Pathology,
    Roi,
    View,
    describe,
)

MASS_CSVS: Final[dict[Literal["train", "test"], str]] = {
    "train": "csv/mass_case_description_train_set.csv",
    "test": "csv/mass_case_description_test_set.csv",
}
MASK_THRESHOLD: Final = 127
# JPEG compression leaves a few grey pixels at the edge of a mask; a mask with more
# than this share of mid-grey pixels (64 to 191) is not a binary mask.
MAX_GREY_SHARE: Final = 0.001

_COLUMNS = [
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
    "ROI mask file path",
]
_IMAGE_ID = re.compile(
    r"Mass-(?P<split>Training|Test)_(?P<patient_id>P_\d+)"
    r"_(?P<side>LEFT|RIGHT)_(?P<view>CC|MLO)"
)
_SPLITS = {"Training": "train", "Test": "test"}
_SERIES = re.compile(r"\d+(\.\d+)+")
_NO_VALUE = {"", "N/A"}  # how the CSVs write a missing shape or margins
_PATHOLOGIES: dict[str, Pathology] = {name: name for name in get_args(Pathology)}


class _Mass(NamedTuple):
    """One row of a mass CSV, checked and converted."""

    image_id: str
    split: Literal["train", "test"]
    patient_id: str
    laterality: Literal["L", "R"]
    view: View
    density: int
    abnormality_id: int
    full_series: str
    mask_series: str
    pathology: Pathology
    assessment: int
    subtlety: int
    shape: str | None
    margins: str | None


def _series(path: str, what: str, problems: list[str], where: str) -> str | None:
    """The series of a CSV path ``<folder>/<study>/<series>/<file>``."""
    parts = path.strip().split("/")
    if len(parts) != 4 or _SERIES.fullmatch(parts[2]) is None:
        problems.append(f"{where}: unexpected {what} path {path.strip()!r}")
        return None
    return parts[2]


def _parse_row(
    row: dict[str, str], split: Literal["train", "test"], where: str
) -> tuple[_Mass | None, list[str]]:
    problems: list[str] = []
    values = {name: (row[name] or "").strip() for name in _COLUMNS}
    image_id = values["image file path"].split("/")[0]
    match = _IMAGE_ID.fullmatch(image_id)
    if match is None:
        return None, [f"{where}: unexpected image folder {image_id!r}"]
    expected = (split, values["patient_id"], values["left or right breast"])
    found = (_SPLITS[match["split"]], match["patient_id"], match["side"])
    if expected != found or values["image view"] != match["view"]:
        problems.append(
            f"{where}: image folder {image_id} does not match the split, patient, "
            "breast and view of the row"
        )
    if values["abnormality type"] != "mass":
        problems.append(
            f"{where}: abnormality type {values['abnormality type']!r}, not 'mass'"
        )
    full_series = _series(values["image file path"], "image", problems, where)
    mask_series = _series(values["ROI mask file path"], "mask", problems, where)
    mask_folder = values["ROI mask file path"].split("/")[0]
    if mask_folder != f"{image_id}_{values['abnormality id']}":
        problems.append(
            f"{where}: mask folder {mask_folder!r} is not that of mass "
            f"{values['abnormality id']} of {image_id}"
        )
    pathology = _PATHOLOGIES.get(values["pathology"])
    if pathology is None:
        problems.append(f"{where}: unknown pathology {values['pathology']!r}")
    numbers = {}
    for name in ("breast_density", "abnormality id", "assessment", "subtlety"):
        try:
            numbers[name] = int(values[name])
        except ValueError:
            problems.append(f"{where}: invalid {name} {values[name]!r}")
    if problems or full_series is None or mask_series is None or pathology is None:
        return None, problems
    return (
        _Mass(
            image_id=image_id,
            split=split,
            patient_id=match["patient_id"],
            laterality="L" if match["side"] == "LEFT" else "R",
            view=View(match["view"]),
            density=numbers["breast_density"],
            abnormality_id=numbers["abnormality id"],
            full_series=full_series,
            mask_series=mask_series,
            pathology=pathology,
            assessment=numbers["assessment"],
            subtlety=numbers["subtlety"],
            shape=None if values["mass shape"] in _NO_VALUE else values["mass shape"],
            margins=(
                None if values["mass margins"] in _NO_VALUE else values["mass margins"]
            ),
        ),
        [],
    )


def _read_masses(dataset_dir: Path, problems: list[str]) -> dict[str, list[_Mass]]:
    """The masses of both mass CSVs, by image."""
    masses: dict[str, list[_Mass]] = defaultdict(list)
    for split, name in MASS_CSVS.items():
        with open(dataset_dir / name, newline="") as file:
            reader = csv.DictReader(file)
            missing = [c for c in _COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                problems.append(f"{name} is missing columns: {missing}")
                continue
            rows = list(reader)
        for line, row in enumerate(rows, start=2):
            mass, row_problems = _parse_row(row, split, f"{name} line {line}")
            problems.extend(row_problems)
            if mass is not None:
                masses[mass.image_id].append(mass)
    return masses


def _check_image(image_id: str, masses: Sequence[_Mass], problems: list[str]) -> None:
    """The rows of one image agree on the image and name each mass once."""
    for field, what in (("full_series", "image series"), ("density", "density")):
        if len({getattr(mass, field) for mass in masses}) > 1:
            problems.append(f"image {image_id}: its masses have more than one {what}")
    ids = [mass.abnormality_id for mass in masses]
    if len(set(ids)) < len(ids):
        problems.append(f"image {image_id}: an abnormality id appears more than once")
    series = [mass.mask_series for mass in masses]
    if len(set(series)) < len(series):
        problems.append(f"image {image_id}: two masses have the same mask series")


def _jpegs(dataset_dir: Path, series: str) -> list[Path]:
    return sorted((dataset_dir / "jpeg" / series).glob("*.jpg"))


def _read_header(path: Path) -> tuple[str, int, int]:
    """The mode, rows and columns of an image, without decoding it."""
    with Image.open(path) as image:
        return image.mode, image.height, image.width


def mask_area(path: Path) -> int:
    """The pixels of the mass in a mask JPEG: those above ``MASK_THRESHOLD``.

    Raises ValueError when the mask is not a nearly binary grey image or is empty.
    """
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"mode {image.mode}, not a grey image")
        pixels = np.asarray(image)
    grey = np.count_nonzero((pixels >= 64) & (pixels <= 191)) / pixels.size
    if grey > MAX_GREY_SHARE:
        raise ValueError(f"{grey:.2%} of the pixels are mid-grey, not a binary mask")
    area = int(np.count_nonzero(pixels > MASK_THRESHOLD))
    if area == 0:
        raise ValueError("the mask is empty")
    return area


def _build_image(
    dataset_dir: Path, image_id: str, masses: Sequence[_Mass], problems: list[str]
) -> ImageRecord | DroppedImage | None:
    """The record of one image; the reason it is dropped; or None, with problems."""
    first = masses[0]
    full = _jpegs(dataset_dir, first.full_series)
    if len(full) != 1:
        problems.append(
            f"image {image_id}: {len(full)} JPEG files for its series "
            f"{first.full_series}, not 1"
        )
        return None
    mode, rows, cols = _read_header(full[0])
    if mode != "L":
        problems.append(f"image {image_id}: mode {mode}, not a grey image")
        return None

    rois, missing = [], []
    for mass in sorted(masses, key=lambda mass: mass.abnormality_id):
        where = f"image {image_id}, mass {mass.abnormality_id}"
        candidates = _jpegs(dataset_dir, mass.mask_series)
        if not candidates:
            problems.append(f"{where}: no JPEG files for its series {mass.mask_series}")
            continue
        masks = [p for p in candidates if _read_header(p)[1:] == (rows, cols)]
        if len(masks) > 1:
            problems.append(f"{where}: {len(masks)} files of the size of the image")
            continue
        if not masks:
            missing.append(mass.abnormality_id)
            continue
        try:
            area = mask_area(masks[0])
        except ValueError as error:
            problems.append(f"{where}: {masks[0].name}: {error}")
            continue
        try:
            rois.append(
                Roi(
                    lesion_type=LesionType.MASS,
                    mask_path=masks[0].relative_to(dataset_dir).as_posix(),
                    mask_area_px=area,
                    abnormality_id=mass.abnormality_id,
                    pathology=mass.pathology,
                    assessment=mass.assessment,
                    subtlety=mass.subtlety,
                    shape=mass.shape,
                    margins=mass.margins,
                )
            )
        except ValidationError as error:
            problems.append(f"{where}: {describe(error)}")
    if missing:
        numbers = ", ".join(str(number) for number in missing)
        return DroppedImage(
            file_id=image_id,
            reason=f"no mask of the size of the image for mass {numbers}",
        )
    if len(rois) < len(masses):
        return None
    try:
        return ImageRecord(
            file_id=image_id,
            patient_id=first.patient_id,
            laterality=first.laterality,
            view=first.view,
            image_path=full[0].relative_to(dataset_dir).as_posix(),
            rows=rows,
            cols=cols,
            pixel_size_mm=None,
            acquisition_date=None,
            acr=first.density,
            birads=None,
            official_split=first.split,
            has_annotation=True,
            rois=tuple(rois),
        )
    except ValidationError as error:
        problems.append(f"image {image_id}: {describe(error)}")
        return None


def build_cbis_ddsm_manifest(dataset_dir: Path) -> Manifest:
    """Build the manifest of the masses of CBIS-DDSM in ``dataset_dir``.

    ``dataset_dir`` is the folder that contains ``csv`` and ``jpeg``. Raises
    ManifestError listing every inconsistency found.
    """
    problems: list[str] = []
    masses = _read_masses(dataset_dir, problems)
    for image_id in sorted(masses):
        _check_image(image_id, masses[image_id], problems)
    if problems:
        raise ManifestError(problems)

    images, dropped = [], []
    for image_id in sorted(masses):
        built = _build_image(dataset_dir, image_id, masses[image_id], problems)
        if isinstance(built, ImageRecord):
            images.append(built)
        elif isinstance(built, DroppedImage):
            dropped.append(built)
    if problems:
        raise ManifestError(problems)
    return Manifest(
        version=SCHEMA_VERSION,
        dataset="cbis-ddsm",
        images=tuple(images),
        dropped=tuple(dropped),
    )
