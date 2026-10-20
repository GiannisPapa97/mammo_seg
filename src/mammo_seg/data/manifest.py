"""Build and validate the manifest of the INbreast dataset.

The manifest has one record per image and combines what the raw release spreads
over several places: the DICOM filename (patient id, laterality, view),
``INbreast.csv`` (date, BI-RADS, ACR density), the DICOM header (image size) and
the ``.xml`` annotations (every lesion with its type and polygon). Later steps
read the manifest instead of the raw files. The schema is in ``schema.py``.

Run it with ``python -m mammo_seg.data.manifest``.
"""

import argparse
import csv
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple

import pandas as pd
import pydicom

from mammo_seg.data.annotations import read_rois
from mammo_seg.data.schema import (
    SCHEMA_VERSION,
    ImageRecord,
    Manifest,
    Roi,
    View,
    image_to_json,
    manifest_from_json,
)

DEFAULT_DATASET_DIR = Path("data/raw/INbreast Release 1.0")
DEFAULT_OUTPUT = Path("data/processed/manifest.json")

IMAGE_COLUMNS = [
    "file_id",
    "patient_id",
    "laterality",
    "view",
    "dicom_path",
    "rows",
    "cols",
    "acquisition_date",
    "acr",
    "birads",
    "has_annotation",
    "n_masses",
    "has_mass",
]
ROI_COLUMNS = ["file_id", "roi_index", "lesion_type", "n_points", "area_px"]

_FILENAME = re.compile(
    r"(?P<file_id>\d+)_(?P<patient_id>[0-9a-f]+)"
    r"_MG_(?P<laterality>[LR])_(?P<view>[A-Z]+)_ANON"
)
_CSV_COLUMNS = ["File Name", "Laterality", "View", "Acquisition date", "ACR", "Bi-Rads"]
# The filenames spell the mediolateral oblique view ML, the CSV and the manifest MLO.
_FILENAME_VIEWS = {"ML": View.MLO}


class ManifestError(ValueError):
    """The raw dataset files are inconsistent. ``problems`` lists every issue."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("invalid dataset:\n- " + "\n- ".join(problems))


class _Dicom(NamedTuple):
    file_id: str
    patient_id: str
    laterality: Literal["L", "R"]
    view: View
    dicom_path: str


class _CsvRow(NamedTuple):
    laterality: str
    view: str
    acquisition_date: str
    acr: int | None
    birads: str


def _list_dicoms(dataset_dir: Path, problems: list[str]) -> dict[str, _Dicom]:
    dicoms = []
    for path in sorted((dataset_dir / "AllDICOMs").glob("*.dcm")):
        match = _FILENAME.fullmatch(path.stem)
        if match is None:
            problems.append(f"unexpected DICOM filename: {path.name}")
            continue
        try:
            view = _FILENAME_VIEWS.get(match["view"]) or View(match["view"])
        except ValueError:
            problems.append(
                f"unexpected view {match['view']!r} in DICOM filename: {path.name}"
            )
            continue
        laterality: Literal["L", "R"] = "L" if match["laterality"] == "L" else "R"
        dicoms.append(
            _Dicom(
                file_id=match["file_id"],
                patient_id=match["patient_id"],
                laterality=laterality,
                view=view,
                dicom_path=path.relative_to(dataset_dir).as_posix(),
            )
        )
    counts = Counter(dicom.file_id for dicom in dicoms)
    for file_id in sorted(file_id for file_id, count in counts.items() if count > 1):
        problems.append(f"file id {file_id} appears in more than one DICOM filename")
    return {dicom.file_id: dicom for dicom in dicoms}


def _read_metadata(dataset_dir: Path, problems: list[str]) -> dict[str, _CsvRow]:
    with open(dataset_dir / "INbreast.csv", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        missing = [n for n in _CSV_COLUMNS if n not in (reader.fieldnames or [])]
        if missing:
            problems.append(f"INbreast.csv is missing columns: {missing}")
            return {}
        rows = list(reader)
    metadata: dict[str, _CsvRow] = {}
    for row in rows:
        values = {name: (row[name] or "").strip() for name in _CSV_COLUMNS}
        file_id = values["File Name"]
        if file_id in metadata:
            problems.append(
                f"file id {file_id} appears in more than one INbreast.csv row"
            )
            continue
        try:
            # One image has a blank density value.
            acr = int(values["ACR"]) if values["ACR"] else None
        except ValueError:
            problems.append(f"image {file_id}: invalid ACR {values['ACR']!r}")
            continue
        metadata[file_id] = _CsvRow(
            laterality=values["Laterality"],
            view=values["View"],
            acquisition_date=values["Acquisition date"],
            acr=acr,
            birads=values["Bi-Rads"],
        )
    return metadata


def _read_annotations(
    dataset_dir: Path, problems: list[str]
) -> dict[str, tuple[Roi, ...]]:
    annotations: dict[str, tuple[Roi, ...]] = {}
    for path in sorted((dataset_dir / "AllXML").glob("*.xml")):
        try:
            annotations[path.stem] = tuple(read_rois(path))
        except ValueError as error:
            problems.append(str(error))
    return annotations


def _read_size(dataset_dir: Path, dicom_path: str) -> tuple[int, int]:
    header = pydicom.dcmread(dataset_dir / dicom_path, stop_before_pixels=True)
    return int(header.Rows), int(header.Columns)


def build_manifest(dataset_dir: Path) -> Manifest:
    """Build the manifest of the INbreast release in ``dataset_dir``.

    ``dataset_dir`` is the folder that contains ``AllDICOMs``, ``AllXML`` and
    ``INbreast.csv``. Raises ManifestError listing every inconsistency found.
    """
    problems: list[str] = []
    dicoms = _list_dicoms(dataset_dir, problems)
    metadata = _read_metadata(dataset_dir, problems)
    annotations = _read_annotations(dataset_dir, problems)

    for file_id in sorted(dicoms.keys() - metadata.keys()):
        problems.append(f"no INbreast.csv row for image {file_id}")
    for file_id in sorted(metadata.keys() - dicoms.keys()):
        problems.append(f"INbreast.csv row {file_id} has no DICOM image")
    for file_id in sorted(annotations.keys() - dicoms.keys()):
        problems.append(f"annotation {file_id}.xml has no DICOM image")
    for file_id in sorted(dicoms.keys() & metadata.keys()):
        dicom, row = dicoms[file_id], metadata[file_id]
        if dicom.laterality != row.laterality:
            problems.append(
                f"image {file_id}: filename laterality {dicom.laterality}, "
                f"INbreast.csv laterality {row.laterality}"
            )
        if dicom.view != row.view:
            problems.append(
                f"image {file_id}: filename view {dicom.view}, "
                f"INbreast.csv view {row.view}"
            )
    if problems:
        raise ManifestError(problems)

    images = []
    for file_id in sorted(dicoms):
        dicom, row = dicoms[file_id], metadata[file_id]
        rows, cols = _read_size(dataset_dir, dicom.dicom_path)
        images.append(
            ImageRecord(
                file_id=file_id,
                patient_id=dicom.patient_id,
                laterality=dicom.laterality,
                view=dicom.view,
                dicom_path=dicom.dicom_path,
                rows=rows,
                cols=cols,
                acquisition_date=row.acquisition_date,
                acr=row.acr,
                birads=row.birads,
                has_annotation=file_id in annotations,
                rois=annotations.get(file_id, ()),
            )
        )
    return Manifest(version=SCHEMA_VERSION, images=tuple(images))


def write_manifest(manifest: Manifest, path: Path) -> None:
    """Write the manifest as JSON, one image per line so it stays readable."""
    lines = ",\n".join("  " + image_to_json(image) for image in manifest.images)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{{"version": {manifest.version}, "images": [\n{lines}\n]}}\n')


def read_manifest(path: Path) -> Manifest:
    """Read a manifest written by ``write_manifest``.

    Raises ``pydantic.ValidationError`` (a ``ValueError``) if the file does not
    match the schema.
    """
    return manifest_from_json(path.read_text())


def images_frame(manifest: Manifest) -> pd.DataFrame:
    """One row per image, with the number of masses and the positive label.

    ``n_masses`` and ``has_mass`` are computed from the ROIs; they are not stored.
    """
    frame = pd.DataFrame(
        [
            {column: getattr(image, column) for column in IMAGE_COLUMNS}
            for image in manifest.images
        ],
        columns=pd.Index(IMAGE_COLUMNS),
    )
    frame["view"] = frame["view"].map(lambda view: view.value)
    frame["acr"] = frame["acr"].astype("Int64")
    return frame


def rois_frame(manifest: Manifest) -> pd.DataFrame:
    """One row per ROI, with its lesion type and polygon area."""
    return pd.DataFrame(
        [
            {
                "file_id": image.file_id,
                "roi_index": index,
                "lesion_type": roi.lesion_type.value,
                "n_points": len(roi.points),
                "area_px": roi.area_px,
            }
            for image in manifest.images
            for index, roi in enumerate(image.rois)
        ],
        columns=pd.Index(ROI_COLUMNS),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    try:
        manifest = build_manifest(args.dataset_dir)
    except ManifestError as error:
        print(error, file=sys.stderr)
        return 1
    write_manifest(manifest, args.output)
    images = images_frame(manifest)
    print(
        f"Wrote {args.output}: {len(images)} images, "
        f"{images.patient_id.nunique()} patients, "
        f"{int(images.has_mass.sum())} with a mass, "
        f"{sum(len(image.rois) for image in manifest.images)} ROIs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
