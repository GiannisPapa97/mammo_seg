"""Helpers that build a tiny synthetic INbreast release; no real data is used."""

import plistlib
from pathlib import Path

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid

CSV_HEADER = (
    "Patient ID;Patient age;Laterality;View;Acquisition date;File Name;ACR;Bi-Rads"
)


def write_dicom(path: Path, rows: int, cols: int) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.7")
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.Rows = rows
    dataset.Columns = cols
    dataset.save_as(path, enforce_file_format=True)


def square(x: float, y: float, side: float) -> list[str]:
    corners = [(x, y), (x + side, y), (x + side, y + side), (x, y + side)]
    return [f"({cx:.6f}, {cy:.6f})" for cx, cy in corners]


def write_annotation(path: Path, rois: list[tuple[str, list[str]]]) -> None:
    """Write an OsiriX-style property list with the given (name, points) ROIs."""
    data = {
        "Images": [
            {
                "ImageIndex": 0,
                "NumberOfROIs": len(rois),
                "ROIs": [
                    {"Name": name, "Point_px": points, "NumberOfPoints": len(points)}
                    for name, points in rois
                ],
            }
        ]
    }
    path.write_bytes(plistlib.dumps(data))
