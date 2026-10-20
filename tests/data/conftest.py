"""Fixtures for the data tests."""

from pathlib import Path

import pytest
from tests.data.synthetic import CSV_HEADER, square, write_annotation, write_dicom


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Four images of two patients.

    - 11111111: left CC of patient a, one 10 x 10 mass and a calcification
    - 11111112: left MLO of patient a, two masses (areas 100 and 25)
    - 22222221: right CC of patient b, calcification only
    - 22222222: right MLO of patient b, no annotation (normal)

    The filenames write the MLO view as ML, the CSV as MLO, as in the real data.
    """
    root = tmp_path / "INbreast Release 1.0"
    (root / "AllDICOMs").mkdir(parents=True)
    (root / "AllXML").mkdir()
    patient_a, patient_b = "a" * 16, "b" * 16
    images = [
        ("11111111", patient_a, "L", "CC", 6, 8),
        ("11111112", patient_a, "L", "ML", 6, 8),
        ("22222221", patient_b, "R", "CC", 10, 12),
        ("22222222", patient_b, "R", "ML", 10, 12),
    ]
    for file_id, patient, laterality, view, rows, cols in images:
        name = f"{file_id}_{patient}_MG_{laterality}_{view}_ANON.dcm"
        write_dicom(root / "AllDICOMs" / name, rows, cols)

    csv_rows = [
        "removed;removed;L;CC;200902;11111111;2;4a",
        "removed;removed;L;MLO;200902;11111112;2;4a",
        "removed;removed;R;CC;201001;22222221;1;2",
        # blank density, as in one image of the real data
        "removed;removed;R;MLO;201001;22222222;   ;1",
    ]
    (root / "INbreast.csv").write_text("\n".join([CSV_HEADER, *csv_rows]) + "\n")

    write_annotation(
        root / "AllXML" / "11111111.xml",
        [("Mass", square(1, 1, 10)), ("Calcification", ["(2.0, 2.0)"])],
    )
    write_annotation(
        root / "AllXML" / "11111112.xml",
        [("Mass", square(0, 0, 10)), ("Mass", square(20, 20, 5))],
    )
    write_annotation(
        root / "AllXML" / "22222221.xml", [("Calcification", square(0, 0, 3))]
    )
    return root
