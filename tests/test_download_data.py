"""scripts/download_data.sh, run on small synthetic zips over file:// URLs."""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "download_data.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("curl") is None or shutil.which("unzip") is None,
    reason="the script needs curl and unzip",
)

INBREAST_FILES = (
    "INbreast Release 1.0/AllDICOMs/1_a_MG_L_CC_ANON.dcm",
    "INbreast Release 1.0/AllDICOMs/2_a_MG_L_ML_ANON.dcm",
    "INbreast Release 1.0/INbreast.csv",
)
CBIS_DDSM_FILES = (
    "jpeg/1.2.3/1-1.jpg",
    "jpeg/1.2.3/1-2.jpg",
    "jpeg/4.5.6/1-1.jpg",
    "csv/dicom_info.csv",
)


def make_zip(path: Path, names: tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, f"synthetic content of {name}\n")
    return path


@pytest.fixture
def sources(tmp_path: Path) -> dict[str, str]:
    """The environment for the script: synthetic zips and a folder for the data."""
    inbreast = make_zip(tmp_path / "source" / "inbreast.zip", INBREAST_FILES)
    cbis_ddsm = make_zip(tmp_path / "source" / "cbis-ddsm.zip", CBIS_DDSM_FILES)
    return {
        "PATH": os.environ["PATH"],
        "DATA_RAW_DIR": str(tmp_path / "raw"),
        "DOWNLOAD_RETRY_SECONDS": "0",
        "INBREAST_URL": inbreast.as_uri(),
        "INBREAST_ZIP_BYTES": str(inbreast.stat().st_size),
        "INBREAST_DCM_COUNT": "2",
        "CBIS_DDSM_URL": cbis_ddsm.as_uri(),
        "CBIS_DDSM_ZIP_BYTES": str(cbis_ddsm.stat().st_size),
        "CBIS_DDSM_JPEG_COUNT": "3",
    }


def run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def raw(env: dict[str, str]) -> Path:
    return Path(env["DATA_RAW_DIR"])


def test_inbreast_is_the_default(sources: dict[str, str]) -> None:
    result = run(sources)

    assert result.returncode == 0, result.stderr
    assert "Done: 2 DICOM files" in result.stdout
    for name in INBREAST_FILES:
        assert (raw(sources) / "inbreast" / name).is_file()
    assert (raw(sources) / "inbreast" / "inbreast-dataset.zip").is_file()
    assert not (raw(sources) / "cbis-ddsm").exists()


def test_cbis_ddsm_goes_into_its_own_folder(sources: dict[str, str]) -> None:
    result = run(sources, "cbis-ddsm")

    assert result.returncode == 0, result.stderr
    assert "Done: 3 JPEG files" in result.stdout
    for name in CBIS_DDSM_FILES:
        assert (raw(sources) / "cbis-ddsm" / name).is_file()
    assert not (raw(sources) / "inbreast").exists()


def test_all_downloads_both(sources: dict[str, str]) -> None:
    result = run(sources, "all")

    assert result.returncode == 0, result.stderr
    assert (raw(sources) / "inbreast" / INBREAST_FILES[0]).is_file()
    assert (raw(sources) / "cbis-ddsm" / CBIS_DDSM_FILES[0]).is_file()


def test_a_second_run_does_nothing(sources: dict[str, str]) -> None:
    assert run(sources, "all").returncode == 0
    offline = sources | {
        "INBREAST_URL": "file:///nonexistent/inbreast.zip",
        "CBIS_DDSM_URL": "file:///nonexistent/cbis-ddsm.zip",
    }

    result = run(offline, "all")

    assert result.returncode == 0, result.stderr
    assert "INbreast is already in" in result.stdout
    assert "CBIS-DDSM is already in" in result.stdout


def test_a_complete_zip_is_not_downloaded_again(sources: dict[str, str]) -> None:
    zip_path = raw(sources) / "cbis-ddsm" / "cbis-ddsm-breast-cancer-image-dataset.zip"
    make_zip(zip_path, CBIS_DDSM_FILES)
    offline = sources | {
        "CBIS_DDSM_URL": "file:///nonexistent/cbis-ddsm.zip",
        "CBIS_DDSM_ZIP_BYTES": str(zip_path.stat().st_size),
    }

    result = run(offline, "cbis-ddsm")

    assert result.returncode == 0, result.stderr
    assert "Zip already downloaded" in result.stdout
    assert (raw(sources) / "cbis-ddsm" / CBIS_DDSM_FILES[0]).is_file()


def test_a_partial_download_is_resumed(sources: dict[str, str]) -> None:
    source = Path(sources["CBIS_DDSM_URL"].removeprefix("file://"))
    zip_path = raw(sources) / "cbis-ddsm" / "cbis-ddsm-breast-cancer-image-dataset.zip"
    zip_path.parent.mkdir(parents=True)
    content = source.read_bytes()
    zip_path.write_bytes(content[: len(content) // 2])

    result = run(sources, "cbis-ddsm")

    assert result.returncode == 0, result.stderr
    assert zip_path.read_bytes() == content
    assert "Done: 3 JPEG files" in result.stdout


def test_a_failing_download_is_retried_then_reported(sources: dict[str, str]) -> None:
    missing = sources | {
        "CBIS_DDSM_URL": "file:///nonexistent/cbis-ddsm.zip",
        "DOWNLOAD_ATTEMPTS": "2",
    }

    result = run(missing, "cbis-ddsm")

    assert result.returncode == 1
    assert "try 1 of 2" in result.stderr
    assert "the download failed 2 times" in result.stderr


def test_a_zip_of_another_size_is_kept_and_reported(sources: dict[str, str]) -> None:
    wrong = sources | {"INBREAST_ZIP_BYTES": "1"}
    zip_path = raw(sources) / "inbreast" / "inbreast-dataset.zip"

    result = run(wrong, "inbreast")

    assert result.returncode == 1
    assert "larger than the expected 1 bytes" not in result.stderr
    assert "expected 1." in result.stderr
    assert zip_path.is_file()

    again = run(wrong, "inbreast")

    assert again.returncode == 1
    assert "larger than the expected 1 bytes" in again.stderr


def test_a_wrong_number_of_images_is_an_error(sources: dict[str, str]) -> None:
    result = run(sources | {"CBIS_DDSM_JPEG_COUNT": "4"}, "cbis-ddsm")

    assert result.returncode == 1
    assert "found 3 JPEG files after unzipping, expected 4" in result.stderr


def test_all_stops_at_the_first_failing_dataset(sources: dict[str, str]) -> None:
    result = run(sources | {"INBREAST_DCM_COUNT": "5"}, "all")

    assert result.returncode == 1
    assert not (raw(sources) / "cbis-ddsm").exists()


@pytest.mark.parametrize("args", [("ddsm",), ("inbreast", "cbis-ddsm")])
def test_unknown_arguments_are_refused(
    sources: dict[str, str], args: tuple[str, ...]
) -> None:
    result = run(sources, *args)

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert not raw(sources).exists()


def test_help_prints_the_usage(sources: dict[str, str]) -> None:
    result = run(sources, "--help")

    assert result.returncode == 0
    assert "inbreast | cbis-ddsm | all" in result.stdout
