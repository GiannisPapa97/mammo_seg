import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.data.manifests import SQUARE, make_manifest

from mammo_seg.data.manifest import write_manifest
from mammo_seg.data.schema import LesionType, Manifest, Roi
from mammo_seg.data.splits import (
    DEFAULT_SEED,
    Splits,
    SplitsFile,
    check_manifest,
    create_splits_file,
    main,
    manifest_fingerprint,
    read_splits,
    strata_summary,
    write_splits,
)
from mammo_seg.data.strata import Stratification


def valid_splits() -> Splits:
    return Splits(
        test_patients=("t1", "t2"),
        folds=(("a1", "a2"), ("b1", "b2")),
        excluded_images=("00000900",),
    )


def valid_file(**changes: object) -> SplitsFile:
    fields: dict[str, object] = {
        "version": 3,
        "manifest_fingerprint": "0" * 64,
        "seed": 0,
        "n_folds": 2,
        "test_fraction": 0.2,
        "stratification": Stratification(),
        "strata": {
            "t1": "mass/small/dense",
            "t2": "no_mass/fatty",
            "a1": "mass/small/dense",
            "a2": "no_mass/fatty",
            "b1": "mass/small/dense",
            "b2": "no_mass/fatty",
        },
        "splits": valid_splits(),
    }
    return SplitsFile(**{**fields, **changes})  # type: ignore[arg-type]


# --- fingerprint ---


def test_the_fingerprint_is_stable_and_follows_the_content() -> None:
    manifest = make_manifest(5, 5)
    assert manifest_fingerprint(manifest) == manifest_fingerprint(manifest)
    assert len(manifest_fingerprint(manifest)) == 64
    first = manifest.images[0]
    changed = replace(first, rois=(*first.rois, Roi(LesionType.MASS, SQUARE)))
    other = Manifest(
        version=2, dataset="inbreast", images=(changed, *manifest.images[1:])
    )
    assert manifest_fingerprint(other) != manifest_fingerprint(manifest)


# --- the splits file ---


def test_create_splits_file_records_how_the_splits_were_made() -> None:
    manifest = make_manifest(20, 30)
    splits_file = create_splits_file(manifest, 5, test_fraction=0.2, seed=3)
    assert splits_file.seed == 3
    assert splits_file.n_folds == 5
    assert splits_file.test_fraction == 0.2
    assert splits_file.manifest_fingerprint == manifest_fingerprint(manifest)
    assert len(splits_file.splits.folds) == 5
    assert splits_file.stratification == Stratification()
    assert splits_file.strata == {
        patient_id: "mass/small/fatty" if index < 20 else "no_mass/fatty"
        for index, patient_id in enumerate(sorted(splits_file.strata))
    }
    assert len(splits_file.strata) == 50
    assert create_splits_file(manifest, 5).seed == DEFAULT_SEED


def test_the_stratification_settings_are_recorded() -> None:
    settings = Stratification(small_mass_mm2=0.01)  # the 0.06 mm² test masses are large
    splits_file = create_splits_file(make_manifest(20, 30), 5, stratification=settings)
    assert splits_file.stratification == settings
    assert "mass/large/fatty" in splits_file.strata.values()


def test_a_splits_file_round_trips_through_json(tmp_path: Path) -> None:
    splits_file = create_splits_file(make_manifest(20, 30), 5, test_fraction=0.2)
    path = tmp_path / "out" / "splits.json"
    write_splits(splits_file, path)
    assert read_splits(path) == splits_file
    data = json.loads(path.read_text())
    assert list(data) == [
        "version",
        "manifest_fingerprint",
        "seed",
        "n_folds",
        "test_fraction",
        "stratification",
        "strata",
        "splits",
    ]
    # The pixel size comes from the manifest, so it is not a setting of the splits.
    assert list(data["stratification"]) == ["small_mass_mm2", "dense_acr"]
    assert list(data["splits"]) == ["test_patients", "folds", "excluded_images"]


def test_splits_of_the_right_manifest_are_returned() -> None:
    manifest = make_manifest(20, 30)
    splits_file = create_splits_file(manifest, 5)
    assert check_manifest(splits_file, manifest) == splits_file.splits


def test_splits_of_another_manifest_are_refused() -> None:
    splits_file = create_splits_file(make_manifest(20, 30), 5)
    with pytest.raises(ValueError, match="different manifest"):
        check_manifest(splits_file, make_manifest(21, 30))


def test_a_file_without_a_test_set_has_no_test_fraction() -> None:
    strata = {"a1": "mass", "a2": "no_mass", "b1": "mass", "b2": "no_mass"}
    splits_file = valid_file(
        test_fraction=None,
        strata=strata,
        splits=replace(valid_splits(), test_patients=()),
    )
    assert splits_file.test_fraction is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"version": 1}, "version"),  # made before the stratification by strata
        ({"version": 2}, "version"),  # made with the pixel size as a setting
        ({"n_folds": 3}, "n_folds is 3 but there are 2 folds"),
        ({"test_fraction": 1.5}, "between 0 and 1"),
        ({"test_fraction": None}, "exactly when a test fraction"),
        ({"manifest_fingerprint": 5}, "manifest_fingerprint"),
        ({"seed": "0"}, "seed"),
        ({"strata": {"t1": "mass"}}, "exactly the patients of the splits"),
    ],
)
def test_an_inconsistent_splits_file_is_rejected(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        valid_file(**changes)


def test_a_test_set_without_a_test_fraction_is_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly when a test fraction"):
        valid_file(
            strata={"a1": "mass", "a2": "no_mass", "b1": "mass", "b2": "no_mass"},
            splits=replace(valid_splits(), test_patients=()),
        )


@pytest.mark.parametrize(
    ("splits", "message"),
    [
        (dict(folds=(("a",),)), "at least 2 folds"),
        (dict(folds=(("a",), ())), "a fold is empty"),
        (dict(folds=(("a", "b"), ("b",))), "patients in more than one set: \\['b'\\]"),
        (dict(test_patients=("a",), folds=(("a",), ("b",))), "more than one set"),
        (dict(excluded_images=("1", "1")), "duplicate excluded images"),
    ],
)
def test_inconsistent_splits_are_rejected(
    splits: dict[str, object], message: str
) -> None:
    fields: dict[str, object] = {
        "test_patients": (),
        "folds": (("a",), ("b",)),
        "excluded_images": (),
    }
    with pytest.raises(ValidationError, match=message):
        Splits(**{**fields, **splits})  # type: ignore[arg-type]


def test_a_file_with_the_pixel_size_as_a_setting_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    write_splits(create_splits_file(make_manifest(20, 30), 5), path)
    data = json.loads(path.read_text())
    data["stratification"]["pixel_size_mm"] = 0.07
    path.write_text(json.dumps({**data, "version": 2}))
    with pytest.raises(ValidationError, match="version"):
        read_splits(path)


def test_a_file_of_the_first_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    write_splits(create_splits_file(make_manifest(20, 30), 5), path)
    data = json.loads(path.read_text())
    old = {key: data[key] for key in data if key not in ("stratification", "strata")}
    path.write_text(json.dumps({**old, "version": 1}))
    with pytest.raises(ValidationError, match="version"):
        read_splits(path)


def test_the_strata_summary_counts_patients_per_set_and_stratum() -> None:
    splits_file = create_splits_file(make_manifest(20, 30), 5, test_fraction=0.2)
    summary = strata_summary(splits_file)
    assert list(summary.columns) == ["mass/small/fatty", "no_mass/fatty", "total"]
    assert list(summary.index) == [*(f"fold_{i}" for i in range(5)), "test", "total"]
    assert summary.loc["total", "total"] == 50
    assert summary.loc["test", "mass/small/fatty"] == 4
    assert summary.loc["total", "mass/small/fatty"] == 20


def test_a_file_with_an_extra_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    write_splits(create_splits_file(make_manifest(20, 30), 5), path)
    data = json.loads(path.read_text())
    data["extra"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="extra"):
        read_splits(path)


# --- the command ---


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    write_manifest(make_manifest(20, 30), path)
    return path


def run(manifest_path: Path, output: Path, *arguments: str) -> int:
    return main(["--manifest", str(manifest_path), "--output", str(output), *arguments])


def test_main_writes_the_splits_and_prints_a_summary(
    manifest_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out" / "splits.json"
    assert run(manifest_path, output, "--folds", "5") == 0
    text = capsys.readouterr().out
    assert f"Wrote {output}" in text
    assert "fold_0" in text and "fold_4" in text
    assert "test" not in text.split("\n", 1)[1]
    assert "Patients per stratum" in text and "no_mass/fatty" in text
    splits_file = read_splits(output)
    assert splits_file.n_folds == 5
    assert splits_file.test_fraction is None
    assert splits_file.seed == DEFAULT_SEED


def test_main_with_a_test_fraction_and_a_seed(
    manifest_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "splits.json"
    code = run(
        manifest_path, output, "--folds", "4", "--test-fraction", "0.2", "--seed", "9"
    )
    assert code == 0
    assert "test" in capsys.readouterr().out
    splits_file = read_splits(output)
    assert (splits_file.n_folds, splits_file.test_fraction, splits_file.seed) == (
        4,
        0.2,
        9,
    )
    assert len(splits_file.splits.test_patients) == 10


def test_main_is_reproducible(manifest_path: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    run(manifest_path, first, "--folds", "5", "--test-fraction", "0.2")
    run(manifest_path, second, "--folds", "5", "--test-fraction", "0.2")
    assert first.read_text() == second.read_text()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--folds", "1"], "at least 2 folds"),
        (["--folds", "5", "--test-fraction", "1.5"], "between 0 and 1"),
        (["--folds", "40"], "has only"),
    ],
)
def test_main_reports_invalid_arguments_and_writes_nothing(
    manifest_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    output = tmp_path / "splits.json"
    assert run(manifest_path, output, *arguments) == 1
    assert not output.exists()
    assert message in capsys.readouterr().err


def test_main_reports_a_missing_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "splits.json"
    assert run(tmp_path / "missing.json", output, "--folds", "5") == 1
    assert not output.exists()
    assert "error:" in capsys.readouterr().err


def test_main_reports_an_invalid_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"version": 2, "images": []}')
    assert run(manifest, tmp_path / "splits.json", "--folds", "5") == 1
    assert "version" in capsys.readouterr().err
