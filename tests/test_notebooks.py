import ast
import json
import re
import shlex
from pathlib import Path
from typing import Any

import nbformat
import pytest

from mammo_seg import cli
from mammo_seg.engine.evaluate import DEFAULT_MIN_AREAS_MM2, DEFAULT_THRESHOLDS

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))
KAGGLE = Path("notebooks/05_kaggle_training.ipynb")
EVALUATION = Path("notebooks/06_kaggle_evaluation.ipynb")
COMPARISON = Path("notebooks/08_kaggle_whole_vs_patches.ipynb")
ERRORS = Path("notebooks/09_error_analysis.ipynb")
GALLERY = Path("notebooks/10_error_gallery.ipynb")
COLAB = Path("notebooks/11_colab_training.ipynb")
COLAB_EVALUATION = Path("notebooks/12_colab_evaluation.ipynb")
DATA_STEPS = {"manifest", "splits", "masks", "preprocess"}


def read(notebook: Path) -> dict[str, Any]:
    content: dict[str, Any] = json.loads(notebook.read_text())
    return content


def commands(notebook: Path) -> list[str]:
    """The shell commands (``!...``) of the code cells, continuation lines joined."""
    found = []
    for cell in read(notebook)["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]).replace("\\\n", " ")
        found += [
            " ".join(line.strip()[1:].split())
            for line in source.splitlines()
            if line.strip().startswith("!")
        ]
    return found


def settings_of(notebook: Path, *names: str) -> dict[str, Any]:
    """The literal values of ``names`` in the first code cell (the settings)."""
    cell = next(c for c in read(notebook)["cells"] if c["cell_type"] == "code")
    found = {}
    for node in ast.parse("".join(cell["source"])).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                found[node.targets[0].id] = ast.literal_eval(node.value)
    return found


def options_of(command: str, capsys: pytest.CaptureFixture[str]) -> set[str]:
    with pytest.raises(SystemExit):
        cli.main([*command.split(), "--help"])
    return set(re.findall(r"--[a-z][a-z-]*", capsys.readouterr().out))


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_a_notebook_is_valid_and_has_no_outputs(notebook: Path) -> None:
    content = read(notebook)
    nbformat.validate(content)
    for cell in content["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


@pytest.mark.parametrize(
    ("notebook", "steps"),
    [
        (KAGGLE, DATA_STEPS | {"train"}),
        (EVALUATION, DATA_STEPS | {"evaluate"}),
        (COMPARISON, DATA_STEPS | {"train", "evaluate"}),
    ],
    ids=lambda value: value.name if isinstance(value, Path) else "",
)
def test_the_kaggle_notebooks_use_options_that_exist(
    notebook: Path, steps: set[str], capsys: pytest.CaptureFixture[str]
) -> None:
    mammo_seg = [c for c in commands(notebook) if c.startswith("mammo-seg ")]
    assert steps == {c.split()[1] for c in mammo_seg}
    for command in mammo_seg:
        name = command.split()[1]
        used = set(re.findall(r"--[a-z][a-z-]*", command))
        assert used <= options_of(f"{name}", capsys), command


def test_the_training_and_the_evaluation_preprocess_alike(
    capsys: pytest.CaptureFixture[str],
) -> None:
    options = {
        notebook: settings_of(notebook, "PREPROCESS_OPTIONS")["PREPROCESS_OPTIONS"]
        for notebook in (KAGGLE, EVALUATION)
    }
    # A model is evaluated on images preprocessed as it was trained.
    assert options[KAGGLE] == options[EVALUATION]
    used = set(re.findall(r"--[a-z][a-z-]*", options[KAGGLE]))
    assert used <= options_of("preprocess", capsys)
    for notebook in options:
        (preprocess,) = [
            c for c in commands(notebook) if c.startswith("mammo-seg preprocess ")
        ]
        assert preprocess.endswith("{PREPROCESS_OPTIONS}")


def test_the_kaggle_notebook_settings_are_valid() -> None:
    settings = settings_of(KAGGLE, "FOLDS", "TRAIN_OPTIONS")
    assert settings["FOLDS"] and set(settings["FOLDS"]) <= set(range(5))
    # The extra options must be accepted by the command they are added to.
    cli.build_parser().parse_args(["train", *shlex.split(settings["TRAIN_OPTIONS"])])


def test_the_evaluation_notebook_settings_are_valid() -> None:
    settings = settings_of(EVALUATION, "FOLDS", "EVALUATE_OPTIONS", "DESIGN")
    assert settings["DESIGN"] in (None, "whole", "patches")
    assert settings["FOLDS"] and set(settings["FOLDS"]) <= set(range(5))
    cli.build_parser().parse_args(
        ["evaluate", *shlex.split(settings["EVALUATE_OPTIONS"])]
    )


def test_the_extra_options_of_the_evaluation_cannot_override_the_folds() -> None:
    (evaluate,) = [
        c for c in commands(EVALUATION) if c.startswith("mammo-seg evaluate ")
    ]
    assert evaluate.index("{EVALUATE_OPTIONS}") < evaluate.index("--folds {folds}")


def test_the_evaluation_does_not_write_into_the_read_only_inputs() -> None:
    (evaluate,) = [
        c for c in commands(EVALUATION) if c.startswith("mammo-seg evaluate ")
    ]
    assert "--output {OUTPUT}" in evaluate
    code = "".join("".join(c["source"]) for c in read(EVALUATION)["cells"])
    assert 'OUTPUT = WORK / f"evaluation_' in code  # WORK is /kaggle/working


def test_the_options_of_the_notebook_cannot_override_the_fold_or_the_time() -> None:
    (train,) = [c for c in commands(KAGGLE) if c.startswith("mammo-seg train ")]
    assert train.index("{TRAIN_OPTIONS}") < train.index("--fold {fold}")
    assert train.index("{TRAIN_OPTIONS}") < train.index("--max-minutes {minutes}")


@pytest.mark.parametrize(
    "notebook",
    [KAGGLE, EVALUATION, COMPARISON, COLAB, COLAB_EVALUATION],
    ids=lambda p: p.name,
)
def test_the_repository_is_not_cloned_where_it_would_hide_the_package(
    notebook: Path,
) -> None:
    source = notebook.read_text()
    assert 'WORK / "mammo_seg"' not in source and "/mammo_seg " not in source


def test_the_comparison_settings_are_valid() -> None:
    settings = settings_of(COMPARISON, "FOLDS", "EPOCHS", "EXPERIMENTS")
    assert settings["FOLDS"] == [0, 1, 2, 3, 4] and settings["EPOCHS"] >= 1
    experiments = settings["EXPERIMENTS"]
    assert set(experiments) == {"whole", "patches"}
    parsed = {
        name: cli.build_parser().parse_args(["train", *shlex.split(options)])
        for name, options in experiments.items()
    }
    assert parsed["whole"].crop_size is None and parsed["patches"].crop_size == 512
    # The inference follows the crops: neither design sets it.
    assert parsed["whole"].inference is None and parsed["patches"].inference is None


def test_the_options_of_a_design_cannot_override_the_fold_or_the_time() -> None:
    (train,) = [c for c in commands(COMPARISON) if c.startswith("mammo-seg train ")]
    assert train.index("{options}") < train.index("--fold {fold}")
    assert train.index("{options}") < train.index("--max-minutes {minutes}")
    assert train.index("--epochs {EPOCHS}") < train.index("{options}")


def test_the_comparison_writes_its_evaluations_into_the_working_folder() -> None:
    (evaluate,) = [
        c for c in commands(COMPARISON) if c.startswith("mammo-seg evaluate ")
    ]
    assert "--output {output}" in evaluate and "--runs-dir {runs}" in evaluate
    code = "".join("".join(c["source"]) for c in read(COMPARISON)["cells"])
    assert 'RUNS = WORK / "runs"' in code  # WORK is /kaggle/working


@pytest.mark.parametrize(
    ("notebook", "folder"),
    [(KAGGLE, "RUNS"), (EVALUATION, "WORK"), (COMPARISON, "RUNS")],
    ids=lambda value: value.name if isinstance(value, Path) else value,
)
def test_a_kaggle_notebook_records_the_commit_of_its_code(
    notebook: Path, folder: str
) -> None:
    recorded = [c for c in commands(notebook) if "rev-parse HEAD" in c]
    assert len(recorded) == 1 and "tee -a {VERSION}" in recorded[0]
    code = "".join("".join(c["source"]) for c in read(notebook)["cells"])
    assert f'VERSION = {folder} / "code_version.txt"' in code


def test_the_error_analysis_looks_at_a_setting_the_evaluation_tries() -> None:
    settings = settings_of(ERRORS, "THRESHOLD", "MIN_AREA_MM2", "NEAR_MM", "CHEST_WALL")
    # Only the settings of the evaluation are in the details file.
    assert settings["THRESHOLD"] in DEFAULT_THRESHOLDS
    assert settings["MIN_AREA_MM2"] in DEFAULT_MIN_AREAS_MM2
    assert settings["NEAR_MM"] > 0 and 0 < settings["CHEST_WALL"] < 1


def test_the_error_gallery_shows_a_setting_the_evaluation_tries() -> None:
    names = (
        "THRESHOLD",
        "MIN_AREA_MM2",
        "FLOOR",
        "ZOOM_MM",
        "NEAR_MM",
        "CHEST_WALL",
        "ROWS_PER_FIGURE",
    )
    settings = settings_of(GALLERY, *names)
    assert set(settings) == set(names)
    assert settings["THRESHOLD"] in DEFAULT_THRESHOLDS
    assert settings["MIN_AREA_MM2"] in DEFAULT_MIN_AREAS_MM2
    # The probabilities are drawn from the floor up, so it must lie below the threshold.
    assert 0 < settings["FLOOR"] < settings["THRESHOLD"]
    assert settings["ZOOM_MM"] > 0 and settings["NEAR_MM"] > 0
    assert 0 < settings["CHEST_WALL"] < 1 and settings["ROWS_PER_FIGURE"] >= 1
    # The gallery groups the false regions as the error analysis does.
    analysis = settings_of(ERRORS, "NEAR_MM", "CHEST_WALL")
    assert {k: settings[k] for k in analysis} == analysis


@pytest.mark.parametrize(
    ("notebook", "step"),
    [(COLAB, "train"), (COLAB_EVALUATION, "evaluate")],
    ids=lambda value: value.name if isinstance(value, Path) else "",
)
def test_the_colab_notebooks_use_options_that_exist(
    notebook: Path, step: str, capsys: pytest.CaptureFixture[str]
) -> None:
    mammo_seg = [c for c in commands(notebook) if c.startswith("mammo-seg ")]
    # The data comes from the archive: no data step runs on Colab.
    assert [c.split()[1] for c in mammo_seg] == [step]
    used = set(re.findall(r"--[a-z][a-z-]*", mammo_seg[0]))
    assert used <= options_of(step, capsys), mammo_seg[0]


def test_the_colab_notebook_settings_are_valid() -> None:
    training = settings_of(COLAB, "FOLDS", "TRAIN_OPTIONS")
    assert training["FOLDS"] and set(training["FOLDS"]) <= set(range(5))
    cli.build_parser().parse_args(["train", *shlex.split(training["TRAIN_OPTIONS"])])
    evaluation = settings_of(COLAB_EVALUATION, "FOLDS", "EVALUATE_OPTIONS")
    assert evaluation["FOLDS"] and set(evaluation["FOLDS"]) <= set(range(5))
    cli.build_parser().parse_args(
        ["evaluate", *shlex.split(evaluation["EVALUATE_OPTIONS"])]
    )


def test_the_colab_training_and_evaluation_read_the_same_data_and_runs() -> None:
    names = ("DATASET", "RUN_NAME")
    assert settings_of(COLAB, *names) == settings_of(COLAB_EVALUATION, *names)
    assert settings_of(COLAB, "DATASET")["DATASET"] in ("inbreast", "cbis-ddsm")
    for notebook in (COLAB, COLAB_EVALUATION):
        code = "".join("".join(c["source"]) for c in read(notebook)["cells"])
        assert 'DRIVE_DIR = Path("/content/drive/MyDrive/' in code


@pytest.mark.parametrize("notebook", [COLAB, COLAB_EVALUATION], ids=lambda p: p.name)
def test_a_colab_notebook_keeps_its_runs_on_drive(notebook: Path) -> None:
    # A session can end at any time and /content goes with it; Drive stays.
    code = "".join("".join(c["source"]) for c in read(notebook)["cells"])
    assert 'RUNS = DRIVE_DIR / "mammo_seg_runs" / RUN_NAME' in code
    (command,) = [c for c in commands(notebook) if c.startswith("mammo-seg ")]
    folder = "--output-dir" if notebook == COLAB else "--runs-dir"
    assert f'{folder} "{{RUNS}}"' in command
    assert "drive.flush_and_unmount()" in code


def test_the_colab_options_cannot_override_the_fold_or_the_time() -> None:
    (train,) = [c for c in commands(COLAB) if c.startswith("mammo-seg train ")]
    assert train.index("{TRAIN_OPTIONS}") < train.index("--fold {fold}")
    assert train.index("{TRAIN_OPTIONS}") < train.index("--max-minutes {minutes}")
    (evaluate,) = [
        c for c in commands(COLAB_EVALUATION) if c.startswith("mammo-seg evaluate ")
    ]
    assert evaluate.index("{EVALUATE_OPTIONS}") < evaluate.index("--folds {folds}")
    assert '--output "{OUTPUT}"' in evaluate


@pytest.mark.parametrize(
    ("notebook", "name"),
    [(COLAB, "code_version.txt"), (COLAB_EVALUATION, "evaluation_code_version.txt")],
    ids=lambda value: value.name if isinstance(value, Path) else value,
)
def test_a_colab_notebook_records_the_commit_of_its_code(
    notebook: Path, name: str
) -> None:
    recorded = [c for c in commands(notebook) if "rev-parse HEAD" in c]
    assert len(recorded) == 1 and 'tee -a "{VERSION}"' in recorded[0]
    code = "".join("".join(c["source"]) for c in read(notebook)["cells"])
    assert f'VERSION = RUNS / "{name}"' in code


def code_of(notebook: Path) -> str:
    return "".join("".join(c["source"]) for c in read(notebook)["cells"])


def test_the_colab_archive_is_that_of_the_dataset() -> None:
    for notebook in (COLAB, COLAB_EVALUATION):
        code = code_of(notebook)
        assert 'PROCESSED = WORK / "data" / "processed" / DATASET' in code
        assert '"mammo_seg_data" / "processed" / DATASET' in code


@pytest.mark.parametrize(
    ("notebook", "step", "options"),
    [
        (COLAB, "train", "--positive-fraction none"),
        (COLAB_EVALUATION, "evaluate", "--min-areas-mm2 0"),
    ],
    ids=lambda value: value.name if isinstance(value, Path) else "",
)
def test_cbis_ddsm_gets_its_options_before_those_of_the_notebook(
    notebook: Path, step: str, options: str
) -> None:
    code = code_of(notebook)
    assert f'DATASET_OPTIONS = "{options}" if DATASET == "cbis-ddsm" else ""' in code
    cli.build_parser().parse_args([step, *options.split()])
    (command,) = [c for c in commands(notebook) if c.startswith("mammo-seg ")]
    extra = "{TRAIN_OPTIONS}" if notebook == COLAB else "{EVALUATE_OPTIONS}"
    assert command.index("{DATASET_OPTIONS}") < command.index(extra)


def test_the_colab_training_can_start_from_a_checkpoint_on_drive() -> None:
    assert settings_of(COLAB, "INIT_FROM") == {"INIT_FROM": None}
    code = code_of(COLAB)
    assert "START = DRIVE_DIR / INIT_FROM if INIT_FROM else None" in code
    assert """DATASET_OPTIONS += f' --init-from "{START}"'""" in code
