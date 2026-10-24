import ast
import json
import re
import shlex
from pathlib import Path
from typing import Any

import nbformat
import pytest

from mammo_seg import cli

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))
KAGGLE = Path("notebooks/05_kaggle_training.ipynb")
EVALUATION = Path("notebooks/06_kaggle_evaluation.ipynb")
COMPARISON = Path("notebooks/08_kaggle_whole_vs_patches.ipynb")
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


def test_the_kaggle_notebook_settings_are_valid() -> None:
    settings = settings_of(KAGGLE, "FOLDS", "TRAIN_OPTIONS")
    assert settings["FOLDS"] and set(settings["FOLDS"]) <= set(range(5))
    # The extra options must be accepted by the command they are added to.
    cli.build_parser().parse_args(["train", *shlex.split(settings["TRAIN_OPTIONS"])])


def test_the_evaluation_notebook_settings_are_valid() -> None:
    settings = settings_of(EVALUATION, "FOLDS", "EVALUATE_OPTIONS")
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
    "notebook", [KAGGLE, EVALUATION, COMPARISON], ids=lambda p: p.name
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
