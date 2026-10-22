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


def test_the_kaggle_notebook_uses_options_that_exist(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mammo_seg = [c for c in commands(KAGGLE) if c.startswith("mammo-seg ")]
    steps = {"manifest", "splits", "masks", "preprocess", "train"}
    assert steps <= {c.split()[1] for c in mammo_seg}
    for command in mammo_seg:
        name = command.split()[1]
        used = set(re.findall(r"--[a-z][a-z-]*", command))
        assert used <= options_of(f"{name}", capsys), command


def test_the_kaggle_notebook_settings_are_valid() -> None:
    settings = settings_of(KAGGLE, "FOLDS", "TRAIN_OPTIONS")
    assert settings["FOLDS"] and set(settings["FOLDS"]) <= set(range(5))
    # The extra options must be accepted by the command they are added to.
    cli.build_parser().parse_args(["train", *shlex.split(settings["TRAIN_OPTIONS"])])


def test_the_options_of_the_notebook_cannot_override_the_fold_or_the_time() -> None:
    (train,) = [c for c in commands(KAGGLE) if c.startswith("mammo-seg train ")]
    assert train.index("{TRAIN_OPTIONS}") < train.index("--fold {fold}")
    assert train.index("{TRAIN_OPTIONS}") < train.index("--max-minutes {minutes}")


def test_the_repository_is_not_cloned_where_it_would_hide_the_package() -> None:
    source = KAGGLE.read_text()
    assert 'WORK / "mammo_seg"' not in source and "/mammo_seg " not in source
