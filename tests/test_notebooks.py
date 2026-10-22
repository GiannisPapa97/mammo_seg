import json
import re
from pathlib import Path
from typing import Any

import nbformat
import pytest

from mammo_seg import cli

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))
CLOUD = Path("notebooks/05_cloud_training.ipynb")


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


def test_the_cloud_notebook_uses_options_that_exist(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mammo_seg = [c for c in commands(CLOUD) if c.startswith("mammo-seg ")]
    steps = {"manifest", "splits", "masks", "preprocess", "train"}
    assert steps <= {c.split()[1] for c in mammo_seg}
    for command in mammo_seg:
        name = command.split()[1]
        used = set(re.findall(r"--[a-z][a-z-]*", command))
        assert used <= options_of(f"{name}", capsys), command
