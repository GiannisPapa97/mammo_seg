"""Training through the ``mammo-seg`` command on synthetic data, on the CPU.

The synthetic images are identical ramps with a fixed 10 x 10 mass, so a small
network learns to find it in a few epochs. These tests check that the whole path
(command, data, model, loop, checkpoints) learns, that a stopped run can be resumed
to the same result as one that was never stopped, and that the best model can be
used afterwards.
"""

import json
from pathlib import Path

import pytest
import torch
from tests.project import Project, make_project

from mammo_seg import cli
from mammo_seg.engine.checkpoint import load_checkpoint, load_model

pytestmark = pytest.mark.integration

EPOCHS = "30"
OPTIONS = [
    "--encoder",
    "tu-mobilenetv3_small_050",
    "--encoder-weights",
    "none",
    "--epochs",
    EPOCHS,
    "--learning-rate",
    "3e-3",
    "--batch-size",
    "4",
    "--samples-per-epoch",
    "16",
    "--patience",
    "0",
    "--no-amp",
]


def history(project: Project) -> list[dict[str, object]]:
    path = project.output / "fold_0" / "history.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def weights(project: Project, name: str) -> dict[str, torch.Tensor]:
    return load_checkpoint(project.output / "fold_0" / name).model_state


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Project:
    """A run that was never stopped; trained once for all the tests."""
    project = make_project(tmp_path_factory.mktemp("whole"))
    assert cli.main(project.train_args(*OPTIONS)) == 0
    return project


def test_the_model_learns_to_find_the_mass(trained: Project) -> None:
    rows = history(trained)
    assert len(rows) == 30
    first, last = rows[0]["train_loss"], rows[-1]["train_loss"]
    assert last < first  # type: ignore[operator]
    best = load_checkpoint(trained.output / "fold_0" / "best.pt").metadata
    assert best["value"] > 0.9  # Dice over all validation images


def test_the_best_model_segments_an_image(trained: Project) -> None:
    model, preprocessing = load_model(trained.output / "fold_0" / "best.pt", "cpu")
    assert preprocessing.size == 64
    from tests.data.preprocessed import SETTINGS

    from mammo_seg.data.dataset import MammogramDataset
    from mammo_seg.data.manifest import read_manifest

    manifest = read_manifest(trained.manifest)
    positive = next(i.file_id for i in manifest.images if i.has_mass)
    sample = MammogramDataset(trained.preprocessed, [positive], SETTINGS, (64, 64))[0]
    with torch.no_grad():
        predicted = torch.sigmoid(model(sample["image"][None])) >= 0.5
    truth = sample["mask"][None] > 0.5
    overlap = (predicted & truth).sum() / truth.sum()
    assert overlap > 0.5


def test_a_run_stopped_by_the_time_limit_and_resumed_equals_a_whole_run(
    trained: Project, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    split = make_project(tmp_path / "split")
    assert cli.main(split.train_args(*OPTIONS, "--max-minutes", "1e-9")) == 0
    out = capsys.readouterr().out
    assert "(time_limit)" in out and "--resume" in out
    assert len(history(split)) == 1

    assert cli.main(split.train_args(*OPTIONS, "--resume")) == 0
    assert "(finished)" in capsys.readouterr().out

    def losses(project: Project) -> list[object]:
        return [row["train_loss"] for row in history(project)]

    assert len(history(split)) == 30
    assert losses(split) == losses(trained)
    for name in ("last.pt", "best.pt"):
        a, b = weights(trained, name), weights(split, name)
        assert all(torch.equal(a[key], b[key]) for key in a)
