import sys
from pathlib import Path
from typing import Any, get_args

import pytest
from tests.data.manifests import make_manifest
from tests.data.preprocessed import SETTINGS
from tests.project import Project, make_project

from mammo_seg import cli
from mammo_seg.data.manifest import write_manifest
from mammo_seg.engine import train as train_module
from mammo_seg.engine.model import Architecture
from mammo_seg.engine.train import SELECTION_METRICS, TrainingResult

TINY = ["--encoder", "tu-mobilenetv3_small_050", "--encoder-weights", "none"]


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return make_project(tmp_path)


def result(**overrides: Any) -> TrainingResult:
    fields: dict[str, Any] = {
        "history": (),
        "best_epoch": 3,
        "best_value": 0.5,
        "stop_reason": "finished",
    }
    return TrainingResult(**{**fields, **overrides})


# --- the command and its options ---


def test_the_choices_are_those_of_the_engine() -> None:
    assert cli.SELECTION_METRICS == SELECTION_METRICS
    assert cli.ARCHITECTURES == get_args(Architecture)


def test_the_defaults_are_the_agreed_training_settings() -> None:
    args = cli.build_parser().parse_args(["train"])
    assert (args.epochs, args.batch_size, args.learning_rate) == (60, 4, 1e-4)
    assert (args.patience, args.positive_fraction, args.fold) == (15, 0.5, 0)
    assert args.selection_metric == "all_images.dice" and args.amp
    assert args.architecture == "unet" and args.encoder == "resnet34"
    assert args.encoder_weights == "imagenet"
    assert args.device is None and not args.resume and args.max_minutes is None
    assert args.output_dir == Path("runs")


def test_training_on_whole_images_is_the_default() -> None:
    args = cli.build_parser().parse_args(["train"])
    assert args.crop_size is None and args.crop_mass_fraction == 0.5


def test_mixed_precision_can_be_switched_off() -> None:
    assert not cli.build_parser().parse_args(["train", "--no-amp"]).amp


def test_the_help_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for name in ("manifest", "splits", "masks", "preprocess", "train"):
        assert name in text


def test_a_command_is_required_and_must_be_known() -> None:
    for arguments in ([], ["fly"]):
        with pytest.raises(SystemExit) as stop:
            cli.main(arguments)
        assert stop.value.code == 2


def test_arguments_come_from_the_command_line_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["mammo-seg", "masks", "--manifest", "m.json"])
    called: list[Any] = []

    def step(argv: Any) -> int:
        called.append(argv)
        return 0

    monkeypatch.setitem(cli.STEPS, "masks", ("", step))
    assert cli.main() == 0
    assert called == [["--manifest", "m.json"]]


# --- the data steps ---


def test_a_data_step_gets_its_own_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.json"
    write_manifest(make_manifest(4, 4), manifest)
    output = tmp_path / "splits.json"
    code = cli.main(
        ["splits", "--folds", "2", "--manifest", str(manifest), "--output", str(output)]
    )
    assert code == 0 and output.is_file()
    assert f"Wrote {output}" in capsys.readouterr().out


def test_the_exit_code_of_a_data_step_is_returned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["masks", "--manifest", str(tmp_path / "missing.json")])
    assert code == 1 and "error" in capsys.readouterr().err


# --- training ---


def test_the_options_become_settings(
    project: Project,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake(*args: Any, **kwargs: Any) -> TrainingResult:
        calls.append((args, kwargs))
        return result()

    monkeypatch.setattr(train_module, "train_fold", fake)
    code = cli.main(
        project.train_args(
            "--fold",
            "1",
            "--epochs",
            "7",
            "--patience",
            "0",
            "--no-amp",
            "--resume",
            "--max-minutes",
            "90",
            "--selection-metric",
            "with_mass.dice",
            "--architecture",
            "fpn",
            *TINY,
        )
    )
    assert code == 0
    ((args, kwargs),) = calls
    _, _, directory, fold, settings, model_settings, preprocessing, crops = args
    assert directory == project.preprocessed and fold == 1
    assert (settings.epochs, settings.batch_size) == (7, 2)
    assert settings.patience is None and not settings.amp
    assert settings.selection_metric == "with_mass.dice"
    assert settings.samples_per_epoch == 4
    assert (model_settings.architecture, model_settings.encoder_weights) == (
        "fpn",
        None,
    )
    assert preprocessing == SETTINGS and crops.crop_size is None
    assert kwargs["output_dir"] == project.output / "fold_1"
    assert kwargs["canvas"] == (64, 64)
    assert kwargs["resume"] and kwargs["time_limit_minutes"] == 90
    assert kwargs["device"].type == "cpu"
    assert "Training ended after 0 epochs (finished)" in capsys.readouterr().out


def test_the_crop_options_become_augmentation_settings(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake(*args: Any, **kwargs: Any) -> TrainingResult:
        calls.append(args)
        return result()

    monkeypatch.setattr(train_module, "train_fold", fake)
    options = ["--crop-size", "32", "--crop-mass-fraction", "0.75", *TINY]
    assert cli.main(project.train_args(*options)) == 0
    augmentation = calls[0][7]
    assert (
        augmentation.crop_size == (32, 32) and augmentation.crop_mass_fraction == 0.75
    )


def test_a_crop_size_of_zero_is_an_error(
    project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(project.train_args("--crop-size", "0", *TINY)) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_the_summary_names_the_best_epoch_and_how_to_continue(
    project: Project,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        train_module, "train_fold", lambda *a, **k: result(stop_reason="time_limit")
    )
    assert cli.main(project.train_args(*TINY)) == 0
    out = capsys.readouterr().out
    assert "best epoch 3 (all_images.dice 0.5000) in" in out
    assert str(Path("fold_0") / "best.pt") in out
    assert "--resume" in out


def test_the_summary_says_when_no_epoch_was_defined(
    project: Project,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        train_module,
        "train_fold",
        lambda *a, **k: result(best_epoch=None, best_value=float("nan")),
    )
    assert cli.main(project.train_args(*TINY)) == 0
    out = capsys.readouterr().out
    assert "all_images.dice was never defined" in out and "--resume" not in out


def test_training_without_pytorch_is_a_clear_error(
    project: Project,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "mammo_seg.engine.train", None)
    assert cli.main(project.train_args(*TINY)) == 1
    assert "training needs PyTorch" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--epochs", "0"], "epochs"),
        (["--device", "tpu"], "unknown device"),
        (["--resume"], "nothing to resume"),
        (["--fold", "5"], "validation fold"),
        (["--encoder", "no-such-encoder"], "no-such-encoder"),
    ],
)
def test_a_bad_run_is_an_error_message_not_a_traceback(
    project: Project,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    arguments = project.train_args(*TINY, *extra)
    assert cli.main(arguments) == 1
    error = capsys.readouterr().err
    assert "\nerror: " in "\n" + error and message in error
    assert "Traceback" not in error


def test_a_missing_input_is_an_error_message(
    project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    project.manifest.unlink()
    assert cli.main(project.train_args(*TINY)) == 1
    assert capsys.readouterr().err.startswith("error:")
