import sys
from pathlib import Path
from typing import Any, get_args

import pytest
from tests.data.manifests import make_manifest
from tests.data.preprocessed import SETTINGS
from tests.project import Project, make_project

from mammo_seg import cli
from mammo_seg.data.loaders import BatchComposition
from mammo_seg.data.manifest import write_manifest
from mammo_seg.data.splits import create_splits_file, write_splits
from mammo_seg.engine import evaluate as evaluate_module
from mammo_seg.engine import train as train_module
from mammo_seg.engine.checkpoint import save_checkpoint
from mammo_seg.engine.inference import INFERENCES
from mammo_seg.engine.model import Architecture, ModelSettings, build_model
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
    assert cli.DEFAULT_THRESHOLDS == evaluate_module.DEFAULT_THRESHOLDS
    assert cli.DEFAULT_MIN_AREAS_MM2 == evaluate_module.DEFAULT_MIN_AREAS_MM2
    assert cli.ARCHITECTURES == get_args(Architecture)
    assert cli.INFERENCES == INFERENCES


def test_the_defaults_are_the_agreed_training_settings() -> None:
    args = cli.build_parser().parse_args(["train"])
    assert (args.epochs, args.batch_size, args.learning_rate) == (60, None, 1e-4)
    assert (args.patience, args.positive_fraction, args.fold) == (15, 0.5, 0)
    assert args.selection_metric == "all_images.dice" and args.amp
    assert args.architecture == "unet" and args.encoder == "resnet34"
    assert args.encoder_weights == "imagenet"
    assert args.device is None and not args.resume and args.max_minutes is None
    assert args.output_dir == Path("runs")


def test_training_on_whole_images_is_the_default() -> None:
    args = cli.build_parser().parse_args(["train"])
    assert args.crop_size is None and args.crop_mass_fraction == 0.5
    assert args.batch_composition is None


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
    assert kwargs["inference"] is None  # follows the crops
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


def without_batch_size(arguments: list[str]) -> list[str]:
    position = arguments.index("--batch-size")
    return arguments[:position] + arguments[position + 2 :]


def captured_settings(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> Any:
    calls: list[tuple[Any, ...]] = []

    def fake(*args: Any, **kwargs: Any) -> TrainingResult:
        calls.append(args)
        return result()

    monkeypatch.setattr(train_module, "train_fold", fake)
    assert cli.main(arguments) == 0
    return calls[0][4]


def test_the_batch_size_defaults_to_4(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = without_batch_size(project.train_args(*TINY))
    settings = captured_settings(monkeypatch, arguments)
    assert settings.batch_size == 4 and settings.composition is None


def test_a_batch_composition_sets_the_batch_size(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = ["--crop-size", "32", "--batch-composition", "3", "1", "2", *TINY]
    arguments = without_batch_size(project.train_args(*options))
    settings = captured_settings(monkeypatch, arguments)
    assert settings.composition == BatchComposition(3, 1, 2)
    assert settings.batch_size == 6


def test_a_batch_size_that_is_not_the_composition_is_an_error(
    project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    options = ["--crop-size", "32", "--batch-composition", "3", "1", "2", *TINY]
    assert cli.main(project.train_args(*options)) == 1  # --batch-size 2
    assert "batch size is 2 but the composition" in capsys.readouterr().err


def test_a_batch_composition_without_crops_is_an_error(
    project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    options = ["--batch-composition", "1", "0", "1", *TINY]
    assert cli.main(project.train_args(*options)) == 1
    assert "made of patches" in capsys.readouterr().err


def test_a_run_with_a_batch_composition_trains(
    project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    options = [
        "--crop-size",
        "32",
        "--batch-composition",
        "1",
        "0",
        "1",
        "--epochs",
        "1",
    ]
    assert cli.main(project.train_args(*options, "--no-amp", *TINY)) == 0
    assert "Training ended after 1 epochs (finished)" in capsys.readouterr().out
    assert (project.output / "fold_0" / "last.pt").is_file()


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


# --- evaluating ---

SMALL = ModelSettings(encoder="tu-mobilenetv3_small_050", encoder_weights=None)


def evaluate_args(project: Project, *extra: str) -> list[str]:
    return [
        "evaluate",
        "--manifest",
        str(project.manifest),
        "--splits",
        str(project.splits),
        "--preprocessed-dir",
        str(project.preprocessed),
        "--runs-dir",
        str(project.output),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--thresholds",
        "0.5",
        "--min-areas-mm2",
        "0",
        "5",
        *extra,
    ]


def save_runs(project: Project, folds: tuple[int, ...] = (0, 1)) -> None:
    for fold in folds:
        save_checkpoint(
            project.output / f"fold_{fold}" / "last.pt",
            build_model(SMALL),
            SMALL,
            SETTINGS,
            metadata={"fold": fold},
        )


def test_the_inference_of_a_training_can_be_chosen(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake(*args: Any, **kwargs: Any) -> TrainingResult:
        calls.append(kwargs)
        return result()

    monkeypatch.setattr(train_module, "train_fold", fake)
    options = ["--crop-size", "32", "--inference", "whole", *TINY]
    assert cli.main(project.train_args(*options)) == 0
    assert calls[0]["inference"] == "whole"


def test_evaluate_can_choose_the_inference(
    project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    save_runs(project)  # checkpoints that record no window
    folds = ["--folds", "0", "1"]
    assert cli.main(evaluate_args(project, *folds, "--inference", "sliding")) == 1
    assert "records no window size" in capsys.readouterr().err
    assert cli.main(evaluate_args(project, *folds, "--inference", "whole")) == 0
    assert "predicted in one pass" in capsys.readouterr().out


def test_the_evaluate_defaults_cover_the_five_folds() -> None:
    args = cli.build_parser().parse_args(["evaluate"])
    assert args.folds == [0, 1, 2, 3, 4] and args.checkpoint_name == "last.pt"
    assert args.thresholds == list(cli.DEFAULT_THRESHOLDS)
    assert args.min_areas_mm2 == list(cli.DEFAULT_MIN_AREAS_MM2)
    assert args.inference is None
    assert not args.test and args.output is None and args.runs_dir == Path("runs")


def test_evaluate_prints_the_sweep_and_writes_the_results(
    project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    save_runs(project)
    assert cli.main(evaluate_args(project, "--folds", "0", "1")) == 0
    out = capsys.readouterr().out
    assert "last.pt on the validation images (16 scored, 2 fold(s) pooled)" in out
    assert "predict nothing" in out and "TEST SET" not in out
    results = project.output / "evaluation_validation.json"
    assert results.is_file() and str(results) in out
    assert '"on": "validation"' in results.read_text()


def test_evaluate_can_write_the_results_elsewhere(
    project: Project, tmp_path: Path
) -> None:
    save_runs(project)
    target = tmp_path / "somewhere" / "results.json"
    code = cli.main(evaluate_args(project, "--folds", "0", "--output", str(target)))
    assert code == 0 and target.is_file()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--folds", "0", "2"], "not a readable checkpoint"),
        (["--folds", "0", "0"], "once each"),
        (["--test", "--folds", "0"], "no test set"),
        (["--thresholds", "1.5"], "threshold"),
        (["--checkpoint-name", "best.pt", "--folds", "0"], "not a readable"),
    ],
)
def test_a_bad_evaluation_is_an_error_message_not_a_traceback(
    project: Project,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    save_runs(project)
    assert cli.main(evaluate_args(project, *extra)) == 1
    assert message in capsys.readouterr().err


def test_evaluating_without_pytorch_is_a_clear_error(
    project: Project,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "mammo_seg.engine.evaluate", None)
    assert cli.main(evaluate_args(project)) == 1
    assert "evaluation needs PyTorch" in capsys.readouterr().err


def test_scoring_the_test_set_says_it_should_be_done_once(
    project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = make_manifest(4, 4, per_patient=2)
    write_splits(create_splits_file(manifest, 2, test_fraction=0.25), project.splits)
    save_runs(project)
    assert cli.main(evaluate_args(project, "--test", "--folds", "0", "1")) == 0
    out = capsys.readouterr().out
    assert "TEST SET: use this once" in out and "on the test images" in out
    assert (project.output / "evaluation_test.json").is_file()
