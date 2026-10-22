import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from tests.data.manifests import make_manifest
from tests.data.preprocessed import SETTINGS, write_preprocessed_for

from mammo_seg.data.dataset import MammogramDataset
from mammo_seg.data.loaders import evaluation_loader, training_loader
from mammo_seg.data.schema import Manifest
from mammo_seg.data.splits import SplitsFile, create_splits_file
from mammo_seg.engine import train as train_module
from mammo_seg.engine.losses import DiceBCELoss
from mammo_seg.engine.metrics import GroupSummary, Summary
from mammo_seg.engine.model import ModelSettings, build_model
from mammo_seg.engine.train import (
    HISTORY_FILE,
    SELECTION_METRICS,
    EpochRecord,
    TrainingSettings,
    describe,
    fit,
    record_to_json,
    selection_value,
    train_fold,
    train_one_epoch,
    validate,
)

CANVAS = (64, 64)
TINY = ModelSettings(encoder="tu-mobilenetv3_small_050", encoder_weights=None)
CPU = torch.device("cpu")
FAST = TrainingSettings(
    epochs=3, batch_size=2, learning_rate=1e-3, samples_per_epoch=4, amp=False
)


@dataclass
class Data:
    manifest: Manifest
    splits_file: SplitsFile
    directory: Path


@pytest.fixture
def data(tmp_path: Path) -> Data:
    """Eight patients (four with a mass), two images each, split into two folds."""
    manifest = make_manifest(4, 4, per_patient=2)
    write_preprocessed_for(tmp_path, SETTINGS, manifest)
    return Data(manifest, create_splits_file(manifest, 2), tmp_path)


def loaders(
    data: Data, settings: TrainingSettings = FAST
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:  # type: ignore[type-arg]
    ids = [image.file_id for image in data.manifest.images]
    dataset = MammogramDataset(data.directory, ids, SETTINGS, CANVAS)
    return (
        training_loader(
            dataset,
            data.manifest,
            settings.batch_size,
            settings.positive_fraction,
            settings.samples_per_epoch,
        ),
        evaluation_loader(dataset, settings.batch_size),
    )


def group(dice: float, images: int = 4) -> GroupSummary:
    return GroupSummary(
        images=images,
        dice=dice,
        iou=dice / 2,
        precision=0.3,
        sensitivity=0.4,
        false_alarm_rate=0.1,
    )


def summary(all_dice: float, with_mass_dice: float = 0.2) -> Summary:
    return Summary(
        with_mass=group(with_mass_dice),
        without_mass=group(0.9),
        all_images=group(all_dice, images=8),
    )


# --- settings ---


def test_the_default_settings_are_the_agreed_ones() -> None:
    settings = TrainingSettings()
    assert settings.epochs == 60 and settings.batch_size == 4
    assert settings.learning_rate == 1e-4 and settings.positive_fraction == 0.5
    assert settings.amp and settings.selection_metric == "all_images.dice"


@pytest.mark.parametrize("name", SELECTION_METRICS)
def test_every_group_and_metric_can_be_selected(name: str) -> None:
    assert TrainingSettings(selection_metric=name).selection_metric == name


@pytest.mark.parametrize(
    "bad",
    [
        {"selection_metric": "dice"},
        {"selection_metric": "all_images.f1"},
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"positive_fraction": 1.0},
        {"samples_per_epoch": 0},
        {"num_workers": -1},
        {"unknown": 1},
    ],
)
def test_invalid_settings_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TrainingSettings(**bad)  # type: ignore[arg-type]


def test_the_sampler_can_be_switched_off() -> None:
    assert TrainingSettings(positive_fraction=None).positive_fraction is None


def test_selection_value_reads_the_group_and_the_metric() -> None:
    result = summary(all_dice=0.6, with_mass_dice=0.25)
    assert selection_value(result, "all_images.dice") == 0.6
    assert selection_value(result, "with_mass.dice") == 0.25
    assert selection_value(result, "with_mass.iou") == 0.125
    assert selection_value(result, "without_mass.sensitivity") == 0.4


def test_selection_value_rejects_an_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown selection metric"):
        selection_value(summary(0.5), "all_images.f1")


# --- one epoch ---


def test_training_an_epoch_changes_the_weights_and_returns_a_finite_loss(
    data: Data,
) -> None:
    torch.manual_seed(0)
    model = build_model(TINY)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    train_loader, _ = loaders(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(
        model,
        train_loader,
        DiceBCELoss(),
        optimizer,
        torch.amp.GradScaler("cpu", enabled=False),
        CPU,
        amp=False,
    )
    assert math.isfinite(loss) and loss > 0
    assert model.training
    changed = [
        not torch.equal(before[key], value)
        for key, value in model.state_dict().items()
        if value.dtype.is_floating_point
    ]
    assert any(changed)


def test_an_epoch_without_a_batch_is_an_error(data: Data) -> None:
    train_loader, _ = loaders(data, replace(FAST, batch_size=2, samples_per_epoch=1))
    with pytest.raises(ValueError, match="no batch"):
        train_one_epoch(
            build_model(TINY),
            train_loader,
            DiceBCELoss(),
            torch.optim.Adam(build_model(TINY).parameters()),
            torch.amp.GradScaler("cpu", enabled=False),
            CPU,
            amp=False,
        )


def test_validation_scores_every_image_and_leaves_the_weights_alone(
    data: Data,
) -> None:
    model = build_model(TINY)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    _, validation_loader = loaders(data)
    result = validate(model, validation_loader, CPU, amp=False)
    assert result.all_images.images == 16
    assert result.with_mass.images == 4 and result.without_mass.images == 12
    assert not model.training
    assert all(torch.equal(before[key], v) for key, v in model.state_dict().items())


def test_a_model_that_predicts_nothing_gets_the_documented_scores(data: Data) -> None:
    class Silent(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.full_like(images, -20.0)

    _, validation_loader = loaders(data)
    result = validate(Silent(), validation_loader, CPU, amp=False)
    assert result.with_mass.dice == 0 and result.with_mass.sensitivity == 0
    assert result.without_mass.dice == 1 and result.without_mass.false_alarm_rate == 0
    assert result.all_images.dice == pytest.approx(12 / 16)


# --- the history ---


def record(epoch: int = 1) -> EpochRecord:
    return EpochRecord(
        epoch=epoch,
        learning_rate=1e-4,
        train_loss=0.5,
        seconds=3.2,
        validation=summary(0.6),
        selection_value=0.6,
    )


def test_an_undefined_metric_is_null_in_the_history_line() -> None:
    line = record_to_json(record())
    assert json.loads(line)["validation"]["all_images"]["dice"] == 0.6
    nan_group = replace(group(0.5), precision=math.nan)
    undefined = replace(record(), validation=Summary(nan_group, nan_group, nan_group))
    line = record_to_json(undefined)
    assert json.loads(line)["validation"]["with_mass"]["precision"] is None
    assert "NaN" not in line


def test_the_log_line_names_the_epoch_and_the_selection_metric() -> None:
    text = describe(record(epoch=7), "all_images.dice")
    assert text.startswith("epoch 7:")
    assert "all_images.dice 0.6000" in text and "train loss 0.5000" in text


# --- fit ---


def test_fit_runs_the_epochs_validates_and_writes_the_history(
    data: Data, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    torch.manual_seed(0)
    train_loader, validation_loader = loaders(data)
    output = tmp_path / "run"
    with caplog.at_level(logging.INFO, logger="mammo_seg.engine.train"):
        result = fit(
            build_model(TINY), train_loader, validation_loader, FAST, CPU, output
        )
    assert [r.epoch for r in result.history] == [1, 2, 3]
    assert all(r.validation.all_images.images == 16 for r in result.history)
    assert all(math.isfinite(r.train_loss) for r in result.history)
    lines = (output / HISTORY_FILE).read_text().splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [1, 2, 3]
    assert len([m for m in caplog.messages if m.startswith("epoch ")]) == 3


def test_the_learning_rate_follows_a_cosine_down_to_zero(data: Data) -> None:
    train_loader, validation_loader = loaders(data)
    settings = replace(FAST, epochs=4, learning_rate=1e-3)
    result = fit(build_model(TINY), train_loader, validation_loader, settings, CPU)
    rates = [r.learning_rate for r in result.history]
    assert rates[0] == 1e-3
    assert rates == sorted(rates, reverse=True)
    assert rates[1] == pytest.approx(1e-3 * (1 + math.cos(math.pi / 4)) / 2)
    assert rates[3] == pytest.approx(1e-3 * (1 + math.cos(3 * math.pi / 4)) / 2)


def test_fit_without_an_output_folder_writes_nothing(
    data: Data, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    train_loader, validation_loader = loaders(data)
    fit(build_model(TINY), train_loader, validation_loader, FAST, CPU)
    assert list(work.iterdir()) == []


def test_amp_requested_on_the_cpu_is_ignored(data: Data) -> None:
    train_loader, validation_loader = loaders(data)
    settings = replace(FAST, epochs=1, amp=True)
    result = fit(build_model(TINY), train_loader, validation_loader, settings, CPU)
    assert math.isfinite(result.history[0].train_loss)


def fake_validation(monkeypatch: pytest.MonkeyPatch, summaries: list[Summary]) -> None:
    results = iter(summaries)
    monkeypatch.setattr(train_module, "validate", lambda *args: next(results))


def test_the_best_epoch_is_the_one_with_the_highest_selection_value(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_validation(monkeypatch, [summary(0.2), summary(0.5), summary(0.4)])
    train_loader, validation_loader = loaders(data)
    result = fit(build_model(TINY), train_loader, validation_loader, FAST, CPU)
    assert result.best_epoch == 2 and result.best_value == 0.5
    assert [r.selection_value for r in result.history] == [0.2, 0.5, 0.4]


def test_the_selection_metric_decides_which_epoch_is_the_best(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_validation(
        monkeypatch,
        [
            summary(0.9, with_mass_dice=0.1),
            summary(0.5, with_mass_dice=0.3),
            summary(0.7),
        ],
    )
    train_loader, validation_loader = loaders(data)
    settings = replace(FAST, selection_metric="with_mass.dice")
    result = fit(build_model(TINY), train_loader, validation_loader, settings, CPU)
    assert result.best_epoch == 2 and result.best_value == 0.3


def test_an_undefined_selection_value_never_wins(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    nan = replace(summary(0.5), all_images=group(math.nan))
    fake_validation(monkeypatch, [nan, summary(0.3), nan])
    train_loader, validation_loader = loaders(data)
    result = fit(build_model(TINY), train_loader, validation_loader, FAST, CPU)
    assert result.best_epoch == 2 and result.best_value == 0.3


def test_without_any_defined_value_there_is_no_best_epoch(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    nan = replace(summary(0.5), all_images=group(math.nan))
    fake_validation(monkeypatch, [nan, nan, nan])
    train_loader, validation_loader = loaders(data)
    result = fit(build_model(TINY), train_loader, validation_loader, FAST, CPU)
    assert result.best_epoch is None and math.isnan(result.best_value)


# --- a fold ---


def test_a_fold_trains_on_the_other_folds_and_validates_on_its_own(
    data: Data, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="mammo_seg.engine.train"):
        result = train_fold(
            data.manifest,
            data.splits_file,
            data.directory,
            fold=1,
            settings=replace(FAST, epochs=1),
            model_settings=TINY,
            preprocessing_settings=SETTINGS,
            device=CPU,
            canvas=CANVAS,
        )
    assert len(result.history) == 1
    assert result.history[0].validation.all_images.images == 8
    assert "fold 1 on cpu: 8 training and 8 validation images" in caplog.messages


def test_a_fold_is_repeatable_with_the_same_seed(data: Data) -> None:
    def run(seed: int) -> list[float]:
        result = train_fold(
            data.manifest,
            data.splits_file,
            data.directory,
            fold=0,
            settings=replace(FAST, epochs=2, seed=seed),
            model_settings=TINY,
            preprocessing_settings=SETTINGS,
            device=CPU,
            canvas=CANVAS,
        )
        return [r.train_loss for r in result.history]

    assert run(0) == run(0)
    assert run(0) != run(1)


def test_a_fold_uses_the_resolved_device_by_default(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAMMO_SEG_DEVICE", "cpu")
    result = train_fold(
        data.manifest,
        data.splits_file,
        data.directory,
        fold=0,
        settings=replace(FAST, epochs=1),
        model_settings=TINY,
        preprocessing_settings=SETTINGS,
        canvas=CANVAS,
    )
    assert len(result.history) == 1


def test_a_fold_outside_the_splits_is_an_error(data: Data) -> None:
    with pytest.raises(ValueError, match="validation fold"):
        train_fold(
            data.manifest,
            data.splits_file,
            data.directory,
            fold=2,
            model_settings=TINY,
            preprocessing_settings=SETTINGS,
            device=CPU,
            canvas=CANVAS,
        )
