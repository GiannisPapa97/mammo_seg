import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from tests.data.manifests import make_manifest
from tests.data.preprocessed import SETTINGS, write_preprocessed_for

from mammo_seg.data.augmentation import AugmentationSettings
from mammo_seg.data.dataset import MammogramDataset
from mammo_seg.data.loaders import BatchComposition, evaluation_loader, training_loader
from mammo_seg.data.preprocessing import PreprocessingSettings
from mammo_seg.data.schema import Manifest
from mammo_seg.data.splits import SplitsFile, create_splits_file
from mammo_seg.engine import train as train_module
from mammo_seg.engine.checkpoint import load_checkpoint, load_model
from mammo_seg.engine.losses import DiceBCELoss
from mammo_seg.engine.metrics import GroupSummary, Summary
from mammo_seg.engine.model import ModelSettings, build_model
from mammo_seg.engine.train import (
    BEST_FILE,
    HISTORY_FILE,
    LAST_FILE,
    SELECTION_METRICS,
    EpochRecord,
    TrainingResult,
    TrainingSettings,
    describe,
    epoch_seed,
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
    directory = tmp_path / "preprocessed"
    write_preprocessed_for(directory, SETTINGS, manifest)
    return Data(manifest, create_splits_file(manifest, 2), directory)


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
            settings.num_workers,
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
        {"seed": -1},
        {"unknown": 1},
    ],
)
def test_invalid_settings_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TrainingSettings(**bad)  # type: ignore[arg-type]


def test_the_sampler_can_be_switched_off() -> None:
    assert TrainingSettings(positive_fraction=None).positive_fraction is None


def test_a_composition_is_off_by_default() -> None:
    assert TrainingSettings().composition is None


def test_the_batch_size_must_be_the_size_of_the_composition() -> None:
    settings = TrainingSettings(batch_size=8, composition=BatchComposition(4, 2, 2))
    assert settings.composition == BatchComposition(4, 2, 2)
    with pytest.raises(ValidationError, match="batch size is 4 but the composition"):
        TrainingSettings(batch_size=4, composition=BatchComposition(4, 2, 2))


# --- seeding ---


def test_an_epoch_seed_is_reproducible_and_fits_torch() -> None:
    assert epoch_seed(0, 1) == epoch_seed(0, 1)
    seeds = [epoch_seed(seed, epoch) for seed in range(5) for epoch in range(1, 61)]
    assert all(0 <= value < 2**64 for value in seeds)
    torch.Generator().manual_seed(max(seeds))  # torch takes the whole range


def test_neighbouring_seeds_do_not_share_epochs() -> None:
    # With seed + epoch, seed 1 at epoch 1 repeated seed 0 at epoch 2.
    assert epoch_seed(1, 1) != epoch_seed(0, 2)
    seeds = [epoch_seed(seed, epoch) for seed in range(10) for epoch in range(1, 61)]
    assert len(set(seeds)) == len(seeds)


def test_a_negative_seed_or_epoch_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        epoch_seed(-1, 1)
    with pytest.raises(ValueError, match="must not be negative"):
        epoch_seed(0, -1)


def test_every_epoch_is_seeded_from_the_seed_and_the_epoch(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[int, int]] = []
    real = train_module.train_one_epoch

    def recording(model: Any, loader: Any, *args: Any) -> float:
        seen.append((loader.generator.initial_seed(), torch.initial_seed()))
        return real(model, loader, *args)

    monkeypatch.setattr(train_module, "train_one_epoch", recording)
    run_fit(data, replace(FAST, seed=7))
    expected = [epoch_seed(7, epoch) for epoch in (1, 2, 3)]
    assert seen == [(seed, seed) for seed in expected]


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


class Constant(torch.nn.Module):
    """Predicts the same logit everywhere; a learning rate of 0 keeps it there."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(logit))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(images) + self.logit


def test_the_training_loss_counts_the_padding_as_background(data: Data) -> None:
    model = Constant(2.0)  # predicts a mass everywhere, the padding too
    train_loader, _ = loaders(data)
    loss_function = DiceBCELoss()

    def mean_loss(everywhere: bool) -> float:
        assert train_loader.generator is not None
        train_loader.generator.manual_seed(0)
        losses = []
        with torch.no_grad():
            for batch in train_loader:
                valid = batch["valid"]
                if everywhere:
                    valid = torch.ones_like(valid)
                logits = model(batch["image"])
                losses.append(float(loss_function(logits, batch["mask"], valid)))
        return sum(losses) / len(losses)

    assert train_loader.generator is not None
    train_loader.generator.manual_seed(0)
    loss = train_one_epoch(
        model,
        train_loader,
        loss_function,
        torch.optim.SGD(model.parameters(), lr=0.0),
        torch.amp.GradScaler("cpu", enabled=False),
        CPU,
        amp=False,
    )
    assert loss == pytest.approx(mean_loss(everywhere=True))
    assert loss != pytest.approx(mean_loss(everywhere=False))  # the data has padding


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


def run_fit(
    data: Data,
    settings: TrainingSettings = FAST,
    output_dir: Path | None = None,
    **options: Any,
) -> TrainingResult:
    torch.manual_seed(0)
    train_loader, validation_loader = loaders(data, settings)
    return fit(
        build_model(TINY),
        train_loader,
        validation_loader,
        settings,
        CPU,
        output_dir,
        model_settings=options.pop("model_settings", TINY),
        preprocessing_settings=options.pop("preprocessing_settings", SETTINGS),
        **options,
    )


def test_fit_runs_the_epochs_validates_and_writes_the_history(
    data: Data, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "run"
    with caplog.at_level(logging.INFO, logger="mammo_seg.engine.train"):
        result = run_fit(data, output_dir=output)
    assert [r.epoch for r in result.history] == [1, 2, 3]
    assert all(r.validation.all_images.images == 16 for r in result.history)
    assert all(math.isfinite(r.train_loss) for r in result.history)
    lines = (output / HISTORY_FILE).read_text().splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [1, 2, 3]
    assert len([m for m in caplog.messages if m.startswith("epoch ")]) == 3
    assert result.stop_reason == "finished"


def test_the_learning_rate_follows_a_cosine_down_to_zero(data: Data) -> None:
    result = run_fit(data, replace(FAST, epochs=4, learning_rate=1e-3))
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
    run_fit(data)
    assert list(work.iterdir()) == []


def test_amp_requested_on_the_cpu_is_ignored(data: Data) -> None:
    result = run_fit(data, replace(FAST, epochs=1, amp=True))
    assert math.isfinite(result.history[0].train_loss)


def test_fit_accepts_a_loader_without_a_generator(data: Data) -> None:
    _, validation_loader = loaders(data)
    plain = torch.utils.data.DataLoader(
        validation_loader.dataset, batch_size=2, drop_last=True
    )
    assert plain.generator is None
    result = fit(
        build_model(TINY),
        plain,
        validation_loader,
        replace(FAST, epochs=1),
        CPU,
        model_settings=TINY,
        preprocessing_settings=SETTINGS,
    )
    assert len(result.history) == 1


def fake_validation(monkeypatch: pytest.MonkeyPatch, summaries: list[Summary]) -> None:
    results = iter(summaries)
    monkeypatch.setattr(train_module, "validate", lambda *args: next(results))


def test_the_best_epoch_is_the_one_with_the_highest_selection_value(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_validation(monkeypatch, [summary(0.2), summary(0.5), summary(0.4)])
    result = run_fit(data)
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
    result = run_fit(data, replace(FAST, selection_metric="with_mass.dice"))
    assert result.best_epoch == 2 and result.best_value == 0.3


def test_an_undefined_selection_value_never_wins(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    nan = replace(summary(0.5), all_images=group(math.nan))
    fake_validation(monkeypatch, [nan, summary(0.3), nan])
    result = run_fit(data)
    assert result.best_epoch == 2 and result.best_value == 0.3


def test_without_any_defined_value_there_is_no_best_epoch(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    nan = replace(summary(0.5), all_images=group(math.nan))
    fake_validation(monkeypatch, [nan, nan, nan])
    result = run_fit(data, replace(FAST, patience=None))
    assert result.best_epoch is None and math.isnan(result.best_value)


# --- checkpoints ---


def test_last_and_best_checkpoints_are_written(
    data: Data, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_validation(monkeypatch, [summary(0.2), summary(0.5), summary(0.4)])
    output = tmp_path / "out"
    run_fit(data, output_dir=output, metadata={"fold": 3})
    last, best = (
        load_checkpoint(output / LAST_FILE),
        load_checkpoint(output / BEST_FILE),
    )
    assert last.metadata["epoch"] == 3 and last.metadata["fold"] == 3
    assert last.training_state is not None
    assert last.training_state["best_epoch"] == 2
    assert [item["epoch"] for item in last.training_state["history"]] == [1, 2, 3]
    assert best.training_state is None  # the weights only
    assert best.metadata["epoch"] == 2 and best.metadata["value"] == 0.5
    assert best.metadata["selection_metric"] == "all_images.dice"
    assert best.model_settings == TINY and best.preprocessing_settings == SETTINGS
    assert not all(
        torch.equal(best.model_state[key], last.model_state[key])
        for key in best.model_state
    )
    assert sorted(path.name for path in output.iterdir()) == [
        BEST_FILE,
        HISTORY_FILE,
        LAST_FILE,
    ]


def test_the_best_checkpoint_is_kept_when_later_epochs_are_worse(
    data: Data, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_validation(monkeypatch, [summary(0.9), summary(0.5), summary(0.4)])
    run_fit(data, output_dir=tmp_path)
    assert load_checkpoint(tmp_path / BEST_FILE).metadata["epoch"] == 1
    assert load_checkpoint(tmp_path / LAST_FILE).metadata["epoch"] == 3


def test_the_best_model_is_usable_for_inference(data: Data, tmp_path: Path) -> None:
    run_fit(data, output_dir=tmp_path)
    model, preprocessing = load_model(tmp_path / BEST_FILE, CPU)
    assert preprocessing == SETTINGS and not model.training
    with torch.no_grad():
        assert model(torch.rand(1, 1, 64, 64)).shape == (1, 1, 64, 64)


# --- stopping ---


def test_training_stops_after_the_patience_without_improvement(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_validation(monkeypatch, [summary(0.5), summary(0.4), summary(0.3)])
    result = run_fit(data, replace(FAST, epochs=6, patience=2))
    assert result.stop_reason == "early_stopping"
    assert [r.epoch for r in result.history] == [1, 2, 3]
    assert result.best_epoch == 1


def test_an_improvement_restarts_the_patience(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_validation(
        monkeypatch, [summary(0.5), summary(0.4), summary(0.6), summary(0.5)] * 2
    )
    result = run_fit(data, replace(FAST, epochs=4, patience=2))
    assert result.stop_reason == "finished" and len(result.history) == 4


def test_without_patience_every_epoch_is_run(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_validation(monkeypatch, [summary(0.5 - 0.1 * i) for i in range(4)])
    result = run_fit(data, replace(FAST, epochs=4, patience=None))
    assert result.stop_reason == "finished" and len(result.history) == 4


def test_the_time_limit_stops_before_an_epoch_that_would_not_fit(
    data: Data,
) -> None:
    result = run_fit(data, time_limit_minutes=1e-9)
    assert result.stop_reason == "time_limit"
    assert [r.epoch for r in result.history] == [1]  # one epoch always runs


def test_a_generous_time_limit_changes_nothing(data: Data) -> None:
    result = run_fit(data, time_limit_minutes=60)
    assert result.stop_reason == "finished" and len(result.history) == 3


# --- resuming ---


def same(record: EpochRecord) -> str:
    """A record as text without its duration: NaN, unlike None, is never equal."""
    return record_to_json(replace(record, seconds=0.0))


@pytest.mark.parametrize("workers", [0, 2])
def test_a_stopped_run_resumed_equals_a_run_that_was_never_stopped(
    data: Data, tmp_path: Path, workers: int
) -> None:
    settings = replace(FAST, epochs=4, num_workers=workers)
    whole = run_fit(data, settings, tmp_path / "whole")
    first = run_fit(data, settings, tmp_path / "split", time_limit_minutes=1e-9)
    assert first.stop_reason == "time_limit" and len(first.history) == 1
    second = run_fit(data, settings, tmp_path / "split", resume=True)
    assert second.stop_reason == "finished"
    assert [r.epoch for r in second.history] == [1, 2, 3, 4]
    assert [same(r) for r in second.history] == [same(r) for r in whole.history]
    assert second.best_epoch == whole.best_epoch
    for name in (LAST_FILE, BEST_FILE):
        a, b = (load_checkpoint(tmp_path / d / name) for d in ("whole", "split"))
        assert all(
            torch.equal(a.model_state[k], b.model_state[k]) for k in a.model_state
        )
    lines = (tmp_path / "split" / HISTORY_FILE).read_text().splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [1, 2, 3, 4]


def test_resuming_a_finished_run_does_nothing(data: Data, tmp_path: Path) -> None:
    done = run_fit(data, output_dir=tmp_path)
    before = (tmp_path / LAST_FILE).read_bytes()
    again = run_fit(data, output_dir=tmp_path, resume=True)
    assert [same(r) for r in again.history] == [same(r) for r in done.history]
    assert again.stop_reason == "finished"
    assert (tmp_path / LAST_FILE).read_bytes() == before


def test_resuming_a_run_stopped_early_stays_stopped(
    data: Data, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = replace(FAST, epochs=6, patience=2)
    fake_validation(monkeypatch, [summary(0.5), summary(0.4), summary(0.3)])
    run_fit(data, settings, tmp_path)
    again = run_fit(data, settings, tmp_path, resume=True)
    assert again.stop_reason == "early_stopping" and len(again.history) == 3


def test_mixed_precision_may_change_between_sessions(
    data: Data, tmp_path: Path
) -> None:
    run_fit(data, FAST, tmp_path, time_limit_minutes=1e-9)
    result = run_fit(data, replace(FAST, amp=True), tmp_path, resume=True)
    assert len(result.history) == 3


def test_resuming_needs_an_output_folder(data: Data) -> None:
    with pytest.raises(ValueError, match="output folder"):
        run_fit(data, resume=True)


def test_resuming_without_a_checkpoint_is_an_error(data: Data, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing to resume"):
        run_fit(data, output_dir=tmp_path, resume=True)


def test_a_checkpoint_without_training_state_cannot_be_resumed(
    data: Data, tmp_path: Path
) -> None:
    run_fit(data, output_dir=tmp_path)
    (tmp_path / BEST_FILE).replace(tmp_path / LAST_FILE)  # the weights only
    with pytest.raises(ValueError, match="no training state"):
        run_fit(data, output_dir=tmp_path, resume=True)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"settings": replace(FAST, epochs=5)}, "epochs: 3 -> 5"),
        ({"settings": replace(FAST, learning_rate=1e-2)}, "learning_rate"),
        ({"settings": replace(FAST, selection_metric="with_mass.dice")}, "selection"),
        # The workers share the samples, so their number changes the draws.
        ({"settings": replace(FAST, num_workers=1)}, "num_workers"),
        (
            {
                "model_settings": ModelSettings(
                    architecture="fpn", encoder=TINY.encoder, encoder_weights=None
                )
            },
            "holds",
        ),
        ({"preprocessing_settings": PreprocessingSettings(size=32)}, "preprocessed"),
        ({"metadata": {"fold": 1}}, "fold = 0, not 1"),
    ],
)
def test_a_run_is_only_resumed_with_the_settings_it_started_with(
    data: Data, tmp_path: Path, options: dict[str, Any], message: str
) -> None:
    run_fit(data, FAST, tmp_path, metadata={"fold": 0}, time_limit_minutes=1e-9)
    options = {"settings": FAST, "metadata": {"fold": 0}, **options}
    with pytest.raises(ValueError, match=message):
        run_fit(data, output_dir=tmp_path, resume=True, **options)


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


# --- crops ---

CROPS = AugmentationSettings(crop_size=(32, 32))


def test_training_canvas_is_the_crop_rounded_up_to_what_the_network_needs() -> None:
    canvas = (1024, 800)
    assert train_module._training_canvas(AugmentationSettings(), canvas) == canvas
    assert train_module._training_canvas(CROPS, canvas) == (32, 32)
    odd = AugmentationSettings(crop_size=(500, 513))
    assert train_module._training_canvas(odd, canvas) == (512, 544)


def test_a_fold_trains_on_crops_and_validates_on_whole_images(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    real_training_loader = training_loader

    def spy(dataset: MammogramDataset, *args: Any, **kwargs: Any) -> Any:
        seen["canvas"] = dataset.canvas
        loader = real_training_loader(dataset, *args, **kwargs)
        seen["batch"] = next(iter(loader))["image"].shape
        return loader

    monkeypatch.setattr(train_module, "training_loader", spy)
    result = train_fold(
        data.manifest,
        data.splits_file,
        data.directory,
        fold=0,
        settings=replace(FAST, epochs=1),
        model_settings=TINY,
        preprocessing_settings=SETTINGS,
        augmentation_settings=CROPS,
        device=CPU,
        canvas=CANVAS,
    )
    assert seen["canvas"] == (32, 32) and seen["batch"] == (2, 1, 32, 32)
    assert result.history[0].validation.all_images.images == 8


def test_a_run_is_not_resumed_with_other_crops(data: Data, tmp_path: Path) -> None:
    def run(augmentation: AugmentationSettings, **options: Any) -> TrainingResult:
        return train_fold(
            data.manifest,
            data.splits_file,
            data.directory,
            fold=0,
            settings=FAST,
            model_settings=TINY,
            preprocessing_settings=SETTINGS,
            augmentation_settings=augmentation,
            device=CPU,
            canvas=CANVAS,
            output_dir=tmp_path / "run",
            **options,
        )

    run(CROPS, time_limit_minutes=1e-9)
    with pytest.raises(ValueError, match="augmentation"):
        run(AugmentationSettings(crop_size=(48, 48)), resume=True)
    assert run(CROPS, resume=True).stop_reason == "finished"


# --- batches of a composition ---

COMPOSED = replace(FAST, composition=BatchComposition(1, 0, 1))  # batches of 2


def test_a_fold_trains_on_batches_of_the_composition(
    data: Data, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    real_training_loader = training_loader

    def spy(dataset: MammogramDataset, *args: Any, **kwargs: Any) -> Any:
        loader = real_training_loader(dataset, *args, **kwargs)
        seen["batches"] = [batch["mask"].flatten(1).sum(1) for batch in loader]
        return loader

    monkeypatch.setattr(train_module, "training_loader", spy)
    result = train_fold(
        data.manifest,
        data.splits_file,
        data.directory,
        fold=0,
        settings=replace(COMPOSED, epochs=1),
        model_settings=TINY,
        preprocessing_settings=SETTINGS,
        augmentation_settings=CROPS,
        device=CPU,
        canvas=CANVAS,
    )
    assert len(seen["batches"]) == 2  # 4 samples per epoch, batches of 2
    for mass_pixels in seen["batches"]:
        assert mass_pixels[0] > 0 and mass_pixels[1] == 0
    assert result.history[0].validation.all_images.images == 8


def test_a_composition_needs_crops(data: Data) -> None:
    with pytest.raises(ValueError, match="made of patches"):
        train_fold(
            data.manifest,
            data.splits_file,
            data.directory,
            fold=0,
            settings=COMPOSED,
            model_settings=TINY,
            preprocessing_settings=SETTINGS,
            device=CPU,
            canvas=CANVAS,
        )


def test_a_run_is_not_resumed_with_another_composition(
    data: Data, tmp_path: Path
) -> None:
    def run(settings: TrainingSettings, **options: Any) -> TrainingResult:
        return train_fold(
            data.manifest,
            data.splits_file,
            data.directory,
            fold=0,
            settings=settings,
            model_settings=TINY,
            preprocessing_settings=SETTINGS,
            augmentation_settings=CROPS,
            device=CPU,
            canvas=CANVAS,
            output_dir=tmp_path / "run",
            **options,
        )

    run(COMPOSED, time_limit_minutes=1e-9)
    other = replace(FAST, composition=BatchComposition(0, 1, 1))
    with pytest.raises(ValueError, match="composition"):
        run(other, resume=True)
    assert run(COMPOSED, resume=True).stop_reason == "finished"
