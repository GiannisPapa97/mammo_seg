import json
import math
from pathlib import Path

import pytest
import torch
from tests.data.manifests import make_manifest
from tests.data.preprocessed import SETTINGS, write_preprocessed_for
from torch import nn

from mammo_seg.data.dataset import MammogramDataset
from mammo_seg.data.folds import fold_ids
from mammo_seg.data.loaders import evaluation_loader
from mammo_seg.data.preprocess_dataset import RECORD_FILE, read_record
from mammo_seg.data.preprocessing import (
    BoundingBox,
    PreprocessingSettings,
    PreprocessParams,
)
from mammo_seg.data.schema import Manifest
from mammo_seg.data.splits import SplitsFile, create_splits_file
from mammo_seg.engine.checkpoint import save_checkpoint
from mammo_seg.engine.evaluate import (
    DEFAULT_MIN_AREAS_MM2,
    DEFAULT_THRESHOLDS,
    Evaluation,
    FoldEvaluation,
    ImageInfo,
    PostProcessing,
    SettingResult,
    best_setting,
    evaluate_folds,
    evaluate_model,
    evaluation_to_dict,
    format_evaluation,
    image_infos,
    pixel_area_mm2,
    remove_small_regions,
    score_batch,
    settings_grid,
    summarize_results,
)
from mammo_seg.engine.model import ModelSettings, build_model

CANVAS = (64, 64)
CPU = torch.device("cpu")
TINY = ModelSettings(encoder="tu-mobilenetv3_small_050", encoder_weights=None)
ONE_MM2 = ImageInfo(pixel_area_mm2=1.0)  # pixels and mm² are the same

# --- post-processing settings and small regions ---


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.2, 1.5])
def test_a_threshold_outside_0_and_1_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        PostProcessing(threshold=threshold)


def test_a_negative_minimum_area_is_rejected() -> None:
    with pytest.raises(ValueError, match="minimum area"):
        PostProcessing(min_area_mm2=-1.0)


def test_the_grid_has_every_threshold_with_every_area() -> None:
    grid = settings_grid((0.4, 0.5), (0, 50, 200))
    assert len(grid) == 6
    assert grid[0] == PostProcessing(0.4, 0) and grid[-1] == PostProcessing(0.5, 200)
    assert len(settings_grid()) == len(DEFAULT_THRESHOLDS) * len(DEFAULT_MIN_AREAS_MM2)


@pytest.mark.parametrize("thresholds, areas", [((), (0,)), ((0.5,), ())])
def test_the_grid_needs_a_threshold_and_an_area(
    thresholds: tuple[float, ...], areas: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        settings_grid(thresholds, areas)


def test_a_pixel_area_follows_the_resize_of_the_breast_box() -> None:
    params = PreprocessParams(
        source_shape=(4000, 3000),
        box=BoundingBox(top=0, bottom=3072, left=0, right=1536),
        shape=(1024, 512),
        flipped=False,
    )
    assert pixel_area_mm2(params) == pytest.approx((3 * 0.07) ** 2)  # 3 x 3 pixels
    assert pixel_area_mm2(params, native_pixel_mm=0.1) == pytest.approx(0.09)


def test_a_minimum_area_in_mm2_is_a_number_of_pixels_per_image() -> None:
    setting = PostProcessing(0.5, 10.0)
    assert setting.min_pixels(1.0) == 10
    assert setting.min_pixels(0.05) == 200  # a finer image needs more pixels
    assert setting.min_pixels(0.3) == 34  # rounded up: 33.3 pixels are too few
    assert PostProcessing(0.5, 0.0).min_pixels(0.05) == 0


def regions() -> torch.Tensor:
    """A 1 pixel spot, a 4 pixel block, a 9 pixel block and two diagonal pixels."""
    image = torch.zeros(12, 12, dtype=torch.bool)
    image[0, 0] = True
    image[0, 5:7] = image[1, 5:7] = True
    image[5:8, 5:8] = True
    image[10, 0] = image[11, 1] = True  # touch only at a corner
    return image


def test_small_regions_are_removed() -> None:
    kept = remove_small_regions(regions(), 5)
    assert kept.sum() == 9 and kept[5:8, 5:8].all()


def test_regions_that_touch_at_a_corner_are_one_region() -> None:
    assert remove_small_regions(regions(), 2).sum() == 4 + 9 + 2  # not the spot
    assert remove_small_regions(regions(), 3).sum() == 4 + 9  # not the two pixels


@pytest.mark.parametrize("area", [0, 1])
def test_an_area_of_0_or_1_keeps_everything(area: int) -> None:
    assert torch.equal(remove_small_regions(regions(), area), regions())


def test_an_empty_prediction_stays_empty() -> None:
    empty = torch.zeros(5, 5, dtype=torch.bool)
    assert not remove_small_regions(empty, 10).any()


# --- scoring a batch ---


def batch(
    mass: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A 16 x 16 image: a 4 x 4 mass (if any), 8 of its pixels predicted at 0.9, a
    2 pixel spot elsewhere at 0.9, one pixel of the mass at 0.4, and an invalid
    column on the right with a confident prediction that must be ignored."""
    probabilities = torch.zeros(1, 1, 16, 16)
    masks = torch.zeros(1, 1, 16, 16)
    valid = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    valid[..., 15] = False
    probabilities[..., :, 15] = 0.99
    if mass:
        masks[0, 0, 2:6, 2:6] = 1
        probabilities[0, 0, 2:4, 2:6] = 0.9  # 8 mass pixels
        probabilities[0, 0, 4, 2] = 0.4
    probabilities[0, 0, 12, 2:4] = 0.9  # a 2 pixel spot outside any mass
    return probabilities, masks, valid


def test_a_spot_costs_precision_until_it_is_removed() -> None:
    kept = score_batch(*batch(True), PostProcessing(0.5, 0), [ONE_MM2])[0]
    assert kept.predicted_pixels == 10
    assert kept.scores.dice == pytest.approx(2 * 8 / (2 * 8 + 2 + 8))
    removed = score_batch(*batch(True), PostProcessing(0.5, 5), [ONE_MM2])[0]
    assert removed.predicted_pixels == 8
    assert removed.scores.dice == pytest.approx(2 * 8 / (2 * 8 + 8))
    assert removed.scores.precision == 1.0


def test_a_lower_threshold_finds_more_of_the_mass() -> None:
    result = score_batch(*batch(True), PostProcessing(0.3, 5), [ONE_MM2])[0]
    assert result.scores.sensitivity == pytest.approx(9 / 16)  # the 0.4 pixel too


def test_a_spot_on_an_image_without_a_mass_is_a_false_alarm_until_removed() -> None:
    alarm = score_batch(*batch(False), PostProcessing(0.5, 0), [ONE_MM2])[0]
    assert alarm.scores.predicted and alarm.scores.dice == 0
    assert alarm.predicted_pixels == 2
    quiet = score_batch(*batch(False), PostProcessing(0.5, 3), [ONE_MM2])[0]
    assert not quiet.scores.predicted and quiet.scores.dice == 1
    assert quiet.predicted_pixels == 0


def test_the_invalid_pixels_never_count() -> None:
    # The confident column 15 is invalid: 16 pixels that would otherwise be predicted.
    assert (
        score_batch(*batch(False), PostProcessing(0.5, 0), [ONE_MM2])[
            0
        ].predicted_pixels
        == 2
    )


def test_each_image_of_a_batch_is_scored_on_its_own() -> None:
    with_mass, masks_a, valid_a = batch(True)
    without, masks_b, valid_b = batch(False)
    results = score_batch(
        torch.cat([with_mass, without]),
        torch.cat([masks_a, masks_b]),
        torch.cat([valid_a, valid_b]),
        PostProcessing(0.5, 5),
        [ONE_MM2, ONE_MM2],
    )
    assert [r.scores.has_mass for r in results] == [True, False]
    assert [r.predicted_pixels for r in results] == [8, 0]


def test_the_same_area_in_mm2_removes_more_pixels_on_a_finer_image() -> None:
    # The 2 pixel spot is 2 mm² at 1 mm² per pixel, but 8 mm² at 4 mm² per pixel.
    coarse = score_batch(*batch(False), PostProcessing(0.5, 5), [ImageInfo(4.0)])[0]
    fine = score_batch(*batch(False), PostProcessing(0.5, 5), [ONE_MM2])[0]
    assert coarse.scores.predicted and not fine.scores.predicted


def test_every_image_needs_its_info() -> None:
    with pytest.raises(ValueError, match="0 infos for 1 images"):
        score_batch(*batch(True), PostProcessing(), [])


def test_tensors_of_different_shapes_are_an_error() -> None:
    probabilities, masks, valid = batch(True)
    with pytest.raises(ValueError, match="shapes differ"):
        score_batch(probabilities[..., :8], masks, valid, PostProcessing(), [ONE_MM2])


def test_the_summary_of_results() -> None:
    results = [
        *score_batch(*batch(True), PostProcessing(0.5, 0), [ONE_MM2]),
        *score_batch(*batch(False), PostProcessing(0.5, 0), [ONE_MM2]),
        *score_batch(*batch(False), PostProcessing(0.5, 3), [ONE_MM2]),
    ]
    result = summarize_results(PostProcessing(0.5, 0), results)
    assert result.false_positive_pixels == pytest.approx(1.0)  # (2 + 0) / 2
    assert result.summary.without_mass.false_alarm_rate == pytest.approx(0.5)
    assert result.summary.without_mass.dice == pytest.approx(0.5)
    expected = (2 * 8 / (2 * 8 + 2 + 8) + 0.5) / 2
    assert result.balanced_dice == pytest.approx(expected)


def test_without_images_of_one_kind_the_balanced_dice_is_undefined() -> None:
    only_masses = summarize_results(
        PostProcessing(), score_batch(*batch(True), PostProcessing(), [ONE_MM2])
    )
    assert math.isnan(only_masses.balanced_dice)
    assert math.isnan(only_masses.false_positive_pixels)


# --- a model over a loader ---


class BrightPixels(nn.Module):
    """Predicts a mass wherever the image is white: a stand-in for a trained model."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images > 0.99).float() * 20 - 10


@pytest.fixture
def manifest(tmp_path: Path) -> Manifest:
    """Eight patients (four with a mass), two images each, split into two folds."""
    manifest = make_manifest(4, 4, per_patient=2)
    write_preprocessed_for(tmp_path / "preprocessed", SETTINGS, manifest)
    return manifest


def test_a_model_is_scored_under_every_setting_and_against_predicting_nothing(
    tmp_path: Path, manifest: Manifest
) -> None:
    splits = create_splits_file(manifest, 2)
    ids = fold_ids(manifest, splits, 0).validation
    dataset = MammogramDataset(tmp_path / "preprocessed", ids, SETTINGS, CANVAS)
    # The white pixels are the 10 x 10 mass (on images that have one), and the last
    # column of every image: a 64 pixel strip that is no mass.
    # A pixel is 6.25 x 5 native pixels of 0.07 mm, 0.153 mm²: the strip is 9.8 mm²
    # and a mass 15.3 mm², so a minimum of 12 mm² removes the strip only.
    settings = [PostProcessing(0.5, 0), PostProcessing(0.5, 12)]
    infos = image_infos(read_record(tmp_path / "preprocessed" / RECORD_FILE), ids)
    results = evaluate_model(
        BrightPixels(), evaluation_loader(dataset, 3), settings, CPU, infos
    )
    assert [len(results.settings[s]) for s in settings] == [8, 8]
    assert len(results.baseline) == 8
    strip, cleaned = (summarize_results(s, results.settings[s]) for s in settings)
    assert strip.summary.with_mass.sensitivity == 1.0
    assert strip.summary.with_mass.dice == pytest.approx(200 / (200 + 64))
    assert strip.summary.without_mass.false_alarm_rate == 1.0
    assert cleaned.summary.with_mass.dice == 1.0
    assert cleaned.summary.without_mass.false_alarm_rate == 0.0
    assert cleaned.balanced_dice == 1.0
    nothing = summarize_results(PostProcessing(), results.baseline)
    assert nothing.summary.with_mass.dice == 0
    assert nothing.summary.without_mass.dice == 1
    assert nothing.false_positive_pixels == 0


# --- the folds ---


def make_runs(
    root: Path, folds: tuple[int, ...] = (0, 1), name: str = "last.pt", **metadata: int
) -> Path:
    runs = root / "runs"
    for fold in folds:
        save_checkpoint(
            runs / f"fold_{fold}" / name,
            build_model(TINY),
            TINY,
            SETTINGS,
            metadata={"fold": fold, **metadata},
        )
    return runs


def evaluate(
    tmp_path: Path,
    manifest: Manifest,
    splits: SplitsFile | None = None,
    **options: object,
) -> Evaluation:
    arguments: dict[str, object] = {
        "manifest": manifest,
        "splits_file": splits or create_splits_file(manifest, 2),
        "preprocessed_dir": tmp_path / "preprocessed",
        "runs_dir": options.get("runs_dir") or make_runs(tmp_path),
        "folds": [0, 1],
        "settings": settings_grid((0.5,), (0, 5)),
        "preprocessing_settings": SETTINGS,
        "device": CPU,
        "canvas": CANVAS,
    }
    return evaluate_folds(**{**arguments, **options})  # type: ignore[arg-type]


def test_each_fold_is_scored_on_its_own_validation_images(
    tmp_path: Path, manifest: Manifest
) -> None:
    evaluation = evaluate(tmp_path, manifest)
    assert evaluation.on == "validation" and evaluation.checkpoint == "last.pt"
    assert list(evaluation.per_fold) == [0, 1]
    assert [e.images for e in evaluation.per_fold.values()] == [8, 8]
    assert evaluation.pooled.images == 16
    assert [r.setting for r in evaluation.pooled.settings] == [
        PostProcessing(0.5, 0),
        PostProcessing(0.5, 5),
    ]


def test_the_pooled_results_are_those_of_all_images_together(
    tmp_path: Path, manifest: Manifest
) -> None:
    evaluation = evaluate(tmp_path, manifest)
    baseline = evaluation.pooled.baseline.summary
    assert baseline.all_images.images == 16 and baseline.with_mass.images == 4
    assert baseline.with_mass.dice == 0 and baseline.without_mass.dice == 1
    assert baseline.all_images.dice == pytest.approx(0.75)  # 12 of 16 are empty


def test_a_checkpoint_of_another_fold_is_refused(
    tmp_path: Path, manifest: Manifest
) -> None:
    runs = make_runs(tmp_path, folds=(0,))
    (runs / "fold_1").mkdir()
    (runs / "fold_0" / "last.pt").rename(runs / "fold_1" / "last.pt")
    with pytest.raises(ValueError, match="trained for fold 0, not fold 1"):
        evaluate(tmp_path, manifest, runs_dir=runs, folds=[1])


def test_a_checkpoint_without_a_fold_is_refused(
    tmp_path: Path, manifest: Manifest
) -> None:
    save_checkpoint(
        tmp_path / "runs" / "fold_0" / "last.pt", build_model(TINY), TINY, SETTINGS
    )
    with pytest.raises(ValueError, match="trained for fold None"):
        evaluate(tmp_path, manifest, runs_dir=tmp_path / "runs", folds=[0])


def test_a_checkpoint_made_with_other_preprocessing_is_refused(
    tmp_path: Path, manifest: Manifest
) -> None:
    other = PreprocessingSettings(size=32)
    runs = tmp_path / "runs"
    save_checkpoint(
        runs / "fold_0" / "last.pt",
        build_model(TINY),
        TINY,
        other,
        metadata={"fold": 0},
    )
    with pytest.raises(ValueError, match="preprocessed with"):
        evaluate(tmp_path, manifest, runs_dir=runs, folds=[0])


def test_the_checkpoint_name_chooses_the_file(
    tmp_path: Path, manifest: Manifest
) -> None:
    runs = make_runs(tmp_path, name="best.pt")
    assert (
        evaluate(
            tmp_path, manifest, runs_dir=runs, checkpoint_name="best.pt"
        ).checkpoint
        == "best.pt"
    )
    with pytest.raises(ValueError, match="not a readable checkpoint"):
        evaluate(tmp_path, manifest, runs_dir=runs)  # there is no last.pt


@pytest.mark.parametrize("folds", [[], [0, 0]])
def test_the_folds_must_be_given_once_each(
    tmp_path: Path, manifest: Manifest, folds: list[int]
) -> None:
    with pytest.raises(ValueError, match="once each"):
        evaluate(tmp_path, manifest, folds=folds)


def test_the_test_set_needs_splits_with_one(tmp_path: Path, manifest: Manifest) -> None:
    with pytest.raises(ValueError, match="no test set"):
        evaluate(tmp_path, manifest, test=True)


def test_the_test_set_is_scored_instead_of_the_validation_folds(
    tmp_path: Path, manifest: Manifest
) -> None:
    splits = create_splits_file(manifest, 2, test_fraction=0.25)
    evaluation = evaluate(tmp_path, manifest, splits, test=True)
    assert evaluation.on == "test"
    test_images = evaluation.per_fold[0].images
    assert test_images == 4 and evaluation.per_fold[1].images == test_images
    validation = evaluate(tmp_path, manifest, splits)
    assert validation.per_fold[0].images == 6  # the other images are in the test set


# --- reporting ---


def sweep() -> FoldEvaluation:
    def result(threshold: float, dice: float, other: float) -> SettingResult:
        base = summarize_results(PostProcessing(threshold), [])
        return SettingResult(base.setting, base.summary, 0.0, dice + other)

    return FoldEvaluation(
        images=0,
        settings=(
            result(0.3, 0.2, 0.1),
            result(0.5, 0.4, 0.3),
            result(0.7, math.nan, 0),
        ),
        baseline=result(0.5, 0.0, 0.0),
    )


def test_the_best_setting_has_the_highest_defined_balanced_dice() -> None:
    assert best_setting(sweep()).setting.threshold == 0.5


def test_with_no_defined_balanced_dice_there_is_no_best_setting() -> None:
    nothing = summarize_results(PostProcessing(), [])
    results = FoldEvaluation(0, (nothing,), nothing)
    with pytest.raises(ValueError, match="no setting"):
        best_setting(results)


def test_the_results_become_json_with_none_for_undefined_values(
    tmp_path: Path, manifest: Manifest
) -> None:
    evaluation = evaluate(tmp_path, manifest)
    text = json.dumps(evaluation_to_dict(evaluation))
    content = json.loads(text)
    assert content["on"] == "validation"
    assert set(content["per_fold"]) == {"0", "1"}
    pooled = content["pooled"]["settings"][0]
    assert pooled["setting"] == {"threshold": 0.5, "min_area_mm2": 0}
    assert content["pooled"]["baseline"]["summary"]["with_mass"]["sensitivity"] == 0.0


def test_the_text_shows_the_sweep_the_baseline_and_the_folds(
    tmp_path: Path, manifest: Manifest
) -> None:
    text = format_evaluation(evaluate(tmp_path, manifest))
    assert "last.pt on the validation images (16 scored, 2 fold(s) pooled)" in text
    assert "predict nothing" in text and "fold 0" in text and "fold 1" in text
    assert text.count("*") == 1  # the best setting is marked once
    assert len({len(line) for line in text.splitlines() if " | " in line}) == 1
