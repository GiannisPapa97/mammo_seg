"""Scoring trained models on validation folds or the held-out test set.

``evaluate_folds`` loads the checkpoint of each fold, predicts the images the model
did not train on (its own validation fold, or the test set) and scores them with
the metrics of ``metrics.py`` under a grid of post-processing settings:

- a **threshold** on the predicted probability, and
- a **minimum region size** in mm²: connected regions of predicted pixels smaller
  than this are removed (8-connectivity), so a stray spot of a few pixels no longer
  counts as a false alarm. It is in mm², not pixels, because a preprocessed pixel
  covers 0.13 to 0.28 mm depending on the size of the breast; its area comes from
  the preprocessing record (``pixel_area_mm2``). For images of unknown pixel size
  (CBIS-DDSM) only a minimum of 0 can be used, and the areas of the masses and
  false regions are left out (None).

Besides the per-image metrics, every setting is scored per lesion: the masses are
the connected regions of the mask, and a mass is **found** when a predicted region
overlaps it with an IoU of at least ``HIT_IOU`` (0.1). A predicted region that
finds no mass is a **false region**. So a region that spills far over a mass counts
as a miss and a false region, and the found masses against the false regions per
image, over the thresholds, make a curve like an FROC. Two masses that touch are one
region of the mask and count as one mass (in INbreast, 116 masses give 115 regions).

Every summary also breaks the scores down by **subgroup**, with the thresholds of
the splits (``strata.py``): images without a mass by kind (normal, BI-RADS 1; a
benign finding, BI-RADS 2 or 3; a suspicious finding that is not a mass, BI-RADS 4a
to 6), masses by size (smaller than 100 mm² or not), and images by breast density
(fatty, ACR 1 or 2; dense, ACR 3 or 4; the one image without a density is in
neither). Images without a mass are not equally hard, so a set's false alarms depend
on its mix; the subgroups make sets with different mixes comparable.

Each model predicts the whole image as it was validated in training: in sliding
windows of its training canvas if it was trained on crops, in one pass otherwise
(the checkpoint records which; ``inference`` overrides it).

The evaluation also keeps every image with its results under every setting
(``Evaluation.images``): its fold, its identity, its scores, each of its masses (area,
found or not, best IoU, centre, highest probability inside it) and each false region
(area, centre, highest probability). The highest probability inside a missed mass
shows whether the model half saw it or did not respond to it at all.
``details_to_dict`` turns them into plain values, for an analysis of the errors.

Scoring a validation fold with the model that did not train on it, for every fold,
gives out-of-fold predictions for all development images; the results of the folds
are pooled by concatenating the per-image scores. Choosing a setting on these
results tunes it on data it was scored on, so it is slightly optimistic: fix the
setting first, then score the test set once.

Every sweep also scores the baseline of predicting nothing, which already gets the
Dice of 1 on every image without a mass.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader

from mammo_seg.data.dataset import DEFAULT_CANVAS, MammogramDataset, Sample
from mammo_seg.data.folds import fold_ids, held_out_ids
from mammo_seg.data.loaders import evaluation_loader
from mammo_seg.data.preprocess_dataset import (
    RECORD_FILE,
    PreprocessingRecord,
    read_record,
)
from mammo_seg.data.preprocessing import PreprocessingSettings, PreprocessParams
from mammo_seg.data.schema import Manifest
from mammo_seg.data.splits import SplitsFile, fold_name
from mammo_seg.data.strata import DEFAULT_DENSE_ACR, DEFAULT_SMALL_MASS_MM2
from mammo_seg.device import resolve_device
from mammo_seg.engine.checkpoint import load_checkpoint, model_from_checkpoint
from mammo_seg.engine.inference import Inference, predict
from mammo_seg.engine.metrics import ImageScores, Summary, image_scores, summarize

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
# The smallest INbreast mass is 14.7 mm², so these never remove a whole mass.
DEFAULT_MIN_AREAS_MM2 = (0.0, 2.0, 5.0, 10.0)
_EIGHT_CONNECTED = np.ones((3, 3), dtype=int)
# A mass is found when a predicted region overlaps it with at least this IoU.
HIT_IOU = 0.1


@dataclass(frozen=True)
class PostProcessing:
    """How a probability map becomes a prediction."""

    threshold: float = 0.5
    min_area_mm2: float = 0.0  # regions smaller than this are removed

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError(f"the threshold must be in (0, 1), got {self.threshold}")
        if self.min_area_mm2 < 0:
            raise ValueError(
                f"the minimum area cannot be negative: {self.min_area_mm2}"
            )

    def min_pixels(self, pixel_area_mm2: float | None) -> int:
        """The minimum region size in pixels of ``pixel_area_mm2`` each.

        Raises ValueError for a minimum above 0 when the pixel area is unknown.
        """
        if pixel_area_mm2 is None:
            if self.min_area_mm2:
                raise ValueError(
                    "a minimum region size in mm² needs the pixel size, which this "
                    "dataset does not have; use a minimum of 0 (--min-areas-mm2 0)"
                )
            return 0
        return math.ceil(self.min_area_mm2 / pixel_area_mm2)


def settings_grid(
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    min_areas_mm2: Sequence[float] = DEFAULT_MIN_AREAS_MM2,
) -> tuple[PostProcessing, ...]:
    """Every threshold with every minimum region size."""
    if not thresholds or not min_areas_mm2:
        raise ValueError("at least one threshold and one minimum area are needed")
    return tuple(PostProcessing(t, a) for t in thresholds for a in min_areas_mm2)


@dataclass(frozen=True)
class ImageInfo:
    """What the scoring needs to know about an image besides its tensors."""

    # The area of one preprocessed pixel; None when the pixel size is unknown.
    pixel_area_mm2: float | None
    birads: str | None  # the BI-RADS category of the image, e.g. "4a"; None if none
    acr: int | None  # the ACR breast density, 1 to 4; None if unknown
    # Which image it is, for the details of an evaluation; empty when unknown.
    file_id: str = ""
    patient_id: str = ""
    view: str = ""


def pixel_area_mm2(params: PreprocessParams, native_pixel_mm: float) -> float:
    """The area in mm² of one pixel of an image preprocessed with ``params``, from
    the side of its native pixel in mm."""
    rows, cols = params.shape
    return (params.box.height / rows) * (params.box.width / cols) * native_pixel_mm**2


def image_infos(
    record: PreprocessingRecord, manifest: Manifest, file_ids: Sequence[str]
) -> dict[str, ImageInfo]:
    """The ``ImageInfo`` of every image of ``file_ids``, from the preprocessing and
    the manifest; the pixel area is None for an image of unknown pixel size."""
    images = {image.file_id: image for image in manifest.images}
    infos = {}
    for file_id in file_ids:
        image = images[file_id]
        area = (
            None
            if image.pixel_size_mm is None
            else pixel_area_mm2(record.images[file_id], image.pixel_size_mm)
        )
        infos[file_id] = ImageInfo(
            area,
            image.birads,
            image.acr,
            file_id,
            image.patient_id,
            image.view.value,
        )
    return infos


def remove_small_regions(prediction: torch.Tensor, min_area: int) -> torch.Tensor:
    """Drop the connected regions (8-connectivity) of a ``[H, W]`` boolean
    prediction that have fewer than ``min_area`` pixels."""
    if min_area <= 1 or not prediction.any():
        return prediction
    labels, count = ndimage.label(prediction.numpy(), structure=_EIGHT_CONNECTED)
    keep = np.bincount(labels.ravel(), minlength=count + 1) >= min_area
    keep[0] = False  # label 0 is the background
    return torch.from_numpy(keep[labels])


@dataclass(frozen=True)
class MassResult:
    """One mass of an image: its area, where it is, and how a predicted region
    found it."""

    area_mm2: float | None  # None when the pixel size is unknown
    found: bool  # a predicted region overlaps it with an IoU of at least HIT_IOU
    iou: float  # the best IoU of a predicted region with it; 0 if none overlaps
    row: float  # its centre, in pixels of the preprocessed image
    col: float
    # The highest predicted probability inside it, whatever the threshold: how
    # close a missed mass came to being found.
    peak: float


@dataclass(frozen=True)
class FalseRegion:
    """A predicted region that found no mass: its area and where it is."""

    area_mm2: float | None  # None when the pixel size is unknown
    row: float  # its centre, in pixels of the preprocessed image
    col: float
    peak: float  # the highest predicted probability inside it


def _centres(labels: npt.NDArray[np.int32], count: int) -> npt.NDArray[np.float64]:
    """The ``[count, 2]`` centres (row, column) of the labelled regions 1..count."""
    if count == 0:
        return np.zeros((0, 2))
    rows, cols = np.indices(labels.shape)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    return np.stack(
        [
            np.bincount(labels.ravel(), axis.ravel(), minlength=count + 1)[1:] / sizes
            for axis in (rows, cols)
        ],
        axis=1,
    )


def _peaks(
    probabilities: npt.NDArray[np.floating[Any]],
    labels: npt.NDArray[np.int32],
    count: int,
) -> npt.NDArray[np.float64]:
    """The highest probability in each of the labelled regions 1..count."""
    if count == 0:
        return np.zeros(0)
    peaks = ndimage.maximum(probabilities, labels, np.arange(1, count + 1))
    return np.asarray(peaks, dtype=np.float64)


def match_lesions(
    predicted: npt.NDArray[np.bool_],
    truth: npt.NDArray[np.bool_],
    pixel_area_mm2: float | None,
    probabilities: npt.NDArray[np.floating[Any]],
) -> tuple[tuple[MassResult, ...], tuple[FalseRegion, ...]]:
    """The masses of a ``[H, W]`` mask, each found or not, and the false regions of
    a ``[H, W]`` prediction (see the module docstring), with the highest of the
    ``[H, W]`` ``probabilities`` inside each. The areas are None when
    ``pixel_area_mm2`` is.

    Raises ValueError if the probabilities do not have the shape of the mask.
    """
    if probabilities.shape != truth.shape:
        raise ValueError(
            f"probabilities of shape {probabilities.shape} for a mask of shape "
            f"{truth.shape}"
        )
    masses, n_masses = ndimage.label(truth, structure=_EIGHT_CONNECTED)
    regions, n_regions = ndimage.label(predicted, structure=_EIGHT_CONNECTED)
    mass_sizes = np.bincount(masses.ravel(), minlength=n_masses + 1)[1:]
    region_sizes = np.bincount(regions.ravel(), minlength=n_regions + 1)[1:]
    iou = np.zeros((n_masses, n_regions))
    if n_masses and n_regions:
        pairs = masses.ravel().astype(np.int64) * (n_regions + 1) + regions.ravel()
        overlap = np.bincount(pairs, minlength=(n_masses + 1) * (n_regions + 1))
        overlap = overlap.reshape(n_masses + 1, n_regions + 1)[1:, 1:]
        iou = overlap / (mass_sizes[:, None] + region_sizes[None, :] - overlap)
    hits = iou >= HIT_IOU
    best = iou.max(axis=1) if n_regions else np.zeros(n_masses)
    mass_centres, region_centres = (
        _centres(masses, n_masses),
        _centres(regions, n_regions),
    )

    def area(size: int) -> float | None:
        return None if pixel_area_mm2 is None else float(size * pixel_area_mm2)

    found = tuple(
        MassResult(
            area(size),
            bool(hit),
            float(score),
            float(r),
            float(c),
            float(peak),
        )
        for size, hit, score, (r, c), peak in zip(
            mass_sizes,
            hits.any(axis=1),
            best,
            mass_centres,
            _peaks(probabilities, masses, n_masses),
            strict=True,
        )
    )
    false = tuple(
        FalseRegion(area(size), float(r), float(c), float(peak))
        for size, matched, (r, c), peak in zip(
            region_sizes,
            hits.any(axis=0),
            region_centres,
            _peaks(probabilities, regions, n_regions),
            strict=True,
        )
        if not matched
    )
    return found, false


@dataclass(frozen=True)
class ImageResult:
    scores: ImageScores
    predicted_pixels: int  # on the valid pixels, after post-processing
    masses: tuple[MassResult, ...]
    false_regions: tuple[FalseRegion, ...]  # predicted regions that found no mass
    info: ImageInfo


def score_batch(
    probabilities: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    setting: PostProcessing,
    infos: Sequence[ImageInfo],
) -> list[ImageResult]:
    """The scores of every image of a batch of ``[B, 1, H, W]`` CPU tensors, with
    the ``ImageInfo`` of each image.

    Raises ValueError for tensors of different shapes or a wrong number of infos.
    """
    if not probabilities.shape == masks.shape == valid.shape:
        raise ValueError(
            f"shapes differ: probabilities {tuple(probabilities.shape)}, masks "
            f"{tuple(masks.shape)}, valid {tuple(valid.shape)}"
        )
    if len(infos) != len(probabilities):
        raise ValueError(f"{len(infos)} infos for {len(probabilities)} images")
    valid = valid.bool()
    predicted = (probabilities >= setting.threshold) & valid
    cleaned = torch.stack(
        [
            remove_small_regions(image[0], setting.min_pixels(info.pixel_area_mm2))
            for image, info in zip(predicted, infos, strict=True)
        ]
    ).unsqueeze(1)
    scores = image_scores(cleaned.float(), masks, valid)
    pixels = cleaned.flatten(1).sum(1).tolist()
    truth = (masks > 0.5) & valid
    results = []
    for index, (score, count, info) in enumerate(
        zip(scores, pixels, infos, strict=True)
    ):
        masses, false_regions = match_lesions(
            cleaned[index, 0].numpy(),
            truth[index, 0].numpy(),
            info.pixel_area_mm2,
            probabilities[index, 0].numpy(),
        )
        results.append(ImageResult(score, int(count), masses, false_regions, info))
    return results


# The kinds of images without a mass, by BI-RADS category.
NEGATIVE_KINDS = {
    "normal": ("1",),
    "benign": ("2", "3"),
    "suspicious": ("4a", "4b", "4c", "5", "6"),
}


@dataclass(frozen=True)
class NegativeGroup:
    """Images without a mass of one kind."""

    images: int
    false_alarm_rate: float  # NaN without images
    false_regions_per_image: float  # NaN without images


@dataclass(frozen=True)
class MassGroup:
    """Masses of one size."""

    masses: int
    found_masses: float  # the share found; NaN without masses


@dataclass(frozen=True)
class DensityGroup:
    """Images of one breast density."""

    images_with_mass: int
    dice_with_mass: float  # NaN without images with a mass
    images_without_mass: int
    false_alarm_rate: float  # NaN without images without a mass


@dataclass(frozen=True)
class Subgroups:
    negatives: dict[str, NegativeGroup]  # "normal", "benign", "suspicious"
    masses: dict[str, MassGroup]  # "small", "large"
    density: dict[str, DensityGroup]  # "fatty", "dense"


def subgroups_of(results: Sequence[ImageResult]) -> Subgroups:
    """The scores of ``results`` by subgroup (see the module docstring)."""
    negatives = [r for r in results if not r.scores.has_mass]
    masses = [mass for r in results for mass in r.masses]

    def density(dense: bool) -> DensityGroup:
        group = [
            r
            for r in results
            if r.info.acr is not None and (r.info.acr >= DEFAULT_DENSE_ACR) == dense
        ]
        with_mass = [r for r in group if r.scores.has_mass]
        without = [r for r in group if not r.scores.has_mass]
        return DensityGroup(
            images_with_mass=len(with_mass),
            dice_with_mass=_mean([r.scores.dice for r in with_mass]),
            images_without_mass=len(without),
            false_alarm_rate=_mean([float(r.scores.predicted) for r in without]),
        )

    def kind(categories: tuple[str, ...]) -> NegativeGroup:
        group = [r for r in negatives if r.info.birads in categories]
        return NegativeGroup(
            images=len(group),
            false_alarm_rate=_mean([float(r.scores.predicted) for r in group]),
            false_regions_per_image=_mean([len(r.false_regions) for r in group]),
        )

    def size(small: bool) -> MassGroup:
        # A mass of unknown area is in neither group.
        group = [
            m
            for m in masses
            if m.area_mm2 is not None and (m.area_mm2 < DEFAULT_SMALL_MASS_MM2) == small
        ]
        return MassGroup(len(group), _mean([float(m.found) for m in group]))

    return Subgroups(
        negatives={
            name: kind(categories) for name, categories in NEGATIVE_KINDS.items()
        },
        masses={"small": size(True), "large": size(False)},
        density={"fatty": density(False), "dense": density(True)},
    )


@dataclass(frozen=True)
class SettingResult:
    """The summary of a set of images under one post-processing setting."""

    setting: PostProcessing
    summary: Summary
    # Mean predicted pixels on the images without a mass; NaN if there are none.
    false_positive_pixels: float
    # The mean of the Dice on images with a mass and on images without one, so both
    # kinds count equally; NaN if either kind is missing.
    balanced_dice: float
    masses: int  # in all the images
    # The share of the masses that were found; NaN without masses.
    found_masses: float
    # The mean number of false regions per image, over all images; NaN without any.
    false_regions_per_image: float
    subgroups: Subgroups


def summarize_results(
    setting: PostProcessing, results: Sequence[ImageResult]
) -> SettingResult:
    summary = summarize([result.scores for result in results])
    negatives = [r.predicted_pixels for r in results if not r.scores.has_mass]
    masses = [mass for result in results for mass in result.masses]
    return SettingResult(
        setting=setting,
        summary=summary,
        false_positive_pixels=_mean(negatives),
        balanced_dice=(summary.with_mass.dice + summary.without_mass.dice) / 2,
        masses=len(masses),
        found_masses=_mean([float(mass.found) for mass in masses]),
        false_regions_per_image=_mean([len(r.false_regions) for r in results]),
        subgroups=subgroups_of(results),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


@dataclass(frozen=True)
class ModelResults:
    """The per-image results of one model, for every setting and for the baseline."""

    settings: dict[PostProcessing, list[ImageResult]]
    baseline: list[ImageResult]


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[Sample],
    settings: Sequence[PostProcessing],
    device: torch.device,
    infos: Mapping[str, ImageInfo],
    amp: bool = False,
    inference: Inference = "whole",
    window: tuple[int, int] | None = None,
) -> ModelResults:
    """Predict every image of the loader once and score it under every setting;
    ``infos`` holds the ``ImageInfo`` of every image, by file id. ``inference`` and
    ``window`` are those of ``inference.predict``."""
    model.eval()
    results: dict[PostProcessing, list[ImageResult]] = {s: [] for s in settings}
    baseline: list[ImageResult] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        probabilities = predict(model, images, inference, window, amp).cpu()
        masks, valid = batch["mask"], batch["valid"]
        batch_infos = [infos[file_id] for file_id in batch["file_id"]]
        for setting in settings:
            results[setting] += score_batch(
                probabilities, masks, valid, setting, batch_infos
            )
        baseline += score_batch(
            torch.zeros_like(probabilities), masks, valid, PostProcessing(), batch_infos
        )
    return ModelResults(results, baseline)


@dataclass(frozen=True)
class FoldEvaluation:
    """Results over a set of images (one fold, or all folds pooled)."""

    images: int
    settings: tuple[SettingResult, ...]
    baseline: SettingResult  # predicting nothing


@dataclass(frozen=True)
class ImageDetail:
    """One scored image: the fold whose model scored it, and its results under
    every setting."""

    fold: int
    results: tuple[ImageResult, ...]


@dataclass(frozen=True)
class Evaluation:
    on: str  # "validation" or "test"
    checkpoint: str
    per_fold: dict[int, FoldEvaluation]
    inference: dict[int, str]  # how each fold's model predicted: "whole" or "sliding"
    pooled: FoldEvaluation
    # Every image with its results, in the order of the settings of the summaries.
    images: tuple[ImageDetail, ...]


def _fold_evaluation(model_results: ModelResults) -> FoldEvaluation:
    return FoldEvaluation(
        images=len(model_results.baseline),
        settings=tuple(
            summarize_results(setting, results)
            for setting, results in model_results.settings.items()
        ),
        baseline=summarize_results(PostProcessing(), model_results.baseline),
    )


def evaluate_folds(
    manifest: Manifest,
    splits_file: SplitsFile,
    preprocessed_dir: Path,
    runs_dir: Path,
    folds: Sequence[int],
    settings: Sequence[PostProcessing],
    preprocessing_settings: PreprocessingSettings = PreprocessingSettings(),  # noqa: B008
    checkpoint_name: str = "last.pt",
    test: bool = False,
    batch_size: int = 4,
    num_workers: int = 0,
    device: torch.device | None = None,
    amp: bool = False,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    inference: Inference | None = None,
) -> Evaluation:
    """Score the checkpoint of each fold on its validation fold, or, with ``test``,
    on the held-out test set.

    The checkpoints are ``<runs_dir>/fold_<k>/<checkpoint_name>``. Raises ValueError
    for no folds or a repeated fold, a checkpoint of another fold or made with other
    preprocessing settings, splits without a test set, or splits made from another
    manifest. Each model predicts as its checkpoint records (in one pass if it
    records nothing), or as ``inference`` says; sliding windows need a checkpoint
    that records its window, ValueError otherwise.
    """
    if not folds or len(set(folds)) != len(folds):
        raise ValueError(f"the folds must be given once each, got {list(folds)}")
    device = device or resolve_device()
    record = read_record(preprocessed_dir / RECORD_FILE)
    per_fold: dict[int, FoldEvaluation] = {}
    used: dict[int, str] = {}
    images: list[ImageDetail] = []
    pooled = ModelResults({s: [] for s in settings}, [])
    for fold in folds:
        path = runs_dir / fold_name(fold) / checkpoint_name
        checkpoint = load_checkpoint(path, device)
        if checkpoint.metadata.get("fold") != fold:
            raise ValueError(
                f"{path} was trained for fold {checkpoint.metadata.get('fold')}, "
                f"not fold {fold}"
            )
        if checkpoint.preprocessing_settings != preprocessing_settings:
            raise ValueError(
                f"{path} was trained on images preprocessed with "
                f"{checkpoint.preprocessing_settings}, not {preprocessing_settings}"
            )
        ids = (
            held_out_ids(manifest, splits_file)
            if test
            else fold_ids(manifest, splits_file, fold).validation
        )
        dataset = MammogramDataset(
            preprocessed_dir, ids, preprocessing_settings, canvas
        )
        chosen: Inference = inference or checkpoint.metadata.get("inference", "whole")
        stored = checkpoint.metadata.get("window")
        if chosen == "sliding" and stored is None:
            raise ValueError(
                f"{path} records no window size, so it cannot predict in sliding "
                "windows; it was trained before they existed"
            )
        window = (int(stored[0]), int(stored[1])) if stored is not None else None
        infos = image_infos(record, manifest, ids)
        for info in infos.values():  # before predicting, so it fails at once
            for setting in settings:
                setting.min_pixels(info.pixel_area_mm2)
        model = model_from_checkpoint(checkpoint).to(device)
        results = evaluate_model(
            model,
            evaluation_loader(dataset, batch_size, num_workers),
            settings,
            device,
            infos,
            amp,
            chosen,
            window,
        )
        used[fold] = chosen
        per_fold[fold] = _fold_evaluation(results)
        images += [
            ImageDetail(fold, tuple(results.settings[s][index] for s in settings))
            for index in range(len(results.baseline))
        ]
        for setting in settings:
            pooled.settings[setting].extend(results.settings[setting])
        pooled.baseline.extend(results.baseline)
    return Evaluation(
        on="test" if test else "validation",
        checkpoint=checkpoint_name,
        per_fold=per_fold,
        inference=used,
        pooled=_fold_evaluation(pooled),
        images=tuple(images),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def evaluation_to_dict(evaluation: Evaluation) -> dict[str, Any]:
    """The summaries of the evaluation as plain values for JSON (without the
    images, see ``details_to_dict``); NaN becomes None."""
    content: dict[str, Any] = _json_safe(asdict(evaluation))
    del content["images"]
    return content


def details_to_dict(evaluation: Evaluation) -> dict[str, Any]:
    """Every image of the evaluation with its results under every setting, as plain
    values for JSON; NaN becomes None. The ``results`` of an image are in the order
    of ``settings``."""
    settings = [result.setting for result in evaluation.pooled.settings]
    images = []
    for image in evaluation.images:
        info = image.results[0].info if image.results else None
        images.append(
            {
                "fold": image.fold,
                **(asdict(info) if info is not None else {}),
                "has_mass": bool(image.results and image.results[0].scores.has_mass),
                "results": [
                    {
                        **{
                            key: value
                            for key, value in asdict(result.scores).items()
                            if key != "has_mass"
                        },
                        "predicted_pixels": result.predicted_pixels,
                        "masses": [asdict(mass) for mass in result.masses],
                        "false_regions": [asdict(r) for r in result.false_regions],
                    }
                    for result in image.results
                ],
            }
        )
    content: dict[str, Any] = _json_safe(
        {
            "on": evaluation.on,
            "checkpoint": evaluation.checkpoint,
            "inference": evaluation.inference,
            "hit_iou": HIT_IOU,
            "settings": [asdict(setting) for setting in settings],
            "images": images,
        }
    )
    return content


# The scores the best setting is chosen by, in order: the first one defined for some
# setting is used.
_SELECTION_SCORES: dict[str, Callable[[SettingResult], float]] = {
    "balanced Dice": lambda result: result.balanced_dice,
    "mass Dice": lambda result: result.summary.with_mass.dice,
}


def selection_score(results: FoldEvaluation) -> str:
    """The score the best setting is chosen by: the balanced Dice, or the Dice on
    the images with a mass when there are no images without one (as on CBIS-DDSM),
    which leaves the balanced Dice undefined."""
    for name, score in _SELECTION_SCORES.items():
        if any(not math.isnan(score(result)) for result in results.settings):
            return name
    raise ValueError("no setting has a defined balanced Dice or mass Dice")


def best_setting(results: FoldEvaluation) -> SettingResult:
    """The setting with the highest selection score (an undefined one never wins)."""
    score = _SELECTION_SCORES[selection_score(results)]
    defined = [r for r in results.settings if not math.isnan(score(r))]
    return max(defined, key=score)


def _show(value: float, digits: int = 3) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


_HEADER = (
    f"{'threshold':>9} {'min mm²':>8}  | {'mass Dice':>9} {'sens.':>6} {'prec.':>6} | "
    f"{'false alarms':>12} {'FP pixels':>9} | {'found':>6} {'FP/image':>8} | "
    f"{'all Dice':>8} {'balanced':>8}"
)


def _row(label: str, result: SettingResult) -> str:
    groups = result.summary
    return (
        f"{label} | {_show(groups.with_mass.dice):>9} "
        f"{_show(groups.with_mass.sensitivity):>6} "
        f"{_show(groups.with_mass.precision):>6} | "
        f"{_show(groups.without_mass.false_alarm_rate):>12} "
        f"{_show(result.false_positive_pixels, 0):>9} | "
        f"{_show(result.found_masses):>6} "
        f"{_show(result.false_regions_per_image, 2):>8} | "
        f"{_show(groups.all_images.dice):>8} {_show(result.balanced_dice):>8}"
    )


def _label(result: SettingResult, mark: str = " ") -> str:
    return f"{result.setting.threshold:>9.2f} {result.setting.min_area_mm2:>8g}{mark}"


def format_evaluation(evaluation: Evaluation) -> str:
    """The pooled sweep (best setting marked ``*``), the baseline, and the folds at
    the best setting, as text."""
    pooled = evaluation.pooled
    best = best_setting(pooled)
    score = selection_score(pooled)
    kinds = sorted(set(evaluation.inference.values()))
    lines = [
        f"{evaluation.checkpoint} on the {evaluation.on} images "
        f"({pooled.images} scored, {len(evaluation.per_fold)} fold(s) pooled), "
        f"predicted {' and '.join(_INFERENCE_TEXT[kind] for kind in kinds)}",
        "",
        _HEADER,
    ]
    for result in pooled.settings:
        lines.append(_row(_label(result, "*" if result is best else " "), result))
    lines.append(_row(f"{'predict nothing':>19}", pooled.baseline))
    lines += [
        f"* the best {score}",
        "",
        f"per fold at threshold {best.setting.threshold}, "
        f"min area {best.setting.min_area_mm2:g} mm²:",
        _HEADER,
    ]
    for fold, evaluated in evaluation.per_fold.items():
        chosen = next(r for r in evaluated.settings if r.setting == best.setting)
        lines.append(_row(f"{'fold ' + str(fold):>18} ", chosen))
    lines += ["", "by subgroup, all folds pooled, at the same setting:"]
    lines += _subgroup_lines(best.subgroups)
    return "\n".join(lines)


_INFERENCE_TEXT = {"whole": "in one pass", "sliding": "in sliding windows"}


def _subgroup_lines(groups: Subgroups) -> list[str]:
    lines = []
    for name, negative in groups.negatives.items():
        lines.append(
            f"  without a mass, {name + ':':<11} {negative.images:>3} images, "
            f"false alarms {_show(negative.false_alarm_rate)}, "
            f"false regions per image {_show(negative.false_regions_per_image, 2)}"
        )
    small, large = groups.masses["small"], groups.masses["large"]
    limit = f"{DEFAULT_SMALL_MASS_MM2:g} mm²"
    lines.append(
        f"  masses under {limit}: {small.masses:>3}, found {_show(small.found_masses)}"
    )
    lines.append(
        f"  masses of {limit} or more: {large.masses:>3}, "
        f"found {_show(large.found_masses)}"
    )
    for name, density in groups.density.items():
        lines.append(
            f"  {name} breasts: {density.images_with_mass:>3} images with a mass, "
            f"Dice {_show(density.dice_with_mass)}; "
            f"{density.images_without_mass:>3} without, "
            f"false alarms {_show(density.false_alarm_rate)}"
        )
    return lines
