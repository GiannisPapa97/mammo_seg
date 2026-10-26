"""Confidence intervals by resampling patients: the bootstrap.

The validation results are one sample of patients; with other patients the scores
would come out differently. The bootstrap estimates by how much. It draws as many
patients as there are, with replacement, from the patients of a details file
(``mammo-seg evaluate --details``), computes every metric on the images of the
drawn patients (a patient drawn twice counts twice), and repeats this
``resamples`` times. The middle ``level`` of the values, between their
percentiles, is the interval. Patients are drawn, not images: the views of one
patient show the same breast and the same mass, so drawing images would treat them
as independent and make the intervals too narrow.

Two evaluations of the same images (two models) are compared **paired**: every draw
scores both on the same patients and keeps the difference, second minus first, so
the luck of a draw (hard or easy patients) cancels. A difference whose interval
excludes 0 is more than the luck of which patients were scored.

The metrics are those of the evaluation, at one post-processing setting, computed
the same way: per-image means for the images with a mass, without one and all
images (undefined values left out), the share of the masses found (all, small and
large ones), and the false regions per image. Each is a ratio of two sums over the
images, so a patient comes down to its sums, and a draw to the sums weighted by how
often each patient was drawn. Nothing is predicted again: the details file is all
it needs, and it runs in seconds without a GPU.

A draw can lack what a metric needs (no small mass among the drawn patients); the
metric is then undefined in that draw and left out of its interval. The intervals
cover which patients were scored, not the randomness of training, which would need
models trained again with other seeds.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from mammo_seg.data.strata import DEFAULT_SMALL_MASS_MM2

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 0
DEFAULT_LEVEL = 0.95

# The metrics, in the order they are reported, with their labels.
METRICS = {
    "with_mass.dice": "Dice, with a mass",
    "with_mass.iou": "IoU, with a mass",
    "with_mass.precision": "precision, with a mass",
    "with_mass.sensitivity": "sensitivity, with a mass",
    "without_mass.false_alarm_rate": "false alarms",
    "all_images.dice": "Dice, all images",
    "masses_found": "masses found",
    "small_masses_found": "small masses found",
    "large_masses_found": "large masses found",
    "false_regions_per_image": "false regions per image",
}
_SCORES = ("dice", "iou", "precision", "sensitivity")


@dataclass(frozen=True)
class Interval:
    value: float  # on the patients as they are; NaN if undefined
    low: float
    high: float


@dataclass(frozen=True)
class Intervals:
    """The interval of every metric of one evaluation."""

    patients: int
    images: int
    resamples: int
    seed: int
    level: float
    metrics: dict[str, Interval]


@dataclass(frozen=True)
class Comparison:
    """The intervals of two evaluations of the same images and of their difference
    (second minus first), from the same draws of patients."""

    patients: int
    images: int
    resamples: int
    seed: int
    level: float
    first: dict[str, Interval]
    second: dict[str, Interval]
    difference: dict[str, Interval]


def _setting_index(
    details: Mapping[str, Any], threshold: float, min_area: float
) -> int:
    for index, setting in enumerate(details["settings"]):
        if math.isclose(setting["threshold"], threshold) and math.isclose(
            setting["min_area_mm2"], min_area
        ):
            return index
    raise ValueError(
        f"the setting threshold {threshold}, minimum area {min_area} mm² is not in "
        f"the details; they have {details['settings']}"
    )


def _defined(value: float | None) -> tuple[float, float]:
    """A per-image score as (sum, count): undefined scores (null) count nothing."""
    return (0.0, 0.0) if value is None else (float(value), 1.0)


def _image_terms(
    image: Mapping[str, Any], result: Mapping[str, Any], small_mass_mm2: float
) -> list[tuple[float, float]]:
    """What one image adds to the numerator and the denominator of every metric."""
    nothing = (0.0, 0.0)
    has_mass = bool(image["has_mass"])
    masses = result["masses"]
    small = [mass for mass in masses if mass["area_mm2"] < small_mass_mm2]
    large = [mass for mass in masses if mass["area_mm2"] >= small_mass_mm2]

    def found(group: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
        return float(sum(mass["found"] for mass in group)), float(len(group))

    terms = {
        **{
            f"with_mass.{score}": _defined(result[score]) if has_mass else nothing
            for score in _SCORES
        },
        "without_mass.false_alarm_rate": (
            nothing if has_mass else (float(result["predicted"]), 1.0)
        ),
        "all_images.dice": _defined(result["dice"]),
        "masses_found": found(masses),
        "small_masses_found": found(small),
        "large_masses_found": found(large),
        "false_regions_per_image": (float(len(result["false_regions"])), 1.0),
    }
    return [terms[name] for name in METRICS]


def patient_sums(
    details: Mapping[str, Any],
    threshold: float,
    min_area_mm2: float,
    small_mass_mm2: float = DEFAULT_SMALL_MASS_MM2,
) -> tuple[tuple[str, ...], npt.NDArray[np.float64]]:
    """The patient ids, sorted, and for each patient the sums of its images for every
    metric, as ``[patients, metrics, 2]``: the numerator, then the denominator.

    Raises ValueError for details without images or without the setting.
    """
    if not details["images"]:
        raise ValueError("the details have no images")
    index = _setting_index(details, threshold, min_area_mm2)
    sums: dict[str, npt.NDArray[np.float64]] = {}
    for image in details["images"]:
        terms = np.array(_image_terms(image, image["results"][index], small_mass_mm2))
        patient = image["patient_id"]
        sums[patient] = sums.get(patient, np.zeros_like(terms)) + terms
    patients = tuple(sorted(sums))
    return patients, np.stack([sums[patient] for patient in patients])


def draw_counts(
    patients: int, resamples: int, seed: int = DEFAULT_SEED
) -> npt.NDArray[np.int64]:
    """How often each patient is drawn in each resample, ``[resamples, patients]``:
    ``patients`` draws with replacement, every patient equally likely.

    Raises ValueError for no patients or no resamples.
    """
    if patients < 1 or resamples < 1:
        raise ValueError(
            f"needs patients and resamples, got {patients} and {resamples}"
        )
    generator = np.random.default_rng(seed)
    counts: npt.NDArray[np.int64] = generator.multinomial(
        patients, np.full(patients, 1 / patients), size=resamples
    )
    return counts


def metric_values(
    sums: npt.NDArray[np.float64], counts: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """Every metric in every resample, ``[resamples, metrics]``: the sums of the
    patients weighted by how often they were drawn; NaN where a resample has
    nothing to divide by."""
    totals = np.einsum("rp,pmk->rmk", counts.astype(np.float64), sums)
    numerators, denominators = totals[..., 0], totals[..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominators > 0, numerators / denominators, np.nan)


def _interval(value: float, draws: npt.NDArray[np.float64], level: float) -> Interval:
    defined = draws[~np.isnan(draws)]
    if math.isnan(value) or defined.size == 0:
        return Interval(value, math.nan, math.nan)
    tail = (1 - level) / 2
    low, high = np.quantile(defined, [tail, 1 - tail])
    return Interval(value, float(low), float(high))


def _check(resamples: int, level: float) -> None:
    if resamples < 1:
        raise ValueError(f"needs at least one resample, got {resamples}")
    if not 0 < level < 1:
        raise ValueError(f"the level must be between 0 and 1, got {level}")


def _intervals_of(
    sums: npt.NDArray[np.float64],
    counts: npt.NDArray[np.int64],
    level: float,
) -> tuple[dict[str, Interval], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    values = metric_values(sums, np.ones((1, len(sums)), dtype=np.int64))[0]
    draws = metric_values(sums, counts)
    intervals = {
        name: _interval(float(values[k]), draws[:, k], level)
        for k, name in enumerate(METRICS)
    }
    return intervals, values, draws


def bootstrap(
    details: Mapping[str, Any],
    threshold: float,
    min_area_mm2: float,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> Intervals:
    """The interval of every metric of one evaluation at one setting.

    Raises ValueError for details without images or without the setting, no
    resamples, or a level outside (0, 1).
    """
    _check(resamples, level)
    patients, sums = patient_sums(details, threshold, min_area_mm2)
    counts = draw_counts(len(patients), resamples, seed)
    intervals, _, _ = _intervals_of(sums, counts, level)
    return Intervals(
        patients=len(patients),
        images=len(details["images"]),
        resamples=resamples,
        seed=seed,
        level=level,
        metrics=intervals,
    )


def _images_of(details: Mapping[str, Any]) -> dict[str, str]:
    return {image["file_id"]: image["patient_id"] for image in details["images"]}


def compare(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    threshold: float,
    min_area_mm2: float,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> Comparison:
    """The intervals of two evaluations at one setting and of their difference,
    second minus first, from the same draws of patients.

    Raises ValueError if the two did not score the same images of the same patients
    on the same set (validation or test), and as ``bootstrap`` does.
    """
    _check(resamples, level)
    if first["on"] != second["on"]:
        raise ValueError(
            f"one evaluation is on the {first['on']} images, the other on the "
            f"{second['on']} images"
        )
    if _images_of(first) != _images_of(second):
        raise ValueError(
            "the two evaluations did not score the same images of the same patients"
        )
    patients, first_sums = patient_sums(first, threshold, min_area_mm2)
    same_patients, second_sums = patient_sums(second, threshold, min_area_mm2)
    assert patients == same_patients  # the same images, so the same patients
    counts = draw_counts(len(patients), resamples, seed)
    first_intervals, first_values, first_draws = _intervals_of(
        first_sums, counts, level
    )
    second_intervals, second_values, second_draws = _intervals_of(
        second_sums, counts, level
    )
    difference = {
        name: _interval(
            float(second_values[k] - first_values[k]),
            second_draws[:, k] - first_draws[:, k],
            level,
        )
        for k, name in enumerate(METRICS)
    }
    return Comparison(
        patients=len(patients),
        images=len(first["images"]),
        resamples=resamples,
        seed=seed,
        level=level,
        first=first_intervals,
        second=second_intervals,
        difference=difference,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def to_dict(result: Intervals | Comparison) -> dict[str, Any]:
    """Plain values for JSON; NaN becomes None."""
    content: dict[str, Any] = _json_safe(asdict(result))
    return content


def _show(interval: Interval) -> str:
    if math.isnan(interval.value):
        return "n/a"
    if math.isnan(interval.low):
        return f"{interval.value:.3f}"
    return f"{interval.value:.3f} [{interval.low:.3f}, {interval.high:.3f}]"


def _difference(interval: Interval) -> str:
    if math.isnan(interval.value) or math.isnan(interval.low):
        return _show(interval)
    sign = "+" if interval.value >= 0 else ""
    if interval.low > 0:
        verdict = "higher"
    elif interval.high < 0:
        verdict = "lower"
    else:
        verdict = "cannot tell"
    return (
        f"{sign}{interval.value:.3f} [{interval.low:+.3f}, {interval.high:+.3f}] "
        f"{verdict}"
    )


def format_result(
    result: Intervals | Comparison, names: Sequence[str] = ("first", "second")
) -> str:
    """A table of every metric: its value with its interval, and for a comparison
    the difference with its interval and whether it excludes 0."""
    level = f"{result.level:.0%}"
    lines = [
        f"{level} intervals from {result.resamples} draws of the {result.patients} "
        f"patients ({result.images} images), seed {result.seed}"
    ]
    width = max(len(label) for label in METRICS.values())
    if isinstance(result, Intervals):
        for name, label in METRICS.items():
            lines.append(f"{label:<{width}}  {_show(result.metrics[name])}")
        return "\n".join(lines)
    lines.append(f"first: {names[0]}\nsecond: {names[1]}")
    column = 26
    lines.append(
        f"{'':<{width}}  {'first':<{column}}{'second':<{column}}second minus first"
    )
    for name, label in METRICS.items():
        lines.append(
            f"{label:<{width}}  {_show(result.first[name]):<{column}}"
            f"{_show(result.second[name]):<{column}}"
            f"{_difference(result.difference[name])}"
        )
    return "\n".join(lines)
