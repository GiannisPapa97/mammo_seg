import json
import math
from typing import Any

import numpy as np
import pytest

from mammo_seg.engine.bootstrap import (
    METRICS,
    Comparison,
    Interval,
    Intervals,
    bootstrap,
    compare,
    draw_counts,
    format_result,
    metric_values,
    patient_sums,
    to_dict,
)

SETTING = {"threshold": 0.5, "min_area_mm2": 0.0}


def mass(area_mm2: float | None, found: bool) -> dict[str, Any]:
    return {"area_mm2": area_mm2, "found": found}


def image(
    file_id: str,
    patient: str,
    *,
    dice: float = 1.0,
    masses: tuple[dict[str, Any], ...] = (),
    false_regions: int = 0,
) -> dict[str, Any]:
    """An image of the details at one setting. With masses, its precision is its
    Dice, or undefined (nothing predicted) when nothing was found; without, anything
    predicted is a false alarm and costs its Dice."""
    has_mass = bool(masses)
    predicted = any(m["found"] for m in masses) or false_regions > 0
    result = {
        "dice": dice,
        "iou": dice / (2 - dice),
        "precision": dice if predicted else None,
        "sensitivity": dice if has_mass else None,
        "predicted": predicted,
        "masses": list(masses),
        "false_regions": [{"area_mm2": 1.0}] * false_regions,
    }
    return {
        "file_id": file_id,
        "patient_id": patient,
        "has_mass": has_mass,
        "results": [result],
    }


def details(*images: dict[str, Any], on: str = "validation") -> dict[str, Any]:
    return {"on": on, "settings": [SETTING], "images": list(images)}


def run(found: dict[str, Any], **options: Any) -> Intervals:
    return bootstrap(found, SETTING["threshold"], SETTING["min_area_mm2"], **options)


def versus(first: dict[str, Any], second: dict[str, Any], **options: Any) -> Comparison:
    return compare(
        first, second, SETTING["threshold"], SETTING["min_area_mm2"], **options
    )


# Three patients: a found large mass in both views, a missed small mass, and two
# images without a mass, one of them with a false region.
EXAMPLE = details(
    image("1", "a", dice=0.8, masses=(mass(500, True),)),
    image("2", "a", dice=0.6, masses=(mass(400, True),)),
    image("3", "b", dice=0.0, masses=(mass(50, False),)),
    image("4", "c", dice=1.0),
    image("5", "c", dice=0.0, false_regions=2),
)


def many_patients(count: int, found_share: float = 0.5) -> dict[str, Any]:
    """``count`` patients with one mass each; the first ``found_share`` are found."""
    return details(
        *(
            image(
                str(k),
                f"p{k}",
                dice=1.0 if k < count * found_share else 0.0,
                masses=(mass(200, k < count * found_share),),
            )
            for k in range(count)
        )
    )


# --- the values on the patients as they are ---


def test_the_values_are_those_of_the_evaluation() -> None:
    values = {name: i.value for name, i in run(EXAMPLE).metrics.items()}
    assert values["with_mass.dice"] == pytest.approx((0.8 + 0.6 + 0.0) / 3)
    # The missed mass predicted nothing: its precision is undefined, left out.
    assert values["with_mass.precision"] == pytest.approx((0.8 + 0.6) / 2)
    assert values["without_mass.false_alarm_rate"] == 0.5
    assert values["all_images.dice"] == pytest.approx(2.4 / 5)
    assert values["masses_found"] == pytest.approx(2 / 3)
    assert values["small_masses_found"] == 0.0
    assert values["large_masses_found"] == 1.0
    assert values["false_regions_per_image"] == pytest.approx(2 / 5)


def test_every_metric_has_an_interval_around_its_value() -> None:
    result = run(EXAMPLE)
    assert list(result.metrics) == list(METRICS)
    for interval in result.metrics.values():
        assert interval.low <= interval.value <= interval.high


def test_a_patient_adds_up_the_sums_of_its_images() -> None:
    patients, sums = patient_sums(EXAMPLE, 0.5, 0.0)
    assert patients == ("a", "b", "c")
    dice = list(METRICS).index("with_mass.dice")
    assert sums[0, dice].tolist() == pytest.approx([1.4, 2.0])
    assert sums[2, dice].tolist() == [0.0, 0.0]  # no image with a mass


# --- the draws ---


def test_every_draw_takes_as_many_patients_as_there_are() -> None:
    counts = draw_counts(5, 200, seed=1)
    assert counts.shape == (200, 5)
    assert (counts.sum(axis=1) == 5).all()
    assert counts.max() > 1  # with replacement: some patients come twice


def test_the_same_seed_draws_the_same_patients() -> None:
    assert (draw_counts(5, 50, seed=3) == draw_counts(5, 50, seed=3)).all()
    assert (draw_counts(5, 50, seed=3) != draw_counts(5, 50, seed=4)).any()


def test_a_draw_takes_the_images_of_a_patient_together() -> None:
    _, sums = patient_sums(EXAMPLE, 0.5, 0.0)
    counts = np.array([[3, 0, 0], [0, 3, 0], [1, 1, 1], [2, 1, 0]])
    found = metric_values(sums, counts)[:, list(METRICS).index("masses_found")]
    # Patient a has both its masses found, b its only one missed.
    assert found.tolist() == pytest.approx([1.0, 0.0, 2 / 3, 4 / 5])


def test_a_metric_without_anything_to_count_in_a_draw_is_undefined() -> None:
    _, sums = patient_sums(EXAMPLE, 0.5, 0.0)
    values = metric_values(sums, np.array([[0, 0, 3]]))  # only patient c
    assert math.isnan(values[0, list(METRICS).index("masses_found")])
    assert values[0, list(METRICS).index("false_regions_per_image")] == 1.0


def test_a_metric_that_is_never_defined_has_no_interval() -> None:
    without_small = details(image("1", "a", dice=0.9, masses=(mass(300, True),)))
    small = run(without_small).metrics["small_masses_found"]
    assert math.isnan(small.value) and math.isnan(small.low)


def test_more_patients_give_narrower_intervals() -> None:
    def width(count: int) -> float:
        found = run(many_patients(count)).metrics["masses_found"]
        return found.high - found.low

    assert width(80) < width(20) < width(5)


def test_a_wider_level_gives_a_wider_interval() -> None:
    found = many_patients(40)
    narrow = run(found, level=0.5).metrics["masses_found"]
    wide = run(found, level=0.99).metrics["masses_found"]
    assert wide.low < narrow.low and narrow.high < wide.high


# --- comparing two evaluations ---


def test_an_evaluation_against_itself_differs_by_exactly_nothing() -> None:
    result = versus(EXAMPLE, EXAMPLE)
    assert all(d == Interval(0.0, 0.0, 0.0) for d in result.difference.values())
    assert result.first == result.second


def test_a_model_better_on_every_patient_is_higher_beyond_doubt() -> None:
    result = versus(many_patients(30, 0.3), many_patients(30, 0.7))
    found = result.difference["masses_found"]
    assert found.value == pytest.approx(0.4) and found.low > 0


def test_the_pairing_is_narrower_than_two_separate_intervals() -> None:
    first, second = many_patients(30, 0.3), many_patients(30, 0.5)
    result = versus(first, second)
    paired = result.difference["masses_found"]
    separate = result.first["masses_found"], result.second["masses_found"]
    # Without pairing the two intervals overlap, the paired difference is clear.
    assert separate[0].high > separate[1].low and paired.low > 0


def test_the_two_evaluations_must_have_scored_the_same_images() -> None:
    other = details(*EXAMPLE["images"][:-1])
    with pytest.raises(ValueError, match="same images"):
        versus(EXAMPLE, other)
    moved = json.loads(json.dumps(EXAMPLE))
    moved["images"][0]["patient_id"] = "b"
    with pytest.raises(ValueError, match="same images"):
        versus(EXAMPLE, moved)


def test_the_two_evaluations_must_be_on_the_same_set() -> None:
    with pytest.raises(ValueError, match="test"):
        versus(EXAMPLE, details(*EXAMPLE["images"], on="test"))


# --- errors ---


def test_the_setting_must_be_in_the_details() -> None:
    with pytest.raises(ValueError, match="not in the details"):
        bootstrap(EXAMPLE, 0.4, 10.0)


@pytest.mark.parametrize("options", [{"resamples": 0}, {"level": 1.0}, {"level": 0}])
def test_invalid_options_are_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        run(EXAMPLE, **options)
    with pytest.raises(ValueError):
        versus(EXAMPLE, EXAMPLE, **options)


def test_details_without_images_are_rejected() -> None:
    with pytest.raises(ValueError, match="no images"):
        run(details())


def test_no_patients_cannot_be_drawn() -> None:
    with pytest.raises(ValueError):
        draw_counts(0, 10)


# --- output ---


def test_the_results_become_plain_json() -> None:
    without_small = details(image("1", "a", dice=0.9, masses=(mass(300, True),)))
    content = json.loads(json.dumps(to_dict(run(without_small, resamples=20))))
    assert content["resamples"] == 20 and content["patients"] == 1
    assert content["metrics"]["small_masses_found"]["value"] is None
    assert set(content["metrics"]) == set(METRICS)


def test_the_table_lists_every_metric_with_its_interval() -> None:
    text = format_result(run(EXAMPLE, resamples=100, seed=2))
    assert text.startswith("95% intervals from 100 draws of the 3 patients (5 images)")
    assert all(label in text for label in METRICS.values())
    assert "masses found              0.667 [" in text


def test_the_table_of_a_comparison_says_whether_the_difference_excludes_0() -> None:
    better = versus(many_patients(30, 0.3), many_patients(30, 0.7))
    text = format_result(better, ("1024", "2048"))
    assert "first: 1024\nsecond: 2048" in text
    assert "higher" in next(
        line for line in text.splitlines() if "masses found" in line
    )
    worse = format_result(versus(many_patients(30, 0.7), many_patients(30, 0.3)))
    assert "lower" in next(
        line for line in worse.splitlines() if "masses found" in line
    )
    same = format_result(versus(many_patients(30, 0.5), many_patients(30, 0.5)))
    line = next(line for line in same.splitlines() if "false regions" in line)
    assert line.endswith("+0.000 [+0.000, +0.000] cannot tell")


def test_the_table_shows_an_undefined_value_and_a_missing_interval() -> None:
    nothing = Interval(math.nan, math.nan, math.nan)
    metrics = dict.fromkeys(METRICS, nothing) | {
        "masses_found": Interval(0.5, math.nan, math.nan)
    }
    text = format_result(Intervals(1, 1, 10, 0, 0.95, metrics))
    assert "masses found              0.500\n" in text
    assert "false regions per image   n/a" in text


def test_masses_of_unknown_area_count_only_among_all_masses() -> None:
    found = details(
        image("1", "p1", masses=(mass(None, True), mass(None, False))),
        image("2", "p2", masses=(mass(None, True),)),
    )
    result = run(found)
    values = {name: metric.value for name, metric in result.metrics.items()}
    assert values["masses_found"] == pytest.approx(2 / 3)
    assert math.isnan(values["small_masses_found"])
    assert math.isnan(values["large_masses_found"])
