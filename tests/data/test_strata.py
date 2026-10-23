import math
from dataclasses import replace
from typing import Any

import pytest
from pydantic import ValidationError
from tests.data.manifests import make_image, pid, square_of

from mammo_seg.data.schema import LesionType, Roi, View
from mammo_seg.data.strata import (
    DEFAULT_DENSE_ACR,
    DEFAULT_SMALL_MASS_MM2,
    INBREAST_PIXEL_SIZE_MM,
    PatientProfile,
    Stratification,
    area_mm2,
    group_name,
    merge_small_groups,
    patient_profiles,
)

# --- areas ---


def test_an_area_is_in_square_millimetres_at_the_native_pixel_size() -> None:
    roi = Roi(LesionType.MASS, ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)))
    assert area_mm2(roi) == pytest.approx(10_000 * 0.07**2)  # 49 mm²
    assert area_mm2(roi, pixel_size_mm=0.14) == pytest.approx(4 * 49)


def test_the_test_squares_have_the_area_asked_for() -> None:
    assert area_mm2(Roi(LesionType.MASS, square_of(100.0))) == pytest.approx(100.0)


def test_the_defaults_are_those_of_the_dataset() -> None:
    assert INBREAST_PIXEL_SIZE_MM == 0.07
    assert DEFAULT_SMALL_MASS_MM2 == 100.0
    assert DEFAULT_DENSE_ACR == 3
    assert Stratification() == Stratification(0.07, 100.0, 3)


@pytest.mark.parametrize(
    "changes",
    [
        {"pixel_size_mm": 0.0},
        {"small_mass_mm2": -1.0},
        {"dense_acr": 0},
        {"dense_acr": 5},
        {"dense_acr": "3"},
    ],
)
def test_invalid_stratification_settings_are_rejected(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        Stratification(**changes)  # type: ignore[arg-type]


# --- profiles ---


def profile_of(*images_args: dict[str, Any]) -> PatientProfile:
    images = [make_image(index, 1, **args) for index, args in enumerate(images_args)]
    (profile,) = patient_profiles(images)
    return profile


def test_a_patient_without_a_mass() -> None:
    profile = profile_of({}, {})
    assert not profile.has_mass and not profile.small_mass
    assert profile.smallest_mass_mm2 is None
    assert profile.group == ("no_mass", "fatty")


@pytest.mark.parametrize(
    ("mm2", "small"), [(99.0, True), (100.5, False), (101.0, False), (20.0, True)]
)
def test_a_mass_is_small_below_the_threshold(mm2: float, small: bool) -> None:
    profile = profile_of({"masses": 1, "mass_mm2": mm2}, {})
    assert profile.has_mass
    assert profile.smallest_mass_mm2 == pytest.approx(mm2)
    assert profile.small_mass is small
    assert profile.group == ("mass", "small" if small else "large", "fatty")


def test_a_mass_exactly_at_the_threshold_is_not_small() -> None:
    image = make_image(1, 1, masses=1, mass_mm2=100.0)
    threshold = area_mm2(image.rois[0])
    (profile,) = patient_profiles([image], Stratification(small_mass_mm2=threshold))
    assert not profile.small_mass


def test_one_small_mass_makes_the_patient_small() -> None:
    # One image with a large mass, another (or the other view) with a small one.
    profile = profile_of(
        {"masses": 1, "mass_mm2": 500.0},
        {"masses": 1, "mass_mm2": 90.0, "view": View.MLO},
    )
    assert profile.small_mass
    assert profile.smallest_mass_mm2 == pytest.approx(90.0)
    assert profile_of({"masses": 2, "mass_mm2": 500.0}).small_mass is False


def test_the_threshold_is_in_square_millimetres_not_pixels() -> None:
    # 2,000 pixels are 9.8 mm² at 0.07 mm, but 49 mm² at 0.157 mm.
    side = math.sqrt(2000)
    roi = Roi(LesionType.MASS, ((0.0, 0.0), (side, 0.0), (side, side), (0.0, side)))
    image = replace(make_image(1, 1), rois=(roi,))
    (fine,) = patient_profiles([image], Stratification(small_mass_mm2=10.0))
    (coarse,) = patient_profiles(
        [image], Stratification(pixel_size_mm=0.157, small_mass_mm2=10.0)
    )
    assert fine.small_mass and not coarse.small_mass


def test_only_mass_rois_count() -> None:
    calcification = Roi(LesionType.CALCIFICATION, square_of(1.0))
    image = replace(make_image(1, 1), rois=(calcification,))
    (profile,) = patient_profiles([image])
    assert not profile.has_mass and profile.group == ("no_mass", "fatty")


@pytest.mark.parametrize(
    ("densities", "dense"),
    [
        ((1, 2), False),
        ((2, 3), True),
        ((4, 1), True),
        ((None, 2), False),
        ((None, 3), True),
    ],
)
def test_a_patient_is_dense_if_any_image_is(
    densities: tuple[int | None, ...], dense: bool
) -> None:
    assert profile_of(*({"acr": acr} for acr in densities)).dense is dense


def test_the_density_threshold_can_change() -> None:
    images = [make_image(1, 1, acr=2)]
    (profile,) = patient_profiles(images, Stratification(dense_acr=2))
    assert profile.dense


def test_profiles_are_one_per_patient_sorted_by_id() -> None:
    images = [make_image(1, 9), make_image(2, 3, masses=1), make_image(3, 9, acr=4)]
    profiles = patient_profiles(images)
    assert [profile.patient_id for profile in profiles] == [pid(3), pid(9)]
    assert [profile.group for profile in profiles] == [
        ("mass", "small", "fatty"),
        ("no_mass", "dense"),
    ]


# --- merging ---


def groups_of(counts: dict[tuple[str, ...], int]) -> dict[str, tuple[str, ...]]:
    """``count`` patients in each group, with made-up patient ids."""
    return {
        f"{group_name(group)}-{index}": group
        for group, count in counts.items()
        for index in range(count)
    }


def sizes(merged: dict[str, str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for name in merged.values():
        found[name] = found.get(name, 0) + 1
    return found


def test_groups_that_are_large_enough_are_kept() -> None:
    groups = groups_of(
        {
            ("no_mass", "fatty"): 39,
            ("mass", "small", "dense"): 6,
            ("mass", "large", "fatty"): 25,
        }
    )
    assert sizes(merge_small_groups(groups, 6)) == {
        "no_mass/fatty": 39,
        "mass/small/dense": 6,
        "mass/large/fatty": 25,
    }


def test_a_small_group_merges_with_the_groups_that_share_its_other_factors() -> None:
    groups = groups_of(
        {
            ("mass", "small", "dense"): 2,
            ("mass", "small", "fatty"): 7,
            ("mass", "large", "dense"): 11,
            ("mass", "large", "fatty"): 30,
        }
    )
    assert sizes(merge_small_groups(groups, 6)) == {
        "mass/small": 9,
        "mass/large/dense": 11,
        "mass/large/fatty": 30,
    }


def test_merging_goes_on_until_every_group_is_large_enough() -> None:
    groups = groups_of(
        {
            ("mass", "small", "dense"): 1,
            ("mass", "small", "fatty"): 2,
            ("mass", "large", "fatty"): 30,
            ("no_mass", "dense"): 2,
            ("no_mass", "fatty"): 40,
        }
    )
    # The small ones first lose their density, then mass/small (3) its size.
    assert sizes(merge_small_groups(groups, 6)) == {"mass": 33, "no_mass": 42}


def test_a_group_of_a_single_factor_is_never_merged() -> None:
    groups = groups_of({("mass", "small", "fatty"): 3, ("no_mass", "fatty"): 40})
    assert sizes(merge_small_groups(groups, 6)) == {"mass": 3, "no_mass/fatty": 40}


def test_merging_keeps_every_patient() -> None:
    groups = groups_of({("mass", "small", "dense"): 1, ("no_mass", "fatty"): 9})
    assert merge_small_groups(groups, 2).keys() == groups.keys()


def test_a_minimum_size_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        merge_small_groups({}, 0)
