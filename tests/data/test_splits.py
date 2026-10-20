from dataclasses import FrozenInstanceError

import pytest
from tests.data.manifests import make_image, make_manifest, pid

from mammo_seg.data.schema import Manifest, View
from mammo_seg.data.splits import (
    DEFAULT_SEED,
    EXCLUDED,
    TEST,
    Splits,
    fold_name,
    make_splits,
    patients_of,
    select_images,
    split_summary,
    splits_frame,
)
from mammo_seg.data.splitting import Patient


def test_select_images_drops_the_excluded_views() -> None:
    images = (
        make_image(1, 1),
        make_image(2, 1, view=View.FB),
        make_image(3, 2, view=View.MLO),
    )
    selected = select_images(Manifest(version=1, images=images))
    assert [image.file_id for image in selected] == ["00000001", "00000003"]


def test_a_patient_has_a_mass_if_any_image_has_one() -> None:
    images = [
        make_image(1, 1),
        make_image(2, 1, masses=2),
        make_image(3, 1),
        make_image(4, 2),
        make_image(5, 2),
    ]
    assert patients_of(images) == [Patient(pid(1), True), Patient(pid(2), False)]


def test_patients_are_sorted_by_id() -> None:
    images = [make_image(1, 9), make_image(2, 3), make_image(3, 5)]
    assert [patient.patient_id for patient in patients_of(images)] == [
        pid(3),
        pid(5),
        pid(9),
    ]


def test_splits_without_a_test_set_put_every_patient_in_a_fold() -> None:
    manifest = make_manifest(20, 30)
    splits = make_splits(manifest, n_folds=5)
    assert splits.test_patients == ()
    assert len(splits.folds) == 5
    flat = [patient_id for fold in splits.folds for patient_id in fold]
    assert len(flat) == len(set(flat)) == 50
    assert splits.excluded_images == ()


def test_splits_with_a_test_set_hold_out_patients_first() -> None:
    manifest = make_manifest(20, 30)
    splits = make_splits(manifest, n_folds=4, test_fraction=0.2)
    folded = {patient_id for fold in splits.folds for patient_id in fold}
    assert len(splits.test_patients) == 10
    assert folded.isdisjoint(splits.test_patients)
    assert len(folded) == 40
    assert len(splits.folds) == 4


def test_no_patient_is_in_two_sets() -> None:
    manifest = make_manifest(20, 30)
    frame = splits_frame(manifest, make_splits(manifest, 5, test_fraction=0.2))
    assert (frame.groupby("patient_id").split.nunique() == 1).all()
    assert set(frame.split) == {TEST, *(fold_name(i) for i in range(5))}


def test_all_the_images_of_a_patient_follow_the_patient() -> None:
    manifest = make_manifest(20, 30, per_patient=4)
    splits = make_splits(manifest, 5)
    frame = splits_frame(manifest, splits)
    assert len(frame) == 200
    expected = splits.split_of_patient()
    assert all(
        split == expected[patient_id]
        for patient_id, split in zip(frame.patient_id, frame.split, strict=True)
    )


def test_the_excluded_view_stays_out_but_its_patient_is_split() -> None:
    manifest = make_manifest(20, 30)
    fb = make_image(900, 0, view=View.FB, masses=1)
    manifest = Manifest(version=1, images=(*manifest.images, fb))
    splits = make_splits(manifest, 5)
    frame = splits_frame(manifest, splits).set_index("file_id")
    assert splits.excluded_images == ("00000900",)
    assert frame.loc["00000900", "split"] == EXCLUDED
    assert frame.loc["00000000", "split"] == splits.split_of_patient()[pid(0)]


def test_a_mass_only_in_an_excluded_image_does_not_make_the_patient_positive() -> None:
    images = [
        make_image(1, 1, view=View.FB, masses=1),
        make_image(2, 1),
        make_image(3, 2),
    ]
    selected = select_images(Manifest(version=1, images=tuple(images)))
    assert patients_of(selected) == [Patient(pid(1), False), Patient(pid(2), False)]


def test_a_patient_with_only_excluded_images_is_not_split() -> None:
    manifest = make_manifest(10, 10)
    only_fb = make_image(900, 99, view=View.FB)
    manifest = Manifest(version=1, images=(*manifest.images, only_fb))
    splits = make_splits(manifest, 5)
    assert pid(99) not in splits.split_of_patient()
    assert splits_frame(manifest, splits).split.eq(EXCLUDED).sum() == 1


def test_splits_are_reproducible_and_depend_on_the_seed() -> None:
    manifest = make_manifest(20, 30)
    assert make_splits(manifest, 5, seed=3) == make_splits(manifest, 5, seed=3)
    assert make_splits(manifest, 5, seed=3) != make_splits(manifest, 5, seed=4)
    assert make_splits(manifest, 5) == make_splits(manifest, 5, seed=DEFAULT_SEED)


def test_the_order_of_the_images_does_not_matter() -> None:
    manifest = make_manifest(20, 30)
    reversed_manifest = Manifest(version=1, images=manifest.images[::-1])
    assert make_splits(manifest, 5, test_fraction=0.2) == make_splits(
        reversed_manifest, 5, test_fraction=0.2
    )


def test_splits_of_the_real_size_are_balanced() -> None:
    manifest = make_manifest(50, 58)  # the real dataset: 50 positive, 58 negative
    splits = make_splits(manifest, 5, test_fraction=0.2)
    positive = {pid(i) for i in range(50)}
    assert 21 <= len(splits.test_patients) <= 22
    assert abs(len(positive & set(splits.test_patients)) - 10) <= 1
    assert {len(fold) for fold in splits.folds} <= {17, 18}
    assert {len(positive & set(fold)) for fold in splits.folds} <= {8}


def test_invalid_arguments_are_rejected() -> None:
    manifest = make_manifest(20, 30)
    with pytest.raises(ValueError, match="at least 2 folds"):
        make_splits(manifest, 1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        make_splits(manifest, 5, test_fraction=1.5)


def test_too_few_patients_with_a_mass_are_rejected() -> None:
    with pytest.raises(ValueError, match="one class has only 3 patient"):
        make_splits(make_manifest(3, 30), 5)


def test_a_manifest_with_no_patient_images_is_rejected() -> None:
    only_fb = Manifest(version=1, images=(make_image(1, 1, view=View.FB),))
    with pytest.raises(ValueError, match="no patients"):
        make_splits(only_fb, 2)


def test_splits_frame_rejects_a_manifest_the_splits_do_not_match() -> None:
    splits = make_splits(make_manifest(20, 30), 5)
    other = make_manifest(21, 30)
    with pytest.raises(ValueError, match="patients missing from the splits"):
        splits_frame(other, splits)


def test_split_summary_counts_patients_and_images_per_set() -> None:
    manifest = make_manifest(20, 30)
    fb = make_image(900, 0, view=View.FB)
    manifest = Manifest(version=1, images=(*manifest.images, fb))
    summary = split_summary(splits_frame(manifest, make_splits(manifest, 5, 0.2)))
    assert list(summary.index) == [
        EXCLUDED,
        "fold_0",
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        TEST,
    ]
    assert summary.loc[EXCLUDED, "images"] == 1
    assert summary.images.sum() == 201
    assert summary.patients.drop(EXCLUDED).sum() == 50
    assert summary.images_with_mass.sum() == 20


def test_splits_are_immutable_and_strictly_typed() -> None:
    splits = Splits(test_patients=(), folds=(("a",), ("b",)), excluded_images=())
    with pytest.raises(FrozenInstanceError):
        splits.folds = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="folds"):
        Splits(test_patients=(), folds=[["a"]], excluded_images=())  # type: ignore[arg-type]
