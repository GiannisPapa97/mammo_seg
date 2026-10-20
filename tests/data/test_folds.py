import pytest
from tests.data.manifests import make_image, make_manifest

from mammo_seg.data.folds import FoldIds, development_ids, fold_ids, held_out_ids
from mammo_seg.data.schema import Manifest, View
from mammo_seg.data.splits import create_splits_file


def patients(manifest: Manifest, ids: tuple[str, ...]) -> set[str]:
    by_id = {image.file_id: image.patient_id for image in manifest.images}
    return {by_id[file_id] for file_id in ids}


@pytest.fixture
def manifest() -> Manifest:
    """20 patients with a mass and 30 without, 4 images each, plus one FB image."""
    base = make_manifest(20, 30)
    fb = make_image(900, 0, view=View.FB)
    return Manifest(version=1, images=(*base.images, fb))


def test_train_and_validation_cover_all_folds_without_overlap(
    manifest: Manifest,
) -> None:
    splits_file = create_splits_file(manifest, 5)
    for fold in range(5):
        ids = fold_ids(manifest, splits_file, fold)
        assert isinstance(ids, FoldIds)
        assert set(ids.train).isdisjoint(ids.validation)
        assert len(ids.train) + len(ids.validation) == 200
        assert patients(manifest, ids.train).isdisjoint(
            patients(manifest, ids.validation)
        )


def test_every_image_is_validated_exactly_once_across_folds(manifest: Manifest) -> None:
    splits_file = create_splits_file(manifest, 5)
    validated = [
        file_id
        for fold in range(5)
        for file_id in fold_ids(manifest, splits_file, fold).validation
    ]
    assert len(validated) == len(set(validated)) == 200


def test_the_validation_fold_is_the_one_in_the_splits(manifest: Manifest) -> None:
    splits_file = create_splits_file(manifest, 5)
    ids = fold_ids(manifest, splits_file, 2)
    assert patients(manifest, ids.validation) == set(splits_file.splits.folds[2])


def test_the_test_set_is_kept_out_of_training_and_validation(
    manifest: Manifest,
) -> None:
    splits_file = create_splits_file(manifest, 4, test_fraction=0.2)
    test = held_out_ids(manifest, splits_file)
    assert len(test) == 40
    assert patients(manifest, test) == set(splits_file.splits.test_patients)
    for fold in range(4):
        ids = fold_ids(manifest, splits_file, fold)
        assert set(test).isdisjoint(ids.train) and set(test).isdisjoint(ids.validation)
    assert set(test).isdisjoint(development_ids(manifest, splits_file))


def test_development_ids_are_every_fold(manifest: Manifest) -> None:
    splits_file = create_splits_file(manifest, 4, test_fraction=0.2)
    development = development_ids(manifest, splits_file)
    assert len(development) == 160
    ids = fold_ids(manifest, splits_file, 0)
    assert set(development) == set(ids.train) | set(ids.validation)


def test_the_excluded_view_is_never_returned(manifest: Manifest) -> None:
    splits_file = create_splits_file(manifest, 4, test_fraction=0.2)
    every = [
        *development_ids(manifest, splits_file),
        *held_out_ids(manifest, splits_file),
    ]
    assert "00000900" not in every
    assert len(every) == 200


def test_the_ids_are_sorted(manifest: Manifest) -> None:
    splits_file = create_splits_file(manifest, 5)
    ids = fold_ids(manifest, splits_file, 1)
    assert list(ids.train) == sorted(ids.train)
    assert list(ids.validation) == sorted(ids.validation)


@pytest.mark.parametrize("fold", [-1, 5, 9])
def test_a_fold_outside_the_splits_is_an_error(manifest: Manifest, fold: int) -> None:
    splits_file = create_splits_file(manifest, 5)
    with pytest.raises(ValueError, match="between 0 and 4"):
        fold_ids(manifest, splits_file, fold)


def test_asking_for_a_test_set_that_was_not_made_is_an_error(
    manifest: Manifest,
) -> None:
    splits_file = create_splits_file(manifest, 5)
    with pytest.raises(ValueError, match="no test set"):
        held_out_ids(manifest, splits_file)


def test_splits_of_another_manifest_are_refused(manifest: Manifest) -> None:
    splits_file = create_splits_file(manifest, 5)
    other = make_manifest(21, 30)
    with pytest.raises(ValueError, match="different manifest"):
        fold_ids(other, splits_file, 0)
    with pytest.raises(ValueError, match="different manifest"):
        development_ids(other, splits_file)
