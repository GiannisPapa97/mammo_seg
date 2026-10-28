"""Which images to train, validate and test on, from the patient-level splits.

Cross-validation trains on all folds but one and validates on the remaining fold;
the optional test set is only used at the end. The functions here turn a splits
file (``splits.py``) into the file ids of those sets, after checking that the
splits were made from the given manifest. Images excluded from the splits (the
``FB`` view) are never returned. All of a patient's images are in the same set,
because the splits assign patients, not images.
"""

from dataclasses import dataclass

from mammo_seg.data.schema import Manifest
from mammo_seg.data.splits import (
    EXCLUDED,
    TEST,
    SplitsFile,
    check_manifest,
    fold_name,
    splits_frame,
)


@dataclass(frozen=True)
class FoldIds:
    """The file ids of one cross-validation round."""

    train: tuple[str, ...]
    validation: tuple[str, ...]


def _ids_by_set(manifest: Manifest, splits_file: SplitsFile) -> dict[str, list[str]]:
    """The file ids of every set (``fold_<i>``, ``test``), in manifest order."""
    splits = check_manifest(splits_file, manifest)
    frame = splits_frame(manifest, splits)
    ids: dict[str, list[str]] = {}
    for file_id, split in zip(frame.file_id, frame.split, strict=True):
        if split != EXCLUDED:
            ids.setdefault(split, []).append(file_id)
    return ids


def fold_ids(
    manifest: Manifest, splits_file: SplitsFile, validation_fold: int
) -> FoldIds:
    """Train on every fold but ``validation_fold`` and validate on that fold.

    Raises ValueError for a fold number outside 0 .. n_folds - 1 or splits made
    from another manifest.
    """
    if not 0 <= validation_fold < splits_file.n_folds:
        raise ValueError(
            f"the validation fold must be between 0 and {splits_file.n_folds - 1}, "
            f"got {validation_fold}"
        )
    ids = _ids_by_set(manifest, splits_file)
    train = [
        file_id
        for index in range(splits_file.n_folds)
        if index != validation_fold
        for file_id in ids.get(fold_name(index), [])
    ]
    return FoldIds(
        train=tuple(sorted(train)),
        validation=tuple(ids.get(fold_name(validation_fold), [])),
    )


def development_ids(manifest: Manifest, splits_file: SplitsFile) -> tuple[str, ...]:
    """Every image of every fold: what a final model is trained on."""
    ids = _ids_by_set(manifest, splits_file)
    return tuple(
        sorted(
            file_id
            for index in range(splits_file.n_folds)
            for file_id in ids.get(fold_name(index), [])
        )
    )


def held_out_ids(manifest: Manifest, splits_file: SplitsFile) -> tuple[str, ...]:
    """The images of the held-out test set; ValueError if there is none."""
    if not splits_file.splits.test_patients:
        raise ValueError("these splits have no test set")
    return tuple(_ids_by_set(manifest, splits_file).get(TEST, []))
