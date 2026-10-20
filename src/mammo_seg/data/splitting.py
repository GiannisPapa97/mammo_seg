"""Stratified splitting of patients into a hold-out set and cross-validation folds.

The unit of splitting is the patient, never the image: all images of a patient
must end up in the same group, otherwise near-identical views (the same mass in
CC and MLO, both breasts, a second exam) leak between training and evaluation.
The functions here work on a list of patients and return groups of patient ids.
Mapping images to groups through their ``patient_id`` is done in ``splits.py``.

Both functions stratify on ``has_mass``: each group keeps about the same share of
patients with a mass as the whole set, and group sizes are as equal as possible.
For a valid input they guarantee that

- every patient id is in exactly one returned group
- no group is empty
- the groups and the ids inside each group are sorted
- the same patients, arguments and seed give the same result, whatever the order
  in which the patients are passed

and they raise ``ValueError`` for an invalid input: no patients, duplicate patient
ids, a fraction outside (0, 1), fewer than 2 folds, or too few patients of one
class to fill the groups.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class Patient:
    """What the splitting needs to know about a patient."""

    patient_id: str
    has_mass: bool  # at least one image of the patient has a mass


def hold_out(
    patients: Sequence[Patient], fraction: float, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split into (train, test) patient ids; test is about ``fraction`` of them."""
    _check_patients(patients)
    if not 0 < fraction < 1:
        raise ValueError(f"the test fraction must be between 0 and 1, got {fraction}")
    ids, labels = _arrays(patients)
    try:
        train, test = train_test_split(
            ids, test_size=fraction, stratify=labels, random_state=seed
        )
    except ValueError as error:
        raise ValueError(
            f"cannot hold out {fraction:.0%} of {len(ids)} patients "
            f"stratified on masses: {error}"
        ) from error
    return _sorted(train), _sorted(test)


def stratified_folds(
    patients: Sequence[Patient], n_folds: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    """Partition the patients into ``n_folds`` groups of patient ids."""
    _check_patients(patients)
    if n_folds < 2:
        raise ValueError(f"at least 2 folds are needed, got {n_folds}")
    ids, labels = _arrays(patients)
    smallest = min(Counter(labels.tolist()).values())
    if smallest < n_folds:
        raise ValueError(
            f"cannot make {n_folds} folds: one class has only {smallest} "
            f"patient(s) and every fold needs at least one of each class"
        )
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return tuple(
        _sorted(ids[test_index]) for _, test_index in splitter.split(ids, labels)
    )


def _check_patients(patients: Sequence[Patient]) -> None:
    if not patients:
        raise ValueError("there are no patients to split")
    counts = Counter(patient.patient_id for patient in patients)
    duplicated = sorted(patient_id for patient_id, n in counts.items() if n > 1)
    if duplicated:
        raise ValueError(f"duplicate patient ids: {duplicated}")


def _arrays(patients: Sequence[Patient]) -> tuple[np.ndarray, np.ndarray]:
    # Sorted by id, so the result does not depend on the order of the input.
    ordered = sorted(patients, key=lambda patient: patient.patient_id)
    ids = np.array([patient.patient_id for patient in ordered])
    labels = np.array([patient.has_mass for patient in ordered])
    return ids, labels


def _sorted(ids: np.ndarray) -> tuple[str, ...]:
    return tuple(sorted(str(patient_id) for patient_id in ids))
