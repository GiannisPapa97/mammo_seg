"""Patient-level splits of the manifest: an optional test set and K folds.

All images of a patient go to the same set, so near-identical views of the same
lesion never leak between training and evaluation. Patients are stratified on
whether they have at least one image with a mass (see ``splitting.py``).

Images with an excluded view are left out of the splits (see ``EXCLUDED_VIEWS``).
All other images are kept, including every image without a mass: the normal ones
and those with only calcifications or other lesions.
"""

from collections.abc import Sequence

import pandas as pd
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from mammo_seg.data.schema import ImageRecord, Manifest, View
from mammo_seg.data.splitting import Patient, hold_out, stratified_folds

DEFAULT_SEED = 0

# The one FB image (a reversed CC) has no mass annotation although its breast has a
# mass in the other views, so it could be a false negative (see docs/dataset.md).
EXCLUDED_VIEWS = frozenset({View.FB})

TEST = "test"
EXCLUDED = "excluded"


def fold_name(index: int) -> str:
    return f"fold_{index}"


@dataclass(frozen=True, config=ConfigDict(strict=True, extra="forbid"))
class Splits:
    """The patients of the test set and of each fold, and the excluded images."""

    test_patients: tuple[str, ...]  # empty when there is no test set
    folds: tuple[tuple[str, ...], ...]
    excluded_images: tuple[str, ...]  # file ids of the images left out

    def split_of_patient(self) -> dict[str, str]:
        """The set of every patient: ``test`` or ``fold_<i>``."""
        sets = {patient_id: TEST for patient_id in self.test_patients}
        for index, fold in enumerate(self.folds):
            sets.update({patient_id: fold_name(index) for patient_id in fold})
        return sets


def select_images(manifest: Manifest) -> tuple[ImageRecord, ...]:
    """The images that take part in the splits: all but the excluded views."""
    return tuple(image for image in manifest.images if image.view not in EXCLUDED_VIEWS)


def patients_of(images: Sequence[ImageRecord]) -> list[Patient]:
    """One patient per ``patient_id``; has a mass if any of the images has one."""
    has_mass: dict[str, bool] = {}
    for image in images:
        has_mass[image.patient_id] = has_mass.get(image.patient_id, False) or (
            image.has_mass
        )
    return [Patient(patient_id, flag) for patient_id, flag in sorted(has_mass.items())]


def make_splits(
    manifest: Manifest,
    n_folds: int,
    test_fraction: float | None = None,
    seed: int = DEFAULT_SEED,
) -> Splits:
    """Split the patients of the manifest into K folds, with an optional test set.

    With ``test_fraction`` set, that share of the patients is held out first, then
    the rest is split into folds; without it, all patients go into the folds.
    Raises ValueError for invalid arguments or classes too small to stratify.
    """
    images = select_images(manifest)
    excluded = tuple(
        sorted(
            image.file_id for image in manifest.images if image.view in EXCLUDED_VIEWS
        )
    )
    patients = patients_of(images)
    test: tuple[str, ...] = ()
    if test_fraction is not None:
        train, test = hold_out(patients, test_fraction, seed)
        kept = set(train)
        patients = [patient for patient in patients if patient.patient_id in kept]
    folds = stratified_folds(patients, n_folds, seed)
    return Splits(test_patients=test, folds=folds, excluded_images=excluded)


def splits_frame(manifest: Manifest, splits: Splits) -> pd.DataFrame:
    """One row per image with its set: ``fold_<i>``, ``test`` or ``excluded``.

    Raises ValueError if the manifest has an image whose patient is not in the
    splits, which means the splits were made from another manifest.
    """
    set_of_patient = splits.split_of_patient()
    excluded = set(splits.excluded_images)
    missing = sorted(
        {
            image.patient_id
            for image in manifest.images
            if image.file_id not in excluded and image.patient_id not in set_of_patient
        }
    )
    if missing:
        raise ValueError(f"patients missing from the splits: {missing}")
    return pd.DataFrame(
        [
            {
                "file_id": image.file_id,
                "patient_id": image.patient_id,
                "has_mass": image.has_mass,
                "split": EXCLUDED
                if image.file_id in excluded
                else set_of_patient[image.patient_id],
            }
            for image in manifest.images
        ],
        columns=pd.Index(["file_id", "patient_id", "has_mass", "split"]),
    )


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Patients, images and images with a mass in every set."""
    summary = frame.groupby("split").agg(
        patients=("patient_id", "nunique"),
        images=("file_id", "size"),
        images_with_mass=("has_mass", "sum"),
    )
    return summary.sort_index()
