"""Patient-level splits of the manifest: an optional test set and K folds.

The test set is either a stratified share of the patients (``--test-fraction``,
INbreast) or the dataset's own test set (``--official-test``, CBIS-DDSM: the
patients of the images whose ``official_split`` is ``test``); the folds are made
from the other patients.

All images of a patient go to the same set, so near-identical views of the same
lesion never leak between training and evaluation. Patients are stratified on
their stratum (see ``strata.py``): with or without a mass, a small mass or not,
dense breasts or not, so the test set and every fold get a similar mix of cases.
A stratum too small to reach every set is merged with its neighbours first, once
for the test set (it needs a patient for the test set and every fold) and once for
the folds.

Images with an excluded view are left out of the splits (see ``EXCLUDED_VIEWS``).
All other images are kept, including every image without a mass: the normal ones
and those with only calcifications or other lesions.

The splits are stored as JSON together with the seed, the arguments, the
stratification settings, the stratum of every patient and a fingerprint of the
manifest they were made from, so a later step can refuse splits that do not belong
to its manifest.

Run it with ``python -m mammo_seg.data.splits --folds 5 [--test-fraction 0.2]``, or
``--dataset cbis-ddsm --folds 5 --official-test``.
"""

import argparse
import hashlib
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, Self

import pandas as pd
from pydantic import ConfigDict, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

from mammo_seg.data.manifest import read_manifest
from mammo_seg.data.paths import INBREAST_PROCESSED_DIR, PROCESSED_DIRS
from mammo_seg.data.schema import Dataset, ImageRecord, Manifest, View, image_to_json
from mammo_seg.data.splitting import Patient, hold_out, stratified_folds
from mammo_seg.data.strata import (
    Group,
    Stratification,
    group_name,
    merge_small_groups,
    patient_profiles,
)

DEFAULT_MANIFEST = INBREAST_PROCESSED_DIR / "manifest.json"
DEFAULT_OUTPUT = INBREAST_PROCESSED_DIR / "splits.json"
DEFAULT_SEED = 0
# 2: stratified on strata, which are stored; 3: the pixel size comes from the
# manifest (version 2), so it is no longer a stratification setting.
SPLITS_VERSION: Final = 3
_CONFIG = ConfigDict(strict=True, extra="forbid")

# The one FB image (a reversed CC) has no mass annotation although its breast has a
# mass in the other views, so it could be a false negative (see docs/dataset.md).
EXCLUDED_VIEWS = frozenset({View.FB})

TEST = "test"
EXCLUDED = "excluded"


def fold_name(index: int) -> str:
    return f"fold_{index}"


@dataclass(frozen=True, config=_CONFIG)
class Splits:
    """The patients of the test set and of each fold, and the excluded images."""

    test_patients: tuple[str, ...]  # empty when there is no test set
    folds: tuple[tuple[str, ...], ...]
    excluded_images: tuple[str, ...]  # file ids of the images left out

    @model_validator(mode="after")
    def _check_consistent(self) -> Self:
        if len(self.folds) < 2:
            raise ValueError(f"at least 2 folds are needed, got {len(self.folds)}")
        if not all(self.folds):
            raise ValueError("a fold is empty")
        counts = Counter(
            patient_id
            for group in (self.test_patients, *self.folds)
            for patient_id in group
        )
        shared = sorted(patient_id for patient_id, n in counts.items() if n > 1)
        if shared:
            raise ValueError(f"patients in more than one set: {shared}")
        if len(set(self.excluded_images)) != len(self.excluded_images):
            raise ValueError("duplicate excluded images")
        return self

    def split_of_patient(self) -> dict[str, str]:
        """The set of every patient: ``test`` or ``fold_<i>``."""
        sets = {patient_id: TEST for patient_id in self.test_patients}
        for index, fold in enumerate(self.folds):
            sets.update({patient_id: fold_name(index) for patient_id in fold})
        return sets


def select_images(manifest: Manifest) -> tuple[ImageRecord, ...]:
    """The images that take part in the splits: all but the excluded views."""
    return tuple(image for image in manifest.images if image.view not in EXCLUDED_VIEWS)


def patient_groups(
    images: Sequence[ImageRecord],
    stratification: Stratification = Stratification(),  # noqa: B008
) -> dict[str, Group]:
    """The stratum of every patient of ``images``, before any merging."""
    return {
        profile.patient_id: profile.group
        for profile in patient_profiles(images, stratification)
    }


def _patients(groups: dict[str, Group], min_size: int) -> list[Patient]:
    merged = merge_small_groups(groups, min_size)
    return [Patient(patient_id, merged[patient_id]) for patient_id in sorted(merged)]


def official_test_patients(images: Sequence[ImageRecord]) -> tuple[str, ...]:
    """The patients of the dataset's own test set, sorted.

    Raises ValueError for an image without an official split or a patient with
    images in both the official train and test sets, which would leak.
    """
    split_of: dict[str, set[str]] = {}
    for image in images:
        if image.official_split is None:
            raise ValueError(f"image {image.file_id} has no official split")
        split_of.setdefault(image.patient_id, set()).add(image.official_split)
    both = sorted(patient for patient, sets in split_of.items() if len(sets) > 1)
    if both:
        raise ValueError(
            f"patients with images in both official sets: {both}; the official "
            "split would leak"
        )
    test = tuple(sorted(p for p, sets in split_of.items() if sets == {TEST}))
    if not test:
        raise ValueError("the official test set is empty")
    return test


def make_splits(
    manifest: Manifest,
    n_folds: int,
    test_fraction: float | None = None,
    seed: int = DEFAULT_SEED,
    stratification: Stratification = Stratification(),  # noqa: B008
    official_test: bool = False,
) -> Splits:
    """Split the patients of the manifest into K folds, with an optional test set.

    With ``test_fraction`` set, that share of the patients is held out first, then
    the rest is split into folds; with ``official_test``, the dataset's own test
    set is held out (``official_test_patients``); with neither, all patients go
    into the folds. The hold-out and the folds are stratified on the patients'
    strata, merged where a stratum has too few patients for the sets. Raises
    ValueError for invalid arguments, both kinds of test set, or strata too small
    to stratify even after merging.
    """
    if n_folds < 2:
        raise ValueError(f"at least 2 folds are needed, got {n_folds}")
    if official_test and test_fraction is not None:
        raise ValueError("use a test fraction or the official test set, not both")
    images = select_images(manifest)
    excluded = tuple(
        sorted(
            image.file_id for image in manifest.images if image.view in EXCLUDED_VIEWS
        )
    )
    groups = patient_groups(images, stratification)
    test: tuple[str, ...] = ()
    if official_test:
        test = official_test_patients(images)
        held = set(test)
        groups = {p: group for p, group in groups.items() if p not in held}
    elif test_fraction is not None:
        # The test set and every fold need a patient of each stratum.
        train, test = hold_out(_patients(groups, n_folds + 1), test_fraction, seed)
        groups = {patient_id: groups[patient_id] for patient_id in train}
    folds = stratified_folds(_patients(groups, n_folds), n_folds, seed)
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


@dataclass(frozen=True, config=_CONFIG)
class SplitsFile:
    """The content of ``splits.json``: the splits and how they were made."""

    version: Literal[3]
    manifest_fingerprint: str  # see manifest_fingerprint()
    seed: int
    n_folds: int
    test_fraction: float | None
    stratification: Stratification
    # The stratum of every patient of the splits, before merging, such as
    # "mass/small/dense".
    strata: dict[str, str]
    splits: Splits
    # The test set is the dataset's own (CBIS-DDSM); False in files made before.
    official_test: bool = False

    @model_validator(mode="after")
    def _check_consistent(self) -> Self:
        if self.n_folds != len(self.splits.folds):
            raise ValueError(
                f"n_folds is {self.n_folds} but there are "
                f"{len(self.splits.folds)} folds"
            )
        if self.test_fraction is not None and not 0 < self.test_fraction < 1:
            raise ValueError(
                f"the test fraction must be between 0 and 1, got {self.test_fraction}"
            )
        if self.official_test and self.test_fraction is not None:
            raise ValueError("a test fraction and the official test set, not both")
        asked = self.test_fraction is not None or self.official_test
        if asked != bool(self.splits.test_patients):
            raise ValueError(
                "there is a test set exactly when a test fraction or the official "
                "test set is asked for"
            )
        if set(self.strata) != set(self.splits.split_of_patient()):
            raise ValueError("the strata must name exactly the patients of the splits")
        return self


_SPLITS_FILE = TypeAdapter(SplitsFile)


def manifest_fingerprint(manifest: Manifest) -> str:
    """A SHA-256 hash of the manifest content, in manifest order."""
    digest = hashlib.sha256()
    for image in manifest.images:
        digest.update(image_to_json(image).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def create_splits_file(
    manifest: Manifest,
    n_folds: int,
    test_fraction: float | None = None,
    seed: int = DEFAULT_SEED,
    stratification: Stratification = Stratification(),  # noqa: B008
    official_test: bool = False,
) -> SplitsFile:
    """Make the splits of a manifest and record how they were made."""
    groups = patient_groups(select_images(manifest), stratification)
    return SplitsFile(
        version=SPLITS_VERSION,
        manifest_fingerprint=manifest_fingerprint(manifest),
        seed=seed,
        n_folds=n_folds,
        test_fraction=test_fraction,
        stratification=stratification,
        strata={patient_id: group_name(group) for patient_id, group in groups.items()},
        splits=make_splits(
            manifest, n_folds, test_fraction, seed, stratification, official_test
        ),
        official_test=official_test,
    )


def strata_summary(splits_file: SplitsFile) -> pd.DataFrame:
    """Patients of every stratum (columns) in every set (rows), with totals."""
    set_of_patient = splits_file.splits.split_of_patient()
    frame = pd.DataFrame(
        {
            "split": [set_of_patient[patient] for patient in splits_file.strata],
            "stratum": list(splits_file.strata.values()),
        }
    )
    return pd.crosstab(frame.split, frame.stratum, margins=True, margins_name="total")


def check_manifest(splits_file: SplitsFile, manifest: Manifest) -> Splits:
    """The splits, if they were made from this manifest; ValueError otherwise."""
    if splits_file.manifest_fingerprint != manifest_fingerprint(manifest):
        raise ValueError(
            "these splits were made from a different manifest; "
            "make them again from the current one"
        )
    return splits_file.splits


def write_splits(splits_file: SplitsFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_SPLITS_FILE.dump_json(splits_file, indent=2) + b"\n")


def read_splits(path: Path) -> SplitsFile:
    """Read a splits file; raises ``pydantic.ValidationError`` if it is invalid."""
    return _SPLITS_FILE.validate_json(path.read_text())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Make patient-level stratified splits of the manifest."
    )
    parser.add_argument("--folds", type=int, required=True, help="number of folds")
    test_set = parser.add_mutually_exclusive_group()
    test_set.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="share of patients to hold out as a test set (default: no test set)",
    )
    test_set.add_argument(
        "--official-test",
        action="store_true",
        help="hold out the dataset's own test set (CBIS-DDSM)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dataset", choices=list(PROCESSED_DIRS), default="inbreast")
    parser.add_argument(
        "--manifest", type=Path, help="default: data/processed/<dataset>/manifest.json"
    )
    parser.add_argument(
        "--output", type=Path, help="default: data/processed/<dataset>/splits.json"
    )
    args = parser.parse_args(argv)
    dataset: Dataset = args.dataset
    output = args.output or PROCESSED_DIRS[dataset] / "splits.json"

    try:
        manifest = read_manifest(
            args.manifest or PROCESSED_DIRS[dataset] / "manifest.json"
        )
        splits_file = create_splits_file(
            manifest,
            args.folds,
            args.test_fraction,
            args.seed,
            official_test=args.official_test,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    write_splits(splits_file, output)
    summary = split_summary(splits_frame(manifest, splits_file.splits))
    print(f"Wrote {output}")
    print(summary.to_string())
    print("\nPatients per stratum:")
    print(strata_summary(splits_file).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
