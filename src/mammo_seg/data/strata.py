"""Groups of patients that the splits keep in the same proportions in every set.

Stratifying only on whether a patient has a mass leaves everything else to chance,
and with about 8 patients with a mass per fold, chance gives folds with very
different cases. Two more properties matter for mass segmentation:

- the size of the masses: small masses are much harder to outline, and a few
  missed pixels ruin their Dice
- the density of the breast: dense tissue looks like a mass and causes false alarms

Every patient therefore gets a group from up to three factors, most important
first:

1. with or without a mass (``mass``, ``no_mass``)
2. for a patient with a mass: whether any of the masses is smaller than
   ``small_mass_mm2`` (``small``, ``large``). 100 mm² is about 11 mm across and
   about the smallest quarter of the INbreast masses; a lower threshold leaves too
   few dense patients with a small mass to spread over the sets
3. whether any image of the patient has an ACR density of ``dense_acr`` or more
   (``dense``, ``fatty``); a missing density counts as neither

Areas are in mm², from the polygons at the native pixel size, never in
preprocessed pixels, whose size varies from image to image. The INbreast DICOM
headers have no pixel spacing; the publication gives 0.07 mm.

A group needs enough patients to put one in every set. ``merge_small_groups``
merges a group that is too small with its neighbours by dropping its last factor:
for example, dense and fatty patients with a small mass become one group
``mass/small``.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass as std_dataclass
from typing import Annotated, Final

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from mammo_seg.data.schema import ImageRecord, LesionType, Roi

INBREAST_PIXEL_SIZE_MM: Final = 0.07  # from the INbreast publication
DEFAULT_SMALL_MASS_MM2: Final = 100.0
DEFAULT_DENSE_ACR: Final = 3
SEPARATOR: Final = "/"  # between the factors of a group name

Group = tuple[str, ...]  # the factors of a group, most important first


@dataclass(frozen=True, config=ConfigDict(strict=True, extra="forbid"))
class Stratification:
    """How patients are grouped; stored with the splits."""

    pixel_size_mm: Annotated[float, Field(gt=0)] = INBREAST_PIXEL_SIZE_MM
    small_mass_mm2: Annotated[float, Field(gt=0)] = DEFAULT_SMALL_MASS_MM2
    dense_acr: Annotated[int, Field(ge=1, le=4)] = DEFAULT_DENSE_ACR


def area_mm2(roi: Roi, pixel_size_mm: float = INBREAST_PIXEL_SIZE_MM) -> float:
    """The area of an ROI's polygon in mm², at the native pixel size."""
    return roi.area_px * pixel_size_mm**2


@std_dataclass(frozen=True)
class PatientProfile:
    """What the grouping knows about a patient."""

    patient_id: str
    dense: bool  # an image has an ACR density of at least the threshold
    # The area of the smallest mass in any image; None without a mass.
    smallest_mass_mm2: float | None
    small_mass: bool  # a mass is smaller than the threshold; False without a mass

    @property
    def has_mass(self) -> bool:
        return self.smallest_mass_mm2 is not None

    @property
    def group(self) -> Group:
        density = "dense" if self.dense else "fatty"
        if not self.has_mass:
            return ("no_mass", density)
        return ("mass", "small" if self.small_mass else "large", density)


def patient_profiles(
    images: Sequence[ImageRecord],
    stratification: Stratification = Stratification(),  # noqa: B008
) -> list[PatientProfile]:
    """One profile per patient of ``images``, sorted by patient id.

    Only mass ROIs count; the same mass seen in two views counts in both, so a
    mass that is small in one view makes the patient's group ``small``.
    """
    dense: dict[str, bool] = {}
    smallest: dict[str, float | None] = {}
    for image in images:
        patient = image.patient_id
        dense[patient] = dense.get(patient, False) or (
            image.acr is not None and image.acr >= stratification.dense_acr
        )
        smallest.setdefault(patient, None)
        for roi in image.rois:
            if roi.lesion_type is LesionType.MASS:
                area = area_mm2(roi, stratification.pixel_size_mm)
                known = smallest[patient]
                smallest[patient] = area if known is None else min(known, area)
    profiles = []
    for patient in sorted(dense):
        least = smallest[patient]
        profiles.append(
            PatientProfile(
                patient_id=patient,
                dense=dense[patient],
                smallest_mass_mm2=least,
                small_mass=least is not None and least < stratification.small_mass_mm2,
            )
        )
    return profiles


def group_name(group: Group) -> str:
    """A group as text, for example ``mass/small/dense``."""
    return SEPARATOR.join(group)


def merge_small_groups(groups: Mapping[str, Group], min_size: int) -> dict[str, str]:
    """The group name of every patient once groups of fewer than ``min_size``
    patients are merged.

    The finest group that is too small loses its last factor, together with every
    group that shares its other factors, until every group is large enough. A group
    of a single factor is never merged, so a split that cannot work still fails
    where the splitting checks its groups. Raises ValueError for a ``min_size``
    below 1.
    """
    if min_size < 1:
        raise ValueError(f"the minimum group size must be at least 1, got {min_size}")
    current = dict(groups)
    while True:
        counts = Counter(current.values())
        small = [
            group for group, n in counts.items() if n < min_size and len(group) > 1
        ]
        if not small:
            return {patient: group_name(group) for patient, group in current.items()}
        finest = max(len(group) for group in small)
        merged = {group[:-1] for group in small if len(group) == finest}
        current = {
            patient: group[: finest - 1] if group[: finest - 1] in merged else group
            for patient, group in current.items()
        }
