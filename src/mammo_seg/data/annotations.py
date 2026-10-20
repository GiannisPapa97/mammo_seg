"""Reading INbreast lesion annotations.

The ``.xml`` files are Apple property lists exported from OsiriX. Each ROI has a
name and a polygon in pixel coordinates, ``(x, y)`` = ``(column, row)``.
"""

import plistlib
import re
from pathlib import Path

from mammo_seg.data.schema import LesionType, Point, Roi

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

# Every ROI name found in the INbreast files, including misspellings, compared
# in lower case. Names that do not say what the region is map to UNKNOWN.
_LESION_TYPES = {
    "mass": LesionType.MASS,
    "calcification": LesionType.CALCIFICATION,
    "calcifications": LesionType.CALCIFICATION,
    "cluster": LesionType.CLUSTER,
    "spiculated region": LesionType.SPICULATED_REGION,
    "espiculated region": LesionType.SPICULATED_REGION,
    "distortion": LesionType.DISTORTION,
    "asymmetry": LesionType.ASYMMETRY,
    "assymetry": LesionType.ASYMMETRY,
    "point 1": LesionType.UNKNOWN,
    "point 3": LesionType.UNKNOWN,
    "unnamed": LesionType.UNKNOWN,
    "": LesionType.UNKNOWN,
}


def parse_lesion_type(name: str) -> LesionType:
    """Map an ROI name to a lesion type; raises ValueError for an unseen name."""
    try:
        return _LESION_TYPES[name.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown lesion name {name!r}") from None


def _parse_point(text: str) -> Point:
    numbers = _NUMBER.findall(text)
    if len(numbers) != 2:
        raise ValueError(f"expected a point like '(x, y)', got {text!r}")
    return float(numbers[0]), float(numbers[1])


def read_rois(path: Path) -> list[Roi]:
    """Read all ROIs of one annotation file.

    Raises ValueError if the file does not describe exactly one image, a lesion
    name is not known, or a point cannot be parsed.
    """
    data = plistlib.loads(path.read_bytes())
    images = data["Images"]
    if len(images) != 1:
        raise ValueError(f"{path.name}: expected 1 image, found {len(images)}")
    rois = []
    for roi in images[0]["ROIs"]:
        try:
            lesion_type = parse_lesion_type(str(roi["Name"]))
            points = tuple(_parse_point(text) for text in roi["Point_px"])
        except ValueError as error:
            raise ValueError(f"{path.name}: {error}") from error
        rois.append(Roi(lesion_type=lesion_type, points=points))
    return rois
