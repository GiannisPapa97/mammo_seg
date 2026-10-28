"""Where the data is by default: one folder per dataset in ``data/raw/`` and
``data/processed/``.

``data/raw/<dataset>/`` holds the downloaded data and is read-only;
``data/processed/<dataset>/`` holds what is derived from it (the manifest, the
splits, the masks and the preprocessed images). The paths are relative to the
working directory, the root of the repository.
"""

from pathlib import Path
from typing import Final

from mammo_seg.data.schema import Dataset

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

INBREAST_RAW_DIR = RAW_DIR / "inbreast" / "INbreast Release 1.0"
INBREAST_PROCESSED_DIR = PROCESSED_DIR / "inbreast"

CBIS_DDSM_RAW_DIR = RAW_DIR / "cbis-ddsm"
CBIS_DDSM_PROCESSED_DIR = PROCESSED_DIR / "cbis-ddsm"

RAW_DIRS: Final[dict[Dataset, Path]] = {
    "inbreast": INBREAST_RAW_DIR,
    "cbis-ddsm": CBIS_DDSM_RAW_DIR,
}
PROCESSED_DIRS: Final[dict[Dataset, Path]] = {
    "inbreast": INBREAST_PROCESSED_DIR,
    "cbis-ddsm": CBIS_DDSM_PROCESSED_DIR,
}
