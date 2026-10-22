"""A small synthetic project on disk, at the paths ``mammo-seg train`` takes."""

from dataclasses import dataclass
from pathlib import Path

from tests.data.manifests import make_manifest
from tests.data.preprocessed import SETTINGS, write_preprocessed_for

from mammo_seg.data.manifest import write_manifest
from mammo_seg.data.splits import create_splits_file, write_splits


@dataclass
class Project:
    """Synthetic files at the paths that ``mammo-seg train`` takes."""

    manifest: Path
    splits: Path
    preprocessed: Path
    output: Path

    def train_args(self, *extra: str) -> list[str]:
        return [
            "train",
            "--manifest",
            str(self.manifest),
            "--splits",
            str(self.splits),
            "--preprocessed-dir",
            str(self.preprocessed),
            "--output-dir",
            str(self.output),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--samples-per-epoch",
            "4",
            *extra,
        ]


def make_project(root: Path) -> Project:
    """Eight patients (four with a mass), two images each, split into two folds."""
    manifest = make_manifest(4, 4, per_patient=2)
    files = Project(
        manifest=root / "manifest.json",
        splits=root / "splits.json",
        preprocessed=root / "preprocessed",
        output=root / "runs",
    )
    write_manifest(manifest, files.manifest)
    write_splits(create_splits_file(manifest, 2), files.splits)
    write_preprocessed_for(files.preprocessed, SETTINGS, manifest)
    return files
