from pathlib import Path

from mammo_seg.data import manifest, masks, paths, preprocess_dataset, splits


def test_inbreast_has_its_own_folders() -> None:
    assert paths.INBREAST_RAW_DIR == Path("data/raw/inbreast/INbreast Release 1.0")
    assert paths.INBREAST_PROCESSED_DIR == Path("data/processed/inbreast")


def test_each_step_reads_by_default_what_the_one_before_writes() -> None:
    assert masks.DEFAULT_MANIFEST == manifest.DEFAULT_OUTPUT
    assert splits.DEFAULT_MANIFEST == manifest.DEFAULT_OUTPUT
    assert preprocess_dataset.DEFAULT_MANIFEST == manifest.DEFAULT_OUTPUT
    assert preprocess_dataset.DEFAULT_MASKS_DIR == masks.DEFAULT_OUTPUT_DIR
    assert preprocess_dataset.DEFAULT_DATASET_DIR == manifest.DEFAULT_DATASET_DIR


def test_the_defaults_stay_in_the_folders_of_inbreast() -> None:
    derived = (
        manifest.DEFAULT_OUTPUT,
        splits.DEFAULT_OUTPUT,
        masks.DEFAULT_OUTPUT_DIR,
        preprocess_dataset.DEFAULT_OUTPUT_DIR,
    )
    for path in derived:
        assert path.parent == paths.INBREAST_PROCESSED_DIR
    assert manifest.DEFAULT_DATASET_DIR.is_relative_to(paths.RAW_DIR / "inbreast")
