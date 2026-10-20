# Data pipeline

How the raw INbreast release is turned into training data. The facts about the raw data are in [dataset.md](dataset.md); this document describes the code. It grows as the pipeline steps are added.

Raw data lives in `data/raw/` and is read-only. Everything derived goes to `data/processed/`. Both are git-ignored.

## 1. Manifest

**Goal:** one validated description of every image and its lesions, so later steps read the manifest instead of parsing the raw files again.

```bash
uv run python -m mammo_seg.data.manifest
```

By default it reads `data/raw/INbreast Release 1.0` and writes `data/processed/manifest.json`. Use `--dataset-dir` and `--output` to change them.

Code, all in `src/mammo_seg/data/`:

- `schema.py`: the schema (`LesionType`, `Roi`, `ImageRecord`, `Manifest`) as frozen pydantic dataclasses, and the JSON conversion.
- `annotations.py`: reads the XML annotation files and normalizes lesion names.
- `manifest.py`: builds and validates the manifest, writes and reads the JSON file, and converts it to pandas tables.

### Format

A JSON file with a schema version and one record per image. Each record holds the image metadata and every annotated lesion (ROI) with a type and a polygon. The file has one image per line so it stays readable and diffs well.

```json
{"version": 1, "images": [
  {"file_id": "20586908", "patient_id": "6c613a14b80a8591", "laterality": "R",
   "view": "CC", "dicom_path": "AllDICOMs/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm",
   "rows": 4084, "cols": 3328, "acquisition_date": "200902", "acr": 2, "birads": "2",
   "has_annotation": true,
   "rois": [{"lesion_type": "mass", "points": [[2290.4, 2835.3], [2292.8, 2835.7]]}]}
]}
```

Image fields:

| Field | Source | Meaning |
|---|---|---|
| `file_id` | DICOM filename | Image id: a string of digits |
| `patient_id` | DICOM filename | Anonymized patient id (lowercase hex); the grouping key for splits |
| `laterality` | DICOM filename | `L` or `R` |
| `view` | DICOM filename | One of `CC`, `MLO` or `FB`; the filenames write `ML` for `MLO`. Any other value is an error |
| `dicom_path` | DICOM filename | Path of the image relative to the dataset folder |
| `rows`, `cols` | DICOM header | Image size in pixels, greater than 0 |
| `acquisition_date` | `INbreast.csv` | Year and semester as six digits, e.g. `200902` |
| `acr` | `INbreast.csv` | Breast density 1 to 4; `null` for one image (any other value is an error) |
| `birads` | `INbreast.csv` | BI-RADS category: one of `1`, `2`, `3`, `4a`, `4b`, `4c`, `5`, `6` |
| `has_annotation` | XML files | An annotation file exists for the image (false for the normal images) |
| `rois` | XML files | The lesions of the image; empty when there is no annotation |

ROI fields:

| Field | Meaning |
|---|---|
| `lesion_type` | One of `mass`, `calcification`, `cluster`, `spiculated_region`, `distortion`, `asymmetry`, `unknown` |
| `points` | Polygon outline in pixels, `[x, y]` = `[column, row]`, finite numbers (no NaN or infinity); a single point for most calcifications |

The lesion type is a closed set. Raw names are mapped to it, merging misspellings (`Assymetry`, `Espiculated Region`, ...). Regions with no usable name (`Point 1`, `Unnamed`, empty) become `unknown`. A name that is not in the mapping is an error, so new data cannot slip in silently.

### Reading and computed values

`read_manifest(path)` returns a `Manifest`. Facts that follow from the ROIs are computed, not stored:

- `ImageRecord.n_masses` and `ImageRecord.has_mass` (the positive label for mass segmentation)
- `Roi.area_px`, the polygon area in pixels (not clipped to the image)

For splitting and analysis, `images_frame(manifest)` gives one row per image (including `n_masses` and `has_mass`) and `rois_frame(manifest)` one row per ROI (lesion type, number of points, area), as pandas tables.

### Validation

`build_manifest` raises `ManifestError` listing every problem it finds, and nothing is written. It checks that:

- DICOM filenames follow the expected pattern, the view is one of `CC`, `MLO` or `FB`, and file ids are unique
- every DICOM image has a row in `INbreast.csv` and every row has an image
- every annotation file belongs to an image
- laterality and view in the filename agree with `INbreast.csv`
- `INbreast.csv` has the expected columns, no duplicate file ids and valid ACR values
- every image record satisfies the schema constraints above (BI-RADS, ACR, date, sizes and so on); violations are reported per image, together with the other problems
- every annotation file describes one image, its lesion names are known and its points can be read

The schema classes are validated by [pydantic](https://docs.pydantic.dev) in strict mode, so a wrong type, an out-of-range value (for example a non-positive image size or an ACR outside 1 to 4), a missing or extra field, an unknown lesion type or view, or another schema version is rejected both when an object is built and when a file is read. `read_manifest` raises `pydantic.ValidationError` (a `ValueError`) that names the offending value.

### On the real data

The manifest of the Kaggle copy has 410 images, 108 patients, 7316 ROIs and 107 images with a mass (303 without), matching `dataset.md`. The file is about 2 MB.

### Decisions

- Only `mass` counts as positive; every other lesion type is kept in the manifest but not used for mass segmentation.
- The one `FB` image (a reversed CC; see the Views section of `dataset.md`) is kept in the manifest but excluded from training and evaluation.
- The manifest keeps all ROIs, including calcifications, so a later calcification task needs no new parsing.
- The manifest is not committed. It is derived from a dataset with an unclear license and is regenerated by the command above.

## 2. Patient splits

**Goal:** assign every patient to a fold, or to an optional test set, so that all images of a patient stay together and training never sees a patient that is evaluated.

```bash
uv run python -m mammo_seg.data.splits --folds 5                      # 5 folds
uv run python -m mammo_seg.data.splits --folds 5 --test-fraction 0.2  # hold out 20% first
```

Arguments:

| Argument | Meaning |
|---|---|
| `--folds K` | Number of cross-validation folds (required, at least 2) |
| `--test-fraction F` | Share of patients to hold out as a test set, between 0 and 1. Off by default; without it all patients go into the folds |
| `--seed N` | Random seed (default 0) |
| `--manifest`, `--output` | Input manifest (default `data/processed/manifest.json`) and output file (default `data/processed/splits.json`) |

Code, in `src/mammo_seg/data/`: `splitting.py` (the stratified splitting of patients) and `splits.py` (splits from the manifest, the file format and the command).

### How the splits are made

1. Images with the `FB` view are left out (see the Views section of `dataset.md`). Every other image is kept, including all images without a mass: the normal ones and those with only calcifications or other lesions.
2. One row per patient, flagged as having a mass if any of the patient's images has one.
3. With `--test-fraction`, that share of patients is held out first (stratified on the flag). The rest is split into K folds with scikit-learn's `StratifiedKFold`, stratified on the same flag, so every fold has about the same share of patients with a mass and about the same size.
4. Each image gets the set of its patient: `fold_<i>`, `test` or `excluded`.

The result does not depend on the order of the patients or images, and the same manifest, arguments and seed give the same file. The command prints a summary of patients, images and images with a mass per set.

On the real manifest, 5 folds with a 20% test set give 22 test patients (10 with a mass) and folds of 17 or 18 patients, each with 8 patients with a mass.

### Format

`splits.json` records the splits and how they were made. It is validated by pydantic like the manifest.

```json
{
  "version": 1,
  "manifest_fingerprint": "414d0377...",
  "seed": 0,
  "n_folds": 5,
  "test_fraction": 0.2,
  "splits": {
    "test_patients": ["036aff49b8ac84f0", "..."],
    "folds": [["...", "..."], ["...", "..."]],
    "excluded_images": ["50994300"]
  }
}
```

- `manifest_fingerprint` is a SHA-256 hash of the manifest content. `check_manifest(splits_file, manifest)` returns the splits only if they were made from that manifest, and raises an error otherwise, so a step never uses splits that belong to another manifest.
- The file is rejected if a patient is in two sets, a fold is empty, `n_folds` does not match the number of folds, or a test set exists without a test fraction (or the reverse).
- `read_splits(path)` reads it, `splits_frame(manifest, splits)` gives one row per image with its set, and `split_summary(frame)` counts patients and images per set.

### Decisions

- Splitting is by patient, stratified on "has at least one mass image", which is the only label that matters for the folds.
- The positive images per fold vary a little, because positive patients have between 1 and 4 positive images; the summary shows them.
- `splits.json` is not committed, like the manifest.
- There is no splitter interface for now. If a second strategy is needed, `splitting.py` is where an interface would go.

## Next steps

- Mask rasterization from the mass polygons.
- Preprocessing: orientation, intensity scaling, breast crop and offline downscale.
