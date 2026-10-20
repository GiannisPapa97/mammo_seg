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

## 3. Masks

**Goal:** a binary segmentation mask for every image, drawn from the mass polygons of the manifest, because the raw data has no mask images.

```bash
uv run python -m mammo_seg.data.masks
```

It reads `data/processed/manifest.json` and writes `data/processed/masks/<file_id>.png` for every image (`--manifest` and `--output-dir` change the paths). It takes about two minutes, mostly for writing the PNG files. Code: `src/mammo_seg/data/masks.py`.

### Format

- One mask per image, at the **native image size** (for example 4084 x 3328), as an 8-bit PNG with the values 0 (background) and 255 (mass). `read_mask(path)` returns a 0/1 array.
- Images without a mass get an empty mask, so every manifest image has a file. This includes the excluded `FB` image; masks do not depend on the splits.
- Native size means preprocessing can downscale image and mask the same way and the target size can change without rebuilding the masks. All 410 masks take about 6 MB, because they are mostly zeros.

### How a mask is drawn

- All `mass` polygons of an image are merged into one mask; overlapping masses become one region. The lesion type is an argument of `rasterize`, so another type can be drawn later. ROIs with fewer than 3 points (single-point calcifications) have no area and leave the mask empty.
- Polygon coordinates are in continuous image space, `(x, y)` = `(column, row)`, with the origin at the top-left corner, so the centre of pixel `(row, column)` is at `(column + 0.5, row + 0.5)`. A pixel is set when its centre is inside the polygon (even-odd rule). This is done with a small scanline routine in NumPy and is tested against an independent brute-force point-in-polygon check.
- Polygons that leave the image are clipped to it. This affects 3 of the 116 masses, which stick out by a few pixels.
- The polygons are drawn as annotated. For 12 of the 116 masses, spiky vertices leave isolated specks of 1 to 47 pixels next to the main region (which has 8,000 to 130,000 pixels); they are kept in the mask.

### Validation

`write_masks` checks every mask against its polygons, raises `MaskError` listing all problems, and does not write the masks that have one. It checks that:

- the mask has the image shape and only the values 0 and 1
- every polygon has at least 3 points and covers at least one pixel of the image
- for polygons fully inside the image, the number of pixels matches the polygon area (shoelace formula) within 0.5% of the area plus 0.1 pixel per pixel of perimeter. On the real masses the difference is at most 0.4% of the area and 0.05 pixel per pixel of perimeter; a polygon that crosses itself differs far more
- the mask has at least as many pixels as the largest polygon and at most the sum of all polygons, and no pixels if there are no polygons

### On the real data

410 masks: 107 non-empty (116 masses, 3 clipped to the image) and 303 empty, matching `has_mass` in the manifest. Overlays on a sample of images (including the clipped ones and an image with three masses) line up with the lesions.

### Decisions

- Masses only, as agreed; calcifications and the other lesion types are not drawn.
- Masks are stored, not generated on the fly, so they are easy to inspect and the dataset class only has to read them.
- Like the manifest and the splits, the masks are not committed.

## 4. Preprocessing

**Goal:** turn a raw 16-bit DICOM into the input of the model, in one chain that training and inference both use: read the DICOM, crop to the breast, rescale so the longer side is 1024 px, flip right breasts, normalize the intensities. Code: `src/mammo_seg/data/preprocessing.py` (the chain) and `src/mammo_seg/data/preprocess_dataset.py` (the cache on disk and the command).

```bash
uv run python -m mammo_seg.data.preprocess_dataset            # about 3.5 minutes
uv run python -m mammo_seg.data.preprocess_dataset --size 768  # another size
```

### The chain

| Step | Function | What it does |
|---|---|---|
| Read | `read_pixels(path)` | The DICOM as a `uint16` array. Only what the dataset contains is accepted (one channel, unsigned up to 16 bits, `MONOCHROME2`, no rescale slope or intercept); anything else raises an error instead of being guessed |
| Crop | `breast_bbox(pixels)` | The box around the breast, found on the raw image |
| Rescale | `preprocess(...)` | The crop is resized so its longer side is `size` (1024), by averaging the source pixels |
| Flip | `preprocess(...)` | Right breasts are flipped horizontally, so the chest wall is on the left in every image |
| Normalize | `normalize_intensity(pixels)` | Scales to 0..1. Not stored: it runs when an image is loaded and at inference |

`preprocess(pixels, laterality, settings)` runs crop, rescale and flip and returns the image together with a `PreprocessParams` record (the crop box, the size of the result and whether it was flipped). `preprocess_mask(mask, params)` applies the same geometry to a mask, and `restore_mask(mask, params)` maps a predicted mask back onto the original image. At inference a DICOM goes through `read_pixels`, `preprocess` and `normalize_intensity`, exactly like the stored training images.

### The breast box

The breast is the largest connected region of pixels brighter than 100 (the background is close to 0, tissue is far above it). The mask is made on every 8th pixel, opened once to cut off thin bright strips along the image border that touch the breast, and the box around the largest region gets a margin of 2% of its size on each side. Without the opening, such a strip made the box run to the image edge in some images. The box is accurate to about 8 px, far less than the margin.

On the real data the box covers a median of 46% of the image (16% to 76%). All 410 boxes were checked on contact sheets (`notebooks/03_preprocessing_review.ipynb`): each encloses the whole breast, tightly, and keeps the pectoral muscle in MLO views. All 107 masses are inside the box; without the margin one mass at the image's left edge would stick out by 2 px, which is what the margin is for.

### Settings

`PreprocessingSettings` holds everything that decides what the preprocessing does. It is stored with the data in `preprocessing.json`, and a trained model must record the same settings; `check_settings(record, expected)` refuses data preprocessed with other settings.

| Setting | Default | Meaning |
|---|---|---|
| `threshold` | 100 | Pixels above this value count as breast |
| `margin` | 0.02 | Margin added around the box, as a fraction of its size |
| `downscale` | 8 | The breast mask is made on every 8th pixel |
| `opening` | 1 | Number of morphological openings of that mask |
| `size` | 1024 | Length of the longer side after rescaling |
| `max_intensity` | 4095 | Divisor of the normalization (the 12-bit range) |
| `percentile` | none | If set, per-image normalization by this percentile of the tissue instead of `max_intensity` |

### Intensity normalization

The default is a fixed divisor of 4095 (clipped at 1): the images come from one scanner with a 12-bit range, and this keeps absolute intensities. I first planned per-image scaling by the 99.5th percentile of the tissue, but on 80 random images it made the tissue brightness vary more across images (coefficient of variation 6.2%, against 3.6% for the fixed divisor), because the brightest pixels are outliers such as calcifications and bright masses, so the scale would depend on the lesions that are to be segmented. Per-image scaling stays available through `percentile` for data from other scanners.

### Stored format

Written to `data/processed/preprocessed/`:

- `images/<file_id>.png`: the preprocessed image, 16-bit (values unchanged apart from the averaging of the rescale)
- `masks/<file_id>.png`: its mass mask, 8-bit with the values 0 and 255 (read it with `read_mask`)
- `preprocessing.json`: the settings and, for every image, `PreprocessParams` (source shape, crop box, result shape, flip)

All 410 images are processed, including the excluded `FB` view, which is only left out of the splits. The 410 images and masks take 168 MB.

### Validation

`preprocess_dataset` raises `PreprocessingError` listing every image that fails; those images are not written and neither is `preprocessing.json`. It checks that:

- the DICOM can be read and has the size the manifest says
- the crop box does not cut the mask (a lesion outside the box is an error)
- the mask is binary and has the shape of the image
- a mask does not vanish when rescaled, and its area after rescaling is within 5% of the expected one plus half a pixel along its edge

### On the real data

- 410 images with a longer side of 1024 and a shorter side of 276 to 784. The breast fills a median of 71% of the preprocessed image (54% to 85%).
- Chest wall on the left in all 410 images (201 right breasts flipped).
- 107 non-empty masks. The rescaled mask has 0.995 to 1.012 times the expected area, none vanishes, and a mask mapped back onto the original image overlaps the original mask with an IoU of at least 0.95 (median 0.99).
- All 107 masks are brighter than the ring of tissue around them.

### Decisions

- Offline preprocessing is a cache, not a design decision: training reads the stored files to avoid decoding DICOMs every epoch, and the same functions run on the fly at inference.
- Geometry is stored, intensity scaling is applied at load time, so it can change without regenerating the files.
- Augmentation (random, so never cached) and padding to a common size belong to the PyTorch dataset.
- The preprocessed data is not committed, like the other derived data.

## 5. PyTorch dataset and loaders

**Goal:** everything between the stored preprocessed data and the model: batches of image and mask tensors for training and evaluation, and the padding of a single image for inference. Code in `src/mammo_seg/`: `device.py`, and in `data/`: `dataset.py`, `folds.py`, `augmentation.py`, `loaders.py`.

```python
from pathlib import Path
from mammo_seg.data.augmentation import Augmentation
from mammo_seg.data.dataset import MammogramDataset
from mammo_seg.data.folds import fold_ids
from mammo_seg.data.loaders import evaluation_loader, training_loader
from mammo_seg.data.manifest import read_manifest
from mammo_seg.data.splits import read_splits

manifest = read_manifest(Path("data/processed/manifest.json"))
ids = fold_ids(manifest, read_splits(Path("data/processed/splits.json")), validation_fold=0)
data = Path("data/processed/preprocessed")
train = training_loader(MammogramDataset(data, ids.train, transform=Augmentation()), manifest, batch_size=8)
validation = evaluation_loader(MammogramDataset(data, ids.validation), batch_size=8)
```

### Installing PyTorch and choosing the device

PyTorch is an extra that picks its build (see the README): `uv sync --extra cpu` locally and in CI, `--extra cu126` or `--extra cu130` on a GPU machine, and no extra on Kaggle or Colab, where a GPU build is already installed. `resolve_device()` picks the device at run time: a CUDA GPU if there is one, then Apple MPS, then the CPU, or what `MAMMO_SEG_DEVICE` (`cpu`, `cuda`, `cuda:1`, `mps`) says. Asking for a GPU that is not there is an error, not a silent fall back to the CPU.

### Samples

`MammogramDataset(preprocessed_dir, file_ids, settings, canvas, transform)` refuses data preprocessed with other settings and unknown file ids. A sample is a dict:

| Key | Content |
|---|---|
| `image` | float32 `[1, 1024, 800]`, normalized to 0..1 by `normalize_intensity`, 0 on the padding |
| `mask` | float32 `[1, 1024, 800]`, 1 on a mass |
| `valid` | bool `[1, 1024, 800]`, true on the image, false on the padding |
| `file_id` | the image id |

Images are padded on the right and at the bottom to the 1024 x 800 canvas. Every preprocessed image is 1024 px high and 276 to 784 px wide; 800 is the next multiple of 32, which a U-Net needs, and the chest wall is on the left, so padding only adds background next to the breast. An image wider than the canvas is an error, not a silent crop. Losses and metrics should only count pixels where `valid` is true, or the padding counts as correctly predicted background.

### Splits

`fold_ids(manifest, splits_file, validation_fold)` gives the training ids (all folds but one) and the validation ids (that fold); `development_ids` gives every fold, for a final model; `held_out_ids` gives the test set. They check the manifest fingerprint and never return the excluded `FB` image. All of a patient's images are in the same set.

### Augmentation

`Augmentation(AugmentationSettings())` is the training transform, built on torchvision `transforms.v2` and applied to image and mask together before padding:

| Setting | Default | Change |
|---|---|---|
| `rotation` | 10 | degrees, either way |
| `scale` | 0.1 | zoom by 1 +- 0.1 |
| `shift` | 0.1 | of the width and height |
| `brightness`, `contrast` | 0.1 | factor 1 +- 0.1, image only |
| `gamma` | 0.2 | exponent 1 +- 0.2, image only |
| `noise` | 0.01 | Gaussian, standard deviation, image only |
| `crop_size` | none | a random patch of this size instead of the whole image (not lesion-aware) |

The mask is resampled by nearest neighbour and stays 0/1. There are no flips: every chest wall is on the left after preprocessing, and flipping would show the model breasts it never sees at inference. On real images, the mask area after augmentation is 0.86 to 1.18 times the original, as the 10% scaling implies.

### Loaders and the imbalance

- `training_loader(dataset, manifest, batch_size, positive_fraction=0.5, samples_per_epoch=None, num_workers=0, seed=0)` draws with PyTorch's `WeightedRandomSampler` so that about `positive_fraction` of the images have a mass; 26% of the INbreast images do. It draws with replacement, so an epoch has a fixed length, by default the number of training images. `positive_fraction=None` turns the sampler off and uses every image once per epoch. The last incomplete batch is dropped.
- `evaluation_loader(dataset, batch_size)` uses every image once, in order, with no sampler and no augmentation: validation and test metrics must reflect the real distribution.
- Everything random runs from one seeded `torch.Generator`, which also seeds the workers, so a run can be repeated.

On fold 0 of the real splits, one training epoch draws 248 images, 48% with a mass (27% in the fold), and no batch of 8 is without one.

### Inference

A model trained on the canvas still works on any image size. `pad_to_multiple(image)` pads a single preprocessed image up to sides divisible by 32 (an 837 px wide image becomes 864) and returns its size, and `unpad(prediction, size)` cuts the prediction back. `restore_mask` (section 4) then maps it onto the original DICOM.

### Review

`notebooks/04_dataset_review.ipynb` shows one image with several augmentations, a training batch as the loader draws it, and the share of images with a mass per epoch.

## Next steps

- The model, the loss and the metrics (Dice, IoU, precision, sensitivity on the valid pixels, positive and all images separately), then the training loop with checkpoints.
