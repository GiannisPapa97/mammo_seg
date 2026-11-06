# MammoSeg

[![CI](https://github.com/GiannisPapa97/mammo_seg/actions/workflows/ci.yaml/badge.svg)](https://github.com/GiannisPapa97/mammo_seg/actions/workflows/ci.yaml)

A reproducible PyTorch research project for segmenting masses in mammograms, on the INbreast dataset, with CBIS-DDSM for pretraining. It trains U-Net and other encoder-decoder architectures on ImageNet-pretrained encoders; the results so far are all from U-Nets with ResNet encoders.

> **Research use only.** This project is not a medical device and its output must not be used for clinical decisions.

**Status:** active development. Results below are from November 2026. The held-out test set has not been used.

## Results

Reference model: a U-Net with a ResNet18 encoder, trained on whole images at a longer side of 1024 pixels and scored on the images it did not train on. These are 5-fold cross-validation results: 326 images of 86 patients, 87 images with a mass (95 masses, 24 of them under 100 mm²) and 239 without. Each number uses the last epoch of each fold, a probability threshold of 0.4 and a minimum region of 10 mm². That setting was chosen on these same folds, so the numbers are slightly optimistic, and they are for full images, so they are not comparable with results on crops around the lesions.

**Finding the masses.** A mass counts as found when one predicted region overlaps it with an IoU of at least 0.1. This criterion asks whether the model points at the mass, and leaves how well it outlines the mass to the Dice and IoU below. It is lenient, so the stricter counts are next to it:

| IoU needed | all masses (of 95) | under 100 mm² (of 24) | 100 mm² or more (of 71) |
|---|---|---|---|
| ≥ 0.1 (found) | 68 (72%) | 6 (25%) | 62 (87%) |
| ≥ 0.3 | 66 (69%) | 6 (25%) | 60 (85%) |
| ≥ 0.5 | 62 (65%) | 4 (17%) | 58 (82%) |
| ≥ 0.75 | 42 (44%) | 2 (8%) | 40 (56%) |

**Outlining them, and false positives** (95% intervals from resampling the patients):

| | |
|---|---|
| Dice, images with a mass | 0.645 [0.560, 0.730] |
| IoU, images with a mass | 0.567 [0.485, 0.648] |
| sensitivity, images with a mass | 0.619 [0.533, 0.703] |
| precision, images with a mass | 0.853 [0.787, 0.901] |
| false alarms, share of images without a mass with any prediction | 0.084 [0.053, 0.120] |
| false regions per image, all images | 0.126 [0.085, 0.169] |
| Dice, all images (predicting nothing scores 0.733) | 0.844 [0.801, 0.882] |

The model finds most large masses and few small ones, which is its main weakness. The fold 0 model finds all 20 small masses of its own training images, so the limit looks like generalizing from 107 images with a mass, not the resolution.

**What was tried.** Each row is a 5-fold run at the same setting, changing one thing (the augmentation row changes several together):

| run | found (IoU ≥ 0.1) | IoU ≥ 0.5 | under 100 mm² found | false regions |
|---|---|---|---|---|
| **Reference:** ResNet18 | 68 | 62 | 6 | 41 |
| ResNet34 encoder | 68 | 65 | 8 | 40 |
| ResNet34, 2048 pixels (96 masses, 25 small, at this size) | 60 | 54 | 6 | 38 |
| ResNet34, sampling toward small masses | 64 | 59 | 8 | 26 |
| ResNet34, stronger augmentation and weight decay | 70 | 64 | 9 | 63 |
| pretrained on CBIS-DDSM | 62 | 59 | 5 | 19 |
| pretrained on CBIS-DDSM, encoder frozen for 10 epochs | 68 | 61 | 7 | 23 |
| CLAHE as a second input channel | 65 | 59 | 7 | 25 |
| Focal Tversky loss | 67 | 56 | 6 | 48 |

None of these finds more masses beyond chance: with 95 masses, a difference needs about 8 or 9 masses to clear it. Several change the false regions instead. Training on 512-pixel patches did no better than whole images. Across the reference and the last four rows, 16 masses, 14 of them small, are found by no run, with a peak probability under 0.05 in every one. Each run's settings, commit and results are in [results/](results/README.md); the counts at stricter IoUs here are computed from the per-mass IoU in its saved details.

## Methods

- **Splits by patient:** 22 of the 108 patients (83 images) are held out as a test set, and the other 86 form 5 folds, stratified on masses, mass size and breast density, so no patient is in two splits.
- **Comparisons between runs:** paired bootstrap over the patients (`mammo-seg compare`), so a difference counts only if its 95% interval excludes 0. Runs are compared on the masses found and the false regions per image, and images with a mass are reported apart from the others.
- **Logged experiments:** every experiment has a README in [results/](results/README.md) with the commit, the settings and the data hashes it used. The result files stay local; only these READMEs are in the repository.
- **Tests:** CI runs linters, strict type checks and the tests on small synthetic images, without a GPU or the real data. Training checkpoints every epoch and resumes, because it runs on temporary Kaggle and Colab GPUs.

## Quick start

Install PyTorch for your machine, get INbreast and build the derived data (see [Development](#development) and [Data](#data) for the details):

```bash
uv sync --extra cpu                 # or cu126, cu130 on a GPU machine
scripts/download_data.sh            # INbreast, into data/raw/inbreast/
uv run mammo-seg manifest
uv run mammo-seg splits --folds 5 --test-fraction 0.2
uv run mammo-seg masks
uv run mammo-seg preprocess
```

Train the reference model on the five folds. This needs a GPU: about 20 minutes per fold on a Kaggle T4.

```bash
for fold in 0 1 2 3 4; do
  uv run mammo-seg train --fold $fold --encoder resnet18 --epochs 60 --batch-size 4 \
    --samples-per-epoch 256 --patience 0
done
```

Score them on their validation folds:

```bash
uv run mammo-seg evaluate --runs-dir runs --details
```

## Roadmap

1. Measure how many masses a first stage can propose at very low thresholds, and how many false candidates come with them.
2. If that leaves room, a two-stage design: propose candidate regions, then segment crops around them at a higher resolution.
3. Score the final design once on the held-out test set.

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 or newer is supported; development uses 3.12.

PyTorch is installed through an extra that picks its build. Choose exactly one:

```bash
uv sync --extra cpu      # local development and CI (no GPU)
uv sync --extra cu126    # a GPU machine with an older NVIDIA driver (CUDA 12.6)
uv sync --extra cu130    # a GPU machine with a recent NVIDIA driver (CUDA 13.0)
```

A plain `uv sync` installs no PyTorch and removes it if it was installed.

**Kaggle and Google Colab (tried) and other cloud notebooks (not tried).** They come with a GPU build of PyTorch already installed, which must not be replaced. Install the package with the `cloud` extra, which adds the model library and leaves PyTorch alone:

```bash
pip install ".[cloud]"
```

The device is chosen at run time: `auto` (default) picks a GPU if there is one, and the `MAMMO_SEG_DEVICE` environment variable (`cpu`, `cuda`, `cuda:1`, `mps`) overrides it.

The checks:

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

## Data

Datasets are not part of this repository. The `data/` directory is git-ignored and has one folder per dataset: raw data goes in `data/raw/<dataset>/` and derived data in `data/processed/<dataset>/`, for example `data/raw/inbreast/` and `data/processed/inbreast/`.

To download and unzip a dataset into its folder in `data/raw/`:

```bash
scripts/download_data.sh             # INbreast, the default
scripts/download_data.sh cbis-ddsm   # CBIS-DDSM
scripts/download_data.sh all         # both
```

- `inbreast`: the [Kaggle copy of INbreast](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset) (about 2 GB, about 9 GB unzipped) into `data/raw/inbreast/`. See [docs/dataset.md](docs/dataset.md) for what it contains.
- `cbis-ddsm`: the [Kaggle copy of CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) as JPEG images with CSV descriptions (about 5.3 GB, about 6.3 GB unzipped) into `data/raw/cbis-ddsm/`. See [docs/dataset-cbis-ddsm.md](docs/dataset-cbis-ddsm.md). It is used for pretraining.

The script needs `curl` and `unzip`, checks the size of each zip and the number of images after unzipping, resumes an interrupted download (up to five tries in a row, then by running it again), and does nothing for a dataset that is already there. The zip is kept; delete it to free the space. The license of the INbreast copy is unclear, so do not redistribute the data. CBIS-DDSM is under CC BY 3.0 (TCIA): cite it whenever it is used; see [docs/dataset-cbis-ddsm.md](docs/dataset-cbis-ddsm.md#license-and-use).

Then build the validated manifest (`data/processed/inbreast/manifest.json`: every image with its metadata and lesions):

```bash
uv run python -m mammo_seg.data.manifest
```

For CBIS-DDSM, `--dataset cbis-ddsm` writes `data/processed/cbis-ddsm/manifest.json`: its masses, with their masks, pathology and the official train and test sets. The masks and preprocessing steps below take the same `--dataset cbis-ddsm`, which reads the mask images of the masses and uses the settings of the scanned films (see [docs/data-pipeline.md](docs/data-pipeline.md#cbis-ddsm)).

Then make patient-level splits (`data/processed/inbreast/splits.json`), stratified on masses, mass size and breast density, for example 5 folds with a 20% test set:

```bash
uv run python -m mammo_seg.data.splits --folds 5 --test-fraction 0.2
```

For CBIS-DDSM, `--dataset cbis-ddsm --folds 5 --official-test` keeps the dataset's own test set and makes the folds from its training set.

Then write the mass masks (`data/processed/inbreast/masks/<file_id>.png`, one per image):

```bash
uv run python -m mammo_seg.data.masks
```

Then preprocess the images once (crop to the breast, rescale to a longer side of 1024 px, flip right breasts), which writes `data/processed/inbreast/preprocessed/`:

```bash
uv run python -m mammo_seg.data.preprocess_dataset
```

`--clahe 0.01` also stores each image after CLAHE (contrast enhancement), which training then reads as a second input channel; see [docs/data-pipeline.md](docs/data-pipeline.md#contrast-enhancement-clahe).

The PyTorch dataset and loaders read these files; see [docs/data-pipeline.md](docs/data-pipeline.md) for the formats, the checks and how to load a fold.

The installed `mammo-seg` command runs the same steps (`mammo-seg manifest`, `splits`, `masks`, `preprocess`) and training.

## Training

Train on one cross-validation fold (a GPU if there is one, else the CPU), writing checkpoints and a history to `runs/fold_0/`:

```bash
uv run mammo-seg train --fold 0
uv run mammo-seg train --fold 0 --resume    # continue after a stop
```

To pretrain on CBIS-DDSM, point `--manifest`, `--splits` and `--preprocessed-dir` at `data/processed/cbis-ddsm/` and add `--positive-fraction none` (every CBIS-DDSM image has a mass); then train on INbreast from its weights with `--init-from <checkpoint>`. See [docs/training.md](docs/training.md#starting-from-another-model).

On Kaggle use [notebooks/05_kaggle_training.ipynb](notebooks/05_kaggle_training.ipynb), which rebuilds the data from the attached dataset, trains one or more folds and stops cleanly before the session limit. To score the trained folds (thresholds, minimum region size, masses found and false regions per image, subgroups, the baseline of predicting nothing) use `uv run mammo-seg evaluate --runs-dir runs`, or [notebooks/06_kaggle_evaluation.ipynb](notebooks/06_kaggle_evaluation.ipynb) on Kaggle. On Google Colab use [notebooks/11_colab_training.ipynb](notebooks/11_colab_training.ipynb) and [notebooks/12_colab_evaluation.ipynb](notebooks/12_colab_evaluation.ipynb): they read the processed data from an archive on Google Drive and keep the runs there, so a run continues after the session ends, also one started on Kaggle. To compare training on whole images with training on patches over the five folds, use [notebooks/08_kaggle_whole_vs_patches.ipynb](notebooks/08_kaggle_whole_vs_patches.ipynb). To look at the individual missed masses and false regions, write the per-image details with `--details` and open them locally in [notebooks/09_error_analysis.ipynb](notebooks/09_error_analysis.ipynb); to see them, with the probabilities of the model, use [notebooks/10_error_gallery.ipynb](notebooks/10_error_gallery.ipynb), which needs the checkpoints too. To tell whether one model is really better than another, `uv run mammo-seg compare <first details> <second details> --threshold 0.4 --min-area-mm2 10` gives the confidence intervals of the metrics and of their paired difference, by resampling the patients. See [docs/training.md](docs/training.md) for the settings, the files a run writes, and what has been tried.

## License

Code: [MIT](LICENSE). Datasets have their own licenses and terms and are not covered by it.
