# MammoSeg

A reproducible PyTorch framework for segmentation of mammography images, starting with the INbreast dataset.

> **Research use only.** This project is not a medical device and its output must not be used for clinical decisions.

**Status:** early development. The data pipeline is in place: a validated manifest of the INbreast dataset, patient-level splits, mass masks, the preprocessing (and, for the masses of CBIS-DDSM, the manifest, the masks, the preprocessing and the splits), and the PyTorch dataset and loaders. The metrics, the loss, the model, checkpoints, the training loop and the evaluation are in place ([docs/training.md](docs/training.md)). Training on all five folds works with crops around the masses, but the model is not yet a useful detector: it beats predicting nothing on the Dice over all images in two of the five folds (a mean Dice of about 0.56 on the images with a mass). The test set has not been used.

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
- `cbis-ddsm`: the [Kaggle copy of CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) as JPEG images with CSV descriptions (about 5.3 GB, about 6.3 GB unzipped) into `data/raw/cbis-ddsm/`. See [docs/dataset-cbis-ddsm.md](docs/dataset-cbis-ddsm.md). Its manifest, masks, preprocessed images and splits are built so far (below); training reads INbreast.

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

The PyTorch dataset and loaders read these files; see [docs/data-pipeline.md](docs/data-pipeline.md) for the formats, the checks and how to load a fold.

The installed `mammo-seg` command runs the same steps (`mammo-seg manifest`, `splits`, `masks`, `preprocess`) and training.

## Training

Train on one cross-validation fold (a GPU if there is one, else the CPU), writing checkpoints and a history to `runs/fold_0/`:

```bash
uv run mammo-seg train --fold 0
uv run mammo-seg train --fold 0 --resume    # continue after a stop
```

On Kaggle use [notebooks/05_kaggle_training.ipynb](notebooks/05_kaggle_training.ipynb), which rebuilds the data from the attached dataset, trains one or more folds and stops cleanly before the session limit. To score the trained folds (thresholds, minimum region size, masses found and false regions per image, subgroups, the baseline of predicting nothing) use `uv run mammo-seg evaluate --runs-dir runs`, or [notebooks/06_kaggle_evaluation.ipynb](notebooks/06_kaggle_evaluation.ipynb) on Kaggle. On Google Colab use [notebooks/11_colab_training.ipynb](notebooks/11_colab_training.ipynb) and [notebooks/12_colab_evaluation.ipynb](notebooks/12_colab_evaluation.ipynb): they read the processed data from an archive on Google Drive and keep the runs there, so a run continues after the session ends, also one started on Kaggle. To compare training on whole images with training on patches over the five folds, use [notebooks/08_kaggle_whole_vs_patches.ipynb](notebooks/08_kaggle_whole_vs_patches.ipynb). To look at the individual missed masses and false regions, write the per-image details with `--details` and open them locally in [notebooks/09_error_analysis.ipynb](notebooks/09_error_analysis.ipynb); to see them, with the probabilities of the model, use [notebooks/10_error_gallery.ipynb](notebooks/10_error_gallery.ipynb), which needs the checkpoints too. To tell whether one model is really better than another, `uv run mammo-seg compare <first details> <second details> --threshold 0.4 --min-area-mm2 10` gives the confidence intervals of the metrics and of their paired difference, by resampling the patients. See [docs/training.md](docs/training.md) for the settings, the files a run writes, and what has been tried.

## Results

Milestone experiments keep their scores and settings locally in `results/<date>_<subject>/` (git-ignored); only each experiment's `README.md`, with the code commit it used and its headline numbers, is tracked. See [results/README.md](results/README.md).

## License

Code: [MIT](LICENSE). Datasets have their own licenses and terms and are not covered by it.
