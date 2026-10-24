# MammoSeg

A reproducible PyTorch framework for segmentation of mammography images, starting with the INbreast dataset.

> **Research use only.** This project is not a medical device and its output must not be used for clinical decisions.

**Status:** early development. The data pipeline is in place: a validated manifest of the INbreast dataset, patient-level splits, mass masks, the preprocessing, and the PyTorch dataset and loaders. The metrics, the loss, the model, checkpoints, the training loop and the evaluation are in place ([docs/training.md](docs/training.md)). Training on all five folds works with crops around the masses, but the model is not yet a useful detector: it beats predicting nothing on the Dice over all images in two of the five folds (a mean Dice of about 0.56 on the images with a mass). The test set has not been used.

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 or newer is supported; development uses 3.12.

PyTorch is installed through an extra that picks its build. Choose exactly one:

```bash
uv sync --extra cpu      # local development and CI (no GPU)
uv sync --extra cu126    # a GPU machine with an older NVIDIA driver (CUDA 12.6)
uv sync --extra cu130    # a GPU machine with a recent NVIDIA driver (CUDA 13.0)
```

A plain `uv sync` installs no PyTorch and removes it if it was installed.

**Kaggle (tried), Google Colab and other cloud notebooks (not tried).** They come with a GPU build of PyTorch already installed, which must not be replaced. Install the package with the `cloud` extra, which adds the model library and leaves PyTorch alone:

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

Datasets are not part of this repository. The `data/` directory is git-ignored; raw data goes in `data/raw/` and derived data in `data/processed/`.

To download and unzip the INbreast dataset (about 2 GB, about 9 GB unzipped) from its [Kaggle copy](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset) into `data/raw/`:

```bash
scripts/download_data.sh
```

It needs `curl` and `unzip`, resumes an interrupted download, and does nothing if the data is already there. The license of this copy is unclear, so do not redistribute the data. See [docs/dataset.md](docs/dataset.md) for what it contains.

Then build the validated manifest (`data/processed/manifest.json`: every image with its metadata and lesions):

```bash
uv run python -m mammo_seg.data.manifest
```

Then make patient-level splits (`data/processed/splits.json`), stratified on masses, mass size and breast density, for example 5 folds with a 20% test set:

```bash
uv run python -m mammo_seg.data.splits --folds 5 --test-fraction 0.2
```

Then write the mass masks (`data/processed/masks/<file_id>.png`, one per image):

```bash
uv run python -m mammo_seg.data.masks
```

Then preprocess the images once (crop to the breast, rescale to a longer side of 1024 px, flip right breasts), which writes `data/processed/preprocessed/`:

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

On Kaggle use [notebooks/05_kaggle_training.ipynb](notebooks/05_kaggle_training.ipynb), which rebuilds the data from the attached dataset, trains one or more folds and stops cleanly before the session limit. To score the trained folds (thresholds, minimum region size, masses found and false regions per image, subgroups, the baseline of predicting nothing) use `uv run mammo-seg evaluate --runs-dir runs`, or [notebooks/06_kaggle_evaluation.ipynb](notebooks/06_kaggle_evaluation.ipynb) on Kaggle. To compare training on whole images with training on patches over the five folds, use [notebooks/08_kaggle_whole_vs_patches.ipynb](notebooks/08_kaggle_whole_vs_patches.ipynb). To look at the individual missed masses and false regions, write the per-image details with `--details` and open them locally in [notebooks/09_error_analysis.ipynb](notebooks/09_error_analysis.ipynb). See [docs/training.md](docs/training.md) for the settings, the files a run writes, and what has been tried.

## Results

Milestone experiments keep their scores and settings locally in `results/<date>_<subject>/` (git-ignored); only each experiment's `README.md`, with the code commit it used and its headline numbers, is tracked. See [results/README.md](results/README.md).

## License

Code: [MIT](LICENSE). Datasets have their own licenses and terms and are not covered by it.
