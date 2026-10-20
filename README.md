# MammoSeg

A reproducible PyTorch framework for segmentation of mammography images, starting with the INbreast dataset.

> **Research use only.** This project is not a medical device and its output must not be used for clinical decisions.

**Status:** early development. The data pipeline is in place: a validated manifest of the INbreast dataset, patient-level splits, mass masks, the preprocessing, and the PyTorch dataset and loaders. Models and training are not implemented yet.

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 or newer is supported; development uses 3.12.

PyTorch is installed through an extra that picks its build. Choose exactly one:

```bash
uv sync --extra cpu      # local development and CI (no GPU)
uv sync --extra cu126    # a GPU machine with an older NVIDIA driver (CUDA 12.6)
uv sync --extra cu130    # a GPU machine with a recent NVIDIA driver (CUDA 13.0)
```

A plain `uv sync` installs no PyTorch and removes it if it was installed. On Kaggle or Colab, where a GPU build of PyTorch is already installed, install the package with `pip install .` (no extra) and the existing PyTorch is used. The device is chosen at run time: `auto` (default) picks a GPU if there is one, and the `MAMMO_SEG_DEVICE` environment variable (`cpu`, `cuda`, `cuda:1`, `mps`) overrides it.

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

Then make patient-level stratified splits (`data/processed/splits.json`), for example 5 folds with a 20% test set:

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

## License

Code: [MIT](LICENSE). Datasets have their own licenses and terms and are not covered by it.
