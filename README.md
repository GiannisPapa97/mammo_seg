# MammoSeg

A reproducible PyTorch framework for segmentation of mammography images, starting with the INbreast dataset.

> **Research use only.** This project is not a medical device and its output must not be used for clinical decisions.

**Status:** early setup. Nothing is implemented yet.

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 or newer is supported; development uses 3.12.

```bash
uv sync
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

## License

Code: [MIT](LICENSE). Datasets have their own licenses and terms and are not covered by it.
