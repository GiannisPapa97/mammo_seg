# Results

Results of milestone experiments, one folder per experiment, named by date and
subject, for example `2026-10-24_whole_vs_patches`. The files stay local (the folder
is git-ignored); only this file and the `README.md` of each experiment are tracked,
so the repository keeps a log of what was run, with which code and what it gave.

Only scores and settings belong here: never images, masks or other data from the
dataset (its license is unclear). Checkpoints are fine locally but are never
committed.

## An experiment folder

```
results/<date>_<subject>/
  README.md                        # tracked: see below
  <design>/evaluation_validation.json
  <design>/fold_<k>/history.jsonl
  <design>/fold_<k>/last.pt        # only when an investigation needs predictions
  code_version.txt                 # written by the Kaggle notebooks
```

Its `README.md` records:

- **What and why:** the question the experiment answers.
- **Code:** the commit the run used (from `code_version.txt`, or the commit of
  `REPO_REF` at the time of the run), and the notebook and its settings.
- **Where it ran:** for example Kaggle, T4 GPU, the date, and the saved notebook
  version.
- **Results:** the headline numbers (aggregate scores only) and the conclusion.
