# Pretraining on CBIS-DDSM

## What and why

Every INbreast run so far overfits: the models fit their training images almost
perfectly whatever the encoder, the augmentation or the weight decay (see
`2026-10-28_whole_1024_resnet18`). INbreast has only 107 images with a mass. This
run trains the ResNet18 U-Net on the masses of CBIS-DDSM, 1,166 images with a mass
in its official training set, so that the INbreast runs can start from it instead
of from ImageNet alone.

It is a pretraining run, not a cross-validation: one model, trained on folds 1 to 4
of the official training set and validated on fold 0 to choose its epoch. The
official test set is scored once, to see how well the model does on CBIS-DDSM
itself. Whether the pretraining helps is decided on INbreast, by fine-tuning from
this model and comparing with the ResNet18 baseline (a later experiment).

## Code

- **Training:** notebook `notebooks/11_colab_training.ipynb` with
  `DATASET = "cbis-ddsm"`, `FOLDS = [0]`, `MAX_MINUTES = 120`, `INIT_FROM = None`,
  `RUN_NAME = "cbis-1024"` and
  `TRAIN_OPTIONS = "--encoder resnet18 --epochs 60 --batch-size 4 --patience 0"`,
  commit `37dcff22d6cedcefbdb8e9ce355c75d3f1b486a0`. For CBIS-DDSM the notebook
  adds `--positive-fraction none`: every image has a mass, so there is nothing to
  balance, and each epoch draws every training image once.
- **Training settings otherwise the defaults of that commit,** the same as the
  INbreast baseline: the encoder pretrained on ImageNet, all layers trained from
  the first epoch, AdamW at 1e-4 with cosine decay and no weight decay, the default
  augmentation and loss, mixed precision, seed 0, the best epoch chosen by the Dice
  over all validation images.
- **Evaluation:** notebook `notebooks/12_colab_evaluation.ipynb`, commit
  `67dc5cd8da3797b72e9e1e0c41b6836d3c240c95`, with `DATASET = "cbis-ddsm"`,
  `FOLDS = [0]` and `RUN_NAME = "cbis-1024"`. For CBIS-DDSM the notebook adds
  `--min-areas-mm2 0`, because the images have no pixel size.
  - validation: `EVALUATE_OPTIONS = "--checkpoint-name best.pt --details"`;
  - test, once, at the threshold chosen on validation:
    `EVALUATE_OPTIONS = "--checkpoint-name best.pt --test --thresholds 0.7 --details"`.

  A first validation run, at commit `c00a807`, wrote its results but failed to
  print the table, because no setting has a balanced Dice without images free of
  masses; `67dc5cd` fixed the table and the run was repeated, overwriting the file.
- **Data on Colab:** the archive of `data/processed/cbis-ddsm/`, made from the
  local files: `splits.json` sha256 `a7d0327c…`, `preprocessing.json` sha256
  `e1491f56…`. Preprocessing at a longer side of 1024 pixels with the CBIS-DDSM
  settings (threshold 20, margin 0.04, chest-wall side found from the image, film
  edges and background cleared; see `docs/data-pipeline.md`).
- **Splits:** the official test set kept as it is (196 patients, 348 images); the
  official training set split into 5 patient folds stratified on mass size (in
  pixels, 40,000) and density. Fold 0 has 132 patients and 245 images; folds 1 to 4
  have 525 patients and 921 images.
- **Checkpoint scored:** `best.pt` (epoch 33). It is the model the INbreast runs
  start from.

## Where it ran

- **Training:** Google Colab, T4 GPU, 2026-10-29, 09:35 to 11:00 UTC. 82 s per
  epoch (median), 82 minutes in all, a peak of 3.1 GB of GPU memory. The browser
  lost its connection to the session once, but the training went on: the history
  has no gap and `code_version.txt` a single session.
- **Evaluation:** Google Colab, T4 GPU, 2026-10-29.

## Files

```
whole/code_version.txt                      # the commit of the training session
whole/evaluation_code_version.txt           # the commits of the evaluation runs
whole/evaluation_validation.json            # fold 0, all thresholds
whole/evaluation_validation_details.json    # every fold 0 image with its results
whole/evaluation_test.json                  # the official test set, threshold 0.7
whole/evaluation_test_details.json          # every test image with its results
whole/fold_0/history.jsonl                  # the training curve
```

The checkpoints stay on Drive, in `mammo_seg_runs/cbis-1024/fold_0/`.

## Results

Every image has a mass, so there are no false alarms and no balanced Dice, and the
Dice over all images is the Dice on the images with a mass. There is no pixel size,
so no masses by size in mm². A mass, here a connected region of a mask, is found at
IoU ≥ 0.1.

### Training

| epoch | training loss | validation loss | validation Dice |
|---|---|---|---|
| 1 | 1.358 | 1.209 | 0.234 |
| 5 | 0.780 | 0.854 | 0.257 |
| 10 | 0.539 | 0.591 | 0.446 |
| 13 | 0.500 | 0.528 | 0.503 |
| 20 | 0.440 | 0.519 | 0.508 |
| 33 (best) | 0.356 | 0.510 | 0.517 |
| 40 | 0.326 | 0.538 | 0.489 |
| 50 | 0.287 | 0.537 | 0.492 |
| 60 | 0.279 | 0.536 | 0.494 |

**Validation stopped improving around epoch 13.** From epoch 13 to 60 the
validation Dice stays between 0.464 and 0.517 (mean 0.493, SD 0.011), with no trend,
while the training loss falls from 0.50 to 0.28. Epoch 33 has both the best Dice
and the lowest validation loss (0.510), but its lead over epochs 13 to 32 is within
the epoch-to-epoch noise of 245 images. The model overfits, more mildly than on
INbreast: the validation loss rises only from 0.510 to about 0.536.

**Early stopping would have saved time.** From this history, with the default
patience of 15 the run stops after epoch 48 with the same `best.pt` (epoch 33);
with a patience of 10, after epoch 32 with epoch 22 (Dice 0.513). Either saves 12 to
28 epochs, 16 to 38 minutes.

### Validation (fold 0, 245 images, 264 masses)

| threshold | Dice | IoU | sensitivity | precision | masses found | false regions per image |
|---|---|---|---|---|---|---|
| 0.3 | 0.515 | 0.411 | 0.630 | 0.528 | 0.735 | 0.93 |
| 0.4 | 0.516 | 0.412 | 0.626 | 0.535 | 0.731 | 0.90 |
| 0.5 | 0.517 | 0.413 | 0.623 | 0.539 | 0.731 | 0.88 |
| 0.6 | 0.518 | 0.414 | 0.619 | 0.543 | 0.731 | 0.87 |
| 0.7 | 0.518 | 0.415 | 0.615 | 0.547 | 0.731 | 0.88 |

The threshold hardly matters: the probabilities are nearly all close to 0 or 1.
0.7 has the best Dice and was used for the test set. Predicting nothing scores 0.

### The official test set (348 images, 369 masses), threshold 0.7

| | validation | test |
|---|---|---|
| Dice | 0.518 | 0.479 |
| IoU | 0.415 | 0.383 |
| sensitivity | 0.615 | 0.590 |
| precision | 0.547 | 0.515 |
| masses found | 0.731 | 0.694 |
| false regions per image | 0.88 | 0.91 |
| Dice, fatty breasts (ACR 1 or 2) | 0.551 (162 images) | 0.543 (205) |
| Dice, dense breasts (ACR 3 or 4) | 0.453 (83 images) | 0.387 (143) |

**On fatty breasts the test set matches validation; the drop is in dense
breasts.** Part of it is the mix: 41% of the test images are dense against 34% in
fold 0, and with the mix of fold 0 the test Dice would be 0.490. The rest is that
the dense test images are harder (0.387 against 0.453), which fits the test set's
larger share of benign masses (see `docs/dataset-cbis-ddsm.md`); with 83 dense
validation images, part of it may be chance. The false regions per image are the
same.

These numbers are for full images of scanned film at a longer side of 1024 pixels,
with the loose outlines of CBIS-DDSM; they are not comparable with INbreast's or
with results on crops around the masses.

## Conclusion

The model learns to segment masses on CBIS-DDSM: a Dice of 0.52 on validation and
0.48 on the official test set, with 69 to 73% of the masses found and about 0.9
false regions per image. It generalizes to new patients of the same dataset
except, partly, in dense breasts. It overfits after about epoch 13, but mildly, and
`best.pt` keeps a model from the plateau.

Whether this helps on INbreast is the next experiment: 5-fold fine-tuning on
INbreast from `best.pt`, with all layers trained from the start and with the
encoder frozen for the first 10 epochs, compared with the ResNet18 baseline at
threshold 0.4 and 10 mm², scoring `last.pt` as for the baseline. The INbreast
splits on Drive have another sha256 than the baseline's (`59a3a718…` against
`6d8d4120…`), because the splits file changed format (version 3) with the common
manifest schema; the folds are the same: every one of the 86 patients is in the
same fold, and the same 326 validation images are scored.

For future pretraining runs: early stopping (patience 10 to 15) or about 30
epochs.

## Data

CBIS-DDSM is used under CC BY 3.0 and must be cited in any report of these
results:

- Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). *Curated Breast
  Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM)* [Data
  set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
- Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., & Rubin, D. L.
  (2017). A curated mammography data set for use in computer-aided detection and
  diagnosis research. *Scientific Data*, 4, 170177.
  https://doi.org/10.1038/sdata.2017.177
- Clark, K., et al. (2013). The Cancer Imaging Archive (TCIA): Maintaining and
  Operating a Public Information Repository. *Journal of Digital Imaging*, 26(6),
  1045–1057. https://doi.org/10.1007/s10278-013-9622-7

Only aggregate scores are here; no images or masks.
