# Whole images against patches

## What and why

After the fix for the flat-zero background, two input designs were left to
choose between:

- **whole:** training on whole preprocessed images (canvas 1024×800) and
  predicting them in one pass
- **patches:** training on 512×512 crops and predicting whole images in sliding
  windows (50% overlap, Gaussian weights)

Each design was trained once on each of the 5 folds and scored on the
validation fold of each, so the two can be compared fold by fold.

## Code

- **Training:** commit `bc59269cc4f559384ee0c8cdd9681f4e276ef825`, notebook
  `notebooks/08_kaggle_whole_vs_patches.ipynb` with `EPOCHS = 60` and:
  - whole: `--batch-size 4 --samples-per-epoch 256 --patience 0`
  - patches: `--crop-size 512 --batch-size 8 --samples-per-epoch 512 --patience 0`
- **Per-image details (whole only):** commit
  `4de65e366864a950d38bee00539784b8a8d5dc34`, notebook
  `notebooks/06_kaggle_evaluation.ipynb` with `DESIGN = "whole"` and
  `EVALUATE_OPTIONS = "--details"`, on the checkpoints of the training run. Its
  summary is identical to the one from the training run.
- **Splits:** the stratified splits (version 2); the Kaggle `splits.json` is
  identical to the local one.
- **Checkpoint scored:** `last.pt` (epoch 60) of each fold.

## Where it ran

Kaggle, T4 GPU, 2026-10-24: notebooks `giannispapatrechas/whole-vs-patch`
(training and evaluation) and `giannispapatrechas/evaluation-whole-details`
(details).

## Files

```
whole/evaluation_validation.json           # the evaluation, all settings
whole/evaluation_validation_details.json   # every validation image with its results
whole/evaluation_code_version.txt          # the commit of the details run
whole/evaluation_details.log               # the log of the details run
whole/fold_<k>/history.jsonl               # the training curves
whole/fold_<k>/last.pt                     # the checkpoints, for notebooks/10_error_gallery.ipynb (download from Kaggle)
whole/evaluation_training_fold0.json       # the training-set check: fold 0 on its own training images
whole/evaluation_training_fold0_details.json
patches/evaluation_validation.json
patches/fold_<k>/history.jsonl
```

## Results

326 validation images over the 5 folds: 87 with a mass (95 masses), 239 without.
Scores are pooled over the folds, except the mean ± SD over the folds. A mass is
found at IoU ≥ 0.1.

At threshold 0.5 and no minimum area:

| | whole | patches |
|---|---|---|
| Dice, images with a mass | 0.647 | 0.593 |
| mean ± SD over the folds | 0.646 ± 0.088 | 0.594 ± 0.083 |
| false alarms, images without a mass | 0.163 | 0.163 |
| masses found | 0.716 | 0.653 |

Per fold, patches minus whole: −0.052 ± 0.090 in Dice; patches did better in 1
of 5 folds.

At the best pooled setting of each design (0.4 and 10 mm² for whole, 0.7 and
10 mm² for patches):

| | whole | patches |
|---|---|---|
| Dice, images with a mass | 0.648 | 0.584 |
| false alarms, images without a mass | 0.113 | 0.059 |
| false regions per image | 0.12 | 0.09 |
| masses found | 0.716 | 0.642 |
| masses under 100 mm² found (24) | 0.333 | 0.333 |
| masses of 100 mm² or more found (71) | 0.845 | 0.746 |
| Dice, fatty breasts (62 with a mass) | 0.717 | 0.673 |
| Dice, dense breasts (25 with a mass) | 0.478 | 0.365 |

## Conclusion

Whole images are better: higher Dice in 4 of 5 folds, more masses found, and
much better in dense breasts. Patches give fewer false alarms at their best
setting, but at a high threshold and at the cost of Dice. The folds vary a lot
(SD 0.09), and with 5 folds the difference is suggestive, not proven.

Next: whole images. Both designs find only 1 in 3 masses under 100 mm², and Dice
in dense breasts is much lower; the per-image details are for analysing these
errors.

## Training-set check

**Question:** are the small masses missed because 1024 is too coarse to show
them, or because the model does not generalize to new ones? A model that cannot
resolve small masses misses them on its own training images too.

**How:** the whole design's fold 0 `last.pt` scored on its own 256 training
images, with the evaluation code of commit
`8c260fd94be746a8547a390f341e355d46ceb006`: `evaluate_model` with the default
settings, whole-image inference, the same preprocessing as the training run.
Locally on the CPU, 2026-10-27, from a script outside the repository; the
output is in the details format, so notebooks 09 and 10 read it with `DETAILS`
pointed at `whole/evaluation_training_fold0_details.json`.

At threshold 0.4 and 10 mm², the fold 0 model on its training and its validation
images:

| | training (256 images) | validation (70 images) |
|---|---|---|
| masses under 100 mm² found | 19 of 20 (0.95) | 1 of 4 (0.25) |
| masses of 100 mm² or more found | 60 of 60 (1.00) | 9 of 11 (0.82) |
| Dice, images with a mass | 0.946 | 0.579 |
| false alarms, images without a mass | 11 of 184 | 8 of 55 |
| false regions per image | 0.047 | 0.143 |

**Conclusion:** at 1024 the model finds and outlines small masses on images it
was trained on, so the input is not too coarse for them; the misses in
validation are mostly a failure to generalize, with a large gap between training
and validation (Dice 0.95 against 0.58). Two limits: recognizing a mass seen
about 100 times in training (60 epochs of 128 draws of the 72 images with a
mass) is easier than recognizing a new one, so a higher resolution may still
help somewhat; and this is one fold, with only 4 small
masses in its validation fold (over all 5 folds, 8 of 24 are found).

This lowers the expectation for 2048 and raises that for what counters
overfitting: stronger augmentation, regularization, an earlier checkpoint, and
more data (pretraining on another dataset). Drawing the few small-mass images
more often may increase the memorizing rather than reduce it.
