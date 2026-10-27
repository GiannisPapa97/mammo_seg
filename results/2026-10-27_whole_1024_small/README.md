# Sampling toward small masses

## What and why

The whole design at 1024 (see `2026-10-24_whole_vs_patches`) finds only 1 in 3
masses under 100 mm², and 2048 did not change that (see `2026-10-26_whole_2048`).
This run asks whether drawing the images with a small mass more often helps: the
same training as the whole design at 1024, with `--small-mass-fraction 0.5` and
nothing else changed. Among the draws of images with a mass, about half are then
images with a small mass instead of the natural share of about 0.28, so each of
them is drawn about 1.8 times as often; the share of images with a mass stays at
0.5.

## Code

- **Training:** commit `64a032084653ed5adf9e4e93719c8b75b8784161`, notebook
  `notebooks/05_kaggle_training.ipynb` with `FOLDS = [0, 1, 2, 3, 4]`,
  `MAX_MINUTES = 11 * 60`, `PREPROCESS_OPTIONS = ""` and
  `TRAIN_OPTIONS = "--epochs 60 --batch-size 4 --samples-per-epoch 256 --patience 0 --small-mass-fraction 0.5"`.
- **Evaluation with per-image details:** the same commit, notebook
  `notebooks/06_kaggle_evaluation.ipynb` with `EVALUATE_OPTIONS = "--details"`
  and `PREPROCESS_OPTIONS = ""`, on the checkpoints of the training run.
- **Splits:** the stratified splits (version 2), the same folds as the 1024 run.
- **Checkpoint scored:** `last.pt` (epoch 60) of each fold.
- **Comparison:** `mammo-seg compare` of the 1024 details of
  `2026-10-24_whole_vs_patches` and these, 2000 draws of the 86 patients, seed 0,
  95% intervals.

## Where it ran

Kaggle, T4 GPU, 2026-10-27: `giannispapatrechas/whole-1024-small` (training) and
`giannispapatrechas/evaluation-1024-small` (evaluation).

Training took 24 s per epoch, about 25 minutes per fold and 2.04 hours for the 5
folds, with a peak of 3.3 GB of GPU memory reserved. The evaluation took about 6
minutes after preprocessing.

## Files

```
whole/evaluation_validation.json           # the evaluation, all settings
whole/evaluation_validation_details.json   # every validation image with its results
whole/evaluation_code_version.txt          # the commit of the evaluation run
whole/evaluation_details.log               # the log of the evaluation run
whole/code_version.txt                     # the commit of the training run
whole/fold_<k>/history.jsonl               # the training curves
```

## Results

326 validation images over the 5 folds: 87 with a mass (95 masses, 24 under
100 mm²), 239 without. A mass is found at IoU ≥ 0.1.

At threshold 0.4 and 10 mm², the setting chosen for 1024 (sampling minus 1024,
with the paired 95% interval of the difference):

| | 1024 | sampling | sampling minus 1024 |
|---|---|---|---|
| masses found | 0.716 | 0.674 | −0.042 [−0.087, +0.000] |
| masses under 100 mm² found | 0.333 | 0.333 | +0.000 [−0.133, +0.133] |
| masses of 100 mm² or more found | 0.845 | 0.789 | −0.056 [−0.109, −0.013] |
| Dice, images with a mass | 0.648 | 0.625 | −0.023 [−0.060, +0.015] |
| sensitivity, images with a mass | 0.644 | 0.602 | −0.042 [−0.080, −0.003] |
| precision, images with a mass | 0.857 | 0.892 | +0.035 [−0.009, +0.084] |
| false alarms, images without a mass | 0.113 | 0.071 | −0.042 [−0.083, −0.004] |
| false regions per image | 0.12 | 0.08 | −0.043 [−0.083, −0.003] |

Dice on the images with a mass per fold: 0.647 ± 0.088 at 1024 and
0.628 ± 0.088 with sampling (mean ± SD over the folds); sampling is higher in 2 of
5 folds.

The best pooled setting with sampling, 0.5 and 10 mm², gives the same masses
found (0.674), false alarms 0.067 and false regions 0.07 per image.

Of the 24 small masses, 7 are found by both runs, and one by each run only. Of
the 71 large masses, 55 are found by both, 5 by 1024 only and 1 by sampling only.

## Conclusion

Sampling toward small masses does not find more of them: the same 8 of 24, nearly
the same masses. It makes the model more conservative instead: fewer false alarms
and false regions, but fewer large masses found and a lower sensitivity, so it is
not better on the main metric, the masses found. Its higher Dice over all images
(0.848 against 0.823 at 0.4 and 10 mm²) comes from the fewer false alarms.

This fits the training-set check of `2026-10-24_whole_vs_patches`: the model
already finds small masses on its training images, so drawing the same few
images more often adds memorizing, not what carries over to new ones. It remains
an option if false positives come to matter more than missed masses.

Next: stay at 1024 without the sampling and address the overfitting with stronger
augmentation and regularization.
