# Whole images at 2048 against 1024

## What and why

The whole design (see `2026-10-24_whole_vs_patches`) finds only 1 in 3 masses
under 100 mm². This run asks whether a larger input helps: the same training as
the whole design, with the longer side of the preprocessed images at 2048 pixels
instead of 1024, and nothing else changed.

## Code

- **Training:** commit `204ae131517f874fd64f18599c21415f3caa3882`, notebook
  `notebooks/05_kaggle_training.ipynb` with `FOLDS = [0, 1, 2, 3, 4]`,
  `MAX_MINUTES = 11 * 60`, `PREPROCESS_OPTIONS = "--size 2048"` and
  `TRAIN_OPTIONS = "--epochs 60 --batch-size 4 --samples-per-epoch 256 --patience 0"`.
- **Evaluation with per-image details:** commit
  `6cafa12eb091f8b2f23bfb09a7cab301d7201e0f`, notebook
  `notebooks/06_kaggle_evaluation.ipynb` with `EVALUATE_OPTIONS = "--details"`
  and `PREPROCESS_OPTIONS = "--size 2048"`, on the checkpoints of the training
  run.
- **Splits:** the stratified splits (version 2), the same folds as the 1024 run.
- **Checkpoint scored:** `last.pt` (epoch 60) of each fold.
- **Comparison:** `mammo-seg compare` of the 1024 and the 2048 details, 2000
  draws of the 86 patients, seed 0, 95% intervals.

## Where it ran

Kaggle, T4 GPU: `giannispapatrechas/whole-2048` (training, 2026-10-26 to 27) and
`giannispapatrechas/evaluation-2048` (evaluation, 2026-10-27).

Preprocessing at 2048 took 472 to 560 s (longer side 2048, shorter side 553 to
1569). Training took 111 s per epoch (24 s at 1024), 1.85 hours per fold and
9.25 hours for the 5 folds, with a peak of 12.5 GB of GPU memory reserved at
batch size 4. The evaluation took about 40 minutes, most of it the
post-processing of the larger masks.

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

326 validation images over the 5 folds. The masks at 2048 count 96 masses (25
under 100 mm²) where those at 1024 count 95 (24): one mass falls apart into two
regions at the finer size. A mass is found at IoU ≥ 0.1.

At threshold 0.4 and 10 mm², the setting chosen for 1024 (2048 minus 1024, with
the paired 95% interval of the difference):

| | 1024 | 2048 | 2048 minus 1024 |
|---|---|---|---|
| masses found | 0.716 | 0.625 | −0.091 [−0.196, +0.011] |
| masses under 100 mm² found | 0.333 | 0.240 | −0.093 [−0.291, +0.154] |
| masses of 100 mm² or more found | 0.845 | 0.761 | −0.085 [−0.186, +0.014] |
| Dice, images with a mass | 0.648 | 0.574 | −0.075 [−0.152, +0.001] |
| sensitivity, images with a mass | 0.644 | 0.543 | −0.101 [−0.180, −0.024] |
| precision, images with a mass | 0.857 | 0.911 | +0.054 [+0.012, +0.105] |
| false alarms, images without a mass | 0.113 | 0.096 | −0.017 [−0.068, +0.034] |
| false regions per image | 0.12 | 0.12 | −0.006 [−0.058, +0.044] |

Dice on the images with a mass per fold: 0.647 ± 0.088 at 1024 and
0.572 ± 0.100 at 2048 (mean ± SD over the folds); 2048 is lower in 4 of 5 folds.

The threshold does not change this: at 2048 the share of masses found is 0.625 at
every threshold from 0.3 to 0.7. Its best pooled setting, 0.6 and 10 mm², gives
Dice 0.568 on the images with a mass (lower than 1024's, interval
[−0.156, −0.003]) and false alarms 0.079.

Among the 17 large masses missed at 2048, 6 have a peak probability of 0.9 or
more: the model responds to them, but the region above the threshold covers
only a small part of the mass (IoU 0.03 to 0.07), or less than 10 mm² and is
removed. Four of these 6 are found at 1024 (IoU 0.72 to 0.92).

## Conclusion

2048 is not better than 1024, and at 4.6 times the training time. Every mass
metric is lower; only the drops in sensitivity and IoU are beyond the luck of
which patients were scored, but none of the differences favours 2048. Small
masses are not found more often, in line with the training-set check of
`2026-10-24_whole_vs_patches`: at 1024 the model finds small masses on its
training images, so the resolution is not what limits it.

The larger input loses mainly on large masses, which come out as small cores.
A likely reason, not tested here: the encoder sees the same number of pixels
around each point, so at 2048 half the tissue around it, too little to fill a
large mass.

Next: stay at 1024 and address the overfitting: sampling toward small masses,
then stronger augmentation and regularization.
