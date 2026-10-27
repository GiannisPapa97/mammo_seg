# Stronger augmentation and weight decay

## What and why

The training-set check of `2026-10-24_whole_vs_patches` shows that the whole
design at 1024 overfits: the fold 0 model finds 19 of 20 small masses on its
training images but 1 of 4 on its validation images, and 2048 and sampling toward
small masses did not help (see `2026-10-26_whole_2048` and
`2026-10-27_whole_1024_small`). This run asks whether regularizing helps: the same
training as the whole design at 1024, with stronger augmentation, an elastic
deformation and weight decay.

| | 1024 | this run |
|---|---|---|
| rotation | ±10° | ±20° |
| scale | ±0.1 | ±0.25 |
| shift | ±0.1 | ±0.1 |
| brightness, contrast | ±0.1 | ±0.2 |
| gamma | ±0.2 | ±0.3 |
| noise (SD) | 0.01 | 0.02 |
| elastic deformation | none | 4 px, control points 64 px apart |
| weight decay (AdamW) | 0 | 0.1 |

The augmentation and the weight decay changed together, so this run does not tell
their effects apart.

## Code

- **Training:** commit `52000891e4c49e6276a03fdfd7b62b09ffcaea34`, notebook
  `notebooks/05_kaggle_training.ipynb` with `FOLDS = [0, 1, 2, 3, 4]`,
  `MAX_MINUTES = 11 * 60`, `PREPROCESS_OPTIONS = ""` and
  `TRAIN_OPTIONS = "--epochs 60 --batch-size 4 --samples-per-epoch 256 --patience 0 --rotation 20 --scale 0.25 --brightness 0.2 --contrast 0.2 --gamma 0.3 --noise 0.02 --elastic 4 --weight-decay 0.1"`.
- **Evaluation with per-image details:** the same commit, notebook
  `notebooks/06_kaggle_evaluation.ipynb` with `EVALUATE_OPTIONS = "--details"`
  and `PREPROCESS_OPTIONS = ""`, on the checkpoints of the training run.
- **Splits:** the stratified splits (version 2), the same folds as the 1024 run.
- **Checkpoint scored:** `last.pt` (epoch 60) of each fold.
- **Comparison:** `mammo-seg compare` of the 1024 details of
  `2026-10-24_whole_vs_patches` and these, 2000 draws of the 86 patients, seed 0,
  95% intervals.

## Where it ran

Kaggle, T4 GPU, 2026-10-27: `giannispapatrechas/whole-1024-aug` (training) and
`giannispapatrechas/evaluation-1024-aug` (evaluation).

Training took 25 s per epoch, about 25 minutes per fold and 2.05 hours for the 5
folds, with a peak of 3.3 GB of GPU memory reserved. The evaluation took about 8
minutes after preprocessing.

## Files

```
whole/evaluation_validation.json               # the evaluation, all settings
whole/evaluation_validation_details.json       # every validation image with its results
whole/evaluation_training_fold0.json           # the training-set check, all settings
whole/evaluation_training_fold0_details.json   # every fold 0 training image with its results
whole/evaluation_code_version.txt              # the commit of the evaluation run
whole/evaluation_details.log                   # the log of the evaluation run
whole/code_version.txt                         # the commit of the training run
whole/fold_<k>/history.jsonl                   # the training curves
```

## Results

326 validation images over the 5 folds: 87 with a mass (95 masses, 24 under
100 mm²), 239 without. A mass is found at IoU ≥ 0.1.

At threshold 0.4 and 10 mm², the setting chosen for 1024 (augmentation minus
1024, with the paired 95% interval of the difference):

| | 1024 | augmentation | augmentation minus 1024 |
|---|---|---|---|
| masses found | 0.716 | 0.737 | +0.021 [−0.063, +0.106] |
| masses under 100 mm² found | 0.333 | 0.375 | +0.042 [−0.200, +0.278] |
| masses of 100 mm² or more found | 0.845 | 0.859 | +0.014 [−0.056, +0.088] |
| Dice, images with a mass | 0.648 | 0.669 | +0.021 [−0.043, +0.085] |
| sensitivity, images with a mass | 0.644 | 0.661 | +0.017 [−0.042, +0.078] |
| precision, images with a mass | 0.857 | 0.861 | +0.004 [−0.039, +0.051] |
| false alarms, images without a mass | 0.113 | 0.146 | +0.033 [−0.017, +0.086] |
| false regions per image | 0.12 | 0.19 | +0.071 [+0.012, +0.130] |

Dice on the images with a mass per fold: 0.647 ± 0.088 at 1024 and
0.673 ± 0.094 with augmentation (mean ± SD over the folds); augmentation is higher
in 3 of 5 folds.

The best pooled setting with augmentation, 0.7 and 10 mm², gives the same masses
found (0.737), false alarms 0.134 and false regions 0.18 per image, so a higher
threshold does not remove the extra false regions. The false alarms are most
frequent on images with a benign finding and no mass (0.159 of them, against 0.075
of the normal images).

Of the 24 small masses, 5 are found by both runs, 3 by 1024 only and 4 by
augmentation only. Of the 71 large masses, 56 are found by both, 4 by 1024 only
and 5 by augmentation only.

## Training-set check

**Question:** does the regularization reduce the memorizing that the training-set
check of `2026-10-24_whole_vs_patches` found?

**How:** as in that check: this run's fold 0 `last.pt` scored on its own 256
training images with the evaluation code of commit
`52000891e4c49e6276a03fdfd7b62b09ffcaea34`, locally on the CPU, 2026-10-27, from
a script outside the repository; the output is in the details format.

At threshold 0.4 and 10 mm², the fold 0 models on their training and validation
images:

| | 1024, training | 1024, validation | augmentation, training | augmentation, validation |
|---|---|---|---|---|
| masses under 100 mm² found | 19 of 20 | 1 of 4 | 19 of 20 | 3 of 4 |
| masses of 100 mm² or more found | 60 of 60 | 9 of 11 | 60 of 60 | 11 of 11 |
| Dice, images with a mass | 0.946 | 0.579 | 0.911 | 0.760 |
| false alarms, images without a mass | 11 of 184 | 8 of 55 | 19 of 184 | 9 of 55 |
| false regions per image | 0.047 | 0.143 | 0.094 | 0.200 |

In fold 0 the gap between training and validation Dice narrows from 0.37 to
0.15: the model fits its training images a little less closely and outlines the
validation masses better. But fold 0 is the fold where the augmentation helped
most (Dice with a mass 0.579 to 0.760); in folds 1 and 2 it is lower than at
1024 (0.726 to 0.689 and 0.624 to 0.550), so this one fold overstates the effect.
The model also has twice the false alarms and false regions on its own training
images, so the extra false regions in validation are not a failure to
generalize: the model is less precise everywhere.

## Conclusion

Stronger augmentation with weight decay is not better on the main metric: 70
instead of 68 of 95 masses found, an interval from −6 to +11 points, and the
small and large masses found change by a few masses each way, mostly different
masses rather than more. It significantly increases the false regions (0.12 to
0.19 per image), and a higher threshold does not remove them. In the one fold
checked it reduces the memorizing, but the gains in validation do not hold over
the folds.

The 1024 run with its original augmentation stays the reference. The options
remain in the code; weight decay alone or a milder augmentation are untested.

Next: a smaller encoder (ResNet18, about 40% fewer parameters than ResNet34) with
the reference settings, and `best.pt` against `last.pt`; then more data, by
pretraining on CBIS-DDSM.
