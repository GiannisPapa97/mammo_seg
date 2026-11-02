# INbreast with a focal Tversky loss

## What and why

The single-pass INbreast runs miss about 27 of the 95 validation masses, most of
them with a peak probability under 0.05, and lowering the threshold does not bring
them back. This run asks whether a loss that makes a missed mass pixel cost more
than a false positive one, the focal Tversky loss (Abraham & Khan, arXiv
1810.07842), finds more masses than the ResNet18 baseline
(`2026-10-28_whole_1024_resnet18`): the same training, with only the region part of
the loss changed.

| | ResNet18 baseline | this run |
|---|---|---|
| region part of the loss | 1 − soft Dice | (1 − Tversky index) ^ 0.75, with the missed mass pixels weighed 0.7 and the false positive pixels 0.3 |

Everything else is the same: the cross-entropy part of the loss, the encoder
pretrained on ImageNet and the decoder random, ResNet18, the augmentation, AdamW at
1e-4 with cosine decay, 60 epochs of 256 images balanced between images with and
without a mass, batches of 4, all layers trained from the first epoch. See
`docs/training.md` (section 2) for the loss.

## Code

- **Training:** notebook `notebooks/05_kaggle_training.ipynb` with
  `FOLDS = [0, 1, 2, 3, 4]`, `MAX_MINUTES = 180`, `PREPROCESS_OPTIONS = ""` and
  `TRAIN_OPTIONS = "--epochs 60 --batch-size 4 --samples-per-epoch 256 --patience 0 --encoder resnet18 --tversky-alpha 0.7 --tversky-beta 0.3 --focal-gamma 0.75"`,
  commit `ae1fd0f88b04efe0eea28cbf0d9900e5e65d2d1d`, one session. Every fold's log
  names the loss it used:
  `DiceBCELoss(dice_weight=1.0, bce_weight=1.0, smooth=1.0, alpha=0.7, beta=0.3, gamma=0.75)`.
- **Evaluation:** notebook `notebooks/06_kaggle_evaluation.ipynb` with
  `EVALUATE_OPTIONS = "--details"`, the same commit.
- **Data:** rebuilt from the raw INbreast dataset on Kaggle by each notebook.
  `splits.json` sha256 `a97a9b9d…`, the file of the frozen-encoder and CLAHE runs,
  which has the baseline's folds. `preprocessing.json` sha256 `f7d2a656…` differs
  from the frozen-encoder run's (`91243aa7…`) by the `clahe` key, here `null`; its
  images have the same crop box, size and flip as the local record of the
  baseline's data, checked locally. The same 326 validation images are scored.
- **Checkpoint scored:** `last.pt` (epoch 60) of each fold, as for the baseline.
- **Comparisons:** `mammo-seg compare` of the baseline's details and these, and of
  the frozen-encoder run's details and these, at threshold 0.4 and 10 mm² (the
  setting chosen for the 1024 run), 2000 draws of the 86 patients, seed 0, 95%
  intervals. The comparison of the five runs below is a count over their details
  files at the same setting, from a script outside the repository.

## Where it ran

- **Training:** Kaggle, T4 GPU, 2026-11-01, the saved version of 19:46 UTC of the
  notebook `mammo-seg-inbreast-tversky`. 21.5 s per epoch (median; the baseline took
  20 s), about 22 minutes per fold and 1 hour 59 minutes in all, including a
  9-minute setup. A peak of 2.9 GB of GPU memory.
- **Evaluation:** Kaggle, T4 GPU, 2026-11-01, the notebook
  `mammo-seg-inbreast-tversky-evaluation`. The files were downloaded with
  `kaggle kernels output`.

## Files

```
whole/code_version.txt                        # the commit of the training session
whole/evaluation_code_version.txt             # the commit of the evaluation
whole/evaluation_validation.json              # the evaluation, all settings
whole/evaluation_validation_details.json      # every validation image with its results
whole/comparison_with_resnet18.json           # mammo-seg compare against the baseline
whole/comparison_with_from_cbis_freeze.json   # mammo-seg compare against the frozen-encoder run
whole/fold_<k>/history.jsonl                  # the training curves
```

## Results

326 validation images over the 5 folds: 87 with a mass (95 masses, 24 under
100 mm²), 239 without. A mass is found at IoU ≥ 0.1.

At threshold 0.4 and 10 mm²; the differences are this run minus the other, with
the paired 95% interval:

| | ResNet18 baseline | frozen encoder | this run | minus baseline | minus frozen encoder |
|---|---|---|---|---|---|
| masses found | 0.716 | 0.716 | 0.705 | −0.011 [−0.066, +0.042] | −0.011 [−0.098, +0.091] |
| masses under 100 mm² found | 0.250 | 0.292 | 0.250 | +0.000 [−0.103, +0.143] | −0.042 [−0.182, +0.097] |
| masses of 100 mm² or more found | 0.873 | 0.859 | 0.859 | −0.014 [−0.078, +0.043] | +0.000 [−0.104, +0.116] |
| Dice, images with a mass | 0.645 | 0.649 | 0.612 | −0.033 [−0.077, +0.010] | −0.038 [−0.106, +0.035] |
| sensitivity, images with a mass | 0.619 | 0.648 | 0.597 | −0.023 [−0.065, +0.017] | −0.051 [−0.123, +0.023] |
| precision, images with a mass | 0.853 | 0.893 | 0.850 | −0.003 [−0.043, +0.038] | **−0.043 [−0.095, −0.002]** |
| false alarms, images without a mass | 0.084 | 0.071 | 0.109 | +0.025 [−0.010, +0.063] | +0.038 [−0.012, +0.089] |
| false regions per image | 0.126 | 0.071 | 0.147 | +0.021 [−0.021, +0.063] | **+0.077 [+0.031, +0.123]** |

**No more masses found, and a few more false positives.** Against the baseline,
nothing differs by more than the luck of which patients were scored. Against the
frozen-encoder run, the false regions are higher and the precision lower beyond
chance. The pixel sensitivity on the images with a mass, which the loss should
raise, is lower than both on the point estimates.

The threshold changes almost nothing: from 0.3 to 0.7 the masses found stay at 65
to 67 of 95 and the Dice with a mass at 0.604 to 0.613. 22 of the 28 missed masses
have a peak probability under 0.05 (22 of the baseline's 27): the loss raises the
probabilities where the model already sees something, not on the masses it misses.

### Per lesion and per image

| | ResNet18 baseline | this run |
|---|---|---|
| masses found (true positives) | 68 | 67 |
| masses missed (false negatives) | 27 | 28 |
| false regions (false positives) | 41 | 48 |
| lesion precision, found / (found + false regions) | 0.624 | 0.583 |
| images with a mass where a mass is found (of 87) | 67 | 66 |
| images with a mass with any prediction | 73 | 72 |
| images without a mass with any prediction (of 239) | 20 | 26 |
| image-level specificity | 0.916 | 0.891 |

There are no lesion-level true negatives, so no lesion-level specificity; the
masses found against the false regions per image take its place. Mass by mass
against the baseline: 64 found by both, 4 by the baseline only, 3 by this run
only, 24 by neither; false-alarm images: 14 in both, 6 in the baseline only, 12 in
this run only. The two runs find nearly the same masses.

### Per fold and by subgroup

Dice on the images with a mass per fold (mean ± SD over the folds): 0.649 ± 0.073
for the baseline and 0.616 ± 0.095 for this run, higher in fold 3 (0.748 against
0.741) and lower in the other 4. The false alarms are higher in 3 folds, lower in
fold 1 and the same in fold 3.

By breast density at the same setting, this run against the baseline: Dice 0.679
against 0.709 in fatty breasts (ACR 1 or 2, 62 images with a mass) and 0.446
against 0.486 in dense ones (ACR 3 or 4, 25). False alarms on the images without a
mass of BI-RADS 1 (53 images): 0.075 against 0.057; of BI-RADS 2 (160): 0.113
against 0.081.

### Training

- **The same course as the baseline:** a Dice with a mass of 0.00 to 0.17 at
  epoch 1 (the baseline: 0.00 to 0.14), and no epoch with a Dice with a mass under
  0.3 after epoch 4 in either run.
- **Best against last epochs:** a mean Dice with a mass over the folds of 0.606 at
  the best epochs (chosen by the Dice over all images, epochs 9 to 41) and 0.620
  over the last 10; the baseline's are 0.641 and 0.650.
- **The training losses do not compare** with the baseline's (0.53 to 0.60 at
  epoch 60, against 0.51 to 0.57): the region part is another function.

## The five single-pass runs together

Mass by mass at threshold 0.4 and 10 mm², over the five 1024-pixel runs from
ImageNet or CBIS-DDSM (the ResNet18 baseline, `2026-10-31_inbreast_from_cbis`,
`2026-10-31_inbreast_from_cbis_freeze`, `2026-11-01_inbreast_clahe` and this run):

| | masses (of 95) |
|---|---|
| found by every run | 49 |
| found by at least 3 runs | 68 |
| found by at least one run | 79 |
| found by no run | 16, 14 of them under 100 mm², each with a peak probability under 0.05 in every run |

The starting weights, the input contrast, the loss and the threshold all leave
the same 16 masses unseen. A majority of the runs finds as many masses as the
baseline alone; their union finds up to 79, but with the false regions of every
run.

## Conclusion

The focal Tversky loss does not find more masses: 67 of 95 against the baseline's
68, within chance, with a few more false regions and false alarms (within chance
against the baseline, beyond it against the frozen-encoder run). It does what it is
built for on the pixels the model already sees and does nothing for the masses it
misses, whose probabilities stay near 0. The ResNet18 baseline and the
frozen-encoder run stay the references.

With this run, every single-pass change tried at 1024 pixels on whole images (the
starting weights, the encoder freezing, the CLAHE channel and the loss) has left
the masses found within chance of 68 of 95, and 16 masses, nearly all small, are
unseen by all of them. Next: the two-stage design (find candidate regions, then
segment crops around them at a higher resolution), starting with an evaluation of
an ensemble of the existing runs, which needs no training.

## Data

Only aggregate scores are here; no images or masks of the dataset.
