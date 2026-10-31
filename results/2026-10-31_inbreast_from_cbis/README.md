# INbreast from the CBIS-DDSM model

## What and why

Every INbreast run so far overfits, whatever the encoder, the augmentation or the
weight decay, and INbreast has only 107 images with a mass. This run asks whether
starting from a model trained on the masses of CBIS-DDSM
(`2026-10-29_cbis_ddsm_pretraining`) instead of from ImageNet alone gives a better
INbreast model: the same training as the ResNet18 baseline
(`2026-10-28_whole_1024_resnet18`), with only the starting weights changed.

| | ResNet18 baseline | this run |
|---|---|---|
| starting weights | encoder pretrained on ImageNet, decoder random | the whole U-Net of `cbis-1024`, fold 0, `best.pt` (epoch 33) |

Everything else is the same: ResNet18, the augmentation, the loss, AdamW at 1e-4
with cosine decay, 60 epochs of 256 images balanced between images with and
without a mass, batches of 4, all layers trained from the first epoch.

## Code

- **Training:** notebook `notebooks/11_colab_training.ipynb` with
  `DATASET = "inbreast"`, `FOLDS = [0, 1, 2, 3, 4]`, `MAX_MINUTES = 180`,
  `RUN_NAME = "inbreast-from-cbis"`,
  `INIT_FROM = "mammo_seg_runs/cbis-1024/fold_0/best.pt"` and
  `TRAIN_OPTIONS = "--epochs 60 --batch-size 4 --samples-per-epoch 256 --patience 0 --encoder resnet18"`,
  commit `5c0beecaf4ee9c091ec4a2ee35e16f955edc3681`, one session.
- **Evaluation:** notebook `notebooks/12_colab_evaluation.ipynb` with
  `EVALUATE_OPTIONS = "--details"`, the same commit.
- **Training-set check:** fold 0's `last.pt` scored on its own 256 training images
  with the default settings grid, from a script run in a cell added to notebook 12
  after its setup: it calls `evaluate_folds` with `fold_ids` replaced by one that
  returns the training ids as the validation ids, and writes the evaluation and
  details formats with `"on": "training"`, like the check of the baseline. Same
  session and commit.
- **Data on Colab:** the archive of `data/processed/inbreast/`. Its `splits.json`
  (sha256 `59a3a718…`) differs from the baseline's (`6d8d4120…`) only in format:
  the splits file became version 3 with the common manifest schema. Every one of
  the 86 patients is in the same fold, and the same 326 validation images are
  scored. `preprocessing.json` (sha256 `2126cdc5…`) is the baseline's.
- **Checkpoint scored:** `last.pt` (epoch 60) of each fold, as for the baseline.
- **Comparison:** `mammo-seg compare` of the baseline's details and these, at
  threshold 0.4 and 10 mm² (the setting chosen for the 1024 run), 2000 draws of
  the 86 patients, seed 0, 95% intervals.

## Where it ran

- **Training:** Google Colab, T4 GPU, 2026-10-31. 25 s per epoch (median), about
  2 hours 7 minutes for the 5 folds, a peak of 2.9 GB of GPU memory. (The baseline
  took 20 s per epoch on a Kaggle T4.)
- **Evaluation and training-set check:** Google Colab, T4 GPU, 2026-10-31.

## Files

```
whole/code_version.txt                        # the commit of the training session
whole/evaluation_code_version.txt             # the commit of the evaluation
whole/evaluation_validation.json              # the evaluation, all settings
whole/evaluation_validation_details.json      # every validation image with its results
whole/evaluation_training_fold0.json          # the training-set check, all settings
whole/evaluation_training_fold0_details.json  # every fold 0 training image with its results
whole/comparison_with_resnet18.json           # the output of mammo-seg compare
whole/fold_<k>/history.jsonl                  # the training curves
```

## Results

326 validation images over the 5 folds: 87 with a mass (95 masses, 24 under
100 mm²), 239 without. A mass is found at IoU ≥ 0.1.

At threshold 0.4 and 10 mm² (this run minus the baseline, with the paired 95%
interval of the difference):

| | ResNet18 baseline | from CBIS-DDSM | difference |
|---|---|---|---|
| masses found | 0.716 | 0.653 | −0.063 [−0.165, +0.022] |
| masses under 100 mm² found | 0.250 | 0.208 | −0.042 [−0.214, +0.133] |
| masses of 100 mm² or more found | 0.873 | 0.803 | −0.070 [−0.188, +0.029] |
| Dice, images with a mass | 0.645 | 0.606 | −0.039 [−0.122, +0.032] |
| sensitivity, images with a mass | 0.619 | 0.595 | −0.025 [−0.101, +0.042] |
| precision, images with a mass | 0.853 | 0.909 | **+0.057 [+0.008, +0.115]** |
| false alarms, images without a mass | 0.084 | 0.046 | **−0.038 [−0.073, −0.004]** |
| false regions per image | 0.126 | 0.058 | **−0.067 [−0.108, −0.030]** |

**Fewer false positives, and no more masses found.** The false alarms, the false
regions and the precision differ by more than the luck of which patients were
scored; nothing on the masses does, and every difference there points down.

The threshold changes almost nothing: from 0.3 to 0.7 the masses found stay at
0.653 and the Dice with a mass at 0.604 to 0.606. The model's probabilities are
nearly all close to 0 or 1, so a lower threshold does not bring the missed masses
back: 32 of its 33 missed masses have a peak probability under 0.05 (22 of the
baseline's 27).

### Per lesion and per image

| | ResNet18 baseline | from CBIS-DDSM |
|---|---|---|
| masses found (true positives) | 68 | 62 |
| masses missed (false negatives) | 27 | 33 |
| false regions (false positives) | 41 | 19 |
| lesion precision, found / (found + false regions) | 0.624 | 0.765 |
| images with a mass where a mass is found (of 87) | 67 | 61 |
| images with a mass with any prediction | 73 | 61 |
| images without a mass with any prediction (of 239) | 20 | 11 |
| image-level specificity | 0.916 | 0.954 |

There are no lesion-level true negatives, so no lesion-level specificity; the
masses found against the false regions per image take its place. Mass by mass:
57 found by both runs, 11 by the baseline only, 5 by this run only, 22 by
neither. False-alarm images: 7 in both, 13 in the baseline only, 4 in this run
only. On every image with a mass where this run predicts anything, it finds a
mass.

### Per fold and by subgroup

Dice on the images with a mass per fold: 0.649 ± 0.073 for the baseline and
0.606 ± 0.079 for this run (mean ± SD over the folds); this run is lower in all 5
folds, by 0.003 to 0.160 (fold 0). The false alarms are lower in 4 folds and the
same in one.

By subgroup at the same setting, this run against the baseline: Dice 0.651
against 0.709 in fatty breasts (62 images with a mass) and 0.493 against 0.486 in
dense ones (25); false alarms 0.019 against 0.057 on normal images (53), 0.037
against 0.085 on images with a benign finding (164), and 0.182 against 0.136 on
suspicious ones (22).

### Training

- **A fast start:** a Dice with a mass of 0.64 to 0.76 at epoch 1, against 0.00 to
  0.14 for the baseline.
- **An unsteady middle:** in 4 of the 5 folds the validation switches between a
  cautious state (nothing predicted on most images, a Dice with a mass under 0.3)
  and a looser one, up to about epoch 30 (fold 2: 15 such epochs). In fold 1 the
  training loss jumps from 0.29 to 0.53 at epochs 24 to 29 before recovering.
  This suggests that 1e-4 is a high learning rate for fine-tuning a model that is
  already trained.
- **Best against last epochs:** a mean Dice with a mass over the folds of 0.614 at
  the best epochs (chosen by the Dice over all images) and 0.612 over the last 10.
- **The training losses of the two runs do not compare:** the soft Dice in the
  loss penalizes small probabilities spread over an image without a mass, so a
  model whose outputs are nearly 0 or 1, as this one's, has a much lower loss
  (0.15 to 0.23 at epoch 60, against about 0.5) without fitting its training
  images better (see below). The loss function is the same in both runs.

## Training-set check

At threshold 0.4 and 10 mm², the fold 0 models on their training and validation
images:

| | baseline, training | baseline, validation | this run, training | this run, validation |
|---|---|---|---|---|
| masses under 100 mm² found | 20 of 20 | 2 of 4 | 9 of 20 | 2 of 4 |
| masses of 100 mm² or more found | 60 of 60 | 10 of 11 | 59 of 60 | 7 of 11 |
| Dice, images with a mass | 0.949 | 0.692 | 0.806 | 0.532 |
| false alarms, images without a mass | 9 of 184 | 6 of 55 | 5 of 184 | 2 of 55 |
| false regions per image | 0.039 | 0.186 | 0.031 | 0.043 |

This run fits its training images less than the baseline: it misses 11 of its 20
small training masses, which the baseline all found. But the gap between training
and validation Dice is the same (0.27 against 0.26), so it does not generalize
better: it is more cautious on every image, not less overfitted.

## Conclusion

Starting from the CBIS-DDSM model does not find more masses. It gives a more
cautious model: about half the false regions and false alarms (both beyond chance)
at the cost of 6 of 95 masses found (within chance), and the threshold cannot
trade one back for the other. For finding masses it is not better than the
ImageNet start; the ResNet18 baseline stays the reference.

A likely reason, not shown directly by these numbers: the masses of CBIS-DDSM are
larger relative to the image and loosely outlined, and it has no images without a
mass, so the pretrained model responds only to clear masses, and fine-tuning all
layers for 60 epochs at 1e-4 does not fully undo it while making the training
unsteady.

Next: the same run with the encoder frozen for the first 10 epochs
(`--freeze-encoder-epochs 10`), which tells whether the caution comes from the
pretrained features (it would then grow) or from fine-tuning every layer at once
(it would then shrink); and possibly a lower learning rate for fine-tuning.

## Data

Only aggregate scores are here; no images or masks of either dataset. The
pretraining used CBIS-DDSM, which must be cited in any report of these results;
see `2026-10-29_cbis_ddsm_pretraining/README.md` for the citations.
