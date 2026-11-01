# INbreast from the CBIS-DDSM model, encoder frozen first

## What and why

Starting INbreast from the CBIS-DDSM model with every layer trained from the first
epoch (`2026-10-31_inbreast_from_cbis`) gave a more cautious model than the
ResNet18 baseline (`2026-10-28_whole_1024_resnet18`): half the false regions, but
6 fewer masses found. This run asks where that caution comes from: the same run
with the encoder frozen for the first 10 epochs, so that only the decoder adapts to
INbreast before the pretrained features change. If the caution came from the
CBIS-DDSM features, it would grow; if it came from fine-tuning every layer at
once, it would shrink.

| | ResNet18 baseline | from CBIS | this run |
|---|---|---|---|
| starting weights | encoder pretrained on ImageNet, decoder random | the whole U-Net of `cbis-1024`, fold 0, `best.pt` (epoch 33) | the same as from CBIS |
| encoder | trained from epoch 1 | trained from epoch 1 | frozen for epochs 1 to 10, trained from epoch 11 |

Everything else is the same: ResNet18, the augmentation, the loss, AdamW at 1e-4
with cosine decay, 60 epochs of 256 images balanced between images with and
without a mass, batches of 4.

## Code

- **Training:** notebook `notebooks/05_kaggle_training.ipynb` with
  `FOLDS = [0, 1, 2, 3, 4]`, `MAX_MINUTES = 180`, `PREPROCESS_OPTIONS = ""` and
  `TRAIN_OPTIONS = "--epochs 60 --batch-size 4 --samples-per-epoch 256 --patience 0 --encoder resnet18 --freeze-encoder-epochs 10 --init-from <the checkpoint>"`,
  commit `7ea7756ca1c3b8e61c70aded27146a87eddce3b4`, one session. The checkpoint is
  `cbis-1024/fold_0/best.pt`, downloaded from Drive and attached as a private
  Kaggle dataset (`mammo-seg-cbis-1024`). Every fold's log says it started from
  those weights, and its history marks epochs 1 to 10 as frozen.
- **Evaluation:** notebook `notebooks/06_kaggle_evaluation.ipynb` with
  `EVALUATE_OPTIONS = "--details"`, the same commit.
- **Training-set check:** fold 0's `last.pt` scored on its own 256 training images
  with the default settings grid, as for the other two runs: a script in cells
  added to notebook 06 calls `evaluate_folds` with `fold_ids` replaced by one that
  returns the training ids as the validation ids, and writes the evaluation and
  details formats with `"on": "training"`. On Kaggle the runs are a read-only
  input, so it writes to `/kaggle/working`. Same session and commit as the
  evaluation.
- **Data:** rebuilt from the raw INbreast dataset on Kaggle by the notebooks.
  `splits.json` sha256 `a97a9b9d…` and `preprocessing.json` sha256 `91243aa7…`
  differ from the baseline's (`6d8d4120…`, `2126cdc5…`) and from the Colab
  archive's (`59a3a718…`, `2126cdc5…`) only in format, checked locally with the
  same commit: the manifest is the same file; the splits have the same folds,
  test patients and excluded images, plus two fields added for CBIS-DDSM
  (`official_test`, `small_mass_px`); and the local `preprocessing.json`, read and
  written again by this commit, gives `91243aa7…`, so every image has the same
  crop box, size and flip. The same 326 validation images are scored.
- **Checkpoint scored:** `last.pt` (epoch 60) of each fold, as for the other runs.
- **Comparisons:** `mammo-seg compare` of the baseline's details and these, and of
  the from-CBIS run's details and these, at threshold 0.4 and 10 mm² (the setting
  chosen for the 1024 run), 2000 draws of the 86 patients, seed 0, 95% intervals.

## Where it ran

- **Training:** Kaggle, T4 GPU, 2026-10-31, the saved version of 19:06 UTC of the
  notebook `notebook1086ad7535`. 16 s per frozen epoch and 20 s per epoch after
  (medians), 1 hour 40 minutes for the 5 folds after a 12-minute setup, a peak of
  3.0 GB of GPU memory. A first version, in the notebook
  `mammo-seg-inbreast-from-cbis-freeze`, gave `--init-from` the folder of the
  Kaggle dataset instead of the file and stopped before training.
- **Evaluation and training-set check:** Kaggle, T4 GPU, 2026-10-31, the notebook
  `mammo-seg-inbreast-from-cbis-freeze-evaluation`. The files were downloaded with
  `kaggle kernels output`.

## Files

```
whole/code_version.txt                        # the commit of the training session
whole/evaluation_code_version.txt             # the commit of the evaluation
whole/evaluation_validation.json              # the evaluation, all settings
whole/evaluation_validation_details.json      # every validation image with its results
whole/evaluation_training_fold0.json          # the training-set check, all settings
whole/evaluation_training_fold0_details.json  # every fold 0 training image with its results
whole/comparison_with_resnet18.json           # mammo-seg compare against the baseline
whole/comparison_with_from_cbis.json          # mammo-seg compare against the from-CBIS run
whole/fold_<k>/history.jsonl                  # the training curves
```

## Results

326 validation images over the 5 folds: 87 with a mass (95 masses, 24 under
100 mm²), 239 without. A mass is found at IoU ≥ 0.1.

At threshold 0.4 and 10 mm²; the differences are this run minus the other, with
the paired 95% interval:

| | ResNet18 baseline | from CBIS | this run | minus baseline | minus from CBIS |
|---|---|---|---|---|---|
| masses found | 0.716 | 0.653 | 0.716 | +0.000 [−0.111, +0.098] | +0.063 [−0.013, +0.144] |
| masses under 100 mm² found | 0.250 | 0.208 | 0.292 | +0.042 [−0.172, +0.250] | +0.083 [+0.000, +0.200] |
| masses of 100 mm² or more found | 0.873 | 0.803 | 0.859 | −0.014 [−0.130, +0.095] | +0.056 [−0.035, +0.154] |
| Dice, images with a mass | 0.645 | 0.606 | 0.649 | +0.004 [−0.079, +0.080] | +0.044 [−0.024, +0.110] |
| sensitivity, images with a mass | 0.619 | 0.595 | 0.648 | +0.028 [−0.053, +0.104] | +0.053 [−0.015, +0.121] |
| precision, images with a mass | 0.853 | 0.909 | 0.893 | +0.040 [−0.001, +0.091] | −0.016 [−0.040, +0.006] |
| false alarms, images without a mass | 0.084 | 0.046 | 0.071 | −0.013 [−0.056, +0.031] | +0.025 [−0.009, +0.059] |
| false regions per image | 0.126 | 0.058 | 0.071 | **−0.055 [−0.107, −0.006]** | +0.012 [−0.017, +0.044] |

**As many masses as the baseline, with fewer false regions.** Against the baseline,
only the false regions differ by more than the luck of which patients were scored.
Against the from-CBIS run nothing does, but the run moves back toward the
baseline, or past it, on every measure: more masses found, a higher Dice and sensitivity, more
false alarms. The interval for the small masses only touches 0 (7 of 24 found
against 5).

The threshold changes almost nothing: from 0.3 to 0.7 the masses found stay at
0.716 and the Dice with a mass at 0.647 to 0.650. All 27 missed masses have a peak
probability under 0.05 (22 of the baseline's 27).

### Per lesion and per image

| | ResNet18 baseline | from CBIS | this run |
|---|---|---|---|
| masses found (true positives) | 68 | 62 | 68 |
| masses missed (false negatives) | 27 | 33 | 27 |
| false regions (false positives) | 41 | 19 | 23 |
| lesion precision, found / (found + false regions) | 0.624 | 0.765 | 0.747 |
| images with a mass where a mass is found (of 87) | 67 | 61 | 67 |
| images with a mass with any prediction | 73 | 61 | 67 |
| images without a mass with any prediction (of 239) | 20 | 11 | 17 |
| image-level specificity | 0.916 | 0.954 | 0.929 |

There are no lesion-level true negatives, so no lesion-level specificity; the
masses found against the false regions per image take its place. Mass by mass
against the baseline: 60 found by both, 8 by the baseline only, 8 by this run
only, 19 by neither; false-alarm images: 6 in both, 14 in the baseline only, 11 in
this run only. Against the from-CBIS run: 59 masses found by both, 3 by it only, 9
by this run only, 24 by neither; false-alarm images: 6 in both, 5 in it only, 11
in this run only. As in the from-CBIS run, on every image with a mass where this
run predicts anything, it finds a mass.

### Per fold and by subgroup

Dice on the images with a mass per fold (mean ± SD over the folds): 0.649 ± 0.073
for the baseline, 0.606 ± 0.079 for the from-CBIS run and 0.646 ± 0.102 for this
run. Against the baseline it is higher in 2 folds (fold 1: +0.119) and lower in 3
(fold 0: −0.096); the false alarms are lower in 3 folds and higher in 2.

By subgroup at the same setting, this run against the baseline: Dice 0.707 against
0.709 in fatty breasts (62 images with a mass) and 0.507 against 0.486 in dense
ones (25); false alarms 0.057 against 0.057 on normal images (53), 0.067 against
0.085 on images with a benign finding (164), and 0.136 against 0.136 on suspicious
ones (22).

### Training

- **The frozen epochs:** a Dice with a mass of 0.54 to 0.84 at epoch 1 and 0.58 to
  0.82 at epoch 10, with false alarms of 0.13 to 0.23 at epoch 10.
- **Unfreezing shakes the model:** at epoch 11 the false alarms rise to 0.30 to
  0.55 in every fold, and up to 0.40 to 0.68 in later epochs, before falling to
  0.02 to 0.15 at epoch 60.
- **A steadier middle than the from-CBIS run:** 4 epochs over the 5 folds with a
  Dice with a mass under 0.3 (none after epoch 40), against 27 in the from-CBIS run
  (15 in fold 2). The baseline has 9, all in its first 4 epochs.
- **Best against last epochs:** a mean Dice with a mass over the folds of 0.643 at
  the best epochs (chosen by the Dice over all images) and 0.630 over the last 10.
  Fold 4's best epoch is epoch 9, still frozen.
- **The training losses** (0.13 to 0.22 at epoch 60) do not compare with the
  baseline's, for the reason given in `2026-10-31_inbreast_from_cbis`.

## Training-set check

At threshold 0.4 and 10 mm², the fold 0 models on their training and validation
images:

| | baseline, training | baseline, validation | from CBIS, training | from CBIS, validation | this run, training | this run, validation |
|---|---|---|---|---|---|---|
| masses under 100 mm² found | 20 of 20 | 2 of 4 | 9 of 20 | 2 of 4 | 5 of 20 | 2 of 4 |
| masses of 100 mm² or more found | 60 of 60 | 10 of 11 | 59 of 60 | 7 of 11 | 57 of 60 | 9 of 11 |
| Dice, images with a mass | 0.949 | 0.692 | 0.806 | 0.532 | 0.768 | 0.596 |
| false alarms, images without a mass | 9 of 184 | 6 of 55 | 5 of 184 | 2 of 55 | 5 of 184 | 4 of 55 |
| false regions per image | 0.039 | 0.186 | 0.031 | 0.043 | 0.027 | 0.057 |

This run fits its training images the least: it misses 15 of its 20 small training
masses. The gap between training and validation Dice is smaller (0.17 against 0.26
and 0.27), but it comes from the lower training Dice, not from a higher validation
one, and it rests on fold 0 alone.

## Conclusion

Freezing the encoder for the first 10 epochs removes most of the caution of the
from-CBIS run: it finds as many masses as the baseline (68 of 95), with the same
Dice and fewer false regions (23 against 41, beyond chance), and its false alarms
(17 images against 20) are within chance. So the caution came from fine-tuning
every layer at once, not from the CBIS-DDSM features alone.

It is not a better model for finding masses: no measure of the masses differs
from the baseline beyond chance, and the threshold still cannot trade the fewer
false regions for more masses, because the missed masses have probabilities near
0. On the point estimates it is the best INbreast run so far, by a small margin.

Next: a lower learning rate for fine-tuning, because the false alarms still jump
when the encoder is unfrozen (for example `--learning-rate 3e-5`, which also slows
the frozen epochs, as there is one learning rate for all layers).

## Data

Only aggregate scores are here; no images or masks of either dataset. The
pretraining used CBIS-DDSM, which must be cited in any report of these results;
see `2026-10-29_cbis_ddsm_pretraining/README.md` for the citations.
