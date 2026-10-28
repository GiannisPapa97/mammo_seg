# A smaller encoder: ResNet18

## What and why

The training-set check of `2026-10-24_whole_vs_patches` shows that the whole
design at 1024 overfits, and stronger augmentation with weight decay did not help
(see `2026-10-27_whole_1024_aug`). This run asks whether a smaller encoder does:
the same training as the whole design at 1024, with a ResNet18 encoder instead of
ResNet34, both pretrained on ImageNet.

| | 1024 | this run |
|---|---|---|
| encoder | ResNet34 | ResNet18 |
| parameters, whole U-Net | 24.4 M | 14.3 M |
| parameters, encoder | 21.3 M | 11.2 M |

Everything else is the same: the augmentation, the loss, the learning rate and its
schedule, 60 epochs of 256 images, batches of 4.

## Code

- **Training:** notebook `notebooks/05_kaggle_training.ipynb` with
  `FOLDS = [0, 1, 2, 3, 4]`, `MAX_MINUTES = 95`, `PREPROCESS_OPTIONS = ""` and
  `TRAIN_OPTIONS = "--epochs 60 --batch-size 4 --samples-per-epoch 256 --patience 0 --encoder resnet18"`,
  commit `52000891e4c49e6276a03fdfd7b62b09ffcaea34`. The session ended during
  fold 4, after epoch 31; epochs 32 to 60 of fold 4 were resumed from its
  `last.pt` with `notebooks/11_colab_training.ipynb`, `FOLDS = [4]`,
  `MAX_MINUTES = 60` and the same `TRAIN_OPTIONS`, commit
  `ed494473d5b43713ff41a1a6d936adc83e48f153`. Between the two commits the code
  under `src/` changed only in the default data paths (one folder per dataset),
  which the notebooks do not use.
- **Evaluation with per-image details:** commit
  `ed494473d5b43713ff41a1a6d936adc83e48f153`, notebook
  `notebooks/12_colab_evaluation.ipynb` with `EVALUATE_OPTIONS = "--details"`.
- **Data on Colab:** the archive of `data/processed/inbreast/`; its `splits.json`
  (sha256 `6d8d4120…`) and `preprocessing.json` (sha256 `2126cdc5…`) are the
  same as those Kaggle rebuilt, so the folds and the images are the same.
- **Splits:** the stratified splits (version 2), the same folds as the 1024 run.
- **Checkpoint scored:** `last.pt` (epoch 60) of each fold.
- **Comparison:** `mammo-seg compare` of the 1024 details of
  `2026-10-24_whole_vs_patches` and these, 2000 draws of the 86 patients, seed 0,
  95% intervals.

## Where it ran

- **Training, folds 0 to 3 and epochs 1 to 31 of fold 4:** Kaggle, T4 GPU,
  2026-10-27, `giannispapatrechas/whole-1024-resnet18`. 20 s per epoch, about
  20 minutes per fold, with a peak of 2.9 GB of GPU memory reserved.
- **Training, epochs 32 to 60 of fold 4:** Google Colab, T4 GPU, 2026-10-27.
  23 s per epoch, 11 minutes. The first run of the Colab notebooks; the resume
  continued the learning-rate schedule, and the first 31 epochs of the history
  are those of Kaggle.
- **Evaluation:** Google Colab, T4 GPU, 2026-10-28.

## Files

```
whole/evaluation_validation.json            # the evaluation, all settings
whole/evaluation_validation_details.json    # every validation image with its results
whole/evaluation_code_version.txt           # the commit of the evaluation run
whole/code_version.txt                      # the commits of the training sessions
whole/comparison_with_1024.json             # the output of mammo-seg compare
whole/evaluation_training_fold0.json        # the training-set check, all settings
whole/evaluation_training_fold0_details.json  # every fold 0 training image with its results
whole/fold_<k>/history.jsonl                # the training curves
```

## Results

326 validation images over the 5 folds: 87 with a mass (95 masses, 24 under
100 mm²), 239 without. A mass is found at IoU ≥ 0.1.

At threshold 0.4 and 10 mm², the setting chosen for 1024 (ResNet18 minus 1024,
with the paired 95% interval of the difference):

| | 1024 | ResNet18 | ResNet18 minus 1024 |
|---|---|---|---|
| masses found | 0.716 | 0.716 | +0.000 [−0.075, +0.083] |
| masses under 100 mm² found | 0.333 | 0.250 | −0.083 [−0.286, +0.114] |
| masses of 100 mm² or more found | 0.845 | 0.873 | +0.028 [−0.039, +0.101] |
| Dice, images with a mass | 0.648 | 0.645 | −0.003 [−0.067, +0.061] |
| sensitivity, images with a mass | 0.644 | 0.619 | −0.025 [−0.085, +0.034] |
| precision, images with a mass | 0.857 | 0.853 | −0.005 [−0.056, +0.048] |
| false alarms, images without a mass | 0.113 | 0.084 | −0.029 [−0.071, +0.013] |
| false regions per image | 0.12 | 0.13 | +0.003 [−0.048, +0.054] |

No interval excludes 0.

Dice on the images with a mass per fold: 0.647 ± 0.088 at 1024 and
0.649 ± 0.073 with ResNet18 (mean ± SD over the folds); ResNet18 is higher in 2
of 5 folds. By subgroup at the same setting, ResNet18 against 1024: Dice 0.709
against 0.717 in fatty breasts (62 images with a mass) and 0.486 against 0.478 in
dense ones (25); false alarms 0.057 against 0.075 on normal images (53), 0.085
against 0.122 on images with a benign finding (164) and 0.136 for both on
suspicious ones (22).

The best pooled setting of ResNet18, 0.3 and 10 mm², finds 0.726 of the masses
with false alarms of 0.084 and 0.12 false regions per image.

Of the 24 small masses, 4 are found by both runs, 4 by 1024 only and 2 by
ResNet18 only. Of the 71 large masses, 58 are found by both, 2 by 1024 only and
4 by ResNet18 only.

**The best epochs are not better than the last ones, as for ResNet34.** The best
epochs by the selection metric (Dice over all validation images) have a mean Dice
with a mass of 0.641 over the folds, the last 10 epochs of each fold 0.650 (0.617
and 0.647 for ResNet34), from the training histories. The best epoch is chosen on
its own validation fold, so even this comparison favours it; `best.pt` against
`last.pt` at the chosen setting is still to be evaluated.

## Training-set check

**Question:** does the smaller encoder memorize its training images less than
ResNet34, as the training-set check of `2026-10-24_whole_vs_patches` found for
ResNet34?

**How:** as in that check: this run's fold 0 `last.pt` scored on its own 256
training images, in the details format, from a cell added to
`notebooks/12_colab_evaluation.ipynb` after its setup (the code of the script of
the earlier checks, on the GPU). Google Colab, T4 GPU, 2026-10-28, 370 s. The
package installed from `main`; the code under `src/` is the same as in commit
`ed494473d5b43713ff41a1a6d936adc83e48f153` (unchanged up to
`694ee9c9e52076c9f53766a7dc124aa57dae9eb6`). The local laptop (3.8 GB of memory)
ran out of memory on this check.

At threshold 0.4 and 10 mm², the fold 0 models on their training and validation
images:

| | 1024, training | 1024, validation | ResNet18, training | ResNet18, validation |
|---|---|---|---|---|
| masses under 100 mm² found | 19 of 20 | 1 of 4 | 20 of 20 | 2 of 4 |
| masses of 100 mm² or more found | 60 of 60 | 9 of 11 | 60 of 60 | 10 of 11 |
| Dice, images with a mass | 0.946 | 0.579 | 0.949 | 0.692 |
| false alarms, images without a mass | 11 of 184 | 8 of 55 | 9 of 184 | 6 of 55 |
| false regions per image | 0.047 | 0.143 | 0.039 | 0.186 |

ResNet18 memorizes its training images as completely as ResNet34: all 80 masses
found, a Dice of 0.949 with a mass, and fewer false alarms than on validation.
The gap between training and validation Dice in fold 0 is 0.26 against 0.37, but
fold 0 is a fold where ResNet18 did better on validation (0.579 to 0.692); over
the 5 folds its validation Dice is the same as ResNet34's, so the narrower gap
does not hold beyond this fold. Both small masses missed on validation got no
response at all (a peak probability under 0.05).

## Conclusion

ResNet18 is as good as ResNet34 on every metric: the same 68 of 95 masses found,
the same Dice on the images with a mass, and intervals of the differences that
are centred near 0. Its fewer false alarms (0.084 against 0.113) are within the
luck of which patients were scored. The small masses stay the weak point with
either encoder (6 or 8 of 24 found), and the two encoders find mostly the same
large masses.

So a smaller encoder neither helps nor hurts: the overfitting is not a matter of
the size of the encoder, and the training-set check confirms that ResNet18 still
fits its training images almost perfectly. ResNet18 costs about 15% less time per epoch on a
Kaggle T4 (20 s against 24 s) and has 40% fewer parameters, so it is the cheaper choice for the next
experiments; the 1024 run with ResNet34 stays the reference, since nothing beat
it.

Next: more data, by pretraining on CBIS-DDSM; and `best.pt` against `last.pt`.
