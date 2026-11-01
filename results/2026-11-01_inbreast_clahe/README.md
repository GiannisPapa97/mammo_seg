# INbreast with a CLAHE channel

## What and why

INbreast has a low contrast: the tissue of a normalized image lies at about 0.27 to
0.43, and several papers enhance the contrast before segmenting. This run asks
whether giving the model the image after CLAHE (contrast-limited adaptive histogram
equalization) as a second input channel finds more masses than the ResNet18
baseline (`2026-10-28_whole_1024_resnet18`): the same training, with only the input
changed.

| | ResNet18 baseline | this run |
|---|---|---|
| input | the normalized image | the normalized image and the same image after CLAHE (clip limit 0.01, 8 × 8 tiles) |
| first layer | ImageNet's, summed over the colour channels | ImageNet's red and green filters, scaled by 3/2 (smp's `in_channels=2`) |

Everything else is the same: the encoder pretrained on ImageNet and the decoder
random, ResNet18, the augmentation (brightness and contrast drawn for each channel
on its own, the gamma shared), the loss, AdamW at 1e-4 with cosine decay, 60 epochs
of 256 images balanced between images with and without a mass, batches of 4, all
layers trained from the first epoch. See `docs/data-pipeline.md` (Contrast
enhancement) for the CLAHE channel and the measurement on the real data that came
before this run: at 0.01, CLAHE stretches the difference between a mass and the
tissue around it, but the texture of that tissue more.

## Code

- **Training:** notebook `notebooks/05_kaggle_training.ipynb` with
  `FOLDS = [0, 1, 2, 3, 4]`, `MAX_MINUTES = 180`, `PREPROCESS_OPTIONS = "--clahe 0.01"`
  and
  `TRAIN_OPTIONS = "--epochs 60 --batch-size 4 --samples-per-epoch 256 --patience 0 --encoder resnet18"`,
  commit `e71e712dee8d7d04925e571481c6de72d5cdd391`, one session. `mammo-seg train`
  sets the model's two input channels from the preprocessing.
- **Evaluation:** notebook `notebooks/06_kaggle_evaluation.ipynb` with
  `EVALUATE_OPTIONS = "--details"` and `PREPROCESS_OPTIONS = "--clahe 0.01"`, the
  same commit.
- **Data:** rebuilt from the raw INbreast dataset on Kaggle by each notebook: 410
  images, 410 CLAHE images. `splits.json` sha256 `a97a9b9d…`, the file of the
  frozen-encoder run (`2026-10-31_inbreast_from_cbis_freeze`), which has the
  baseline's folds. `preprocessing.json` sha256 `e3e0d0f4…` differs from that
  run's (`91243aa7…`) by the `clahe` key; its images have the same crop box, size
  and flip as the local record of the baseline's data, checked locally. The same
  326 validation images are scored.
- **Checkpoint scored:** `last.pt` (epoch 60) of each fold, as for the baseline.
- **Comparisons:** `mammo-seg compare` of the baseline's details and these, and of
  the frozen-encoder run's details and these, at threshold 0.4 and 10 mm² (the
  setting chosen for the 1024 run), 2000 draws of the 86 patients, seed 0, 95%
  intervals.

## Where it ran

- **Training:** Kaggle, T4 GPU, 2026-11-01, the saved version of 14:42 UTC of the
  notebook `mammo-seg-inbreast-clahe`. 22 s per epoch (median; the baseline took
  20 s), about 22 minutes per fold and 1 hour 59 minutes in all, including a
  6-minute setup (the preprocessing with CLAHE took 251 s). A peak of 2.9 GB of GPU
  memory.
- **Evaluation:** Kaggle, T4 GPU, 2026-11-01, the notebook
  `mammo-seg-inbreast-clahe-evaluation`. The version ends in an error from an extra
  cell for the training-set check (see below), after the evaluation had written its
  files. The files were downloaded with `kaggle kernels output`.

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
| masses found | 0.716 | 0.716 | 0.684 | −0.032 [−0.120, +0.051] | −0.032 [−0.143, +0.091] |
| masses under 100 mm² found | 0.250 | 0.292 | 0.292 | +0.042 [−0.100, +0.179] | +0.000 [−0.182, +0.158] |
| masses of 100 mm² or more found | 0.873 | 0.859 | 0.817 | −0.056 [−0.149, +0.038] | −0.042 [−0.180, +0.104] |
| Dice, images with a mass | 0.645 | 0.649 | 0.604 | −0.041 [−0.111, +0.028] | −0.045 [−0.126, +0.045] |
| sensitivity, images with a mass | 0.619 | 0.648 | 0.577 | −0.043 [−0.112, +0.028] | −0.071 [−0.152, +0.014] |
| precision, images with a mass | 0.853 | 0.893 | 0.890 | +0.037 [−0.034, +0.101] | −0.003 [−0.059, +0.040] |
| false alarms, images without a mass | 0.084 | 0.071 | 0.067 | −0.017 [−0.060, +0.024] | −0.004 [−0.045, +0.035] |
| false regions per image | 0.126 | 0.071 | 0.077 | **−0.049 [−0.096, −0.003]** | +0.006 [−0.034, +0.043] |

**No more masses found, and fewer false regions.** Against the baseline, only the
false regions differ by more than the luck of which patients were scored; against
the frozen-encoder run, nothing does. On the masses, the point estimates are lower
than both, except for the small masses (7 of 24 found, against 6 and 7).

The threshold changes almost nothing: from 0.3 to 0.7 the masses found stay at 65
of 95 and the Dice with a mass at 0.600 to 0.606. 23 of the 30 missed masses have a
peak probability under 0.05 (22 of the baseline's 27).

### Per lesion and per image

| | ResNet18 baseline | this run |
|---|---|---|
| masses found (true positives) | 68 | 65 |
| masses missed (false negatives) | 27 | 30 |
| false regions (false positives) | 41 | 25 |
| lesion precision, found / (found + false regions) | 0.624 | 0.722 |
| images with a mass where a mass is found (of 87) | 67 | 64 |
| images with a mass with any prediction | 73 | 66 |
| images without a mass with any prediction (of 239) | 20 | 16 |
| image-level specificity | 0.916 | 0.933 |

There are no lesion-level true negatives, so no lesion-level specificity; the
masses found against the false regions per image take its place. Mass by mass
against the baseline: 60 found by both, 8 by the baseline only, 5 by this run
only, 22 by neither, so 73 of the 95 are found by at least one of the two;
false-alarm images: 7 in both, 13 in the baseline only, 9 in this run only.
Against the frozen-encoder run: 56 masses found by both, 12 by it only, 9 by this
run only, 18 by neither.

### Per fold and by subgroup

Dice on the images with a mass per fold (mean ± SD over the folds): 0.649 ± 0.073
for the baseline and 0.606 ± 0.101 for this run. It is lower in 4 folds, by 0.030
to 0.076 (fold 4), and higher in fold 1 (+0.033). The false alarms are lower in 3
folds, the same in one and higher in fold 4 (0.150 against 0.125).

By breast density at the same setting, this run against the baseline: Dice 0.652
against 0.709 in fatty breasts (ACR 1 or 2, 62 images with a mass) and 0.485
against 0.486 in dense ones (ACR 3 or 4, 25). The dense breasts, where a contrast
enhancement would help most, show no change; the loss is in the fatty ones.

### Training

- **The same course as the baseline:** a Dice with a mass of 0.03 to 0.21 at
  epoch 1 (the baseline: 0.00 to 0.14), and no epoch with a Dice with a mass under
  0.3 after epoch 4 in either run.
- **The training images are fitted as well:** a training loss of 0.50 to 0.57 at
  epoch 60, against 0.51 to 0.57. Both runs start from ImageNet with the same loss,
  so here the losses compare.
- **Best against last epochs:** a mean Dice with a mass over the folds of 0.620 at
  the best epochs (chosen by the Dice over all images, epochs 18 to 38) and 0.615
  over the last 10; the baseline's are 0.641 and 0.650.

## Training-set check

Not run. The cell added to notebook 06 for it imported the package in the notebook's
own Python process, which had loaded numpy before the notebook's `pip install`
replaced it, and it failed on importing scipy. A rerun would cost another GPU
session and cannot change the conclusion; such cells must run their code in a new
process (`!python <script>`), as `mammo-seg evaluate` does.

## Conclusion

The CLAHE channel does not find more masses: 65 of 95 against the baseline's 68,
within chance, with every measure of the masses within chance and most point
estimates lower. Like the CBIS-DDSM start, it gives a more cautious model, with
fewer false regions (25 against 41, beyond chance), and the threshold cannot trade
them for more masses, because the missed masses have probabilities near 0. It does
not help the dense breasts, where it was meant to. The ResNet18 baseline and the
frozen-encoder run stay the references.

So contrast enhancement is not a large lever for this model, and CBIS-DDSM
pretraining with a CLAHE channel is not worth running. The two runs miss partly
different masses (73 of 95 are found by at least one), which bears on an ensemble
of the runs.

Next: a Focal Tversky loss on the baseline setting, the last single-pass experiment
before the two-stage design (detect, crop, segment again).

## Data

Only aggregate scores are here; no images or masks of the dataset.
