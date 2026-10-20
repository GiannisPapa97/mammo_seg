# Training

How the model is trained and judged. The data side (manifest, splits, masks, preprocessing, dataset and loaders) is in [data-pipeline.md](data-pipeline.md). This document grows as the model and the training loop are added.

Code: `src/mammo_seg/engine/`.

## 1. Metrics

`image_scores(probabilities, masks, valid)` scores every image of a batch; `summarize(scores)` turns the scores of a validation or test set into one summary. All tensors are `[B, 1, H, W]`: the predicted probabilities (after a sigmoid), the 0/1 masks and the valid pixels from the dataset.

### What is measured

For each image, on its valid pixels only (the padding never counts), with a pixel predicted as mass when its probability is at least 0.5:

| Metric | Formula | Meaning |
|---|---|---|
| Dice | 2 TP / (2 TP + FP + FN) | Overlap of prediction and mask; 1 is perfect |
| IoU | TP / (TP + FP + FN) | Overlap, stricter than Dice |
| Precision | TP / (TP + FP) | Of the predicted pixels, how many are mass |
| Sensitivity | TP / (TP + FN) | Of the mass pixels, how many are found |

TP, FP and FN count the pixels that are correctly predicted as mass, wrongly predicted as mass, and missed.

### Images without a mass or without a prediction

| Case | Dice, IoU | Precision | Sensitivity |
|---|---|---|---|
| No mass, nothing predicted | 1 (a correct "nothing here") | undefined | undefined |
| No mass, something predicted (a false alarm) | 0 | 0 | undefined |
| Mass, nothing predicted (a miss) | 0 | undefined | 0 |

Undefined values are NaN and are left out of the means. On images without a mass, the **false-alarm rate** is the share of images where anything is predicted.

### Averaging: per image, then the mean

A validation set is summarized by computing each metric per image and taking the mean over the images, so every image counts once. Pooling the pixels of all images into one giant image instead would let large masses dominate. For example, with a small mass of 1,000 pixels of which 100 are found (Dice about 0.10) and a large one of 50,000 pixels of which 45,000 are found (Dice about 0.90):

- per image, then the mean: (0.10 + 0.90) / 2 = **0.50**
- pooled: about **0.90**, although the small mass is almost entirely missed

The INbreast masses range from about 3,000 to 750,000 pixels at native size, and the small ones are the hard ones, so the per-image mean is used. It is not a per-patient mean: a patient with more images counts more.

### The summary

`summarize` reports the same metrics for three groups, so good scores on the lesions cannot hide false alarms on healthy images, or the other way round:

- `with_mass`: the images with a mass. Dice, IoU, precision and sensitivity on these say how well the lesions are found and outlined.
- `without_mass`: the images without one. Their Dice is the share without any prediction, and the false-alarm rate is its complement.
- `all_images`: everything together.

## 2. Loss

`DiceBCELoss(dice_weight=1, bce_weight=1)` takes the raw model outputs (logits), the masks and the valid pixels, and returns `dice_weight * (1 - soft Dice) + bce_weight * binary cross-entropy`:

- **Soft Dice** uses the probabilities instead of a thresholded prediction, so it has gradients. It is computed per image and averaged, like the metric, and measures overlap, so the background, about 99% of a positive image, does not swamp it. A smoothing term of 1 keeps it defined on images without a mass: predicting nothing there gives no loss, matching the metric.
- **Binary cross-entropy** gives a signal on every pixel, which keeps training stable early on, when the predicted masses are still empty.
- Both are computed on valid pixels only: the padding gives neither loss nor gradients.

### Sanity check on real data

A small convolutional network trained on two real positive images (at a quarter of the resolution) brings the loss from 1.63 to 0.36 in 400 steps, and the Dice on those images from 0 to 0.83: the loss drives the model to find the masses. On a batch of one positive and three negative images, a weaker network first settles on predicting nothing, which scores the three negatives perfectly. That is why training balances positive and negative images (the sampler in `loaders.py`) and the summary reports images with a mass separately.

## Next steps

- The model: a U-Net with a pretrained encoder and one input channel.
- The training loop: training per fold, validation with these metrics, checkpoints with resuming, and the device choice.
