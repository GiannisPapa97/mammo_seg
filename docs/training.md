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

## 3. The model

`build_model(ModelSettings())` builds a network from `segmentation_models_pytorch` (smp): an architecture on top of an encoder, with one input channel and one output channel of raw logits (apply a sigmoid for probabilities). The default is a U-Net with a ResNet-34 encoder pretrained on ImageNet: 24.4 million parameters, and a forward pass on the full 1024 x 800 canvas takes about 2.6 s for one image on a laptop CPU.

| Setting | Default | Meaning |
|---|---|---|
| `architecture` | `unet` | `unet`, `unetplusplus`, `deeplabv3`, `deeplabv3plus` or `fpn` |
| `encoder` | `resnet34` | Any smp encoder name, for example `resnet18`, `efficientnet-b0` or a timm model as `tu-<name>` |
| `encoder_weights` | `imagenet` | `imagenet` for pretrained weights, `None` for random ones |

smp adapts the pretrained first layer to a grayscale image by summing its weights over the colour channels. The input sides do not have to divide by 32 (smp handles it), but training and validation use the 1024 x 800 canvas and inference pads to a multiple of 32 (`pad_to_multiple`), like the data loaders.

### Free cloud GPUs (Kaggle, Colab)

The model code runs the same on a CPU and on a GPU, so tests and smoke runs happen locally and real training on a cloud notebook:

1. **Install without touching PyTorch.** The notebook comes with a GPU build of PyTorch. Install the package with `pip install ".[cloud]"` (see the README); a simulated install into an environment that already had PyTorch left it untouched.
2. **The device** is chosen by `resolve_device()`: `auto` picks the GPU. Asking for `cuda` on a notebook without a GPU is an error, not a silent run on the CPU.
3. **Pretrained weights need internet the first time.** They come from the Hugging Face hub and are cached (by default in `~/.cache/huggingface`, or where `HF_HOME` points). On Kaggle, internet is off by default: enable it in the notebook settings. For a notebook without internet, build the model once somewhere with internet, upload that cache folder as a Kaggle dataset and set `HF_HOME` to it (not tried on Kaggle). A failed download raises an error that says so. Models rebuilt from a checkpoint never download anything.
4. **Sessions end without warning**, so training state goes into checkpoints (below), saved into the notebook's output folder and resumed in the next session.

### Checkpoints

`save_checkpoint(path, model, model_settings, preprocessing_settings, training_state, metadata)` writes one file with the weights, the model settings, the preprocessing settings the model was trained with, and optionally the training state (optimizer, epoch, best Dice) and metadata (such as the fold). It is built for the cloud:

- **Atomic:** it writes to a temporary file and renames it, so a session that dies during a save leaves the last good checkpoint in place.
- **Portable:** it loads onto a CPU whatever device it was saved from (`load_checkpoint(path, device="cpu")`), so a model trained on a GPU can be used on a laptop.
- **Safe:** it is loaded with `weights_only=True`, so a checkpoint downloaded from somewhere cannot run code. Settings are stored as JSON text and everything else is tensors and plain values.
- **Self-contained:** `load_model(path, device)` rebuilds the network without downloading pretrained weights, puts it in evaluation mode and returns it with the preprocessing settings that images must be given at inference.

A checkpoint is about 100 MB of weights for the default model, and about 300 MB with the Adam optimizer state.

## Next steps

- The training loop: training per fold with the sampler and augmentation, validation with the metrics above, checkpoints with resuming (best and last), logging, and a command to run it on a few images locally and on a cloud GPU.
