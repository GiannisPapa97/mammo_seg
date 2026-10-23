# Training

How the model is trained and judged. The data side (manifest, splits, masks, preprocessing, dataset and loaders) is in [data-pipeline.md](data-pipeline.md).

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

### Free cloud GPUs (Kaggle)

The model code runs the same on a CPU and on a GPU, so tests and smoke runs happen locally and real training on a cloud notebook (only Kaggle was tried):

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

## 4. The training loop

`train_fold(manifest, splits_file, preprocessed_dir, fold, ...)` (`engine/train.py`) trains on every fold of the patient-level splits but one and validates on that one. Each epoch is a pass of training and a pass of validation:

- **Training** draws batches balanced between images with and without a mass (the sampler of `loaders.py`, 50% by default), applies the augmentation, and minimizes the Dice + cross-entropy loss with Adam. The learning rate starts at 1e-4 and follows a cosine curve down to 0 over the epochs.
- **Crops** (`--crop-size N`, off by default): training can use square patches of N pixels instead of whole images, half of them (`--crop-mass-fraction`) placed around a mass when the image has one (see the augmentation in [data-pipeline.md](data-pipeline.md)). The training canvas is then the crop size rounded up to a multiple of 32, so batches are cheaper and a larger `--batch-size` fits. Validation always uses whole images, so crop and whole-image runs are scored the same way. The crop settings are saved in the checkpoint, and a run cannot be resumed with other ones.
- **Batch composition** (`--batch-composition MASS NEG_FROM_MASS_IMAGES NEG_FROM_NORMAL_IMAGES`, for example `4 2 2`, needs `--crop-size`): instead of drawing images at random, every batch has that many patches that contain a mass, negative patches (no mass) from images with a mass, and negative patches from images without one. The batch size is their sum, `--positive-fraction` is not used, and the composition is saved in the checkpoint like the other settings, so a run cannot be resumed with another one. Patch kinds are described in [data-pipeline.md](data-pipeline.md). The negative patches of images with a mass are only mostly negative: when no patch avoids the mass, the one with the least of it is used, which happened for about a fifth of them on fold 0 at 512 px (a median of 0.23% of the patch), so about 55% of all patches hold mass pixels, not 50%. Implemented and tested on synthetic data; not yet run on the real data.
- **Validation** runs on every image of the validation fold once and computes the metrics of section 1 for images with a mass, without one, and all images. All of them are logged every epoch.
- **Mixed precision** (float16) is used on a CUDA GPU only: the network runs in it, the loss in float32, with a gradient scaler. On the CPU it is off whatever is asked.
- **The best epoch** is the one with the highest value of the selection metric, by default the mean Dice over all validation images (`all_images.dice`). Another metric can be chosen, for example `with_mass.dice`; an undefined value (NaN) never wins. Note that on this dataset a model that predicts nothing already scores well on all images, because most images have no mass, so look at the images with a mass too.

| Setting | Default |
|---|---|
| `epochs` | 60 |
| `batch_size` | 4 |
| `learning_rate` | 1e-4 (cosine decay) |
| `positive_fraction` | 0.5 |
| `composition` | none (4 2 2 with `--batch-composition 4 2 2`) |
| `patience` | 15 epochs without improvement, then stop (`0` or `None` never stops early) |
| `selection_metric` | `all_images.dice` |
| `amp` | on (GPU only) |
| `seed` | 0 |

### What a run writes

In `<output-dir>/fold_<k>/`:

| File | Content |
|---|---|
| `history.jsonl` | One JSON line per epoch: learning rate, training loss, duration, all validation metrics (`null` where undefined) |
| `last.pt` | After every epoch: weights, optimizer, learning-rate schedule, history; all that is needed to continue |
| `best.pt` | The weights of the best epoch, with the model and preprocessing settings; load it with `load_model` |

Both are written atomically (section 3).

### Stopping and resuming

A run stops when the epochs are done, when the selection metric has not improved for `patience` epochs, or, with `--max-minutes`, before an epoch that would not end within that time, so a session with a time limit ends with a saved state instead of a lost epoch. At least one epoch always runs, so repeated sessions make progress.

`--resume` continues from `last.pt`. It refuses a checkpoint made with other training settings, model, preprocessing or fold (only the number of workers and mixed precision may differ), so a run is never continued with something else than it started with. The random draws of an epoch (the sampler and the augmentation) depend only on the seed and the epoch number, so a stopped and resumed run gives the same history and weights as one that was never stopped; a test checks this on the CPU.

## 5. Running it

The `mammo-seg` command (installed with the package) runs the data steps and the training; the options are listed by `mammo-seg train --help`.

```bash
mammo-seg train --fold 0 --output-dir runs                   # a GPU if there is one, else the CPU
mammo-seg train --fold 0 --output-dir runs --resume          # continue after a stop
mammo-seg train --epochs 30 --selection-metric with_mass.dice --device cuda
```

The data steps are the same commands as `python -m mammo_seg.data.<step>`: `mammo-seg manifest`, `splits`, `masks` and `preprocess`. Training reads the preprocessing settings and the canvas from the preprocessed data, so they cannot disagree with it.

On Kaggle, use [notebooks/05_kaggle_training.ipynb](../notebooks/05_kaggle_training.ipynb). Its first cell holds the experiment: the folds (`FOLDS`), the time shared by them (`MAX_MINUTES`) and any other `mammo-seg train` options (`TRAIN_OPTIONS`, set to the crop settings that worked, see below; `FOLDS` is `[0]`, use `[0, 1, 2, 3, 4]` for all five). The next cells install the package, rebuild the manifest, splits, masks and preprocessed images from the attached INbreast dataset (a few minutes, so nothing derived has to be uploaded), train each fold with the time that is left, and print the best epoch of each. To continue in a later session, save a version, add its output as an input and set `PREVIOUS_RUNS`; a fold with a `last.pt` is resumed and a finished fold returns at once. The notebook asks for `--device cuda`, so a notebook without a GPU stops with an error.

### Evaluating a trained model

`mammo-seg evaluate` (`engine/evaluate.py`) scores the checkpoint of each fold, `<runs-dir>/fold_<k>/last.pt` by default (`--checkpoint-name best.pt` for the best epoch), on the images that model did not train on: its own validation fold, or with `--test` the held-out test set. Scoring every validation fold with its own model gives out-of-fold predictions for all development images, and the results of the folds are pooled by putting the per-image scores together.

```bash
mammo-seg evaluate --runs-dir runs                       # all five folds, a grid of settings
mammo-seg evaluate --runs-dir runs --folds 0 1 --thresholds 0.4 0.5 --min-areas 0 500
```

Each probability map is turned into a prediction under every combination of a **threshold** (0.3 to 0.7 by default) and a **minimum region size** (0, 100, 300 and 1000 pixels by default): connected regions (8-connectivity) of predicted pixels smaller than the minimum are removed, so a stray spot of a few pixels no longer makes a false alarm. For each setting the table gives the metrics of section 1 on images with a mass, the false-alarm rate with that rule, the mean number of predicted pixels on images without a mass (`FP pixels`), the Dice over all images, and the **balanced Dice**: the mean of the Dice on images with a mass and on images without one, so both kinds count equally. A row for the baseline of **predicting nothing** is always printed, since it scores the share of images without a mass on the all-images Dice. `*` marks the best balanced Dice, and the folds are shown at that setting. Everything is also written to `<runs-dir>/evaluation_validation.json` (or `_test.json`; `--output` changes it).

- A checkpoint trained for another fold, or on images preprocessed with other settings, is refused.
- Choosing the setting on the validation predictions tunes it on data it was scored on, so it is slightly optimistic. Choose one setting for all folds (not one per fold, which would fit 15 to 19 images with a mass each), then score the test set once.
- The command runs on a CPU, but a GPU is much faster. On Kaggle use [notebooks/06_kaggle_evaluation.ipynb](../notebooks/06_kaggle_evaluation.ipynb) with the saved output of the training notebook as an input.

### What has and has not been tried

- Tested on the CPU with synthetic images (CI does the same): the loop, checkpoints, stopping, resuming and the command, including a run through the command that learns a synthetic mass.
- Run once on the real data on a laptop CPU for one epoch with a small encoder, which wrote the checkpoints and the history. That machine has 4 GB of RAM and cannot hold a batch of 4 full-size images, so the default batch size is untested here.
- Run once on Kaggle on a T4 GPU, fold 0, with the default settings (batch size 4, mixed precision): about 24 s per epoch, no memory problem, and early stopping after 25 epochs (the best was epoch 10). **The validation scores are poor**: the training loss fell from 1.57 to 0.60, but the validation Dice over all images stayed at or below 0.045 and, on the images with a mass, at or below 0.17 although the sensitivity there reached 0.89, so most masses were found but the predicted areas were far too large. Every epoch also had a false-alarm rate of 1.000, which counts an image without a mass as soon as one pixel is predicted on it; such an image scores a Dice of 0, which is what holds the all-images Dice down. A second run gave the same numbers to four decimals, so a run is reproducible.
- Three checks on that fold, to find out why:
  - **Training longer does not help.** A run of 60 epochs without early stopping, choosing the epoch by the Dice on images with a mass, stayed between 0.02 and 0.08 on those images from epoch 12 to 60 (the best, 0.165 at epoch 10, was a single high value). The training loss flattened between 0.5 and 0.6.
  - **It is not overfitting.** Scored by the validation code on the training images of the last epoch, the images with a mass had a Dice of 0.095 (validation 0.073), a sensitivity of 0.95 and a precision of 0.054: the predicted area is about 18 times the mass, on data the model had trained on for 60 epochs.
  - **The code can fit masks.** Trained on four training images with a mass, the same model and loss reached a Dice of 0.84 to 0.93 on each of them in 200 steps. This shows that the data path, the model and the loss work together; it does not show that the masks are placed correctly, because a network can also memorize wrong masks. On these four images, the masses cover 0.2 to 0.4 % of the image.
  - A training loss that stays high is not by itself a sign of underfitting here: on an image without a mass the Dice part is `S / (S + 1)` for `S`, the sum of the predicted probabilities over about 800,000 pixels, so it is close to 1 unless the background probabilities are tiny. At step 200 of the four-image check the loss was still 1.17 while the Dice was above 0.84.
- **A higher learning rate (5e-4) does not help.** Fold 0, 60 epochs without early stopping, choosing the epoch by the Dice on images with a mass:
  - Epoch 11 looked like a success: Dice on images with a mass 0.300, over all images 0.718, false alarms 0.125. The next epoch was back at 0.09 with false alarms of 1.000, and from epoch 16 to 60 the sensitivity and the Dice on images with a mass were exactly 0 in every epoch, with false alarms of 1.000. The training loss kept falling to about 0.53.
  - **Epoch 11 was an artifact of the BatchNorm statistics.** The network normalizes with running means and variances in evaluation mode, and here they are far from the statistics of the training batches (probably because they lag behind weights that change fast, at a batch of 4 and this learning rate; this was not tested). With the statistics saved in the checkpoint, the epoch-11 weights predict almost nothing (precision 0.93, sensitivity 0.22, false alarms 0.125). With statistics recomputed from 80 training images (half of them with a mass), the same weights reach a Dice of 0.27 on images with a mass (sensitivity 0.60, precision 0.21), but predict something on every image without a mass (false alarms 1.000, Dice over all images 0.07). Train-mode and evaluation-mode predictions of the same weights differ a lot at both checkpoints, so the validation scores of these runs depend heavily on those statistics.
  - **The last epoch (60) is not a model that predicts nothing.** On eight validation images (four with a mass, four without), it predicted 53,000 to 123,000 pixels per image in evaluation mode and 14,000 to 26,000 in training mode, with none inside a mass. Recomputing the statistics changed nothing. The training at 5e-4 is unstable.
  - These two checks were done with throwaway scripts that are not in the repository.
- **The masks lie on the lesions, as far as a look can tell.** Overlays of the preprocessed masks on the preprocessed images, for six of the 68 training images with a mass of fold 0 (picked at even spacing, not at random): in five, the outline encloses a distinct dense region of the shape of a mass, and the chest wall is on the left, so the flip of right breasts worked. In the sixth, one mask is cut by the left image border and a second one is tiny, so it was inconclusive. This rules out a gross flip or offset, but is not a proof for every image. The masses are small against the whole breast and many dense regions look like them, which is what makes the task hard.
- **Training on crops works.** All five folds, with `--crop-size 512 --batch-size 8 --samples-per-epoch 512 --patience 0 --selection-metric with_mass.dice` (half of the crops around a mass, the rest at random; whole images for validation; 60 epochs, about 19 s per epoch on a T4, about 19 minutes per fold). Whole-image training gave a Dice of 0.02 to 0.08 on the images with a mass of fold 0; with crops no fold collapses, and the late epochs are stable. The mean of the last 10 epochs of each fold (rounded; the all-images row compares with a model that predicts nothing, which scores the share of images without a mass):

  | Fold | Dice, images with a mass | False-alarm rate | Dice, all images | Predict nothing |
  |---|---|---|---|---|
  | 0 | 0.50 | 0.35 | 0.61 | 0.73 |
  | 1 | 0.52 | 0.09 | 0.80 | 0.73 |
  | 2 | 0.63 | 0.24 | 0.73 | 0.75 |
  | 3 | 0.37 | 0.69 | 0.32 | 0.71 |
  | 4 | 0.81 | 0.05 | 0.92 | 0.75 |
  | mean | about 0.56 | about 0.28 | about 0.68 | |

- **What a batch of that run holds** (one epoch of fold 0, 512 draws, seed 0, measured afterwards): 3.9 images with a mass per batch of 8 on average, from 1 to 7. 86% of the 251 patches of images with a mass contained at least one mass pixel (a mass fills a median of 1.2% of such a patch), none of the 261 patches of images without one did, so 42% of all patches had a mass pixel, and mass pixels were 1.1% of all patch pixels (0.27% in the whole dataset).
- **Padding was counted as valid in these runs.** 262 of the 410 preprocessed images are narrower than 512 px, so most crops were padded with zeros on the right, and that padding counted as correct background in the loss; validation excluded its own padding. This was fixed afterwards (the padding of a crop is not valid), and the effect on the scores was not measured, so the table above is the old behavior.
- **How to read it.**
  - The folds differ a lot (0.37 to 0.81), and each validation fold has only 15 to 19 images with a mass, so no single fold gives a safe estimate.
  - On the Dice over all images, the model clearly beats predicting nothing in folds 1 and 4 only; fold 2 is level, and folds 0 and 3 are worse. It is not yet a useful detector.
  - **Do not quote the mean of the best epochs** (0.723 in the Results cell of the notebook). Each best epoch is chosen on its own validation fold by the Dice on images with a mass alone, which favours epochs that over-predict (fold 0's best epoch has a false-alarm rate of 0.73), so it is optimistic. Use the converged values above.
  - Fold 3 is the weak one. Its false-alarm rate jumps between 0.04 and 0.96 from one epoch to the next, even at a small learning rate, so on normal tissue the predicted probabilities probably lie close to the threshold of 0.5 (not tested). A threshold and a minimum size of the predicted regions are the cheapest things to try.
  - Crops, a batch of 8 and 512 samples per epoch were changed together, so the effect of the crop alone is not separated. Nothing here uses the held-out test set.
- The false-alarm rate counts one predicted pixel as a false alarm, so it was 1.000 in almost every whole-image run and does not discriminate well. A companion that ignores very small predicted regions, or the predicted area on images without a mass, is needed.
- **Not tried:** batches with a fixed composition on the real data (implemented, see section 4), the training after the padding fix, other GPUs, the held-out test set, thresholds other than 0.5, post-processing of the predicted regions, changes of the loss, a warm-up or lower encoder learning rate, averaging of weights, and other crop sizes.

## Next steps

- Run `mammo-seg evaluate` on the five trained folds to see whether the false alarms can be reduced without retraining (not run on real checkpoints yet).
- Judge the folds by the converged values and the paired differences between settings, not by the best epoch. Change one thing at a time.
- Evaluation on the held-out test set, with the simple baselines the protocol calls for.
