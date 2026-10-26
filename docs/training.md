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
- Both are computed on the pixels the caller marks as valid. Training marks every pixel, so the padding counts as background: no mass can be there, the network sees padding at inference too, and it teaches the network that a flat area of zeros is not a mass. The metrics still leave the padding out (section 1). Leaving it out of the loss too left the flat background untrained, the likely cause of validation predicting masses all over it (see [the flat background](#the-flat-background)).

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

- **Training** draws batches balanced between images with and without a mass (the sampler of `loaders.py`, 50% by default), applies the augmentation, and minimizes the Dice + cross-entropy loss with Adam, on every pixel including the padding (section 2). The learning rate starts at 1e-4 and follows a cosine curve down to 0 over the epochs.
- **Crops** (`--crop-size N`, off by default): training can use square patches of N pixels instead of whole images, half of them (`--crop-mass-fraction`) placed around a mass when the image has one (see the augmentation in [data-pipeline.md](data-pipeline.md)). The training canvas is then the crop size rounded up to a multiple of 32, so batches are cheaper and a larger `--batch-size` fits. Validation scores whole images either way, but a model trained on crops predicts them in sliding windows of the training canvas (see *How a whole image is predicted* below). The crop settings are saved in the checkpoint, and a run cannot be resumed with other ones.
- **Batch composition** (`--batch-composition MASS NEG_FROM_MASS_IMAGES NEG_FROM_NORMAL_IMAGES`, for example `4 2 2`, needs `--crop-size`): instead of drawing images at random, every batch has that many patches that contain a mass, negative patches (no mass) from images with a mass, and negative patches from images without one. The batch size is their sum, `--positive-fraction` is not used, and the composition is saved in the checkpoint like the other settings, so a run cannot be resumed with another one. Patch kinds are described in [data-pipeline.md](data-pipeline.md). The negative patches of images with a mass are only mostly negative: when no patch avoids the mass, the one with the least of it is used, which happened for about a fifth of them on fold 0 at 512 px (a median of 0.23% of the patch), so about 55% of all patches hold mass pixels, not 50%. Implemented and tested on synthetic data; not yet run on the real data.
- **Validation** runs on every image of the validation fold once and computes the metrics of section 1 for images with a mass, without one, and all images. All of them are logged every epoch.
- **How a whole image is predicted** (`--inference`): a model should see at validation and at test what it saw in training, so there are two consistent designs.
  - **Whole images**: trained without `--crop-size`, it predicts a whole image in one pass (`whole`).
  - **Patches**: trained with `--crop-size`, it predicts a whole image in **sliding windows** of the training canvas (`sliding`, `engine/inference.py`). The windows overlap by half (the last one touches the far edge: on the 1024 x 800 canvas with 512 px windows, rows start at 0, 256 and 512 and columns at 0, 256 and 288, so 9 windows), and their probabilities are averaged with Gaussian weights (a standard deviation of an eighth of the window) that trust the centre of a window more than its borders, as nnU-Net does. It costs about 3 times the computation of one pass.

  By default the inference follows the training: `sliding` with `--crop-size`, `whole` without. `--inference whole` with `--crop-size` gives the mixed design of the earlier crop runs (patches in training, whole images at validation), kept only to compare with. The choice and the window are stored in the checkpoints, a run is not resumed with other ones, and `mammo-seg evaluate` predicts the same way unless told otherwise.
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
| `inference` | sliding windows with crops, one pass without |

### What a run writes

In `<output-dir>/fold_<k>/`:

| File | Content |
|---|---|
| `history.jsonl` | One JSON line per epoch: learning rate, training loss, duration, all validation metrics (`null` where undefined), and on a GPU the peak memory of the epoch in GB (`peak_gpu_memory_gb`, the memory reserved by PyTorch; `null` on a CPU) |
| `last.pt` | After every epoch: weights, optimizer, learning-rate schedule, history; all that is needed to continue |
| `best.pt` | The weights of the best epoch, with the model and preprocessing settings; load it with `load_model` |

Both are written atomically (section 3).

### Stopping and resuming

A run stops when the epochs are done, when the selection metric has not improved for `patience` epochs, or, with `--max-minutes`, before an epoch that would not end within that time, so a session with a time limit ends with a saved state instead of a lost epoch. At least one epoch always runs, so repeated sessions make progress.

`--resume` continues from `last.pt`. It refuses a checkpoint made with other training settings, model, preprocessing or fold (only mixed precision may differ), so a run is never continued with something else than it started with. The number of workers must stay the same too: the workers share the samples of an epoch, so their number changes the random draws.

The random draws of an epoch (the sampler and the augmentation) depend only on the seed, the epoch number and the number of workers, so a stopped and resumed run gives the same history and weights as one that was never stopped; tests check this on the CPU with no workers and with two. Two things make it hold:

- **Every epoch has its own seed**, mixed from the run's seed and the epoch number by NumPy's `SeedSequence` (`epoch_seed`). It seeds PyTorch and the loader's generator before the epoch. An earlier version used the sum of the two, so a run with seed 1 drew at epoch 1 exactly what a run with seed 0 drew at epoch 2: runs with neighbouring seeds shared their batches and looked more alike than independent runs.
- **The data loader workers are started again for every epoch**, not kept between epochs. A kept worker goes on with the random state it started with, so reseeding before an epoch did not reach it. With `--num-workers 2`, as on Kaggle, a resumed run then drew other images and other augmentations than one that never stopped, so the earlier runs that went on in a later session cannot be repeated exactly. Restarting the workers costs a moment per epoch.

## 5. Running it

The `mammo-seg` command (installed with the package) runs the data steps and the training; the options are listed by `mammo-seg train --help`.

```bash
mammo-seg train --fold 0 --output-dir runs                   # a GPU if there is one, else the CPU
mammo-seg train --fold 0 --output-dir runs --resume          # continue after a stop
mammo-seg train --epochs 30 --selection-metric with_mass.dice --device cuda
```

The data steps are the same commands as `python -m mammo_seg.data.<step>`: `mammo-seg manifest`, `splits`, `masks` and `preprocess`. Training reads the preprocessing settings and the canvas from the preprocessed data, so they cannot disagree with it.

On Kaggle, use [notebooks/05_kaggle_training.ipynb](../notebooks/05_kaggle_training.ipynb). Its first cell holds the experiment: the folds (`FOLDS`), the time shared by them (`MAX_MINUTES`) and any other `mammo-seg train` options (`TRAIN_OPTIONS`, set to the crop settings that worked, see below; `FOLDS` is `[0]`, use `[0, 1, 2, 3, 4]` for all five), and any `mammo-seg preprocess` options (`PREPROCESS_OPTIONS`, for example `--size 2048` for a longer side of 2048 pixels instead of 1024). The next cells install the package, rebuild the manifest, splits, masks and preprocessed images from the attached INbreast dataset (a few minutes, so nothing derived has to be uploaded), train each fold with the time that is left, and print the best epoch of each, with the median time per epoch and the peak GPU memory. A short run (a few epochs with a small `--samples-per-epoch`) measures what a larger input or batch costs before a long one. To continue in a later session, save a version, add its output as an input and set `PREVIOUS_RUNS`; a fold with a `last.pt` is resumed and a finished fold returns at once. The notebook asks for `--device cuda`, so a notebook without a GPU stops with an error. Every session appends its time and the commit of the code it installed to `runs/code_version.txt` (the evaluation notebook to `code_version.txt` in its output), so the results of a run name the exact code that made them; copy that file with the results into `results/` (see [results/README.md](../results/README.md)).

**Whole images or patches.** [notebooks/08_kaggle_whole_vs_patches.ipynb](../notebooks/08_kaggle_whole_vs_patches.ipynb) trains all five folds of both consistent designs (whole images predicted in one pass; patches predicted in sliding windows), with 60 epochs of 64 steps each, evaluates the last epoch of every fold, and compares the designs by the mean and the standard deviation over the folds and by the difference per fold. It resumes in a later session like the training notebook.

### Evaluating a trained model

`mammo-seg evaluate` (`engine/evaluate.py`) scores the checkpoint of each fold, `<runs-dir>/fold_<k>/last.pt` by default (`--checkpoint-name best.pt` for the best epoch), on the images that model did not train on: its own validation fold, or with `--test` the held-out test set. Scoring every validation fold with its own model gives out-of-fold predictions for all development images, and the results of the folds are pooled by putting the per-image scores together.

```bash
mammo-seg evaluate --runs-dir runs                       # all five folds, a grid of settings
mammo-seg evaluate --runs-dir runs --folds 0 1 --thresholds 0.4 0.5 --min-areas-mm2 0 5
```

Each probability map is turned into a prediction under every combination of a **threshold** (0.3 to 0.7 by default) and a **minimum region size** in mm² (`--min-areas-mm2`; 0, 2, 5 and 10 mm² by default): connected regions (8-connectivity) of predicted pixels smaller than the minimum are removed, so a stray spot of a few pixels no longer makes a false alarm. The size is in mm², not pixels, because a preprocessed pixel covers 0.017 to 0.078 mm² depending on the size of the breast (its area comes from `preprocessing.json` and the native pixel of 0.07 mm): 10 mm² is 129 to 579 pixels. The smallest mass is 14.7 mm², so the defaults never remove a whole mass; the earlier default of 1000 pixels removed 12 of the 115.

For each setting the table gives:

- the metrics of section 1 on images with a mass, the false-alarm rate with that rule, and the mean number of predicted pixels on images without a mass (`FP pixels`);
- **per lesion**: the share of the masses that were **found** and the **false regions per image** (`FP/image`, over all images). The masses are the connected regions of the mask (two masses that touch count as one: 116 masses give 115 regions). A mass is found when a predicted region overlaps it with an IoU of at least 0.1, and a predicted region that finds no mass is a false region. So a region that spills far over a mass counts as a miss and a false region, which the per-image sensitivity hides. Over the thresholds, the found masses against the false regions per image make a curve like an FROC;
- the Dice over all images, and the **balanced Dice**: the mean of the Dice on images with a mass and on images without one, so both kinds count equally.

A row for the baseline of **predicting nothing** is always printed, since it scores the share of images without a mass on the all-images Dice. `*` marks the best balanced Dice, and the folds are shown at that setting.

Below, the same setting **by subgroup**, all folds pooled, with the thresholds of the splits: images without a mass by kind (normal, BI-RADS 1; a benign finding, BI-RADS 2 or 3; a suspicious finding that is not a mass, BI-RADS 4a to 6) with their false alarms and false regions per image; masses under 100 mm² and of 100 mm² or more with the share found; and fatty (ACR 1 or 2) and dense (ACR 3 or 4) breasts with the Dice on images with a mass and the false alarms on images without one. Images without a mass are not equally hard, so a set's false alarms depend on its mix; the subgroups make sets with different mixes comparable.

Everything, the subgroups of every setting included, is also written to `<runs-dir>/evaluation_validation.json` (or `_test.json`; `--output` changes it).

**Per image** (`--details`): every scored image with its results under every setting is also written to `<output>_details.json` (for example `evaluation_validation_details.json`), for an analysis of the errors. An image has its fold, file id, patient id, view, BI-RADS, ACR density and pixel area, and per setting its scores, its masses (area in mm², found or not, the best IoU of a predicted region with it, its centre in pixels of the preprocessed image, and its peak: the highest predicted probability inside it, whatever the threshold) and its false regions (area in mm², centre and peak). The peak of a missed mass tells a weak response just under the threshold from no response at all. The `results` of an image are in the order of `settings`. It holds scores and positions only, no images: a few MB for the five folds. [notebooks/09_error_analysis.ipynb](../notebooks/09_error_analysis.ipynb) reads it locally, at one setting, to show which masses are missed (by size, density, BI-RADS and view, and whether the other view of the breast finds one), where the false regions are and whether they lie on an annotated finding that is not a mass, and the worst images by id. [notebooks/10_error_gallery.ipynb](../notebooks/10_error_gallery.ipynb) shows them: it predicts the images again on the CPU with the checkpoints of the details, checks the scores against the details file, and draws every missed mass and false region with a zoom, the probabilities around it and, for a missed mass, the other view of the breast. Its figures show dataset images, so its outputs must stay stripped.

- A checkpoint trained for another fold, or on images preprocessed with other settings, is refused.
- Each model predicts as its checkpoint records: in sliding windows if it was trained on crops, in one pass otherwise. `--inference whole` or `--inference sliding` overrides that, for example to score a patch model in one pass; sliding windows need a checkpoint that records its window, which those made before this option do not.
- Choosing the setting on the validation predictions tunes it on data it was scored on, so it is slightly optimistic. Choose one setting for all folds (not one per fold, which would fit 15 to 19 images with a mass each), then score the test set once.
- The command runs on a CPU, but a GPU is much faster. On Kaggle use [notebooks/06_kaggle_evaluation.ipynb](../notebooks/06_kaggle_evaluation.ipynb) with the saved output of the training notebook as an input, and the same `PREPROCESS_OPTIONS` as the training: the evaluation stops if the checkpoints were trained on images preprocessed otherwise.

### What has and has not been tried

**All the runs below used the earlier splits** (version 1 of `splits.json`, stratified on masses only). The splits are now also stratified on mass size and breast density (see [data-pipeline.md](data-pipeline.md#2-patient-splits)), so the folds have other patients, and the scores below cannot be compared fold by fold with new runs; the mean over the folds is only a rough comparison. Checkpoints of these runs must not be evaluated with the new splits: their training images are now partly in other validation folds.

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
- **Padding was counted as valid in these runs.** 262 of the 410 preprocessed images are narrower than 512 px, so most crops were padded with zeros on the right, and that padding counted as correct background in the loss; validation excluded its own padding. A later change (PR #22) left the padding of a crop out of the loss, and with it the same settings collapsed again on folds 0 and 1 (below). The padding now counts in the loss again, on purpose.
- **How to read it.**
  - The folds differ a lot (0.37 to 0.81), and each validation fold has only 15 to 19 images with a mass, so no single fold gives a safe estimate.
  - On the Dice over all images, the model clearly beats predicting nothing in folds 1 and 4 only; fold 2 is level, and folds 0 and 3 are worse. It is not yet a useful detector.
  - **Do not quote the mean of the best epochs** (0.723 in the Results cell of the notebook). Each best epoch is chosen on its own validation fold by the Dice on images with a mass alone, which favours epochs that over-predict (fold 0's best epoch has a false-alarm rate of 0.73), so it is optimistic. Use the converged values above.
  - Fold 3 is the weak one. Its false-alarm rate jumps between 0.04 and 0.96 from one epoch to the next, even at a small learning rate, so on normal tissue the predicted probabilities probably lie close to the threshold of 0.5 (not tested). A threshold and a minimum size of the predicted regions are the cheapest things to try.
  - Crops, a batch of 8 and 512 samples per epoch were changed together, so the effect of the crop alone is not separated. Nothing here uses the held-out test set.
- **The crop runs collapsed again when the padding left the loss.** With the padding of a crop not valid (PR #22) and otherwise the same settings and patches (the image and mask tensors were checked to be identical on 51 patches; only the valid pixels differed, 12.7% of them on average), folds 0 and 1 gave a validation Dice of 0.02 on images with a mass and a false-alarm rate of 1.000 from epoch 1 to 60, while the training loss was almost the same as in the run that worked (0.625 against 0.624 at epoch 60).
- The false-alarm rate counts one predicted pixel as a false alarm, so it was 1.000 in almost every whole-image run and does not discriminate well. `mammo-seg evaluate` now adds a minimum region size, the predicted area on images without a mass, and the false regions per image (see above).
- **Not tried:** batches with a fixed composition on the real data (implemented, see section 4), training with the background kept at 0 and the padding in the loss (the fix below), other GPUs, the held-out test set, thresholds other than 0.5, post-processing of the predicted regions, changes of the loss, a warm-up or lower encoder learning rate, averaging of weights, and other crop sizes.

#### The flat background

The likely cause of that collapse, and of the earlier failure of whole-image training:

- **The background is exactly 0.** The scanner sets the air around the breast to 0 (in the raw images, 59% to 80% of the pixels are 0 and almost none lie between 1 and 100). In every preprocessed image a median of 29% of the pixels is one flat region of zeros, and validation scores it.
- **Training never had it where the loss counts.** The noise of the augmentation (and the contrast changes) turned the background into a speckle of small values: on fold 0, no pixel of the image part of a training patch lay in a flat area of zeros (5 x 5), against 29% of the pixels of a validation image. The only flat zeros in training were the padding. The crop runs that worked counted it in the loss, by accident; whole-image training and the crop runs after PR #22 did not, so the network never learned what to predict on the flat background.
- **An untrained input sits at the threshold.** On flat zeros, an untrained U-Net gives probabilities of about 0.4 to 0.6, and on which side of 0.5 they land depends on the BatchNorm statistics (with a ResNet-34 encoder: 6% above 0.5 in training mode, 97% in evaluation mode). Predictions above 0.5 on the background make every image without a mass a false alarm and drag the Dice of the others down, while the training loss, which never sees these pixels, looks normal.
- **It fits the earlier observations:** the predicted area of about 18 times the mass on training images scored without augmentation; at a learning rate of 5e-4, 53,000 to 123,000 predicted pixels per image, none inside a mass, against about 140,000 background pixels in an image of median width; and an epoch whose scores depended entirely on the BatchNorm statistics.
- **What is not shown yet:** that the false positives of the collapsed runs lie on the background (a check of their checkpoints), and that training with the fix recovers. Two short CPU runs on real patches at a quarter of the resolution showed the effect early in training only, so they are weak evidence.

The fix: the augmentation keeps the background exactly 0 (see the augmentation in [data-pipeline.md](data-pipeline.md)), and the loss counts the padding as background (section 2). On fold 0, 19% of the pixels of the image part of a training patch now lie in a flat area of zeros.

## Next steps

- Check where the false positives of the collapsed fold-0 run lie (on the zero background or in the breast), then train folds 0 and 1 with the fix for the flat background and compare them with the run that worked.
- Run `mammo-seg evaluate` on the folds trained with the fix, to see whether a threshold and a minimum region size reduce the false alarms (not run on real checkpoints yet; the earlier checkpoints belong to the earlier splits).
- Judge the folds by the converged values and the paired differences between settings, not by the best epoch. Change one thing at a time.
- Evaluation on the held-out test set, with the simple baselines the protocol calls for.
