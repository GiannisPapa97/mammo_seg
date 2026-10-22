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

## 4. The training loop

`train_fold(manifest, splits_file, preprocessed_dir, fold, ...)` (`engine/train.py`) trains on every fold of the patient-level splits but one and validates on that one. Each epoch is a pass of training and a pass of validation:

- **Training** draws batches balanced between images with and without a mass (the sampler of `loaders.py`, 50% by default), applies the augmentation, and minimizes the Dice + cross-entropy loss with Adam. The learning rate starts at 1e-4 and follows a cosine curve down to 0 over the epochs.
- **Validation** runs on every image of the validation fold once and computes the metrics of section 1 for images with a mass, without one, and all images. All of them are logged every epoch.
- **Mixed precision** (float16) is used on a CUDA GPU only: the network runs in it, the loss in float32, with a gradient scaler. On the CPU it is off whatever is asked.
- **The best epoch** is the one with the highest value of the selection metric, by default the mean Dice over all validation images (`all_images.dice`). Another metric can be chosen, for example `with_mass.dice`; an undefined value (NaN) never wins. Note that on this dataset a model that predicts nothing already scores well on all images, because most images have no mass, so look at the images with a mass too.

| Setting | Default |
|---|---|
| `epochs` | 60 |
| `batch_size` | 4 |
| `learning_rate` | 1e-4 (cosine decay) |
| `positive_fraction` | 0.5 |
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

On Kaggle or Colab, use [notebooks/05_cloud_training.ipynb](../notebooks/05_cloud_training.ipynb): it installs the package, rebuilds the manifest, splits, masks and preprocessed images from the attached INbreast dataset (a few minutes, so nothing derived has to be uploaded), and trains with `--max-minutes`. To continue in a later session, add the previous version's output as an input and set `PREVIOUS_RUNS`.

### What has and has not been tried

- Tested on the CPU with synthetic images (CI does the same): the loop, checkpoints, stopping, resuming and the command, including a run through the command that learns a synthetic mass.
- Run once on the real data on a laptop CPU for one epoch with a small encoder, which wrote the checkpoints and the history. That machine has 4 GB of RAM and cannot hold a batch of 4 full-size images, so the default batch size is untested here.
- **Not tried:** a GPU, mixed precision, the CUDA PyTorch builds, a real Kaggle or Colab session, and the memory use and speed of the default model on a GPU. Results of real training runs are not reported here yet.

## Next steps

- A first real training run on a GPU, to set the batch size and see what the metrics look like.
- Evaluation on the held-out test set, with the simple baselines the protocol calls for.
