"""Data loaders for training and evaluation, and padding for inference.

Training draws images with ``WeightedRandomSampler`` so that about
``positive_fraction`` of them have a mass (26% of the INbreast images do). It
draws with replacement, so an epoch has a fixed length, by default the number of
training images. Validation and test loaders keep the natural distribution and
the original order: their metrics have to reflect the real data.

Everything random is driven by one seeded ``torch.Generator``, so a run can be
repeated. The augmentation uses PyTorch's random numbers, and the loader seeds
every worker from that generator.
"""

from collections.abc import Sequence

import torch
from torch.nn import functional
from torch.utils.data import DataLoader, WeightedRandomSampler

from mammo_seg.data.dataset import DEFAULT_MULTIPLE, MammogramDataset, Sample
from mammo_seg.data.schema import Manifest

DEFAULT_POSITIVE_FRACTION = 0.5


def has_mass_of(manifest: Manifest, file_ids: Sequence[str]) -> list[bool]:
    """Whether each image has a mass, in the order of ``file_ids``."""
    by_id = {image.file_id: image.has_mass for image in manifest.images}
    missing = sorted(set(file_ids) - by_id.keys())
    if missing:
        raise ValueError(f"images not in the manifest: {missing}")
    return [by_id[file_id] for file_id in file_ids]


def balanced_sampler(
    has_mass: Sequence[bool],
    positive_fraction: float = DEFAULT_POSITIVE_FRACTION,
    num_samples: int | None = None,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """Draw indices so that about ``positive_fraction`` of them have a mass.

    Raises ValueError for a fraction outside (0, 1) or when there are no positive
    or no negative images to balance.
    """
    if not 0 < positive_fraction < 1:
        raise ValueError(
            f"the positive fraction must be between 0 and 1, got {positive_fraction}"
        )
    positives = sum(has_mass)
    negatives = len(has_mass) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"cannot balance {positives} images with a mass and {negatives} without"
        )
    weights = [
        positive_fraction / positives if flag else (1 - positive_fraction) / negatives
        for flag in has_mass
    ]
    return WeightedRandomSampler(
        weights,
        num_samples=num_samples if num_samples is not None else len(has_mass),
        replacement=True,
        generator=generator,
    )


def training_loader(
    dataset: MammogramDataset,
    manifest: Manifest,
    batch_size: int,
    positive_fraction: float | None = DEFAULT_POSITIVE_FRACTION,
    samples_per_epoch: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[Sample]:
    """Batches for training, balanced by the sampler (``None`` turns it off).

    Without the sampler, every image is used once per epoch in a shuffled order.
    The last incomplete batch is dropped, so batch normalization never sees a
    batch of one image.
    """
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if positive_fraction is not None:
        sampler = balanced_sampler(
            has_mass_of(manifest, dataset.file_ids),
            positive_fraction,
            samples_per_epoch,
            generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        generator=generator,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def evaluation_loader(
    dataset: MammogramDataset, batch_size: int, num_workers: int = 0
) -> DataLoader[Sample]:
    """Batches for validation and test: every image once, in order, nothing random."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def pad_to_multiple(
    image: torch.Tensor, multiple: int = DEFAULT_MULTIPLE
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad a ``[..., H, W]`` image with zeros on the right and at the bottom up to
    sides divisible by ``multiple``, for inference on a single image of any size.

    Returns the padded image and the original ``(H, W)``, which ``unpad`` uses to
    cut the padding off the prediction.
    """
    if multiple < 1:
        raise ValueError(f"the multiple must be at least 1, got {multiple}")
    height, width = image.shape[-2:]
    rows = -(-height // multiple) * multiple
    cols = -(-width // multiple) * multiple
    padded = functional.pad(image, (0, cols - width, 0, rows - height))
    return padded, (height, width)


def unpad(prediction: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Cut a ``[..., H, W]`` prediction back to the ``size`` of the original image."""
    height, width = size
    if prediction.shape[-2] < height or prediction.shape[-1] < width:
        raise ValueError(
            f"a prediction of {tuple(prediction.shape)} is smaller than {size}"
        )
    return prediction[..., :height, :width]
