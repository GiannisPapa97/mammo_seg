"""Data loaders for training and evaluation, and padding for inference.

Training draws images with ``WeightedRandomSampler`` so that about
``positive_fraction`` of them have a mass (26% of the INbreast images do). It
draws with replacement, so an epoch has a fixed length, by default the number of
training images. Validation and test loaders keep the natural distribution and
the original order: their metrics have to reflect the real data.

Training can instead build every batch from a ``BatchComposition``: so many patches
that contain a mass, so many negative patches from images with a mass and so many
from images without one. A batch then always has the same make-up, where random
draws give a different number of images with a mass in each. This needs the crop of
the augmentation (``AugmentationSettings.crop_size``).

Everything random is driven by one seeded ``torch.Generator``, so a run can be
repeated. The augmentation uses PyTorch's random numbers, and the loader seeds
every worker from that generator. Workers are started again for every epoch rather
than kept (no persistent workers): a kept worker goes on with the random state it
started with, so reseeding the generator before an epoch would not reach it, and a
run resumed at a later epoch would draw other samples than one that never stopped.
The draws still depend on the number of workers, which share the samples between
them.
"""

from collections.abc import Iterator, Sequence
from typing import Annotated, Self

import torch
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from torch.nn import functional
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from mammo_seg.data.augmentation import PatchKind
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


@dataclass(frozen=True, config=ConfigDict(strict=True, extra="forbid"))
class BatchComposition:
    """What every training batch holds, in patches; the batch size is their sum."""

    mass: Annotated[int, Field(ge=0)] = 4  # patches that contain a mass
    negative_from_mass_images: Annotated[int, Field(ge=0)] = 2  # no mass in the patch
    negative_from_normal_images: Annotated[int, Field(ge=0)] = 2

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if self.size < 1:
            raise ValueError("a batch needs at least one patch")
        return self

    @property
    def size(self) -> int:
        return (
            self.mass
            + self.negative_from_mass_images
            + self.negative_from_normal_images
        )


class TypedBatchSampler(Sampler[list[tuple[int, PatchKind]]]):
    """Batches of ``(index, kind)`` with the make-up of a ``BatchComposition``.

    A batch holds, in this order, ``mass`` patches of images with a mass, then
    ``negative_from_mass_images`` negative patches of images with a mass, then
    ``negative_from_normal_images`` negative patches of images without one. Images
    are drawn with replacement. ``num_batches`` batches make an epoch.

    Raises ValueError for fewer than one batch, or a composition that asks for
    patches of images that there are none of.
    """

    def __init__(
        self,
        has_mass: Sequence[bool],
        composition: BatchComposition,
        num_batches: int,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if num_batches < 1:
            raise ValueError(f"an epoch needs at least one batch, got {num_batches}")
        self.positives = [index for index, flag in enumerate(has_mass) if flag]
        self.negatives = [index for index, flag in enumerate(has_mass) if not flag]
        wants_positives = composition.mass + composition.negative_from_mass_images
        if wants_positives and not self.positives:
            raise ValueError("the batches need images with a mass, and there are none")
        if composition.negative_from_normal_images and not self.negatives:
            raise ValueError(
                "the batches need images without a mass, and there are none"
            )
        self.composition = composition
        self.num_batches = num_batches
        self.generator = generator

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[tuple[int, PatchKind]]]:
        composition = self.composition
        for _ in range(self.num_batches):
            yield [
                *self._draw(self.positives, composition.mass, "mass"),
                *self._draw(
                    self.positives, composition.negative_from_mass_images, "negative"
                ),
                *self._draw(
                    self.negatives, composition.negative_from_normal_images, "negative"
                ),
            ]

    def _draw(
        self, pool: list[int], count: int, kind: PatchKind
    ) -> list[tuple[int, PatchKind]]:
        if count == 0:
            return []
        picks = torch.randint(len(pool), (count,), generator=self.generator)
        return [(pool[pick], kind) for pick in picks.tolist()]


def training_loader(
    dataset: MammogramDataset,
    manifest: Manifest,
    batch_size: int,
    positive_fraction: float | None = DEFAULT_POSITIVE_FRACTION,
    samples_per_epoch: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    composition: BatchComposition | None = None,
) -> DataLoader[Sample]:
    """Batches for training, balanced by the sampler (``None`` turns it off).

    Without the sampler, every image is used once per epoch in a shuffled order.
    The last incomplete batch is dropped, so batch normalization never sees a
    batch of one image.

    With a ``composition``, every batch has that make-up instead (see
    ``TypedBatchSampler``) and ``positive_fraction`` is not used; the dataset must
    cut patches. ``batch_size`` must equal the size of the composition, and an
    epoch has ``samples_per_epoch // batch_size`` batches. Raises ValueError
    otherwise.
    """
    generator = torch.Generator().manual_seed(seed)
    if composition is not None:
        if composition.size != batch_size:
            raise ValueError(
                f"the batch size is {batch_size} but the composition has "
                f"{composition.size} patches"
            )
        draws = samples_per_epoch if samples_per_epoch is not None else len(dataset)
        return DataLoader(
            dataset,
            batch_sampler=TypedBatchSampler(
                has_mass_of(manifest, dataset.file_ids),
                composition,
                draws // batch_size,
                generator,
            ),
            generator=generator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
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
