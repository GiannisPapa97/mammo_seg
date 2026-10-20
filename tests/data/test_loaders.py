from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tests.data.manifests import make_image
from tests.data.preprocessed import CANVAS, SETTINGS, WIDTHS, write_preprocessed

from mammo_seg.data.dataset import MammogramDataset
from mammo_seg.data.loaders import (
    DEFAULT_POSITIVE_FRACTION,
    balanced_sampler,
    evaluation_loader,
    has_mass_of,
    pad_to_multiple,
    training_loader,
    unpad,
)
from mammo_seg.data.schema import Manifest

IDS = list(WIDTHS)  # image 00000001 has a mass, the other two do not


@pytest.fixture
def dataset(tmp_path: Path) -> MammogramDataset:
    write_preprocessed(tmp_path, SETTINGS)
    return MammogramDataset(tmp_path, IDS, SETTINGS, CANVAS)


@pytest.fixture
def manifest() -> Manifest:
    images = [
        make_image(number, number, masses=int(number == 1)) for number in (1, 2, 3)
    ]
    return Manifest(version=1, images=tuple(images))


def drawn_ids(loader: torch.utils.data.DataLoader) -> list[str]:  # type: ignore[type-arg]
    return [file_id for batch in loader for file_id in batch["file_id"]]


# --- the sampler ---


def test_the_sampler_draws_about_the_positive_fraction() -> None:
    has_mass = [True] * 10 + [False] * 90
    generator = torch.Generator().manual_seed(0)
    for fraction in (0.5, 0.3):
        sampler = balanced_sampler(
            has_mass, fraction, num_samples=20000, generator=generator
        )
        drawn = [has_mass[i] for i in sampler]
        assert sum(drawn) / len(drawn) == pytest.approx(fraction, abs=0.02)


def test_the_default_epoch_has_as_many_draws_as_images() -> None:
    sampler = balanced_sampler([True, False, False, False])
    assert len(sampler) == 4 and len(list(sampler)) == 4
    assert DEFAULT_POSITIVE_FRACTION == 0.5


def test_the_sampler_is_reproducible_with_a_seeded_generator() -> None:
    has_mass = [True, False] * 20

    def draws(seed: int) -> list[int]:
        generator = torch.Generator().manual_seed(seed)
        return list(balanced_sampler(has_mass, generator=generator))

    assert draws(1) == draws(1)
    assert draws(1) != draws(2)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5])
def test_a_fraction_outside_0_and_1_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        balanced_sampler([True, False], fraction)


def test_one_class_alone_cannot_be_balanced() -> None:
    with pytest.raises(ValueError, match="0 images with a mass and 3 without"):
        balanced_sampler([False, False, False])
    with pytest.raises(ValueError, match="2 images with a mass and 0 without"):
        balanced_sampler([True, True])


def test_has_mass_follows_the_order_of_the_ids(manifest: Manifest) -> None:
    assert has_mass_of(manifest, ["00000003", "00000001"]) == [False, True]
    with pytest.raises(ValueError, match="not in the manifest: \\['00000009'\\]"):
        has_mass_of(manifest, ["00000009"])


# --- the loaders ---


def test_the_training_loader_balances_the_draws(
    dataset: MammogramDataset, manifest: Manifest
) -> None:
    loader = training_loader(dataset, manifest, batch_size=4, samples_per_epoch=2000)
    counts = Counter(drawn_ids(loader))
    assert sum(counts.values()) == 2000
    assert counts["00000001"] / 2000 == pytest.approx(0.5, abs=0.04)


def test_the_training_batches_have_the_canvas_shape(
    dataset: MammogramDataset, manifest: Manifest
) -> None:
    batch = next(iter(training_loader(dataset, manifest, batch_size=2)))
    assert batch["image"].shape == (2, 1, *CANVAS)
    assert batch["mask"].shape == (2, 1, *CANVAS)
    assert batch["valid"].shape == (2, 1, *CANVAS)


def test_the_last_incomplete_batch_is_dropped(
    dataset: MammogramDataset, manifest: Manifest
) -> None:
    loader = training_loader(dataset, manifest, batch_size=2, samples_per_epoch=5)
    assert [len(batch["file_id"]) for batch in loader] == [2, 2]


def test_without_the_sampler_every_image_is_used_once(
    dataset: MammogramDataset, manifest: Manifest
) -> None:
    loader = training_loader(dataset, manifest, batch_size=1, positive_fraction=None)
    assert sorted(drawn_ids(loader)) == IDS


def test_the_training_order_depends_on_the_seed(
    dataset: MammogramDataset, manifest: Manifest
) -> None:
    def order(seed: int) -> list[str]:
        loader = training_loader(dataset, manifest, 1, samples_per_epoch=30, seed=seed)
        return drawn_ids(loader)

    assert order(3) == order(3)
    assert order(3) != order(4)


def test_the_training_loader_works_with_workers(
    dataset: MammogramDataset, manifest: Manifest
) -> None:
    loader = training_loader(dataset, manifest, 2, samples_per_epoch=4, num_workers=2)
    assert len(drawn_ids(loader)) == 4


def test_the_evaluation_loader_uses_every_image_once_in_order(
    dataset: MammogramDataset,
) -> None:
    loader = evaluation_loader(dataset, batch_size=2)
    assert drawn_ids(loader) == IDS
    assert [len(batch["file_id"]) for batch in loader] == [2, 1]


def test_the_training_loader_needs_the_images_in_the_manifest(
    dataset: MammogramDataset, manifest: Manifest
) -> None:
    other = Manifest(version=1, images=(replace(manifest.images[0]),))
    with pytest.raises(ValueError, match="not in the manifest"):
        training_loader(dataset, other, batch_size=2)


# --- padding for inference ---


@pytest.mark.parametrize(
    ("shape", "padded"),
    [
        ((1, 1024, 837), (1, 1024, 864)),
        ((1, 1024, 800), (1, 1024, 800)),
        ((1, 5, 33), (1, 32, 64)),
    ],
)
def test_an_image_is_padded_to_multiples_of_32(
    shape: tuple[int, int, int], padded: tuple[int, int, int]
) -> None:
    image = torch.rand(shape)
    result, size = pad_to_multiple(image)
    assert result.shape == padded
    assert size == shape[1:]
    assert torch.equal(result[:, : shape[1], : shape[2]], image)
    assert result[:, shape[1] :, :].sum() == 0 and result[:, :, shape[2] :].sum() == 0


def test_unpad_restores_the_original_size() -> None:
    image = torch.rand(2, 1, 50, 70)
    padded, size = pad_to_multiple(image, 16)
    assert padded.shape == (2, 1, 64, 80)
    assert torch.equal(unpad(padded, size), image)


def test_padding_and_unpadding_check_their_arguments() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        pad_to_multiple(torch.rand(1, 4, 4), 0)
    with pytest.raises(ValueError, match="smaller than"):
        unpad(torch.rand(1, 4, 4), (8, 4))
