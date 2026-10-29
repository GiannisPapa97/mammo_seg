from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from tests.data.manifests import make_image
from tests.data.preprocessed import CANVAS, SETTINGS, WIDTHS, write_preprocessed

from mammo_seg.data.augmentation import Augmentation, AugmentationSettings
from mammo_seg.data.dataset import MammogramDataset
from mammo_seg.data.loaders import (
    DEFAULT_POSITIVE_FRACTION,
    BatchComposition,
    TypedBatchSampler,
    balanced_sampler,
    evaluation_loader,
    has_mass_of,
    has_small_mass_of,
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
    return Manifest(version=2, dataset="inbreast", images=tuple(images))


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
    with pytest.raises(ValueError, match="2 images with a mass and 0 without; "):
        balanced_sampler([True, True])
    with pytest.raises(ValueError, match="--positive-fraction none"):
        balanced_sampler([True, True])


# --- small masses ---

# 4 images with a small mass, 16 with only larger ones, 80 without a mass.
HAS_MASS = [True] * 20 + [False] * 80
HAS_SMALL = [True] * 4 + [False] * 96


def shares(sampler: torch.utils.data.Sampler[int]) -> tuple[float, float]:
    """The share of the draws with a mass, and of those the share with a small one."""
    drawn = list(sampler)
    with_mass = [i for i in drawn if HAS_MASS[i]]
    small = [i for i in with_mass if HAS_SMALL[i]]
    return len(with_mass) / len(drawn), len(small) / len(with_mass)


def test_without_a_small_mass_fraction_the_draws_are_as_before() -> None:
    sampler = balanced_sampler(HAS_MASS, 0.5, has_small_mass=HAS_SMALL)
    weights = sampler.weights.tolist()
    assert weights == pytest.approx([0.5 / 20] * 20 + [0.5 / 80] * 80)


def test_the_sampler_draws_about_the_small_mass_fraction() -> None:
    generator = torch.Generator().manual_seed(0)
    for fraction in (0.5, 0.7):
        sampler = balanced_sampler(
            HAS_MASS,
            0.5,
            num_samples=40000,
            generator=generator,
            has_small_mass=HAS_SMALL,
            small_mass_fraction=fraction,
        )
        with_mass, small = shares(sampler)
        # The share of the images with a mass stays what it was.
        assert with_mass == pytest.approx(0.5, abs=0.02)
        assert small == pytest.approx(fraction, abs=0.02)


def test_the_small_mass_weights_add_up_to_the_positive_fraction() -> None:
    sampler = balanced_sampler(
        HAS_MASS, 0.4, has_small_mass=HAS_SMALL, small_mass_fraction=0.5
    )
    weights = sampler.weights.tolist()
    assert weights[:4] == pytest.approx([0.4 * 0.5 / 4] * 4)
    assert weights[4:20] == pytest.approx([0.4 * 0.5 / 16] * 16)
    assert weights[20:] == pytest.approx([0.6 / 80] * 80)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
def test_a_small_mass_fraction_outside_0_and_1_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match="small-mass fraction"):
        balanced_sampler(
            HAS_MASS, has_small_mass=HAS_SMALL, small_mass_fraction=fraction
        )


@pytest.mark.parametrize(
    "has_small", [[False] * 100, [True] * 20 + [False] * 80], ids=["none", "all"]
)
def test_small_masses_cannot_be_balanced_without_both_kinds(
    has_small: list[bool],
) -> None:
    with pytest.raises(ValueError, match="cannot balance"):
        balanced_sampler(HAS_MASS, has_small_mass=has_small, small_mass_fraction=0.5)


@pytest.mark.parametrize("has_small", [None, [True] * 4], ids=["missing", "short"])
def test_a_small_mass_fraction_needs_the_small_mass_of_every_image(
    has_small: list[bool] | None,
) -> None:
    with pytest.raises(ValueError, match="small mass of every image"):
        balanced_sampler(HAS_MASS, has_small_mass=has_small, small_mass_fraction=0.5)


def test_a_small_mass_is_one_under_100_mm2_on_its_polygon() -> None:
    images = [
        make_image(1, 1, masses=1, mass_mm2=99.0),
        make_image(2, 2, masses=1, mass_mm2=101.0),
        make_image(3, 3),
        make_image(4, 4, masses=2, mass_mm2=50.0),
    ]
    manifest = Manifest(version=2, dataset="inbreast", images=tuple(images))
    ids = ["00000004", "00000003", "00000002", "00000001"]
    assert has_small_mass_of(manifest, ids) == [True, False, False, True]
    assert has_small_mass_of(manifest, ids, small_mass_mm2=200.0) == [
        True,
        False,
        True,
        True,
    ]
    with pytest.raises(ValueError, match="not in the manifest"):
        has_small_mass_of(manifest, ["00000009"])


@pytest.fixture
def small_and_large() -> Manifest:
    """Image 1 with a small mass, image 2 with a large one, image 3 without."""
    images = [
        make_image(1, 1, masses=1, mass_mm2=50.0),
        make_image(2, 2, masses=1, mass_mm2=500.0),
        make_image(3, 3),
    ]
    return Manifest(version=2, dataset="inbreast", images=tuple(images))


def test_the_training_loader_draws_the_small_masses_more_often(
    dataset: MammogramDataset, small_and_large: Manifest
) -> None:
    def counts(fraction: float | None) -> Counter[str]:
        loader = training_loader(
            dataset,
            small_and_large,
            batch_size=4,
            samples_per_epoch=2000,
            small_mass_fraction=fraction,
        )
        return Counter(drawn_ids(loader))

    natural, more = counts(None), counts(0.8)
    assert natural["00000001"] / 2000 == pytest.approx(0.25, abs=0.04)
    assert more["00000001"] / 2000 == pytest.approx(0.4, abs=0.04)
    assert more["00000003"] / 2000 == pytest.approx(0.5, abs=0.04)


def test_a_small_mass_fraction_needs_the_sampler_without_a_composition(
    dataset: MammogramDataset, small_and_large: Manifest
) -> None:
    with pytest.raises(ValueError, match="needs the sampler"):
        training_loader(
            dataset, small_and_large, 4, positive_fraction=None, small_mass_fraction=0.5
        )
    with pytest.raises(ValueError, match="needs the sampler"):
        training_loader(
            dataset,
            small_and_large,
            4,
            composition=BatchComposition(2, 1, 1),
            small_mass_fraction=0.5,
        )


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


def epoch_of(
    loader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    seed: int,
) -> list[tuple[tuple[str, ...], float]]:
    """One epoch, after reseeding as the training loop does before every epoch."""
    torch.manual_seed(seed)
    assert loader.generator is not None
    loader.generator.manual_seed(seed)
    return [(tuple(batch["file_id"]), float(batch["image"].sum())) for batch in loader]


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("composition", [None, BatchComposition(1, 1, 0)])
def test_an_epoch_depends_only_on_its_seed_with_or_without_workers(
    tmp_path: Path,
    manifest: Manifest,
    workers: int,
    composition: BatchComposition | None,
) -> None:
    # A resumed run starts a new loader at a later epoch; it must draw what the
    # loader of an uninterrupted run draws at that epoch, augmentation included.
    write_preprocessed(tmp_path, SETTINGS)
    crops = Augmentation(AugmentationSettings(crop_size=(32, 24)))
    dataset = MammogramDataset(tmp_path, IDS, SETTINGS, (32, 32), transform=crops)

    def loader() -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
        return training_loader(
            dataset,
            manifest,
            2,
            samples_per_epoch=8,
            num_workers=workers,
            composition=composition,
        )

    uninterrupted = loader()
    first = epoch_of(uninterrupted, 1)
    second = epoch_of(uninterrupted, 2)
    assert first != second
    assert epoch_of(loader(), 2) == second


def test_the_evaluation_loader_uses_every_image_once_in_order(
    dataset: MammogramDataset,
) -> None:
    loader = evaluation_loader(dataset, batch_size=2)
    assert drawn_ids(loader) == IDS
    assert [len(batch["file_id"]) for batch in loader] == [2, 1]


def test_the_training_loader_needs_the_images_in_the_manifest(
    dataset: MammogramDataset, manifest: Manifest
) -> None:
    other = Manifest(
        version=2, dataset="inbreast", images=(replace(manifest.images[0]),)
    )
    with pytest.raises(ValueError, match="not in the manifest"):
        training_loader(dataset, other, batch_size=2)


# --- batches of a known composition ---

NOTHING = AugmentationSettings(
    rotation=0, scale=0, shift=0, brightness=0, contrast=0, gamma=0, noise=0
)


def test_a_composition_has_the_size_of_its_parts() -> None:
    assert BatchComposition().size == 8  # 4 + 2 + 2
    assert BatchComposition(3, 1, 2).size == 6
    assert BatchComposition(1, 0, 0).size == 1


@pytest.mark.parametrize("parts", [(0, 0, 0), (-1, 2, 2), (4, -2, 2), (4, 2, -2)])
def test_a_composition_needs_a_batch_and_no_negative_count(
    parts: tuple[int, int, int],
) -> None:
    with pytest.raises((ValidationError, ValueError)):
        BatchComposition(*parts)


def test_every_batch_has_the_make_up_of_the_composition() -> None:
    has_mass = [True] * 3 + [False] * 7
    sampler = TypedBatchSampler(has_mass, BatchComposition(4, 2, 2), 50)
    batches = list(sampler)
    assert len(sampler) == len(batches) == 50
    for batch in batches:
        assert len(batch) == 8
        masses, from_mass_images, from_normal_images = batch[:4], batch[4:6], batch[6:]
        assert {kind for _, kind in masses} == {"mass"}
        assert {kind for _, kind in from_mass_images + from_normal_images} == {
            "negative"
        }
        assert all(has_mass[index] for index, _ in masses + from_mass_images)
        assert not any(has_mass[index] for index, _ in from_normal_images)


def test_the_images_of_a_batch_are_drawn_with_replacement() -> None:
    # One image with a mass: every mass patch of an epoch comes from it.
    sampler = TypedBatchSampler([True, False, False], BatchComposition(4, 0, 0), 5)
    assert all(index == 0 for batch in sampler for index, _ in batch)
    # Many images: the draws spread over them.
    many = TypedBatchSampler([True] * 50, BatchComposition(4, 0, 0), 20)
    assert len({index for batch in many for index, _ in batch}) > 10


def test_a_part_of_zero_patches_needs_no_such_images() -> None:
    only_masses = TypedBatchSampler([True, True], BatchComposition(2, 0, 0), 2)
    assert all(len(batch) == 2 for batch in only_masses)
    only_normal = TypedBatchSampler([False, False], BatchComposition(0, 0, 3), 2)
    assert all(kind == "negative" for batch in only_normal for _, kind in batch)


def test_the_batches_are_reproducible_with_a_seeded_generator() -> None:
    def draw(seed: int) -> list[list[tuple[int, str]]]:
        generator = torch.Generator().manual_seed(seed)
        sampler = TypedBatchSampler(
            [True] * 5 + [False] * 5, BatchComposition(), 4, generator
        )
        return list(sampler)  # type: ignore[arg-type]

    assert draw(1) == draw(1)
    assert draw(1) != draw(2)


@pytest.mark.parametrize(
    ("has_mass", "composition", "message"),
    [
        ([False, False], BatchComposition(4, 2, 2), "images with a mass"),
        ([False, False], BatchComposition(0, 2, 2), "images with a mass"),
        ([True, True], BatchComposition(4, 2, 2), "images without a mass"),
    ],
)
def test_a_composition_needs_the_images_it_asks_for(
    has_mass: list[bool], composition: BatchComposition, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TypedBatchSampler(has_mass, composition, 1)


def test_an_epoch_needs_a_batch() -> None:
    with pytest.raises(ValueError, match="at least one batch"):
        TypedBatchSampler([True, False], BatchComposition(1, 0, 1), 0)


@pytest.fixture
def patch_dataset(tmp_path: Path) -> MammogramDataset:
    """Patches of 32 x 32 of the three images; only image 1 has a mass."""
    write_preprocessed(tmp_path, SETTINGS)
    crop = replace(NOTHING, crop_size=(32, 32))
    return MammogramDataset(
        tmp_path, IDS, SETTINGS, (32, 32), transform=Augmentation(crop)
    )


def test_the_training_loader_builds_batches_of_the_composition(
    patch_dataset: MammogramDataset, manifest: Manifest
) -> None:
    composition = BatchComposition(2, 1, 1)
    loader = training_loader(
        patch_dataset, manifest, 4, samples_per_epoch=40, composition=composition
    )
    batches = list(loader)
    assert len(batches) == 10
    for batch in batches:
        assert batch["image"].shape == (4, 1, 32, 32)
        assert batch["valid"].shape == (4, 1, 32, 32)
        mass_pixels = batch["mask"].flatten(1).sum(1)
        assert batch["file_id"][:3] == ["00000001"] * 3
        assert batch["file_id"][3] in ("00000002", "00000003")
        assert (mass_pixels[:2] > 0).all()  # the mass patches contain the mass
        assert (mass_pixels[2:] == 0).all()  # the negative ones do not


def test_the_batches_of_a_composition_depend_on_the_seed(
    patch_dataset: MammogramDataset, manifest: Manifest
) -> None:
    def order(seed: int) -> list[str]:
        loader = training_loader(
            patch_dataset,
            manifest,
            4,
            samples_per_epoch=40,
            seed=seed,
            composition=BatchComposition(2, 1, 1),
        )
        return drawn_ids(loader)

    assert order(3) == order(3)
    assert order(3) != order(4)


def test_the_default_epoch_of_a_composition_has_as_many_patches_as_images(
    patch_dataset: MammogramDataset, manifest: Manifest
) -> None:
    loader = training_loader(
        patch_dataset, manifest, 2, composition=BatchComposition(1, 0, 1)
    )
    assert len(loader) == 1  # 3 images, batches of 2


def test_the_loader_of_a_composition_works_with_workers(
    patch_dataset: MammogramDataset, manifest: Manifest
) -> None:
    loader = training_loader(
        patch_dataset,
        manifest,
        4,
        samples_per_epoch=8,
        num_workers=2,
        composition=BatchComposition(2, 1, 1),
    )
    assert len(drawn_ids(loader)) == 8


def test_the_batch_size_must_be_the_size_of_the_composition(
    patch_dataset: MammogramDataset, manifest: Manifest
) -> None:
    with pytest.raises(ValueError, match="batch size is 8 but .* 4 patches"):
        training_loader(
            patch_dataset, manifest, 8, composition=BatchComposition(2, 1, 1)
        )


def test_a_composition_needs_a_dataset_that_cuts_patches(
    dataset: MammogramDataset, manifest: Manifest
) -> None:
    loader = training_loader(
        dataset, manifest, 2, samples_per_epoch=2, composition=BatchComposition(1, 0, 1)
    )
    with pytest.raises(ValueError, match="needs a transform that cuts patches"):
        next(iter(loader))


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
