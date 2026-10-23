from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from tests.data.preprocessed import CANVAS, SETTINGS, write_preprocessed
from torchvision.transforms import v2

from mammo_seg.data.augmentation import Augmentation, AugmentationSettings
from mammo_seg.data.dataset import MammogramDataset

NOTHING = AugmentationSettings(
    rotation=0, scale=0, shift=0, brightness=0, contrast=0, gamma=0, noise=0
)


def lesion_sample() -> tuple[torch.Tensor, torch.Tensor]:
    """A gray 64 x 48 image with a bright square, and the square's mask."""
    image = torch.full((1, 64, 48), 0.3)
    image[0, 20:36, 16:32] = 0.9
    mask = torch.zeros(1, 64, 48)
    mask[0, 20:36, 16:32] = 1
    return image, mask


def overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a & b).sum() / (a | b).sum())


def test_the_defaults_are_mild() -> None:
    settings = AugmentationSettings()
    assert (settings.rotation, settings.scale, settings.shift) == (10.0, 0.1, 0.1)
    assert (settings.brightness, settings.contrast, settings.gamma) == (0.1, 0.1, 0.2)
    assert settings.noise == 0.01
    assert settings.crop_size is None and settings.crop_mass_fraction == 0.5


def test_shapes_and_types_are_kept() -> None:
    torch.manual_seed(0)
    image, mask = Augmentation()(*lesion_sample())
    assert image.shape == mask.shape == (1, 64, 48)
    assert type(image) is torch.Tensor and type(mask) is torch.Tensor
    assert image.dtype == torch.float32 and mask.dtype == torch.float32


def test_the_image_stays_in_0_to_1_and_the_mask_stays_binary() -> None:
    strong = AugmentationSettings(rotation=30, brightness=0.5, contrast=0.5, noise=0.2)
    for seed in range(10):
        torch.manual_seed(seed)
        image, mask = Augmentation(strong)(*lesion_sample())
        assert image.min() >= 0 and image.max() <= 1
        assert ((mask == 0) | (mask == 1)).all()


def test_switched_off_augmentation_changes_nothing() -> None:
    image, mask = lesion_sample()
    new_image, new_mask = Augmentation(NOTHING)(image, mask)
    assert torch.equal(new_image, image) and torch.equal(new_mask, mask)


def test_the_mask_moves_with_the_image() -> None:
    geometric_only = AugmentationSettings(
        rotation=20, scale=0.2, shift=0.2, brightness=0, contrast=0, gamma=0, noise=0
    )
    for seed in range(10):
        torch.manual_seed(seed)
        image, mask = Augmentation(geometric_only)(*lesion_sample())
        assert overlap(image[0] > 0.6, mask[0] > 0.5) > 0.85


def test_the_geometry_really_changes() -> None:
    torch.manual_seed(1)
    _, mask = Augmentation(AugmentationSettings(shift=0.3))(*lesion_sample())
    assert not torch.equal(mask, lesion_sample()[1])


def test_intensity_changes_leave_the_mask_alone() -> None:
    intensity_only = AugmentationSettings(
        rotation=0,
        scale=0,
        shift=0,
        brightness=0.5,
        contrast=0.5,
        gamma=0.5,
        noise=0.05,
    )
    torch.manual_seed(2)
    image, mask = Augmentation(intensity_only)(*lesion_sample())
    assert torch.equal(mask, lesion_sample()[1])
    assert not torch.equal(image, lesion_sample()[0])


def test_gamma_keeps_the_order_of_intensities() -> None:
    gamma_only = AugmentationSettings(
        rotation=0, scale=0, shift=0, brightness=0, contrast=0, gamma=0.5, noise=0
    )
    ramp = torch.linspace(0, 1, 48).repeat(1, 64, 1)
    torch.manual_seed(3)
    image, _ = Augmentation(gamma_only)(ramp, torch.zeros(1, 64, 48))
    assert (image[0, 0].diff() >= 0).all()
    assert image[0, 0, 0] == 0 and image[0, 0, -1] == 1


def test_there_is_no_flip() -> None:
    transforms = [
        *Augmentation().geometric.transforms,
        *Augmentation().intensity.transforms,
    ]
    flips = (v2.RandomHorizontalFlip, v2.RandomVerticalFlip)
    assert not any(isinstance(transform, flips) for transform in transforms)


def test_the_same_seed_gives_the_same_result() -> None:
    torch.manual_seed(7)
    first = Augmentation()(*lesion_sample())
    torch.manual_seed(7)
    second = Augmentation()(*lesion_sample())
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_a_random_crop_returns_a_patch_of_image_and_mask() -> None:
    crop = AugmentationSettings(
        rotation=0, scale=0, shift=0, brightness=0, contrast=0, gamma=0, noise=0,
        crop_size=(32, 24),
    )  # fmt: skip
    for seed in range(10):
        torch.manual_seed(seed)
        image, mask = Augmentation(crop)(*lesion_sample())
        assert image.shape == mask.shape == (1, 32, 24)
        assert torch.equal(image > 0.6, mask > 0.5)


def crops(fraction: float, seeds: int = 40) -> list[tuple[torch.Tensor, torch.Tensor]]:
    settings = AugmentationSettings(
        rotation=0, scale=0, shift=0, brightness=0, contrast=0, gamma=0, noise=0,
        crop_size=(16, 16), crop_mass_fraction=fraction,
    )  # fmt: skip
    found = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        found.append(Augmentation(settings)(*lesion_sample()))
    return found


def test_crops_around_a_mass_always_contain_part_of_it() -> None:
    patches = crops(1.0)
    assert all(mask.sum() > 0 for _, mask in patches)
    assert all(image.shape == mask.shape == (1, 16, 16) for image, mask in patches)
    # The mass is not always at the same place of the patch.
    assert len({tuple(torch.nonzero(mask[0])[0].tolist()) for _, mask in patches}) > 3


def test_crops_at_random_often_miss_a_small_mass() -> None:
    patches = crops(0.0)
    misses = sum(1 for _, mask in patches if mask.sum() == 0)
    assert 0 < misses < len(patches)  # random, not always off and not always on


def test_the_share_of_crops_around_a_mass_follows_the_setting() -> None:
    # The 16 x 16 mass sits in a 64 x 48 image: a random 16 x 16 patch holds part of
    # it far less often than half of the time, a mass-centred one half of the time
    # or more.
    share = sum(1 for _, mask in crops(0.5, 200) if mask.sum() > 0) / 200
    assert 0.5 <= share <= 0.8


def test_a_crop_of_an_image_without_a_mass_is_placed_at_random() -> None:
    image, mask = torch.rand(1, 64, 48), torch.zeros(1, 64, 48)
    settings = AugmentationSettings(
        rotation=0, scale=0, shift=0, brightness=0, contrast=0, gamma=0, noise=0,
        crop_size=(16, 16), crop_mass_fraction=1.0,
    )  # fmt: skip
    torch.manual_seed(0)
    patch, patch_mask = Augmentation(settings)(image, mask)
    assert patch.shape == patch_mask.shape == (1, 16, 16) and patch_mask.sum() == 0


def test_crops_are_repeatable_with_the_same_seed() -> None:
    first, second = crops(0.5, 5), crops(0.5, 5)
    for (a, b), (c, d) in zip(first, second, strict=True):
        assert torch.equal(a, c) and torch.equal(b, d)


def test_a_crop_larger_than_the_image_is_padded() -> None:
    crop = AugmentationSettings(crop_size=(80, 60))
    torch.manual_seed(0)
    image, mask = Augmentation(crop)(*lesion_sample())
    assert image.shape == mask.shape == (1, 80, 60)


@pytest.mark.parametrize(
    "changes",
    [
        {"rotation": -1.0},
        {"rotation": 181.0},
        {"scale": 1.0},
        {"shift": 1.5},
        {"brightness": 1.0},
        {"gamma": -0.1},
        {"noise": -0.01},
        {"crop_size": (0, 10)},
        {"crop_mass_fraction": -0.1},
        {"crop_mass_fraction": 1.1},
    ],
)
def test_invalid_settings_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AugmentationSettings(**changes)  # type: ignore[arg-type]


def test_the_dataset_applies_the_augmentation_before_padding(tmp_path: Path) -> None:
    write_preprocessed(tmp_path, SETTINGS)
    crop = AugmentationSettings(crop_size=(32, 24))
    dataset = MammogramDataset(
        tmp_path, ["00000001"], SETTINGS, CANVAS, transform=Augmentation(crop)
    )
    torch.manual_seed(0)
    sample = dataset[0]
    assert sample["image"].shape == (1, 64, 48)
    assert sample["valid"][0, :32, :24].all() and not sample["valid"][0, 32:].any()
