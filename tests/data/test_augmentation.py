from dataclasses import replace
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from tests.data.preprocessed import CANVAS, SETTINGS, write_preprocessed
from torchvision.transforms import v2

from mammo_seg.data.augmentation import (
    BREAST_THRESHOLD,
    Augmentation,
    AugmentationSettings,
    PatchKind,
    elastic,
    elastic_displacement,
)
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


def flat_zero(image: torch.Tensor) -> torch.Tensor:
    """The pixels of a ``[1, H, W]`` image whose 5 x 5 neighbourhood is all 0."""
    zero = (image[0] == 0).float()[None, None]
    return (torch.nn.functional.avg_pool2d(zero, 5, stride=1) == 1)[0, 0]


def test_the_zero_background_stays_zero() -> None:
    strong = AugmentationSettings(
        rotation=0, scale=0, shift=0, brightness=0.5, contrast=0.5, gamma=0.5,
        noise=0.05,
    )  # fmt: skip
    for seed in range(10):
        torch.manual_seed(seed)
        image, _ = Augmentation(strong)(*breast_sample())
        breast = breast_sample()[0] > 0
        assert (image[~breast] == 0).all()
        assert not torch.equal(image[breast], breast_sample()[0][breast])


def test_the_area_brought_in_by_the_geometry_stays_zero() -> None:
    # The geometric changes draw their random numbers first, so the same seed gives
    # the same geometry with and without intensity changes.
    geometry = replace(NOTHING, rotation=30, shift=0.2)
    everything = replace(geometry, brightness=0.5, contrast=0.5, gamma=0.5, noise=0.05)
    brought_in = 0
    for seed in range(10):
        torch.manual_seed(seed)
        moved, _ = Augmentation(geometry)(
            torch.full((1, 64, 48), 0.4), torch.zeros(1, 64, 48)
        )
        torch.manual_seed(seed)
        changed, _ = Augmentation(everything)(
            torch.full((1, 64, 48), 0.4), torch.zeros(1, 64, 48)
        )
        assert (changed[moved == 0] == 0).all()
        assert not torch.equal(changed[moved > 0], moved[moved > 0])
        brought_in += int((moved == 0).sum())
    assert brought_in > 0


def test_the_default_augmentation_keeps_a_flat_background() -> None:
    # Validation images keep their flat zero background; training patches must too.
    for seed in range(10):
        torch.manual_seed(seed)
        image, _ = Augmentation()(*breast_sample())
        assert flat_zero(image).any()


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


def test_the_elastic_deformation_is_off_by_default() -> None:
    settings = AugmentationSettings()
    assert settings.elastic == 0 and settings.elastic_spacing == 64


def test_the_elastic_deformation_bends_the_mask_with_the_image() -> None:
    bend = replace(NOTHING, elastic=2.0, elastic_spacing=16)
    changed = 0
    for seed in range(10):
        torch.manual_seed(seed)
        image, mask = Augmentation(bend)(*lesion_sample())
        assert ((mask == 0) | (mask == 1)).all()
        assert overlap(image[0] > 0.6, mask[0] > 0.5) > 0.85
        changed += not torch.equal(mask, lesion_sample()[1])
    assert changed == 10


def test_the_elastic_displacement_has_the_asked_size_and_is_smooth() -> None:
    torch.manual_seed(0)
    shift = torch.stack([elastic_displacement(256, 200, 4.0, 32) for _ in range(20)])
    assert shift.shape == (20, 2, 256, 200)
    # Bicubic interpolation between the control points lowers the spread a little.
    assert 0.7 * 4.0 < float(shift.std()) < 4.0
    # Neighbouring pixels move almost alike: a smooth bend, not noise, for which
    # their difference would spread 1.4 times as much as the displacement itself.
    neighbours = float((shift[..., 1:] - shift[..., :-1]).std())
    assert neighbours < 0.1 * float(shift.std())


def test_the_area_brought_in_by_the_elastic_deformation_stays_zero() -> None:
    image = torch.full((1, 64, 48), 0.4)
    torch.manual_seed(3)
    bent, _ = elastic(image, torch.zeros(1, 64, 48), strength=3.0, spacing=16)
    assert (bent == 0).any()  # some border pixels come from outside the image
    assert bent.min() >= 0 and bent.max() <= 0.4 + 1e-6
    # Away from the borders a flat image stays flat: nothing is invented.
    assert torch.allclose(bent[0, 16:48, 12:36], torch.tensor(0.4))


def test_a_strong_elastic_deformation_keeps_the_mask_binary() -> None:
    # Values in between would mean a mask that was interpolated.
    torch.manual_seed(0)
    _, mask = elastic(*lesion_sample(), strength=10.0, spacing=8)
    assert ((mask == 0) | (mask == 1)).all() and mask.sum() > 0


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


def test_a_crop_larger_than_the_image_keeps_the_whole_image_on_that_side() -> None:
    sample = lesion_sample()  # 64 x 48
    for crop_size, expected in [
        ((80, 60), (64, 48)),  # larger on both sides
        ((80, 24), (64, 24)),  # larger in height only
        ((32, 60), (32, 48)),  # larger in width only
    ]:
        crop = replace(NOTHING, crop_size=crop_size)
        torch.manual_seed(0)
        image, mask = Augmentation(crop)(*sample)
        assert image.shape == mask.shape == (1, *expected)
    # With nothing to move, a patch as large as the image is the image.
    torch.manual_seed(0)
    image, mask = Augmentation(replace(NOTHING, crop_size=(80, 60)))(*sample)
    assert torch.equal(image, sample[0]) and torch.equal(mask, sample[1])


# --- patches of a kind ---


def breast_sample() -> tuple[torch.Tensor, torch.Tensor]:
    """A 64 x 48 image that is black except for a breast (rows 8 to 56, columns 0 to
    24) with a 16 x 16 mass in it; the mask marks the mass."""
    image = torch.zeros(1, 64, 48)
    image[0, 8:56, :24] = 0.4
    image[0, 24:40, 4:20] = 0.9
    mask = torch.zeros(1, 64, 48)
    mask[0, 24:40, 4:20] = 1
    return image, mask


def typed(
    kind: PatchKind, seeds: int = 40, **changes: object
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    settings = replace(NOTHING, crop_size=(16, 16), **changes)  # type: ignore[arg-type]
    found = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        found.append(Augmentation(settings)(*breast_sample(), kind))
    return found


def test_a_mass_patch_always_contains_part_of_the_mass() -> None:
    patches = typed("mass")
    assert all(mask.sum() > 0 for _, mask in patches)
    assert all(image.shape == mask.shape == (1, 16, 16) for image, mask in patches)
    assert len({tuple(torch.nonzero(mask[0])[0].tolist()) for _, mask in patches}) > 3


def test_a_negative_patch_never_contains_the_mass() -> None:
    patches = typed("negative")
    assert all(mask.sum() == 0 for _, mask in patches)
    assert all(image.shape == mask.shape == (1, 16, 16) for image, mask in patches)


def test_a_negative_patch_always_holds_breast_where_random_patches_do_not() -> None:
    # The breast covers the left half of the image: a random patch of 16 x 16 is
    # sometimes only background, one centred on a breast pixel never is.
    assert all((image > BREAST_THRESHOLD).any() for image, _ in typed("negative"))
    random = Augmentation(replace(NOTHING, crop_size=(16, 16)))
    background_only = 0
    for seed in range(40):
        torch.manual_seed(seed)
        image, _ = random(*breast_sample())
        background_only += int(not (image > BREAST_THRESHOLD).any())
    assert background_only > 0


def test_a_negative_patch_is_centred_on_the_breast_with_intensity_changes() -> None:
    # Gamma below 1 lifts small values, which could push a noised background above
    # the breast threshold; the background stays 0, so the centre is always breast.
    patches = typed("negative", brightness=0.5, contrast=0.5, gamma=0.5, noise=0.05)
    for image, _ in patches:
        assert image[0, 8, 8] > BREAST_THRESHOLD


def test_the_kind_overrides_the_share_of_patches_around_a_mass() -> None:
    assert all(mask.sum() > 0 for _, mask in typed("mass", crop_mass_fraction=0.0))
    assert all(mask.sum() == 0 for _, mask in typed("negative", crop_mass_fraction=1.0))


def test_patches_of_a_kind_are_repeatable_with_the_same_seed() -> None:
    for kind in ("mass", "negative"):
        for (a, b), (c, d) in zip(typed(kind, 5), typed(kind, 5), strict=True):
            assert torch.equal(a, c) and torch.equal(b, d)


def test_a_negative_patch_of_an_image_without_a_mass() -> None:
    image, _ = breast_sample()
    torch.manual_seed(0)
    patch, patch_mask = Augmentation(replace(NOTHING, crop_size=(16, 16)))(
        image, torch.zeros_like(image), "negative"
    )
    assert patch.shape == patch_mask.shape == (1, 16, 16) and patch_mask.sum() == 0


def test_a_mass_patch_of_an_image_without_a_mass_is_placed_at_random() -> None:
    image, _ = breast_sample()
    torch.manual_seed(0)
    patch, patch_mask = Augmentation(replace(NOTHING, crop_size=(16, 16)))(
        image, torch.zeros_like(image), "mass"
    )
    assert patch.shape == patch_mask.shape == (1, 16, 16) and patch_mask.sum() == 0


def test_a_negative_patch_of_an_image_without_breast_is_placed_anywhere() -> None:
    blank = torch.zeros(1, 64, 48)
    torch.manual_seed(0)
    patch, patch_mask = Augmentation(replace(NOTHING, crop_size=(16, 16)))(
        blank, blank.clone(), "negative"
    )
    assert patch.shape == patch_mask.shape == (1, 16, 16)


def test_a_negative_patch_is_drawn_again_until_it_avoids_the_mass() -> None:
    # The mass fills the left half of the image: only patches on the right are
    # free, and a third of the possible centres lead to one.
    image = torch.full((1, 64, 48), 0.4)
    mask = torch.zeros(1, 64, 48)
    mask[0, :, :24] = 1
    for seed in range(20):
        torch.manual_seed(seed)
        _, patch_mask = Augmentation(replace(NOTHING, crop_size=(16, 16)))(
            image, mask, "negative"
        )
        assert patch_mask.sum() == 0


def test_when_no_patch_avoids_the_mass_the_one_with_the_least_is_used() -> None:
    # The mass fills all of the image but the last 8 columns: no patch is free, and
    # a patch at the right edge holds half as many mass pixels as one elsewhere.
    image = torch.full((1, 64, 48), 0.4)
    mask = torch.zeros(1, 64, 48)
    mask[0, :, :40] = 1
    held = []
    for seed in range(20):
        torch.manual_seed(seed)
        patch, patch_mask = Augmentation(replace(NOTHING, crop_size=(16, 16)))(
            image, mask, "negative"
        )
        assert patch.shape == (1, 16, 16)
        held.append(int(patch_mask.sum()))
    assert all(count < 16 * 16 for count in held)
    assert min(held) == 8 * 16


def test_a_patch_of_a_kind_on_a_small_image_is_the_whole_image_on_that_side() -> None:
    image, mask = torch.full((1, 10, 8), 0.4), torch.zeros(1, 10, 8)
    for kind in ("mass", "negative"):
        patch, patch_mask = Augmentation(replace(NOTHING, crop_size=(16, 16)))(
            image, mask, kind
        )
        assert patch.shape == patch_mask.shape == (1, 10, 8)


@pytest.mark.parametrize("kind", ["mass", "negative"])
def test_a_kind_needs_a_crop_size(kind: PatchKind) -> None:
    with pytest.raises(ValueError, match="needs crop_size"):
        Augmentation(NOTHING)(*lesion_sample(), kind)


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
        {"elastic": -1.0},
        {"elastic_spacing": 1},
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


def test_the_padding_of_a_crop_of_a_small_image_is_not_valid(tmp_path: Path) -> None:
    write_preprocessed(tmp_path, SETTINGS)  # 64 high; 30, 48 and 20 px wide
    crop = replace(NOTHING, crop_size=(32, 48))
    dataset = MammogramDataset(
        tmp_path,
        ["00000001", "00000002", "00000003"],
        SETTINGS,
        (32, 48),
        transform=Augmentation(crop),
    )
    torch.manual_seed(0)
    narrow, full, narrowest = dataset[0], dataset[1], dataset[2]
    assert narrow["image"].shape == (1, 32, 48)
    assert narrow["valid"][0, :, :30].all() and not narrow["valid"][0, :, 30:].any()
    assert (narrow["image"][0, :, 30:] == 0).all()
    assert full["valid"].all()
    assert narrowest["valid"][0, :, :20].all()
    assert not narrowest["valid"][0, :, 20:].any()
