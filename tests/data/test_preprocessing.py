from functools import partial

import numpy as np
import pytest
from pydantic import ValidationError

from mammo_seg.data.preprocessing import BoundingBox, breast_bbox


def image(
    shape: tuple[int, int] = (100, 80), **blobs: tuple[int, int, int, int]
) -> np.ndarray:
    """A black uint16 image with one bright rectangle per keyword.

    Each rectangle is given as (top, bottom, left, right), slice bounds.
    """
    pixels = np.zeros(shape, dtype=np.uint16)
    for top, bottom, left, right in blobs.values():
        pixels[top:bottom, left:right] = 2000
    return pixels


# The simplest settings: every pixel, no opening, no margin.
exact_bbox = partial(breast_bbox, downscale=1, opening=0, margin=0.0)


# --- BoundingBox ---


def test_a_box_has_a_size() -> None:
    box = BoundingBox(top=10, bottom=40, left=5, right=25)
    assert (box.height, box.width) == (30, 20)


def test_crop_returns_the_pixels_inside_the_box() -> None:
    pixels = np.arange(60, dtype=np.uint16).reshape(6, 10)
    cropped = BoundingBox(top=1, bottom=4, left=2, right=7).crop(pixels)
    assert cropped.shape == (3, 5)
    assert np.array_equal(cropped, pixels[1:4, 2:7])
    assert cropped.dtype == np.uint16
    assert np.shares_memory(cropped, pixels)


def test_crop_rejects_a_box_larger_than_the_image() -> None:
    with pytest.raises(ValueError, match="does not fit an image of \\(6, 10\\)"):
        BoundingBox(top=0, bottom=7, left=0, right=5).crop(np.zeros((6, 10)))
    with pytest.raises(ValueError, match="does not fit"):
        BoundingBox(top=0, bottom=5, left=0, right=11).crop(np.zeros((6, 10)))


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"top": -1, "bottom": 5, "left": 0, "right": 5}, "cannot start before"),
        ({"top": 0, "bottom": 5, "left": -2, "right": 5}, "cannot start before"),
        ({"top": 5, "bottom": 5, "left": 0, "right": 5}, "positive size"),
        ({"top": 0, "bottom": 5, "left": 6, "right": 5}, "positive size"),
    ],
)
def test_an_invalid_box_is_rejected(fields: dict[str, int], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        BoundingBox(**fields)


def test_a_box_with_float_bounds_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(top=0.5, bottom=5, left=0, right=5)  # type: ignore[arg-type]


# --- finding the breast ---


def test_the_box_is_the_extent_of_the_bright_region() -> None:
    box = exact_bbox(image(breast=(20, 70, 10, 50)))
    assert box == BoundingBox(top=20, bottom=70, left=10, right=50)


def test_the_box_can_crop_its_own_image() -> None:
    pixels = image(breast=(20, 70, 10, 50))
    cropped = exact_bbox(pixels).crop(pixels)
    assert cropped.shape == (50, 40)
    assert (cropped == 2000).all()


def test_the_margin_is_a_fraction_of_the_box_size_on_every_side() -> None:
    pixels = image((200, 300), breast=(50, 150, 100, 200))  # 100 x 100
    box = exact_bbox(pixels, margin=0.1)
    assert box == BoundingBox(top=40, bottom=160, left=90, right=210)


def test_the_margin_follows_each_side_of_a_non_square_box() -> None:
    pixels = image((300, 300), breast=(100, 200, 50, 250))  # 100 high, 200 wide
    box = exact_bbox(pixels, margin=0.1)
    assert (box.top, box.bottom) == (90, 210)
    assert (box.left, box.right) == (30, 270)


def test_the_margin_stops_at_the_image_border() -> None:
    pixels = image((100, 80), breast=(0, 60, 20, 80))
    box = exact_bbox(pixels, margin=0.5)
    assert box.top == 0
    assert box.right == 80
    assert (box.bottom, box.left) == (90, 0)


def test_stray_bright_pixels_do_not_widen_the_box() -> None:
    pixels = image(breast=(20, 70, 10, 50))
    pixels[2, 75] = 4000  # a speck in the corner
    pixels[95, 2] = 4000
    assert exact_bbox(pixels) == BoundingBox(top=20, bottom=70, left=10, right=50)


def test_the_largest_region_wins() -> None:
    pixels = image((120, 120), breast=(10, 100, 10, 60), label=(105, 115, 90, 118))
    assert exact_bbox(pixels) == BoundingBox(top=10, bottom=100, left=10, right=60)


def test_regions_touching_at_a_corner_are_one_region() -> None:
    pixels = image((60, 60), a=(10, 20, 10, 20), b=(20, 30, 20, 30))
    assert exact_bbox(pixels) == BoundingBox(top=10, bottom=30, left=10, right=30)


def test_only_pixels_above_the_threshold_count() -> None:
    pixels = image(breast=(20, 70, 10, 50))
    pixels[20:70, 50:60] = 100  # exactly at the default threshold: background
    pixels[20:70, 60:70] = 50
    assert exact_bbox(pixels).right == 50
    assert exact_bbox(pixels, threshold=99).right == 60
    assert exact_bbox(pixels, threshold=40).right == 70


def test_dim_background_noise_below_the_threshold_is_ignored() -> None:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 60, size=(100, 80), dtype=np.uint16)
    pixels[20:70, 10:50] = 1500
    assert exact_bbox(pixels) == BoundingBox(top=20, bottom=70, left=10, right=50)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32])
def test_any_numeric_image_type_works(dtype: type) -> None:
    pixels: np.ndarray = np.zeros((60, 60), dtype=dtype)
    pixels[10:40, 10:30] = 200
    assert exact_bbox(pixels) == BoundingBox(top=10, bottom=40, left=10, right=30)


# --- the default settings: fast mask and opening ---


def test_the_fast_mask_is_accurate_to_the_downscale() -> None:
    pixels = image((256, 256), breast=(43, 205, 61, 190))
    box = breast_bbox(pixels, downscale=8, opening=0, margin=0.0)
    assert abs(box.top - 43) < 8 and abs(box.bottom - 205) < 8
    assert abs(box.left - 61) < 8 and abs(box.right - 190) < 8


def test_opening_cuts_off_a_thin_strip_that_touches_the_breast() -> None:
    # a breast and a 6-pixel strip along the border that continues below it
    pixels = image((512, 384), breast=(40, 300, 100, 380), strip=(300, 512, 376, 382))
    with_strip = breast_bbox(pixels, downscale=8, opening=0, margin=0.0)
    without = breast_bbox(pixels, margin=0.0)  # the default opening
    assert with_strip.bottom >= 504
    assert without.bottom <= 304


def test_the_default_box_follows_the_breast_in_a_noisy_image() -> None:
    rng = np.random.default_rng(1)
    rows, cols = np.mgrid[0:400, 0:300]
    pixels: np.ndarray = rng.integers(0, 50, size=(400, 300)).astype(np.uint16)
    breast = ((rows - 200) / 150.0) ** 2 + ((cols - 40) / 200.0) ** 2 < 1
    pixels[breast] = 1200
    pixels[5:25, 250:290] = 4000  # a label in the corner
    box = breast_bbox(pixels, margin=0.0)
    assert abs(box.top - 50) <= 8 and abs(box.bottom - 350) <= 8
    assert box.left == 0
    assert abs(box.right - 240) <= 8


def test_the_default_margin_keeps_a_breast_that_fills_the_image() -> None:
    pixels = np.full((100, 80), 1500, dtype=np.uint16)
    box = breast_bbox(pixels)
    assert (box.top, box.left) == (0, 0)
    assert (box.bottom, box.right) == (100, 80)


# --- errors ---


def test_an_image_without_a_breast_is_rejected() -> None:
    with pytest.raises(ValueError, match="no breast found"):
        exact_bbox(np.zeros((50, 50), dtype=np.uint16))
    with pytest.raises(ValueError, match="no breast found"):
        exact_bbox(image(breast=(20, 70, 10, 50)), threshold=5000)


def test_a_breast_smaller_than_the_opening_is_rejected() -> None:
    pixels = image((100, 100), speck=(40, 42, 40, 42))
    with pytest.raises(ValueError, match="no breast found"):
        breast_bbox(pixels, downscale=1, opening=1, margin=0.0)


def test_an_image_that_is_not_two_dimensional_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected a 2-D image, got 3 dimensions"):
        breast_bbox(np.zeros((5, 5, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="expected a 2-D image, got 1 dimensions"):
        breast_bbox(np.zeros(5, dtype=np.uint16))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"margin": -0.1}, "margin cannot be negative"),
        ({"downscale": 0}, "downscale must be at least 1"),
        ({"opening": -1}, "opening at least 0"),
    ],
)
def test_invalid_arguments_are_rejected(
    arguments: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        breast_bbox(image(breast=(20, 70, 10, 50)), **arguments)  # type: ignore[arg-type]
