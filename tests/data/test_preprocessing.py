from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import TypeAdapter, ValidationError
from tests.data.synthetic import write_dicom_pixels

from mammo_seg.data.masks import rasterize
from mammo_seg.data.preprocessing import (
    DEFAULT_MARGIN,
    DEFAULT_MAX_INTENSITY,
    DEFAULT_SIZE,
    BoundingBox,
    Breast,
    Edges,
    PreprocessingSettings,
    PreprocessParams,
    breast_bbox,
    breast_region,
    bright_mask,
    chest_wall_on_right,
    clear_edges,
    film_edges,
    find_breast,
    kept_area,
    largest_region,
    normalize_intensity,
    preprocess,
    preprocess_mask,
    preprocess_points,
    read_pixels,
    region_bbox,
    restore_mask,
    target_shape,
)
from mammo_seg.data.schema import LesionType, Roi


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


def test_a_border_leaves_out_bright_strips_along_the_edges() -> None:
    # A strip along the top edge touches the breast; leaving out the outer
    # 3 pixels separates them.
    pixels = image(breast=(2, 60, 10, 50), strip=(0, 3, 0, 80))
    assert exact_bbox(pixels) == BoundingBox(top=0, bottom=60, left=0, right=80)
    assert exact_bbox(pixels, border=3) == BoundingBox(
        top=3, bottom=60, left=10, right=50
    )


def test_the_region_is_the_breast_on_every_downscaled_pixel() -> None:
    region = breast_region(image(breast=(20, 60, 10, 50)), downscale=2, opening=0)
    assert region.shape == (50, 40)
    assert region.sum() == 20 * 20
    box = region_bbox(region, (100, 80), downscale=2, margin=0.0)
    assert box == BoundingBox(top=20, bottom=60, left=10, right=50)
    with pytest.raises(ValueError, match="margin cannot be negative"):
        region_bbox(region, (100, 80), margin=-0.1)


def wall_image(wall: str) -> np.ndarray:
    """A breast filling the height on one side, brightest there, fading to the
    nipple on the other."""
    pixels = np.zeros((60, 80), dtype=np.uint16)
    pixels[5:55, 5:45] = np.linspace(3000, 300, 40, dtype=np.uint16)
    return pixels if wall == "left" else np.ascontiguousarray(pixels[:, ::-1])


@pytest.mark.parametrize("wall", ["left", "right"])
def test_the_chest_wall_is_where_the_breast_is_brightest(wall: str) -> None:
    pixels = wall_image(wall)
    region = breast_region(pixels, downscale=1, opening=0)
    assert chest_wall_on_right(pixels, region, downscale=1) is (wall == "right")


def test_dark_background_on_one_side_does_not_decide_the_side() -> None:
    # Bright labels outside the breast region do not count.
    pixels = wall_image("left")
    pixels[50:58, 70:78] = 4000
    region = breast_region(pixels, downscale=1, opening=0)
    assert not chest_wall_on_right(pixels, region, downscale=1)


@pytest.mark.parametrize(("wall", "laterality"), [("left", "R"), ("right", "L")])
def test_by_image_the_flip_follows_the_chest_wall_not_the_laterality(
    wall: str, laterality: str
) -> None:
    settings = PreprocessingSettings(
        downscale=1, opening=0, margin=0.0, size=40, orientation="image"
    )
    result, params = preprocess(wall_image(wall), laterality, settings)  # type: ignore[arg-type]
    assert params.flipped is (wall == "right")
    assert result[:, :5].mean() > result[:, -5:].mean()  # the wall is on the left


def film(wall: str = "left") -> np.ndarray:
    """A 100 x 80 film: a breast at the chest wall, bright film edges along the top
    and the far side touching it, and a label in the background."""
    pixels = image(breast=(20, 80, 0, 40), label=(85, 92, 60, 70))
    pixels[:3] = 2000  # the top edge of the film
    pixels[:, 77:] = 2000  # the far side
    pixels[40:45, 40:77] = 2000  # a strip from the breast to the far edge
    return pixels if wall == "left" else np.ascontiguousarray(pixels[:, ::-1])


def edges_of(mask: np.ndarray, wall_on_right: bool = False, **kwargs: float) -> Edges:
    return film_edges(mask, depth=0.05, wall_on_right=wall_on_right, **kwargs)


@pytest.mark.parametrize("wall", ["left", "right"])
def test_film_edges_on_the_sides_away_from_the_chest_wall_are_found(
    wall: str,
) -> None:
    mask = bright_mask(film(wall), downscale=1)
    edges = edges_of(mask, wall == "right")
    assert edges == Edges(top=3, bottom=0, far=3)
    cleared = clear_edges(mask, edges, wall == "right")
    assert not cleared[:3].any()  # the top edge
    far = slice(77, 80) if wall == "left" else slice(0, 3)
    assert not cleared[:, far].any()  # the far side
    breast = slice(0, 40) if wall == "left" else slice(40, 80)
    assert cleared[20:80, breast].all()  # the breast is kept
    assert mask[:3].all()  # the input is not changed


def test_the_chest_wall_side_is_never_a_film_edge() -> None:
    mask = np.zeros((100, 80), dtype=bool)
    mask[:, :2] = True  # a full column at the chest wall
    assert edges_of(mask) == Edges(0, 0, 0)
    assert edges_of(mask, wall_on_right=True).far == 2  # there, it is the far side


def test_a_bright_run_deeper_than_the_depth_is_tissue_and_kept() -> None:
    mask = np.zeros((100, 80), dtype=bool)
    mask[:8] = True  # 8% of the height: deeper than 5%
    mask[-3:] = True  # 3%: a film edge
    assert edges_of(mask) == Edges(top=0, bottom=3, far=0)


def test_rows_less_bright_than_the_coverage_are_not_a_film_edge() -> None:
    mask = np.zeros((100, 80), dtype=bool)
    mask[:3, :60] = True  # 75% of each row
    assert edges_of(mask).top == 0
    assert edges_of(mask, coverage=0.7).top == 3


def test_an_image_filled_with_bright_rows_has_no_film_edge() -> None:
    assert edges_of(np.ones((100, 80), dtype=bool)) == Edges(0, 0, 0)


def test_film_edges_that_touch_the_breast_are_cleared_before_finding_it() -> None:
    pixels = film()
    plain = PreprocessingSettings(downscale=1, opening=0, margin=0.0)
    region, flipped, _ = find_breast(pixels, "L", plain)
    assert region[:3].any() and not flipped
    with_edges = replace(plain, film_edge_depth=0.05)
    region = find_breast(pixels, "L", with_edges).region
    box = region_bbox(region, pixels.shape, downscale=1, margin=0.0)
    # The strip from the breast to the far edge is still there, but no longer
    # reaches the cleared top edge or the far side.
    assert (box.top, box.bottom, box.left, box.right) == (20, 80, 0, 77)


def test_the_kept_area_is_the_region_widened_and_scaled_up() -> None:
    region = np.zeros((10, 8), dtype=bool)
    region[4:6, 3:5] = True
    breast = Breast(region, False, region.copy())
    kept = kept_area(breast, (79, 62), downscale=8, widen=1)
    assert kept.shape == (79, 62)
    expected = np.zeros((79, 62), dtype=bool)
    expected[3 * 8 : 7 * 8, 2 * 8 : 6 * 8] = True
    assert np.array_equal(kept, expected)
    assert kept_area(breast, (80, 64), downscale=8, widen=0).sum() == 4 * 64


def test_dim_speckled_tissue_next_to_the_breast_is_kept_and_a_label_is_not() -> None:
    region = np.zeros((40, 40), dtype=bool)
    region[10:30, :15] = True
    bright = region.copy()
    bright[10:30, 15:25:2] = True  # speckle: every other column, gaps of 1 pixel
    bright[34:37, 34:37] = True  # a label apart from the breast
    bright[2:5, 20:23] = True  # a ring around a hole is filled
    kept = kept_area(Breast(region, False, bright), (40, 40), downscale=1, widen=0)
    assert kept[10:30, :24].all()
    assert not kept[34:37, 34:37].any()
    assert not kept[2:5, 20:23].any()  # not connected to the breast


def test_clearing_the_background_removes_labels_and_keeps_the_breast() -> None:
    pixels = image(breast=(20, 80, 0, 40), label=(85, 95, 50, 78))
    settings = PreprocessingSettings(
        downscale=1, opening=0, margin=0.5, size=100, clear_background=2
    )
    result, params = preprocess(pixels, "L", settings)
    assert params.box.bottom == 100  # the margin reaches the label
    rows = slice(85 - params.box.top, 95 - params.box.top)
    assert not result[rows, 50:78].any()  # the label is cleared
    kept = preprocess(pixels, "L", replace(settings, clear_background=None))[0]
    assert kept[rows, 50:78].all()
    assert np.array_equal(result[: 80 - params.box.top], kept[: 80 - params.box.top])


def test_the_mask_and_region_helpers_reject_bad_arguments() -> None:
    with pytest.raises(ValueError, match="downscale must be at least 1"):
        bright_mask(image(), downscale=0)
    with pytest.raises(ValueError, match="border at least 0"):
        largest_region(np.ones((4, 4), dtype=bool), border=-1)
    with pytest.raises(ValueError, match="no breast found"):
        largest_region(np.zeros((4, 4), dtype=bool))


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
        ({"border": -1}, "border at least 0"),
    ],
)
def test_invalid_arguments_are_rejected(
    arguments: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        breast_bbox(image(breast=(20, 70, 10, 50)), **arguments)  # type: ignore[arg-type]


# --- the chain: crop, rescale, flip ---

# Every pixel of the breast, no margin and no opening, so geometry is easy to follow.
EXACT_SETTINGS = PreprocessingSettings(margin=0.0, downscale=1, opening=0, size=100)


def two_tone_breast() -> np.ndarray:
    """A 400 x 300 image with a 200 x 100 breast: dark left half, bright right half."""
    pixels = np.zeros((400, 300), dtype=np.uint16)
    pixels[100:300, 100:150] = 1000
    pixels[100:300, 150:200] = 3000
    return pixels


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (2000, 800, (1024, 410)),
        (800, 2000, (410, 1024)),
        (1024, 1024, (1024, 1024)),
        (100, 100000, (1, 1024)),
    ],
)
def test_the_longer_side_gets_the_target_size(
    height: int, width: int, expected: tuple[int, int]
) -> None:
    assert target_shape(height, width, 1024) == expected


def test_the_result_is_cropped_and_rescaled_by_averaging() -> None:
    result, params = preprocess(two_tone_breast(), "L", EXACT_SETTINGS)
    assert result.shape == (100, 50)  # the 200 x 100 crop at half size
    assert result.dtype == np.uint16
    assert (result[:, :25] == 1000).all()
    assert (result[:, 25:] == 3000).all()
    assert params.box == BoundingBox(top=100, bottom=300, left=100, right=200)
    assert params.shape == (100, 50)
    assert params.source_shape == (400, 300)


def test_averaging_blends_pixels_inside_each_target_pixel() -> None:
    pixels = np.zeros((40, 40), dtype=np.uint16)
    pixels[10:30, 10:30] = np.tile(
        np.array([[1000, 3000], [3000, 1000]], dtype=np.uint16), (10, 10)
    )
    result, _ = preprocess(
        pixels, "L", PreprocessingSettings(margin=0.0, downscale=1, opening=0, size=10)
    )
    assert result.shape == (10, 10)
    assert (result == 2000).all()


def test_a_left_breast_is_not_flipped_and_a_right_breast_is() -> None:
    left, left_params = preprocess(two_tone_breast(), "L", EXACT_SETTINGS)
    right, right_params = preprocess(two_tone_breast(), "R", EXACT_SETTINGS)
    assert not left_params.flipped and right_params.flipped
    assert left[0, 0] == 1000 and left[0, -1] == 3000
    assert right[0, 0] == 3000 and right[0, -1] == 1000
    assert np.array_equal(right, left[:, ::-1])


def test_the_longer_side_is_the_size_in_the_settings() -> None:
    pixels = two_tone_breast()
    for size in (64, 200, 300):
        result, params = preprocess(
            pixels,
            "L",
            PreprocessingSettings(margin=0.0, size=size, downscale=1, opening=0),
        )
        assert max(result.shape) == size
        assert result.shape == params.shape


def test_the_default_settings_make_a_1024_pixel_image() -> None:
    pixels = np.zeros((4084, 3328), dtype=np.uint16)
    pixels[300:3800, 1500:3328] = 1500
    result, params = preprocess(pixels, "R")
    assert max(result.shape) == DEFAULT_SIZE
    assert result.shape == params.shape
    assert params.box.top < 300 and params.box.bottom > 3800  # the margin


def test_preprocessing_is_deterministic() -> None:
    first, first_params = preprocess(two_tone_breast(), "R", EXACT_SETTINGS)
    second, second_params = preprocess(two_tone_breast(), "R", EXACT_SETTINGS)
    assert np.array_equal(first, second)
    assert first_params == second_params


def test_the_input_is_not_modified() -> None:
    pixels = two_tone_breast()
    before = pixels.copy()
    preprocess(pixels, "R", EXACT_SETTINGS)
    assert np.array_equal(pixels, before)


def test_an_unknown_laterality_is_rejected() -> None:
    with pytest.raises(ValueError, match="laterality must be L or R"):
        preprocess(two_tone_breast(), "X", EXACT_SETTINGS)  # type: ignore[arg-type]


def test_an_image_without_a_breast_is_rejected_by_the_chain() -> None:
    with pytest.raises(ValueError, match="no breast found"):
        preprocess(np.zeros((100, 100), dtype=np.uint16), "L", EXACT_SETTINGS)


# --- settings and parameters ---


def test_the_default_settings_are_the_documented_ones() -> None:
    settings = PreprocessingSettings()
    assert (settings.threshold, settings.downscale, settings.opening) == (100, 8, 1)
    assert settings.margin == DEFAULT_MARGIN
    assert settings.size == DEFAULT_SIZE == 1024
    assert settings.max_intensity == DEFAULT_MAX_INTENSITY == 4095
    assert settings.percentile is None
    assert (settings.border, settings.orientation) == (0, "laterality")
    assert (settings.film_edge_depth, settings.film_edge_coverage) == (0.0, 0.8)
    assert settings.clear_background is None


def test_settings_written_before_border_and_orientation_still_read() -> None:
    old = (
        '{"threshold": 100, "margin": 0.02, "downscale": 8, "opening": 1, '
        '"size": 1024, "max_intensity": 4095, "percentile": null}'
    )
    read = TypeAdapter(PreprocessingSettings).validate_json(old)
    assert read == PreprocessingSettings()


@pytest.mark.parametrize(
    "changes",
    [
        {"threshold": -1},
        {"margin": -0.1},
        {"downscale": 0},
        {"opening": -1},
        {"size": 0},
        {"max_intensity": 0},
        {"percentile": 0},
        {"percentile": 100.5},
        {"border": -1},
        {"orientation": "nipple"},
        {"film_edge_depth": -0.1},
        {"film_edge_depth": 1.0},
        {"film_edge_coverage": 0},
        {"clear_background": -1},
    ],
)
def test_invalid_settings_are_rejected(changes: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        PreprocessingSettings(**changes)  # type: ignore[arg-type]


def test_settings_do_not_convert_types() -> None:
    with pytest.raises(ValidationError):
        PreprocessingSettings(size="1024")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PreprocessingSettings(size=1024.0)  # type: ignore[arg-type]


def test_params_reject_a_box_outside_the_source_image() -> None:
    box = BoundingBox(top=0, bottom=500, left=0, right=100)
    with pytest.raises(ValidationError, match="does not fit"):
        PreprocessParams(source_shape=(400, 300), box=box, shape=(10, 5), flipped=False)


def test_params_reject_an_empty_shape() -> None:
    box = BoundingBox(top=0, bottom=100, left=0, right=100)
    with pytest.raises(ValidationError):
        PreprocessParams(source_shape=(400, 300), box=box, shape=(0, 5), flipped=False)


# --- masks follow the image ---


def lesion_image_and_mask() -> tuple[np.ndarray, np.ndarray]:
    """A breast with a bright lesion near its right side, and the lesion's mask."""
    pixels = np.zeros((600, 500), dtype=np.uint16)
    pixels[100:500, 100:400] = 1000
    pixels[250:330, 300:380] = 4000
    mask = np.zeros((600, 500), dtype=np.uint8)
    mask[250:330, 300:380] = 1
    return pixels, mask


@pytest.mark.parametrize("laterality", ["L", "R"])
def test_the_mask_stays_on_the_lesion_after_preprocessing(laterality: str) -> None:
    pixels, mask = lesion_image_and_mask()
    settings = PreprocessingSettings(margin=0.05, downscale=1, opening=0, size=200)
    image, params = preprocess(pixels, laterality, settings)  # type: ignore[arg-type]
    small_mask = preprocess_mask(mask, params)
    assert small_mask.shape == image.shape
    assert small_mask.dtype == np.uint8
    assert set(np.unique(small_mask)) == {0, 1}
    lesion = image > 2500
    overlap = (lesion & (small_mask == 1)).sum()
    assert overlap / (lesion | (small_mask == 1)).sum() > 0.9


def test_a_flipped_mask_is_on_the_other_side() -> None:
    pixels, mask = lesion_image_and_mask()
    settings = PreprocessingSettings(margin=0.0, downscale=1, opening=0, size=100)
    _, left_params = preprocess(pixels, "L", settings)
    _, right_params = preprocess(pixels, "R", settings)
    left = preprocess_mask(mask, left_params)
    right = preprocess_mask(mask, right_params)
    assert left.sum() == right.sum() > 0
    assert np.array_equal(right, left[:, ::-1])
    assert np.where(left)[1].mean() > left.shape[1] / 2  # the lesion is on the right
    assert np.where(right)[1].mean() < right.shape[1] / 2


def test_a_small_mask_keeps_its_area_in_proportion() -> None:
    pixels, mask = lesion_image_and_mask()
    settings = PreprocessingSettings(margin=0.0, downscale=1, opening=0, size=150)
    _, params = preprocess(pixels, "L", settings)
    scale = params.shape[0] / params.box.height
    small = preprocess_mask(mask, params)
    assert small.sum() == pytest.approx(mask.sum() * scale**2, rel=0.05)


def test_the_mask_must_have_the_shape_of_the_source_image() -> None:
    pixels, mask = lesion_image_and_mask()
    _, params = preprocess(pixels, "L", EXACT_SETTINGS)
    with pytest.raises(ValueError, match="mask shape"):
        preprocess_mask(mask[:-1], params)


def test_the_mask_must_be_binary() -> None:
    pixels, mask = lesion_image_and_mask()
    _, params = preprocess(pixels, "L", EXACT_SETTINGS)
    mask[250, 300] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        preprocess_mask(mask, params)


def test_a_crop_that_cuts_the_mask_is_an_error() -> None:
    pixels, mask = lesion_image_and_mask()
    _, params = preprocess(pixels, "L", EXACT_SETTINGS)
    mask[10:20, 10:20] = 1  # outside the breast box
    with pytest.raises(ValueError, match="cuts the mask"):
        preprocess_mask(mask, params)


def box_params(flipped: bool) -> PreprocessParams:
    """A 50 x 20 crop scaled by 2 in both directions."""
    box = BoundingBox(top=10, bottom=60, left=20, right=40)
    return PreprocessParams(
        source_shape=(100, 80), box=box, shape=(100, 40), flipped=flipped
    )


@pytest.mark.parametrize(
    ("flipped", "expected"),
    [
        (False, [[0.0, 0.0], [40.0, 100.0], [10.0, 50.0]]),
        (True, [[40.0, 0.0], [0.0, 100.0], [30.0, 50.0]]),
    ],
)
def test_points_are_cropped_scaled_and_flipped(
    flipped: bool, expected: list[list[float]]
) -> None:
    # The corners of the box and a point inside it, as (x, y).
    points = [(20.0, 10.0), (40.0, 60.0), (25.0, 35.0)]
    assert np.allclose(preprocess_points(points, box_params(flipped)), expected)


def test_points_outside_the_box_map_outside_the_result() -> None:
    ((x, y),) = preprocess_points([(0.0, 70.0)], box_params(False))
    assert x < 0 and y > 100


@pytest.mark.parametrize("laterality", ["L", "R"])
def test_a_polygon_lands_on_its_preprocessed_mask(laterality: str) -> None:
    pixels, mask = lesion_image_and_mask()
    settings = PreprocessingSettings(margin=0.05, downscale=1, opening=0, size=200)
    _, params = preprocess(pixels, laterality, settings)  # type: ignore[arg-type]
    corners = [(300.0, 250.0), (380.0, 250.0), (380.0, 330.0), (300.0, 330.0)]
    moved = preprocess_points(corners, params)
    roi = Roi(LesionType.MASS, tuple((float(x), float(y)) for x, y in moved))
    drawn = rasterize([roi], params.shape) == 1
    small_mask = preprocess_mask(mask, params) == 1
    assert (drawn & small_mask).sum() / (drawn | small_mask).sum() > 0.9


@pytest.mark.parametrize("shape", [(3,), (2, 3), (1, 2, 2)])
def test_points_must_be_pairs(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"\[n, 2\] points"):
        preprocess_points(np.zeros(shape), box_params(False))


@pytest.mark.parametrize("laterality", ["L", "R"])
def test_a_mask_can_be_restored_onto_the_original_image(laterality: str) -> None:
    pixels, mask = lesion_image_and_mask()
    settings = PreprocessingSettings(margin=0.05, downscale=1, opening=0, size=200)
    _, params = preprocess(pixels, laterality, settings)  # type: ignore[arg-type]
    restored = restore_mask(preprocess_mask(mask, params), params)
    assert restored.shape == mask.shape
    assert restored.dtype == np.uint8
    overlap = ((restored == 1) & (mask == 1)).sum()
    assert overlap / ((restored == 1) | (mask == 1)).sum() > 0.9
    assert restored[: params.box.top].sum() == 0  # nothing outside the crop


def test_restoring_checks_the_mask() -> None:
    pixels, mask = lesion_image_and_mask()
    _, params = preprocess(pixels, "L", EXACT_SETTINGS)
    with pytest.raises(ValueError, match="mask shape"):
        restore_mask(np.zeros((5, 5), dtype=np.uint8), params)
    wrong = np.full(params.shape, 3, dtype=np.uint8)
    with pytest.raises(ValueError, match="only 0 and 1"):
        restore_mask(wrong, params)


# --- reading DICOM files ---


def test_a_dicom_is_read_as_a_uint16_array(tmp_path: Path) -> None:
    pixels = np.arange(12, dtype=np.uint16).reshape(3, 4) * 300
    write_dicom_pixels(tmp_path / "a.dcm", pixels)
    read = read_pixels(tmp_path / "a.dcm")
    assert read.dtype == np.uint16
    assert np.array_equal(read, pixels)


@pytest.mark.parametrize("name", ["a.jpg", "a.JPEG", "a.png"])
def test_an_8_bit_grey_image_file_is_read_as_uint16(tmp_path: Path, name: str) -> None:
    pixels = np.zeros((8, 16), dtype=np.uint8)
    pixels[:, 8:] = 255
    Image.fromarray(pixels).save(tmp_path / name, quality=100)
    read = read_pixels(tmp_path / name)
    assert read.dtype == np.uint16
    assert np.array_equal(read, pixels)  # the values stay 0 to 255


def test_an_image_file_that_is_not_8_bit_grey_is_rejected(tmp_path: Path) -> None:
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(tmp_path / "a.png")
    with pytest.raises(ValueError, match="expected an 8-bit grey image, got mode RGB"):
        read_pixels(tmp_path / "a.png")


def test_an_8_bit_dicom_is_widened(tmp_path: Path) -> None:
    pixels = np.arange(12, dtype=np.uint8).reshape(3, 4)
    write_dicom_pixels(tmp_path / "a.dcm", pixels)
    read = read_pixels(tmp_path / "a.dcm")
    assert read.dtype == np.uint16
    assert np.array_equal(read, pixels)


@pytest.mark.parametrize(
    ("photometric", "elements", "message"),
    [
        ("MONOCHROME1", {}, "expected MONOCHROME2"),
        ("MONOCHROME2", {"RescaleSlope": 2}, "rescale slope or intercept"),
        ("MONOCHROME2", {"RescaleIntercept": -1024}, "rescale slope or intercept"),
        ("MONOCHROME2", {"SamplesPerPixel": 3}, "one channel"),
    ],
)
def test_an_unsupported_dicom_is_rejected(
    tmp_path: Path, photometric: str, elements: dict[str, object], message: str
) -> None:
    pixels = np.ones((3, 4), dtype=np.uint16)
    write_dicom_pixels(tmp_path / "a.dcm", pixels, photometric, **elements)
    with pytest.raises(ValueError, match=message):
        read_pixels(tmp_path / "a.dcm")


def test_a_signed_dicom_is_rejected(tmp_path: Path) -> None:
    write_dicom_pixels(tmp_path / "a.dcm", np.ones((3, 4), dtype=np.int16))
    with pytest.raises(ValueError, match="unsigned image"):
        read_pixels(tmp_path / "a.dcm")


# --- intensity normalization ---

# Per-image scaling by the 99.5th percentile of the tissue (an option, not the default).
PER_IMAGE = PreprocessingSettings(percentile=99.5)


def test_by_default_pixels_are_divided_by_the_12_bit_maximum() -> None:
    pixels = np.array([[0, 100, 2047, 4095, 4096, 65535]], dtype=np.uint16)
    result = normalize_intensity(pixels)
    assert result.dtype == np.float32
    assert result[0, :4] == pytest.approx([0.0, 100 / 4095, 2047 / 4095, 1.0])
    assert (result[0, 4:] == 1.0).all()


def test_the_default_scaling_does_not_depend_on_the_rest_of_the_image() -> None:
    pixels = ramp_image()
    padded = np.pad(pixels, ((300, 300), (300, 300)))
    brighter = pixels.copy()
    brighter[0, 0] = 60000
    assert np.array_equal(
        normalize_intensity(padded)[300:500, 300:500], normalize_intensity(pixels)
    )
    assert np.array_equal(
        normalize_intensity(brighter)[1:, 1:], normalize_intensity(pixels)[1:, 1:]
    )


def test_a_different_maximum_changes_the_scale() -> None:
    pixels = np.array([[1000, 2000]], dtype=np.uint16)
    result = normalize_intensity(pixels, PreprocessingSettings(max_intensity=2000))
    assert result[0] == pytest.approx([0.5, 1.0])


def test_the_default_scaling_accepts_an_image_without_tissue() -> None:
    assert (normalize_intensity(np.zeros((4, 4), dtype=np.uint16)) == 0).all()


def ramp_image() -> np.ndarray:
    """A black background around a 100 x 100 block of 10000 distinct tissue values."""
    pixels = np.zeros((200, 200), dtype=np.uint16)
    pixels[50:150, 50:150] = np.arange(200, 10200, dtype=np.uint16).reshape(100, 100)
    return pixels


def test_the_result_is_a_float32_image_between_0_and_1() -> None:
    result = normalize_intensity(ramp_image())
    assert result.dtype == np.float32
    assert result.shape == (200, 200)
    assert result.min() >= 0.0 and result.max() == 1.0


def test_the_background_stays_at_zero() -> None:
    result = normalize_intensity(ramp_image())
    assert (result[:50] == 0).all()
    assert (result[:, :50] == 0).all()


def test_the_tissue_percentile_maps_to_one_and_brighter_pixels_are_cut_off() -> None:
    pixels = ramp_image()
    result = normalize_intensity(pixels, PER_IMAGE)
    tissue = result[pixels > 100]
    assert (tissue == 1.0).mean() == pytest.approx(0.005, abs=0.0005)
    high = np.percentile(pixels[pixels > 100], 99.5)
    assert result[pixels == 5000][0] == pytest.approx(5000 / high)


def test_brighter_means_higher_and_the_order_is_kept() -> None:
    pixels = ramp_image()
    result = normalize_intensity(pixels, PER_IMAGE)
    tissue = np.sort(pixels[pixels > 100])
    values = normalize_intensity(tissue.reshape(1, -1), PER_IMAGE)[0]
    assert (np.diff(values) >= 0).all()
    assert result[60, 60] < result[100, 100] < result[140, 140]


def test_images_of_different_brightness_get_the_same_result() -> None:
    pixels = ramp_image()
    dimmer = (pixels // 2).astype(np.uint16)
    dimmer_tissue = dimmer > 100
    a = normalize_intensity(pixels, PER_IMAGE)
    b = normalize_intensity(dimmer, PER_IMAGE)
    assert np.abs(a[dimmer_tissue] - b[dimmer_tissue]).max() < 0.01


def test_more_background_does_not_change_the_result() -> None:
    pixels = ramp_image()
    padded = np.pad(pixels, ((300, 300), (300, 300)))
    assert np.array_equal(
        normalize_intensity(padded, PER_IMAGE)[300:500, 300:500],
        normalize_intensity(pixels, PER_IMAGE),
    )


def test_a_higher_percentile_scales_by_a_higher_value() -> None:
    pixels = ramp_image()
    p100 = normalize_intensity(pixels, PreprocessingSettings(percentile=100.0))
    p50 = normalize_intensity(pixels, PreprocessingSettings(percentile=50.0))
    assert p100.max() == 1.0 and (p100[pixels > 100] == 1.0).mean() < 0.001
    assert (p50 > p100).sum() > 0
    assert p50.max() == 1.0 and (p50[pixels > 100] == 1.0).mean() > 0.4


def test_the_threshold_decides_what_counts_as_tissue_in_percentile_mode() -> None:
    pixels = ramp_image()
    pixels[0:20, 0:20] = 60  # dim background noise
    low = normalize_intensity(
        pixels, PreprocessingSettings(percentile=99.5, threshold=10)
    )
    high = normalize_intensity(
        pixels, PreprocessingSettings(percentile=99.5, threshold=100)
    )
    assert not np.array_equal(low, high)


def test_the_threshold_does_not_matter_in_the_default_mode() -> None:
    pixels = ramp_image()
    pixels[0:20, 0:20] = 60
    low = normalize_intensity(pixels, PreprocessingSettings(threshold=10))
    high = normalize_intensity(pixels, PreprocessingSettings(threshold=100))
    assert np.array_equal(low, high)


def test_normalizing_is_deterministic_and_leaves_the_input_alone() -> None:
    pixels = ramp_image()
    before = pixels.copy()
    assert np.array_equal(
        normalize_intensity(pixels, PER_IMAGE), normalize_intensity(pixels, PER_IMAGE)
    )
    assert np.array_equal(pixels, before)


def test_a_full_range_16_bit_image_does_not_overflow() -> None:
    pixels = np.full((50, 50), 65535, dtype=np.uint16)
    assert (normalize_intensity(pixels, PER_IMAGE) == 1.0).all()


def test_an_image_without_tissue_is_rejected() -> None:
    with pytest.raises(ValueError, match="no tissue found"):
        normalize_intensity(np.full((10, 10), 100, dtype=np.uint16), PER_IMAGE)
    with pytest.raises(ValueError, match="no tissue found"):
        normalize_intensity(np.zeros((10, 10), dtype=np.uint16), PER_IMAGE)


def test_a_non_2d_image_is_rejected_by_the_normalization() -> None:
    with pytest.raises(ValueError, match="expected a 2-D image, got 3 dimensions"):
        normalize_intensity(np.ones((4, 4, 3), dtype=np.uint16) * 500)


def test_preprocessing_then_normalizing_gives_the_model_input() -> None:
    pixels, _ = lesion_image_and_mask()
    image, _ = preprocess(
        pixels, "R", PreprocessingSettings(margin=0.0, downscale=1, opening=0, size=100)
    )
    result = normalize_intensity(image, PER_IMAGE)
    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert 0.0 <= result.min() and result.max() == 1.0
