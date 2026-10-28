import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image
from tests.data.manifests import make_image

from mammo_seg.data.manifest import write_manifest
from mammo_seg.data.masks import (
    MaskError,
    main,
    mask_path,
    mask_problems,
    polygon_pixels,
    rasterize,
    read_mask,
    write_mask,
    write_masks,
)
from mammo_seg.data.schema import ImageRecord, LesionType, Manifest, Roi


def mass(*points: tuple[float, float]) -> Roi:
    return Roi(LesionType.MASS, tuple(points))


def rectangle(x1: float, y1: float, x2: float, y2: float) -> Roi:
    return mass((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def inside_brute_force(
    points: list[tuple[float, float]], shape: tuple[int, int]
) -> npt.NDArray[np.uint8]:
    """Pixels whose centre is inside the polygon, by the even-odd ray rule.

    An independent, slow implementation to check the rasterizer against.
    """
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols] + 0.5
    inside = np.zeros(shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(points, [*points[1:], points[0]], strict=True):
        crosses = (y1 <= yy) & (yy < y2) | (y2 <= yy) & (yy < y1)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1 + (yy - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (xx < x_at)
    return inside.astype(np.uint8)


# --- rasterizing polygons ---


def test_an_axis_aligned_rectangle_covers_exactly_its_pixels() -> None:
    mask = rasterize([rectangle(2, 1, 6, 4)], (6, 8))
    expected = np.zeros((6, 8), dtype=np.uint8)
    expected[1:4, 2:6] = 1
    assert np.array_equal(mask, expected)
    assert mask.dtype == np.uint8


def test_a_pixel_is_set_when_its_centre_is_inside() -> None:
    # centres are at c + 0.5, so a polygon from 0.5 to 2.5 covers pixels 0 and 1
    mask = rasterize([rectangle(0.5, 0.5, 2.5, 2.5)], (4, 4))
    assert mask.sum() == 4
    assert mask[:2, :2].all()


def test_x_is_the_column_and_y_is_the_row() -> None:
    mask = rasterize([rectangle(4, 0, 6, 1)], (3, 8))  # a 1 x 2 strip in row 0
    assert mask.shape == (3, 8)
    assert mask[0, 4:6].all()
    assert mask.sum() == 2


def test_the_winding_direction_does_not_matter() -> None:
    points = [(2.0, 1.0), (9.0, 2.0), (6.0, 8.0), (1.0, 6.0)]
    clockwise = rasterize([mass(*points)], (10, 12))
    counter = rasterize([mass(*points[::-1])], (10, 12))
    assert np.array_equal(clockwise, counter)


def test_a_concave_polygon_has_its_notch_empty() -> None:
    # an L: the 2 x 2 corner at the top right is not part of the shape
    shape = mass((0, 0), (2, 0), (2, 2), (4, 2), (4, 4), (0, 4))
    mask = rasterize([shape], (4, 4))
    expected = np.ones((4, 4), dtype=np.uint8)
    expected[:2, 2:] = 0
    assert np.array_equal(mask, expected)


def test_a_vertex_on_a_pixel_centre_row_is_counted_once() -> None:
    diamond = [(3.0, 0.0), (6.0, 2.5), (3.0, 5.0), (0.0, 2.5)]
    mask = rasterize([mass(*diamond)], (6, 7))
    assert np.array_equal(mask, inside_brute_force(diamond, (6, 7)))


@pytest.mark.parametrize(
    "points",
    [
        [(3.2, 4.7), (40.1, 5.3), (36.4, 29.9), (7.7, 33.3)],
        [(10.0, 10.0), (50.5, 12.5), (30.0, 45.0)],
        [
            (5.5, 5.5),
            (25.5, 5.5),
            (25.5, 15.5),
            (15.5, 15.5),
            (15.5, 30.5),
            (5.5, 30.5),
        ],
        [
            (30 + 25 * math.cos(angle), 28 + 20 * math.sin(angle))
            for angle in np.linspace(0, 2 * math.pi, 40, endpoint=False)
        ],
    ],
)
def test_the_rasterizer_agrees_with_a_brute_force_point_in_polygon_test(
    points: list[tuple[float, float]],
) -> None:
    mask = rasterize([mass(*points)], (60, 60))
    assert np.array_equal(mask, inside_brute_force(points, (60, 60)))


def test_the_pixel_count_matches_the_polygon_area() -> None:
    circle = mass(
        *[
            (100 + 50 * math.cos(angle), 80 + 50 * math.sin(angle))
            for angle in np.linspace(0, 2 * math.pi, 200, endpoint=False)
        ]
    )
    assert polygon_pixels(circle, (200, 220)) == pytest.approx(
        circle.area_px, rel=0.005
    )


def test_a_polygon_is_clipped_to_the_image() -> None:
    mask = rasterize([rectangle(-2, -1, 3, 2)], (4, 6))
    expected = np.zeros((4, 6), dtype=np.uint8)
    expected[0:2, 0:3] = 1
    assert np.array_equal(mask, expected)
    mask = rasterize([rectangle(4, 2, 9, 7)], (4, 6))
    expected = np.zeros((4, 6), dtype=np.uint8)
    expected[2:4, 4:6] = 1
    assert np.array_equal(mask, expected)


def test_a_polygon_outside_the_image_leaves_the_mask_empty() -> None:
    assert not rasterize([rectangle(10, 10, 14, 14)], (6, 6)).any()
    assert not rasterize([rectangle(-8, -8, -2, -2)], (6, 6)).any()


def test_overlapping_lesions_are_merged() -> None:
    mask = rasterize([rectangle(0, 0, 4, 4), rectangle(2, 2, 6, 6)], (8, 8))
    assert mask.sum() == 16 + 16 - 4


def test_only_the_chosen_lesion_type_is_drawn() -> None:
    cluster = Roi(LesionType.CLUSTER, rectangle(0, 0, 3, 3).points)
    mask = rasterize([cluster, rectangle(4, 4, 6, 6)], (8, 8))
    assert mask.sum() == 4
    assert (
        rasterize([cluster, rectangle(4, 4, 6, 6)], (8, 8), LesionType.CLUSTER).sum()
        == 9
    )


def test_rois_without_an_area_leave_the_mask_empty() -> None:
    point = Roi(LesionType.CALCIFICATION, ((2.0, 2.0),))
    line = Roi(LesionType.CALCIFICATION, ((1.0, 1.0), (5.0, 5.0)))
    assert not rasterize([point, line], (8, 8), LesionType.CALCIFICATION).any()
    assert polygon_pixels(point, (8, 8)) == 0


def test_no_rois_give_an_empty_mask() -> None:
    mask = rasterize([], (5, 7))
    assert mask.shape == (5, 7)
    assert not mask.any()


# --- checking a mask against its polygons ---


def test_a_mask_made_from_its_polygons_has_no_problems() -> None:
    rois = [rectangle(2, 2, 30, 20), rectangle(20, 10, 45, 40)]  # they overlap
    mask = rasterize(rois, (50, 60))
    assert mask_problems(mask, rois, (50, 60)) == []


def test_a_clipped_polygon_is_not_a_problem() -> None:
    rois = [rectangle(-5, -5, 20, 20)]
    assert mask_problems(rasterize(rois, (30, 30)), rois, (30, 30)) == []


def test_an_empty_mask_without_polygons_has_no_problems() -> None:
    assert mask_problems(rasterize([], (5, 5)), [], (5, 5)) == []


def test_a_mask_of_the_wrong_shape_is_a_problem() -> None:
    rois = [rectangle(0, 0, 4, 4)]
    problems = mask_problems(np.zeros((5, 5), dtype=np.uint8), rois, (6, 6))
    assert problems == ["mask shape (5, 5) is not the image shape (6, 6)"]


def test_a_mask_that_is_not_binary_is_a_problem() -> None:
    rois = [rectangle(0, 0, 4, 4)]
    mask = rasterize(rois, (6, 6))
    mask[0, 0] = 2
    assert any("not binary" in problem for problem in mask_problems(mask, rois, (6, 6)))
    mask = rasterize(rois, (6, 6)).astype(np.int64)
    assert any("not binary" in problem for problem in mask_problems(mask, rois, (6, 6)))


def test_a_polygon_that_crosses_itself_is_a_problem() -> None:
    bow_tie = mass((0, 0), (20, 20), (20, 0), (0, 20))
    problems = mask_problems(rasterize([bow_tie], (30, 30)), [bow_tie], (30, 30))
    assert len(problems) == 1
    assert "mass 0: covers" in problems[0]
    assert "cross itself" in problems[0]


def test_a_polygon_outside_the_image_is_a_problem() -> None:
    far = rectangle(50, 50, 60, 60)
    problems = mask_problems(rasterize([far], (20, 20)), [far], (20, 20))
    assert problems == ["mass 0: outside the image or smaller than a pixel"]


def test_a_mass_with_fewer_than_three_points_is_a_problem() -> None:
    line = mass((1, 1), (5, 5))
    problems = mask_problems(rasterize([line], (8, 8)), [line], (8, 8))
    assert problems == ["mass 0: fewer than 3 points, so no polygon"]


def test_pixels_that_no_polygon_covers_are_a_problem() -> None:
    rois = [rectangle(0, 0, 4, 4)]
    mask = rasterize(rois, (20, 20))
    mask[10:15, 10:15] = 1
    problems = mask_problems(mask, rois, (20, 20))
    assert len(problems) == 1
    assert "outside the 16 to 16 covered by its polygons" in problems[0]


def test_a_mask_with_pixels_but_no_polygons_is_a_problem() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = 1
    assert mask_problems(mask, [], (5, 5)) == [
        "mask has 1 pixels but there are no polygons"
    ]


# --- files ---


def test_a_mask_is_stored_as_a_png_with_0_and_255(tmp_path: Path) -> None:
    mask = rasterize([rectangle(2, 1, 6, 4)], (6, 8))
    path = tmp_path / "masks" / "x.png"
    write_mask(mask, path)
    with Image.open(path) as image:
        assert image.mode == "L"
        assert image.size == (8, 6)  # width, height
        assert set(np.unique(np.asarray(image))) == {0, 255}
    assert np.array_equal(read_mask(path), mask)


def test_an_empty_mask_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "empty.png"
    write_mask(rasterize([], (6, 8)), path)
    assert not read_mask(path).any()
    assert read_mask(path).shape == (6, 8)


def test_the_mask_path_uses_the_file_id(tmp_path: Path) -> None:
    assert mask_path(tmp_path, "20586908") == tmp_path / "20586908.png"


# --- writing the masks of a manifest ---


def image_with(
    file_id: int, rois: tuple[Roi, ...], rows: int = 30, cols: int = 40
) -> ImageRecord:
    return replace(make_image(file_id, file_id), rows=rows, cols=cols, rois=rois)


def make_small_manifest() -> Manifest:
    return Manifest(
        version=2,
        dataset="inbreast",
        images=(
            image_with(
                1,
                (rectangle(5, 5, 15, 12), Roi(LesionType.CALCIFICATION, ((2.0, 2.0),))),
            ),
            image_with(2, ()),
            image_with(3, (rectangle(-4, 20, 6, 40), rectangle(20, 5, 30, 15))),
            image_with(4, (Roi(LesionType.CLUSTER, rectangle(1, 1, 9, 9).points),)),
        ),
    )


def test_write_masks_writes_one_mask_per_image(tmp_path: Path) -> None:
    manifest = make_small_manifest()
    summary = write_masks(manifest, tmp_path / "masks")
    assert sorted(path.name for path in (tmp_path / "masks").iterdir()) == [
        "00000001.png",
        "00000002.png",
        "00000003.png",
        "00000004.png",
    ]
    assert summary.images == 4
    assert summary.images_with_masks == 2
    assert summary.masses == 3
    assert summary.clipped_masses == 1
    first = read_mask(tmp_path / "masks" / "00000001.png")
    assert first.shape == (30, 40)
    assert first.sum() == 70
    assert not read_mask(tmp_path / "masks" / "00000002.png").any()
    assert not read_mask(
        tmp_path / "masks" / "00000004.png"
    ).any()  # a cluster, not a mass
    third = read_mask(tmp_path / "masks" / "00000003.png")
    assert third.sum() == summary.mask_pixels - first.sum()
    assert third[20:30, 0:6].all() and third[5:15, 20:30].all()


def test_write_masks_reports_every_problem_and_skips_those_masks(
    tmp_path: Path,
) -> None:
    bow_tie = mass((0, 0), (20, 20), (20, 0), (0, 20))
    manifest = Manifest(
        version=2,
        dataset="inbreast",
        images=(
            image_with(1, (rectangle(5, 5, 15, 12),)),
            image_with(2, (bow_tie,)),
            image_with(3, (rectangle(100, 100, 110, 110),)),
        ),
    )
    with pytest.raises(MaskError) as error:
        write_masks(manifest, tmp_path)
    assert len(error.value.problems) == 2
    assert error.value.problems[0].startswith("image 00000002: mass 0: covers")
    assert error.value.problems[1] == (
        "image 00000003: mass 0: outside the image or smaller than a pixel"
    )
    assert [path.name for path in tmp_path.iterdir()] == ["00000001.png"]


def test_lesions_given_as_mask_images_are_refused_not_skipped(tmp_path: Path) -> None:
    mask = Roi(LesionType.MASS, mask_path="jpeg/1.2/1-2.jpg", mask_area_px=40)
    manifest = Manifest(
        version=2,
        dataset="cbis-ddsm",
        images=(image_with(1, (rectangle(5, 5, 15, 12),)), image_with(2, (mask,))),
    )
    with pytest.raises(MaskError) as error:
        write_masks(manifest, tmp_path)
    assert error.value.problems == [
        "image 00000002: its lesions are mask images, which this step does not read"
    ]
    assert [path.name for path in tmp_path.iterdir()] == ["00000001.png"]


# --- the command ---


def test_main_writes_the_masks_and_prints_a_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(make_small_manifest(), manifest_path)
    output = tmp_path / "out"
    code = main(["--manifest", str(manifest_path), "--output-dir", str(output)])
    assert code == 0
    assert len(list(output.iterdir())) == 4
    text = capsys.readouterr().out
    assert "Wrote 4 masks" in text
    assert "2 with masses (3 masses, 1 clipped to the image), 2 empty" in text


def test_main_reports_problems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bow_tie = mass((0, 0), (20, 20), (20, 0), (0, 20))
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        Manifest(version=2, dataset="inbreast", images=(image_with(1, (bow_tie,)),)),
        manifest_path,
    )
    code = main(
        ["--manifest", str(manifest_path), "--output-dir", str(tmp_path / "out")]
    )
    assert code == 1
    assert "cross itself" in capsys.readouterr().err


def test_main_reports_a_missing_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["--manifest", str(tmp_path / "missing.json"), "--output-dir", str(tmp_path)]
    )
    assert code == 1
    assert "error:" in capsys.readouterr().err
