import pytest
import torch
from torch import nn

from mammo_seg.engine.inference import (
    INFERENCES,
    gaussian_weights,
    predict,
    window_starts,
)


@pytest.mark.parametrize(
    ("length", "window", "starts"),
    [
        (1024, 512, [0, 256, 512]),  # the height of the canvas
        (800, 512, [0, 256, 288]),  # the width: the last window ends at the edge
        (512, 512, [0]),
        (300, 512, [0]),  # an image smaller than the window: one window
        (10, 4, [0, 2, 4, 6]),
    ],
)
def test_windows_overlap_by_half_and_end_at_the_edge(
    length: int, window: int, starts: list[int]
) -> None:
    assert window_starts(length, window) == starts


def test_the_windows_cover_every_pixel() -> None:
    for length, window in [(1024, 512), (800, 512), (37, 8), (5, 5)]:
        covered = torch.zeros(length, dtype=torch.bool)
        for start in window_starts(length, window):
            covered[start : start + window] = True
        assert covered.all()


@pytest.mark.parametrize(("length", "window"), [(0, 4), (4, 0)])
def test_an_empty_window_or_length_is_rejected(length: int, window: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        window_starts(length, window)


def test_the_weights_trust_the_centre_and_are_never_zero() -> None:
    weights = gaussian_weights((64, 32))
    assert weights.shape == (64, 32)
    assert weights.max() == 1.0 and weights.min() > 0
    assert weights[32, 16] > weights[32, 0] > weights[0, 0]
    assert torch.allclose(weights, weights.flip(0)) and torch.allclose(
        weights, weights.flip(1)
    )


class PixelWise(nn.Module):
    """A model whose output at a pixel depends only on that pixel."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images * 8 - 4


class Counting(nn.Module):
    """Counts its calls and the shapes it sees."""

    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[tuple[int, ...]] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(images.shape))
        return torch.zeros_like(images)


def test_sliding_windows_of_a_pixelwise_model_equal_one_pass() -> None:
    images = torch.rand(2, 1, 40, 30)
    whole = predict(PixelWise(), images, "whole")
    sliding = predict(PixelWise(), images, "sliding", window=(16, 16))
    assert torch.allclose(whole, sliding, atol=1e-6)
    assert whole.shape == images.shape and whole.dtype == torch.float32


def test_the_model_sees_only_windows_of_the_window_size() -> None:
    model = Counting()
    probabilities = predict(model, torch.rand(3, 1, 40, 30), "sliding", window=(16, 16))
    # Rows start at 0, 8, 16 and 24, columns at 0, 8 and 14.
    assert len(model.shapes) == 4 * 3
    assert set(model.shapes) == {(3, 1, 16, 16)}
    assert torch.allclose(probabilities, torch.full((3, 1, 40, 30), 0.5))


def test_a_window_larger_than_the_image_is_cut_to_it() -> None:
    model = Counting()
    predict(model, torch.rand(1, 1, 20, 10), "sliding", window=(512, 512))
    assert model.shapes == [(1, 1, 20, 10)]


def test_one_pass_runs_the_model_once() -> None:
    model = Counting()
    predict(model, torch.rand(2, 1, 40, 30), "whole", window=(16, 16))
    assert model.shapes == [(2, 1, 40, 30)]


def test_sliding_windows_need_a_window() -> None:
    with pytest.raises(ValueError, match="need a window size"):
        predict(PixelWise(), torch.rand(1, 1, 8, 8), "sliding")


def test_an_unknown_inference_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown inference"):
        predict(PixelWise(), torch.rand(1, 1, 8, 8), "tiles")  # type: ignore[arg-type]


def test_the_inferences_are_whole_and_sliding() -> None:
    assert INFERENCES == ("whole", "sliding")
