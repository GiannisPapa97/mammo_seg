from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from tests.data.preprocessed import CANVAS, SETTINGS, write_preprocessed
from torch import nn
from torch.utils.data import DataLoader

from mammo_seg.data.dataset import MammogramDataset
from mammo_seg.engine import model as model_module
from mammo_seg.engine.losses import DiceBCELoss
from mammo_seg.engine.metrics import image_scores, summarize
from mammo_seg.engine.model import (
    ARCHITECTURES,
    ModelSettings,
    build_model,
    count_parameters,
)

# A small encoder and random weights: fast, and nothing is downloaded.
SMALL = ModelSettings(encoder="resnet18", encoder_weights=None)


def test_the_defaults_are_a_unet_with_a_pretrained_resnet34() -> None:
    settings = ModelSettings()
    assert settings.architecture == "unet"
    assert settings.encoder == "resnet34"
    assert settings.encoder_weights == "imagenet"


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_every_architecture_maps_one_channel_to_one_channel(architecture: str) -> None:
    settings = ModelSettings(
        architecture=architecture,  # type: ignore[arg-type]
        encoder="resnet18",
        encoder_weights=None,
    )
    model = build_model(settings).eval()
    with torch.no_grad():
        output = model(torch.rand(2, 1, 64, 64))
    assert output.shape == (2, 1, 64, 64)
    assert output.dtype == torch.float32


def test_the_output_is_raw_logits() -> None:
    model = build_model(SMALL).eval()
    with torch.no_grad():
        output = model(torch.rand(1, 1, 64, 64))
    assert output.min() < 0 or output.max() > 1  # not squashed into 0..1 by a sigmoid


def test_the_canvas_size_works() -> None:
    model = build_model(SMALL).eval()
    with torch.no_grad():
        output = model(torch.rand(1, 1, 1024, 800))
    assert output.shape == (1, 1, 1024, 800)


def test_the_same_seed_gives_the_same_weights() -> None:
    torch.manual_seed(0)
    first = build_model(SMALL).state_dict()
    torch.manual_seed(0)
    second = build_model(SMALL).state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_the_model_has_the_expected_size() -> None:
    assert 10e6 < count_parameters(build_model(SMALL)) < 20e6  # resnet18 U-Net
    frozen = build_model(SMALL)
    for parameter in frozen.parameters():
        parameter.requires_grad = False
    assert count_parameters(frozen) == 0
    assert count_parameters(frozen, trainable_only=False) > 10e6


# --- the settings ---


def test_the_settings_reject_unknown_values() -> None:
    with pytest.raises(ValidationError):
        ModelSettings(architecture="resnet")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ModelSettings(encoder_weights="random")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ModelSettings(encoder="")


def test_an_unknown_encoder_is_an_error() -> None:
    with pytest.raises(ValueError, match="cannot build unet with encoder 'nonsense'"):
        build_model(ModelSettings(encoder="nonsense", encoder_weights=None))


# --- the pretrained weights, without downloading ---


class Recorder(nn.Module):
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        Recorder.calls.append(kwargs)


def test_the_pretrained_weights_and_one_channel_are_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Recorder.calls = []
    monkeypatch.setitem(model_module.ARCHITECTURES, "unet", Recorder)
    build_model(ModelSettings())
    assert Recorder.calls == [
        {
            "encoder_name": "resnet34",
            "encoder_weights": "imagenet",
            "in_channels": 1,
            "classes": 1,
        }
    ]


def test_a_failed_download_explains_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(**kwargs: Any) -> nn.Module:
        raise OSError("Network is unreachable")

    monkeypatch.setitem(model_module.ARCHITECTURES, "unet", offline)
    with pytest.raises(RuntimeError, match="needs internet.*encoder_weights=None"):
        build_model(ModelSettings())
    with pytest.raises(OSError, match="Network is unreachable"):
        build_model(ModelSettings(encoder_weights=None))  # no download involved: as is


# --- with the loss and the metrics ---


def test_a_few_training_steps_lower_the_loss() -> None:
    torch.manual_seed(0)
    model = build_model(SMALL)
    image = torch.rand(2, 1, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[0, 0, 20:40, 20:40] = 1
    valid = torch.ones_like(mask, dtype=torch.bool)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        loss = loss_fn(model(image), mask, valid)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8


def test_a_batch_from_the_data_loader_runs_through_model_loss_and_metrics(
    tmp_path: Path,
) -> None:
    write_preprocessed(tmp_path, SETTINGS)
    dataset = MammogramDataset(
        tmp_path, ["00000001", "00000002", "00000003"], SETTINGS, CANVAS
    )
    batch = next(iter(DataLoader(dataset, batch_size=3)))
    model = build_model(SMALL).eval()
    with torch.no_grad():
        logits = model(batch["image"])
        loss = DiceBCELoss()(logits, batch["mask"], batch["valid"])
        scores = image_scores(torch.sigmoid(logits), batch["mask"], batch["valid"])
    assert logits.shape == batch["mask"].shape
    assert loss.item() > 0
    assert summarize(scores).all_images.images == 3
