"""Saving and loading trained models.

A checkpoint is one file with everything needed to use a model or to resume
training: the weights, the model settings (so the network can be rebuilt), the
preprocessing settings the model was trained with (so inference preprocesses
images the same way), and optionally the training state (optimizer, epoch, ...).

Built for free cloud GPUs, where a session can end at any time and a model is often
trained on a GPU and used on a CPU:

- a checkpoint is written to a temporary file and renamed, so a session that dies
  during the save never leaves a broken file in place of the last good one
- it loads on a CPU whatever device it was saved from
- it is loaded with ``weights_only=True``, so a downloaded checkpoint cannot run
  code; settings are stored as JSON text and the rest are tensors and plain values
- the pretrained encoder weights are not downloaded when a model is rebuilt from a
  checkpoint, because the checkpoint already holds all the weights
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import TypeAdapter
from torch import nn

from mammo_seg.data.preprocessing import PreprocessingSettings
from mammo_seg.engine.model import ModelSettings, build_model

FORMAT_VERSION = 1
_MODEL_SETTINGS = TypeAdapter(ModelSettings)
_PREPROCESSING_SETTINGS = TypeAdapter(PreprocessingSettings)
_KEYS = {
    "format_version",
    "model_settings",
    "preprocessing_settings",
    "model_state",
    "training_state",
    "metadata",
}


@dataclass(frozen=True)
class Checkpoint:
    model_settings: ModelSettings
    preprocessing_settings: PreprocessingSettings
    model_state: dict[str, torch.Tensor]
    training_state: dict[str, Any] | None  # optimizer, epoch, ... for resuming
    metadata: dict[str, Any]  # plain values such as the fold or the best Dice


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_settings: ModelSettings,
    preprocessing_settings: PreprocessingSettings,
    training_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a checkpoint atomically; the folder is created if needed.

    ``training_state`` and ``metadata`` may hold tensors and plain Python values
    (numbers, strings, booleans, lists and dicts of these), which is what
    ``optimizer.state_dict()`` contains.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "format_version": FORMAT_VERSION,
        "model_settings": _MODEL_SETTINGS.dump_json(model_settings).decode(),
        "preprocessing_settings": _PREPROCESSING_SETTINGS.dump_json(
            preprocessing_settings
        ).decode(),
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "training_state": dict(training_state) if training_state is not None else None,
        "metadata": dict(metadata or {}),
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(content, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> Checkpoint:
    """Read a checkpoint onto ``device`` (the CPU by default).

    Raises ValueError if the file is not a checkpoint of this format, and
    pydantic's ``ValidationError`` (a ValueError) for invalid settings.
    """
    try:
        content = torch.load(path, map_location=device, weights_only=True)
    except Exception as error:
        raise ValueError(
            f"{path.name} is not a readable checkpoint: {error}"
        ) from error
    if not isinstance(content, dict) or set(content) != _KEYS:
        raise ValueError(f"{path.name} is not a checkpoint of this project")
    if content["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"{path.name} has format version {content['format_version']}, "
            f"expected {FORMAT_VERSION}"
        )
    return Checkpoint(
        model_settings=_MODEL_SETTINGS.validate_json(content["model_settings"]),
        preprocessing_settings=_PREPROCESSING_SETTINGS.validate_json(
            content["preprocessing_settings"]
        ),
        model_state=content["model_state"],
        training_state=content["training_state"],
        metadata=content["metadata"],
    )


def model_from_checkpoint(checkpoint: Checkpoint) -> nn.Module:
    """Rebuild the network of a checkpoint and load its weights.

    No pretrained weights are downloaded: random ones are replaced by the saved ones.
    """
    model = build_model(
        ModelSettings(
            architecture=checkpoint.model_settings.architecture,
            encoder=checkpoint.model_settings.encoder,
            encoder_weights=None,
        )
    )
    model.load_state_dict(checkpoint.model_state)
    return model


def load_model(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[nn.Module, PreprocessingSettings]:
    """A model ready for inference on ``device``, and the preprocessing it expects.

    The model is in evaluation mode. Images must be preprocessed with the returned
    settings.
    """
    checkpoint = load_checkpoint(path, device)
    model = model_from_checkpoint(checkpoint).to(device).eval()
    return model, checkpoint.preprocessing_settings
