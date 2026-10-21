"""Building the segmentation network.

Models come from ``segmentation_models_pytorch`` (smp): an architecture (U-Net,
DeepLabV3+ and others) on top of a pretrained encoder. The input has one channel
(smp sums the pretrained weights of the first layer over the colour channels) and
the output is one channel of raw logits for the mass mask, ready for
``DiceBCELoss``; apply a sigmoid to get probabilities.

Pretrained encoder weights are downloaded the first time a model is built. That
needs internet, which is off by default on Kaggle notebooks: enable it in the
notebook settings, or build the model with ``encoder_weights=None`` and load a
trained checkpoint (see ``checkpoint.py``), which holds all the weights. Tests
and CI use ``None`` and never download anything.
"""

from typing import Annotated, Literal

import segmentation_models_pytorch as smp
from pydantic import ConfigDict, StringConstraints
from pydantic.dataclasses import dataclass
from torch import nn

ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "fpn": smp.FPN,
}
Architecture = Literal["unet", "unetplusplus", "deeplabv3", "deeplabv3plus", "fpn"]


@dataclass(frozen=True, config=ConfigDict(strict=True, extra="forbid"))
class ModelSettings:
    """What decides the network; stored with a trained model."""

    architecture: Architecture = "unet"
    encoder: Annotated[str, StringConstraints(min_length=1)] = "resnet34"
    # "imagenet" downloads pretrained encoder weights; None starts from random ones.
    encoder_weights: Literal["imagenet"] | None = "imagenet"


def build_model(settings: ModelSettings = ModelSettings()) -> nn.Module:  # noqa: B008
    """A network mapping ``[B, 1, H, W]`` images to ``[B, 1, H, W]`` logits.

    Raises ValueError for an encoder smp does not know, and RuntimeError with a
    hint when the pretrained weights cannot be downloaded.
    """
    architecture = ARCHITECTURES[settings.architecture]
    try:
        model: nn.Module = architecture(
            encoder_name=settings.encoder,
            encoder_weights=settings.encoder_weights,
            in_channels=1,
            classes=1,
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"cannot build {settings.architecture} with encoder "
            f"{settings.encoder!r}: {error}"
        ) from error
    except Exception as error:
        if settings.encoder_weights is None:
            raise
        raise RuntimeError(
            f"could not get the pretrained {settings.encoder_weights} weights for "
            f"{settings.encoder!r}: {error}. They are downloaded on first use, which "
            "needs internet (on Kaggle: enable it in the notebook settings); or use "
            "encoder_weights=None and load a checkpoint."
        ) from error
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """The number of (trainable) parameters of a model."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )
