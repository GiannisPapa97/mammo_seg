"""Choosing where tensors and models run: the CPU or a GPU.

The same code runs locally on a CPU and on GPU machines such as Kaggle or Colab.
The device is chosen at run time, either explicitly or through the
``MAMMO_SEG_DEVICE`` environment variable, and defaults to the best one available.
"""

import os

import torch

DEVICE_ENV = "MAMMO_SEG_DEVICE"


def resolve_device(requested: str | None = None) -> torch.device:
    """The device to use.

    ``requested`` is ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:<index>"`` or
    ``"mps"`` (Apple GPUs). When it is ``None``, the ``MAMMO_SEG_DEVICE``
    environment variable is used, and ``"auto"`` when that is not set. ``"auto"``
    picks a CUDA GPU if there is one, then MPS, then the CPU.

    Asking for a GPU that is not available raises ValueError instead of silently
    falling back to the CPU, so a run meant for a GPU does not quietly run for hours
    on a CPU.
    """
    name = (requested or os.environ.get(DEVICE_ENV) or "auto").strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("an MPS device was requested but none is available")
        return torch.device("mps")
    if name == "cuda" or name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"{name!r} was requested but no CUDA GPU is available; install a "
                "CUDA build of PyTorch or use 'cpu'"
            )
        device = torch.device(name)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"{name!r} was requested but there are only "
                f"{torch.cuda.device_count()} CUDA devices"
            )
        return device
    raise ValueError(
        f"unknown device {name!r}: use 'auto', 'cpu', 'cuda', 'cuda:<index>' or 'mps'"
    )
