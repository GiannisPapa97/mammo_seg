import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from mammo_seg.device import DEVICE_ENV, resolve_device


@pytest.fixture
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.delenv(DEVICE_ENV, raising=False)


@pytest.fixture
def two_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.delenv(DEVICE_ENV, raising=False)


# --- the installed libraries ---


def test_torch_runs_on_the_cpu() -> None:
    x = torch.ones(2, 3)
    assert (x @ x.T).tolist() == [[3.0, 3.0], [3.0, 3.0]]


def test_torchvision_transforms_image_and_mask_together() -> None:
    image = tv_tensors.Image(torch.zeros(1, 4, 6))
    mask = tv_tensors.Mask(torch.zeros(1, 4, 6, dtype=torch.uint8))
    image[0, 1, 0] = 1.0
    mask[0, 1, 0] = 1
    flipped_image, flipped_mask = v2.RandomVerticalFlip(p=1.0)(image, mask)
    assert flipped_image[0, 2, 0] == 1.0
    assert flipped_mask[0, 2, 0] == 1
    assert isinstance(flipped_mask, tv_tensors.Mask)


# --- choosing the device ---


def test_auto_falls_back_to_the_cpu(no_gpu: None) -> None:
    assert resolve_device() == torch.device("cpu")
    assert resolve_device("auto") == torch.device("cpu")


def test_auto_picks_a_cuda_gpu_when_there_is_one(two_gpus: None) -> None:
    assert resolve_device() == torch.device("cuda")


def test_auto_picks_mps_when_there_is_no_cuda(
    no_gpu: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device() == torch.device("mps")


def test_the_cpu_can_be_chosen_on_a_gpu_machine(two_gpus: None) -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_a_specific_gpu_can_be_chosen(two_gpus: None) -> None:
    assert resolve_device("cuda:1") == torch.device("cuda", 1)
    assert resolve_device(" CUDA ") == torch.device("cuda")


def test_the_environment_variable_sets_the_default(
    two_gpus: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DEVICE_ENV, "cpu")
    assert resolve_device() == torch.device("cpu")
    assert resolve_device("cuda") == torch.device("cuda")  # an argument wins


def test_a_missing_gpu_is_an_error_not_a_silent_fallback(no_gpu: None) -> None:
    with pytest.raises(ValueError, match="no CUDA GPU is available"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="MPS device was requested but none"):
        resolve_device("mps")


def test_a_gpu_index_beyond_the_available_ones_is_an_error(two_gpus: None) -> None:
    with pytest.raises(ValueError, match="only 2 CUDA devices"):
        resolve_device("cuda:2")


def test_an_unknown_device_is_an_error(no_gpu: None) -> None:
    with pytest.raises(ValueError, match="unknown device 'tpu'"):
        resolve_device("tpu")


def test_mps_can_be_chosen_when_available(
    no_gpu: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("mps") == torch.device("mps")
