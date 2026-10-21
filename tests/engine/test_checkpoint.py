import pickle
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from mammo_seg.data.preprocessing import PreprocessingSettings
from mammo_seg.engine import checkpoint as checkpoint_module
from mammo_seg.engine.checkpoint import (
    FORMAT_VERSION,
    Checkpoint,
    load_checkpoint,
    load_model,
    model_from_checkpoint,
    save_checkpoint,
)
from mammo_seg.engine.model import ModelSettings, build_model

SETTINGS = ModelSettings(encoder="resnet18", encoder_weights=None)
PREPROCESSING = PreprocessingSettings(size=512, margin=0.05)


@pytest.fixture
def model() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_model(SETTINGS)


def outputs(model: torch.nn.Module) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        result: torch.Tensor = model(
            torch.linspace(0, 1, 64 * 64).reshape(1, 1, 64, 64)
        )
    return result


# --- saving and loading ---


def test_a_checkpoint_round_trips(tmp_path: Path, model: torch.nn.Module) -> None:
    path = tmp_path / "runs" / "model.pt"
    save_checkpoint(path, model, SETTINGS, PREPROCESSING)
    checkpoint = load_checkpoint(path)
    assert isinstance(checkpoint, Checkpoint)
    assert checkpoint.model_settings == SETTINGS
    assert checkpoint.preprocessing_settings == PREPROCESSING
    assert checkpoint.training_state is None and checkpoint.metadata == {}
    assert set(checkpoint.model_state) == set(model.state_dict())
    assert all(
        torch.equal(checkpoint.model_state[key], value)
        for key, value in model.state_dict().items()
    )


def test_a_rebuilt_model_gives_the_same_output(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    save_checkpoint(tmp_path / "model.pt", model, SETTINGS, PREPROCESSING)
    rebuilt = model_from_checkpoint(load_checkpoint(tmp_path / "model.pt"))
    assert torch.equal(outputs(rebuilt), outputs(model))


def test_a_rebuilt_model_never_asks_for_pretrained_weights(
    tmp_path: Path,
    model: torch.nn.Module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[object] = []
    real = build_model

    def spy(settings: ModelSettings) -> torch.nn.Module:
        requested.append(settings.encoder_weights)
        return real(settings)

    monkeypatch.setattr(checkpoint_module, "build_model", spy)
    pretrained = ModelSettings(encoder="resnet18", encoder_weights="imagenet")
    save_checkpoint(tmp_path / "model.pt", model, pretrained, PREPROCESSING)
    model_from_checkpoint(load_checkpoint(tmp_path / "model.pt"))
    assert requested == [None]


def test_load_model_returns_an_evaluation_model_and_the_preprocessing(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    save_checkpoint(tmp_path / "model.pt", model, SETTINGS, PREPROCESSING)
    loaded, preprocessing = load_model(tmp_path / "model.pt", "cpu")
    assert not loaded.training
    assert preprocessing == PREPROCESSING
    assert torch.equal(outputs(loaded), outputs(model))


def test_the_training_state_and_metadata_are_kept(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.rand(1, 1, 64, 64)).sum().backward()
    optimizer.step()
    state = {"optimizer": optimizer.state_dict(), "epoch": 7, "best_dice": 0.61}
    save_checkpoint(
        tmp_path / "model.pt",
        model,
        SETTINGS,
        PREPROCESSING,
        training_state=state,
        metadata={"fold": 2, "note": "first run"},
    )
    checkpoint = load_checkpoint(tmp_path / "model.pt")
    assert checkpoint.training_state is not None
    assert checkpoint.training_state["epoch"] == 7
    assert checkpoint.training_state["best_dice"] == 0.61
    assert checkpoint.metadata == {"fold": 2, "note": "first run"}
    fresh = torch.optim.Adam(build_model(SETTINGS).parameters(), lr=1e-3)
    fresh.load_state_dict(checkpoint.training_state["optimizer"])  # resumes
    assert len(fresh.state_dict()["state"]) == len(optimizer.state_dict()["state"])


def test_a_checkpoint_loads_onto_the_requested_device(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    save_checkpoint(tmp_path / "model.pt", model, SETTINGS, PREPROCESSING)
    checkpoint = load_checkpoint(tmp_path / "model.pt", torch.device("cpu"))
    assert all(
        tensor.device.type == "cpu" for tensor in checkpoint.model_state.values()
    )


def test_weights_on_another_device_are_saved_from_the_cpu(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    save_checkpoint(tmp_path / "model.pt", model, SETTINGS, PREPROCESSING)
    raw = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=True)
    assert all(tensor.device.type == "cpu" for tensor in raw["model_state"].values())


# --- safety on a free cloud GPU ---


def test_no_temporary_file_is_left_behind(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    save_checkpoint(tmp_path / "model.pt", model, SETTINGS, PREPROCESSING)
    assert [path.name for path in tmp_path.iterdir()] == ["model.pt"]


def test_a_failed_save_keeps_the_previous_checkpoint(
    tmp_path: Path, model: torch.nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, SETTINGS, PREPROCESSING, metadata={"epoch": 1})
    before = path.read_bytes()

    def dying_session(content: object, target: Path) -> None:
        Path(target).write_bytes(b"half a checkpoint")
        raise OSError("the session was killed")

    monkeypatch.setattr(torch, "save", dying_session)
    with pytest.raises(OSError, match="killed"):
        save_checkpoint(path, model, SETTINGS, PREPROCESSING, metadata={"epoch": 2})
    assert path.read_bytes() == before
    assert load_checkpoint(path).metadata == {"epoch": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["model.pt"]


def test_a_checkpoint_with_code_in_it_is_not_loaded(tmp_path: Path) -> None:
    class Payload:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return (print, ("this must never run",))

    path = tmp_path / "evil.pt"
    path.write_bytes(pickle.dumps(Payload()))
    with pytest.raises(ValueError, match="not a readable checkpoint"):
        load_checkpoint(path)


# --- bad files ---


def test_a_missing_or_garbled_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a readable checkpoint"):
        load_checkpoint(tmp_path / "missing.pt")
    (tmp_path / "garbled.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="not a readable checkpoint"):
        load_checkpoint(tmp_path / "garbled.pt")


def test_a_file_of_another_kind_is_an_error(tmp_path: Path) -> None:
    torch.save({"weights": torch.zeros(2)}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a checkpoint of this project"):
        load_checkpoint(tmp_path / "other.pt")
    torch.save([1, 2, 3], tmp_path / "list.pt")
    with pytest.raises(ValueError, match="not a checkpoint of this project"):
        load_checkpoint(tmp_path / "list.pt")


def test_another_format_version_is_an_error(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, SETTINGS, PREPROCESSING)
    content = torch.load(path, weights_only=True)
    content["format_version"] = FORMAT_VERSION + 1
    torch.save(content, path)
    with pytest.raises(ValueError, match="format version 2, expected 1"):
        load_checkpoint(path)


def test_invalid_settings_in_a_file_are_an_error(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, SETTINGS, PREPROCESSING)
    content = torch.load(path, weights_only=True)
    content["model_settings"] = content["model_settings"].replace("unet", "resnet")
    torch.save(content, path)
    with pytest.raises(ValidationError, match="architecture"):
        load_checkpoint(path)


def test_weights_that_do_not_fit_the_model_are_an_error(
    tmp_path: Path, model: torch.nn.Module
) -> None:
    other = ModelSettings(architecture="fpn", encoder="resnet18", encoder_weights=None)
    save_checkpoint(tmp_path / "model.pt", model, other, PREPROCESSING)
    with pytest.raises(RuntimeError, match="size mismatch|Missing key|Unexpected key"):
        model_from_checkpoint(load_checkpoint(tmp_path / "model.pt"))
