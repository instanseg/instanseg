from types import SimpleNamespace

import numpy as np
import pytest
import torch

from instanseg.scripts import train
from instanseg.utils import AI_utils
from instanseg.utils.training_checkpoint import (
    atomic_torch_save,
    load_training_checkpoint,
    validate_resume_config,
)


def _training_args(tmp_path):
    args = train.parser.parse_args([])
    args.output_path = tmp_path
    args.layers = [32, 64, 128, 256]
    args.source_dataset = ["cpdmi_2023"]
    args.cells_and_nuclei = False
    args.model_folder = None
    args.resume_checkpoint = None
    args.optimize_hyperparameters = False
    args.on_cluster = True
    return args


def test_atomic_save_preserves_previous_checkpoint_on_failed_write(tmp_path, monkeypatch):
    target = tmp_path / "latest_checkpoint.pth"
    atomic_torch_save({"value": "previous"}, target)

    def failed_save(payload, path):
        path.write_bytes(b"partial")
        raise RuntimeError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", failed_save)
    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_torch_save({"value": "replacement"}, target)

    assert torch.load(target, weights_only=False) == {"value": "previous"}
    assert not (tmp_path / ".latest_checkpoint.pth.tmp").exists()


def test_main_keeps_latest_and_best_checkpoints_separate(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()

    scores = iter([0.8, 0.5])

    def fake_train_epoch(model, *args, **kwargs):
        with torch.no_grad():
            model.weight.add_(1)
        return 1.0, 0.0

    def fake_test_epoch(*args, **kwargs):
        return 1.0, np.array([next(scores)]), 0.0

    monkeypatch.setattr(AI_utils, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(AI_utils, "test_epoch", fake_test_epoch)

    train.args = _training_args(tmp_path)
    train.device = torch.device("cpu")
    train.method = SimpleNamespace(postprocessing=None)
    train.iou_threshold = np.array([0.5])
    train.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train.scheduler = None

    train.main(
        model,
        loss_fn=None,
        train_loader=[None],
        test_loader=[None],
        num_epochs=2,
        phase="main",
    )

    best = load_training_checkpoint(tmp_path / "best_model_weights.pth", "cpu")
    latest = load_training_checkpoint(tmp_path / "latest_checkpoint.pth", "cpu")
    compatibility = load_training_checkpoint(tmp_path / "model_weights.pth", "cpu")

    assert best["epoch"] == 0
    assert best["best_f1_score"] == pytest.approx(0.8)
    assert best["model_state_dict"]["weight"].item() == pytest.approx(1.0)
    assert compatibility["model_state_dict"]["weight"].item() == pytest.approx(1.0)
    assert latest["epoch"] == 1
    assert latest["best_f1_score"] == pytest.approx(0.8)
    assert latest["model_state_dict"]["weight"].item() == pytest.approx(2.0)
    assert latest["train_losses"] == [1.0, 1.0]

    resumed_model = torch.nn.Linear(1, 1, bias=False)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.001)
    train.args.resume_checkpoint = str(tmp_path / "latest_checkpoint.pth")
    resumed_state = train._restore_training_checkpoint(
        resumed_model,
        resumed_optimizer,
        scheduler=None,
        checkpoint_path=train.args.resume_checkpoint,
        device="cpu",
        args=train.args,
    )
    assert resumed_model.weight.item() == pytest.approx(2.0)
    assert resumed_state["epoch"] == 1

    scores = iter([0.9])
    train.optimizer = resumed_optimizer
    resumed_result = train.main(
        resumed_model,
        loss_fn=None,
        train_loader=[None],
        test_loader=[None],
        num_epochs=3,
        phase="main",
        start_epoch=resumed_state["epoch"] + 1,
        resume_state=resumed_state,
    )
    assert resumed_result[1] == [1.0, 1.0, 1.0]
    assert resumed_result[5] == pytest.approx(0.9)
    resumed_latest = load_training_checkpoint(tmp_path / "latest_checkpoint.pth", "cpu")
    assert resumed_latest["epoch"] == 2
    assert resumed_latest["model_state_dict"]["weight"].item() == pytest.approx(3.0)


def test_resume_config_mismatch_is_rejected():
    with pytest.raises(ValueError, match="tile_size"):
        validate_resume_config(
            {"tile_size": 256, "requested_pixel_size": 0.325},
            {"tile_size": 384, "requested_pixel_size": 0.325},
        )
