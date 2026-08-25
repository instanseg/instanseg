import os
import random
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1


def atomic_torch_save(payload, path):
    """Write a torch checkpoint without exposing a partially written target file."""
    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["cuda"]])


def make_training_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    phase,
    epoch,
    best_f1_score,
    train_losses,
    test_losses,
    f1_list,
    f1_list_cells,
    training_config,
):
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "phase": phase,
        "epoch": int(epoch),
        "f1_score": float(best_f1_score),
        "best_f1_score": float(best_f1_score),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "train_losses": list(train_losses),
        "test_losses": list(test_losses),
        "f1_list": list(f1_list),
        "f1_list_cells": list(f1_list_cells),
        "rng_state": capture_rng_state(),
        "training_config": dict(training_config),
    }


def load_training_checkpoint(path, device):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    version = checkpoint.get("checkpoint_schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported training checkpoint schema {version!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    required = {
        "phase",
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "best_f1_score",
        "training_config",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Training checkpoint is missing required fields: {missing}")
    if checkpoint["phase"] not in {"hotstart", "main"}:
        raise ValueError(f"Invalid checkpoint phase: {checkpoint['phase']!r}")
    return checkpoint


def validate_resume_config(saved_config, current_config):
    mismatches = []
    for key, saved_value in saved_config.items():
        current_value = current_config.get(key)
        if current_value != saved_value:
            mismatches.append(f"{key}: saved={saved_value!r}, current={current_value!r}")
    if mismatches:
        raise ValueError("Resume configuration mismatch: " + "; ".join(mismatches))
