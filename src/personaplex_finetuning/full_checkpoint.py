"""Complete PersonaPlex LM checkpoints for full-parameter training and inference."""

from __future__ import annotations

import json
import os
from pathlib import Path


def resolve_full_checkpoint(checkpoint: Path) -> tuple[Path, dict]:
    checkpoint = Path(checkpoint).expanduser()
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    weights = directory / "model.safetensors"
    metadata_path = directory / "checkpoint.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"full checkpoint metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("method") != "full":
        raise ValueError(f"invalid full checkpoint metadata: {metadata_path}")
    if not isinstance(metadata.get("base_model_root"), str) or not metadata["base_model_root"]:
        raise ValueError(f"full checkpoint lacks base_model_root: {metadata_path}")
    if isinstance(metadata.get("step"), bool) or not isinstance(metadata.get("step"), int) or metadata["step"] < 0:
        raise ValueError(f"full checkpoint has invalid step: {metadata_path}")
    if not weights.is_file():
        raise FileNotFoundError(f"full checkpoint weights not found: {weights}")
    return weights, metadata


def save_full_weights(model, checkpoint_dir: Path) -> Path:
    """Write all LM tensors, publishing the weights file only after serialization succeeds."""
    from safetensors.torch import save_model

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights = checkpoint_dir / "model.safetensors"
    temporary = checkpoint_dir / "model.safetensors.tmp"
    save_model(model, str(temporary))
    os.replace(temporary, weights)
    return weights


def write_full_metadata(checkpoint_dir: Path, *, base_model_root: Path, step: int) -> None:
    """Publish the completion marker after weights and optimizer state exist."""
    metadata = checkpoint_dir / "checkpoint.json"
    temporary = checkpoint_dir / "checkpoint.json.tmp"
    temporary.write_text(
        json.dumps({"method": "full", "base_model_root": str(Path(base_model_root).expanduser().resolve()), "step": step}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, metadata)


def load_full_weights(model, checkpoint: Path) -> tuple[Path, dict]:
    from safetensors.torch import load_model

    weights, metadata = resolve_full_checkpoint(checkpoint)
    load_model(model, str(weights), strict=True)
    return weights, metadata
