"""Small configuration reader with deterministic, config-relative paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    path: Path
    model_root: Path
    personaplex_source: Path
    manifest: Path
    output_dir: Path
    seed: int = 42
    window_seconds: float = 30.0
    shuffle: bool = False
    max_steps: int = 300
    learning_rate: float = 2e-5
    lora_rank: int = 16
    lora_alpha: int = 32
    device: str = "cuda"


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("PyYAML is required for non-JSON YAML configs") from exc
    else:
        parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError("config root must be a mapping")
    return parsed


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    raw = _read_yaml_or_json(path)
    model = raw.get("model", {})
    data = raw.get("data", {})
    train = raw.get("train", {})
    lora = raw.get("lora", {})
    if not isinstance(model, dict) or not isinstance(data, dict):
        raise ValueError("model and data config sections must be mappings")
    root = path.parent
    def resolve(section: dict[str, Any], key: str, default: str | None = None) -> Path:
        value = section.get(key, default)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty path")
        return (root / value).resolve() if not Path(value).is_absolute() else Path(value)
    return Config(
        path=path,
        model_root=resolve(model, "root"),
        personaplex_source=resolve(model, "source"),
        manifest=resolve(data, "manifest"),
        output_dir=resolve(train if isinstance(train, dict) else {}, "output_dir", "../runs/overfit_10"),
        seed=int(raw.get("seed", 42)),
        window_seconds=float(data.get("window_seconds", 30.0)),
        shuffle=bool(data.get("shuffle", False)),
        max_steps=int(train.get("max_steps", 300)) if isinstance(train, dict) else 300,
        learning_rate=float(train.get("learning_rate", 2e-5)) if isinstance(train, dict) else 2e-5,
        lora_rank=int(lora.get("rank", 16)) if isinstance(lora, dict) else 16,
        lora_alpha=int(lora.get("alpha", 32)) if isinstance(lora, dict) else 32,
        device=str(model.get("device", "cuda")),
    )
