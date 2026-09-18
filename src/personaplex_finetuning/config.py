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
    prepared_dir: Path
    output_dir: Path
    seed: int = 42
    window_seconds: float = 30.0
    shuffle: bool = False
    max_steps: int = 300
    learning_rate: float = 2e-5
    lora_rank: int = 16
    lora_alpha: int = 32
    qlora: bool = False
    quant_type: str = "nf4"
    device: str = "cuda"
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    eval_every_steps: int = 0
    save_every_steps: int = 50
    val_ratio: float = 0.05
    random_crop: bool = False
    prompt_aug_prob: float = 0.0
    val_manifest_path: Path | None = None

    @property
    def manifest(self) -> Path:
        """Canonical local manifest inside the externally prepared dataset root."""
        return self.prepared_dir / "train.jsonl"

    @property
    def val_manifest(self) -> Path | None:
        if self.val_manifest_path is not None:
            return self.val_manifest_path
        val_default = self.prepared_dir / "val.jsonl"
        return val_default if val_default.is_file() else None

    def replace(self, **kwargs) -> Config:
        import dataclasses
        return dataclasses.replace(self, **kwargs)


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
    prepared_dir = resolve(data, "prepared_dir") if "prepared_dir" in data else resolve(data, "manifest").parent
    qlora = bool(lora.get("qlora", False)) if isinstance(lora, dict) else False
    quant_type = str(lora.get("quant_type", "nf4")).lower() if isinstance(lora, dict) else "nf4"
    if quant_type not in {"nf4", "fp4"}:
        raise ValueError("lora.quant_type must be nf4 or fp4")
    val_manifest_raw = data.get("val_manifest")
    val_manifest_path = resolve(data, "val_manifest") if isinstance(val_manifest_raw, str) and val_manifest_raw else None
    return Config(
        path=path,
        model_root=resolve(model, "root"),
        personaplex_source=resolve(model, "source"),
        prepared_dir=prepared_dir,
        output_dir=resolve(train if isinstance(train, dict) else {}, "output_dir", "../runs/overfit_10"),
        seed=int(raw.get("seed", 42)),
        window_seconds=float(data.get("window_seconds", 30.0)),
        shuffle=bool(data.get("shuffle", False)),
        max_steps=int(train.get("max_steps", 300)) if isinstance(train, dict) else 300,
        learning_rate=float(train.get("learning_rate", 2e-5)) if isinstance(train, dict) else 2e-5,
        lora_rank=int(lora.get("rank", 16)) if isinstance(lora, dict) else 16,
        lora_alpha=int(lora.get("alpha", 32)) if isinstance(lora, dict) else 32,
        qlora=qlora,
        quant_type=quant_type,
        device=str(model.get("device", "cuda")),
        gradient_accumulation_steps=max(1, int(train.get("gradient_accumulation_steps", 1))) if isinstance(train, dict) else 1,
        warmup_steps=max(0, int(train.get("warmup_steps", 0))) if isinstance(train, dict) else 0,
        eval_every_steps=max(0, int(train.get("eval_every_steps", 0))) if isinstance(train, dict) else 0,
        save_every_steps=max(1, int(train.get("save_every_steps", 50))) if isinstance(train, dict) else 50,
        val_ratio=float(data.get("val_ratio", 0.05)),
        random_crop=bool(data.get("random_crop", False)),
        prompt_aug_prob=float(data.get("prompt_aug_prob", 0.0)),
        val_manifest_path=val_manifest_path,
    )
