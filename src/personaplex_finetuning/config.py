"""Small configuration reader with deterministic, config-relative paths and OmegaConf/Hydra support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


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
    depformer_learning_rate: float | None = None
    train_stage: str = "joint"
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
    static_chunking: bool = False
    swap_roles_after_pass: bool = False
    val_manifest_path: Path | None = None
    gradient_checkpointing: bool = False
    mixed_precision: str = "bf16"

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


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")

    # Load with Hydra or OmegaConf to support modular configs and dotlist overrides
    try:
        conf = OmegaConf.load(str(path))
        if isinstance(conf, DictConfig) and "defaults" in conf:
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra

            # Normalize common convenience overrides for user simplicity
            hydra_overrides = []
            for o in (overrides or []):
                if o.startswith("gpus=") or o.startswith("gpu=") or o.startswith("devices=") or o.startswith("device_ids="):
                    continue
                elif o.startswith("device="):
                    hydra_overrides.append(f"model.{o}")
                elif o.startswith("window_seconds=") or o.startswith("shuffle="):
                    hydra_overrides.append(f"data.{o}")
                elif o.startswith("depformer_lr=") or o.startswith("depformer_learning_rate="):
                    val = o.split("=", 1)[1]
                    hydra_overrides.append(f"+train.depformer_learning_rate={val}")
                elif o.startswith("tempformer_lr=") or o.startswith("tempformer_learning_rate="):
                    val = o.split("=", 1)[1]
                    hydra_overrides.append(f"train.learning_rate={val}")
                elif o.startswith("stage=") or o.startswith("train_stage="):
                    val = o.split("=", 1)[1]
                    hydra_overrides.append(f"+train.stage={val}")
                elif o.startswith("freeze_depformer=") or o.startswith("freeze_tempformer="):
                    key, val = o.split("=", 1)
                    if val.lower() in {"true", "1"}:
                        st = "temporal_only" if "depformer" in key else "depth_only"
                        hydra_overrides.append(f"+train.stage={st}")
                elif o.startswith("learning_rate=") or o.startswith("max_steps=") or o.startswith("output_dir="):
                    hydra_overrides.append(f"train.{o}")
                elif o.startswith("rank=") or o.startswith("alpha=") or o.startswith("qlora="):
                    hydra_overrides.append(f"lora.{o}")
                else:
                    hydra_overrides.append(o)

            GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=str(path.parent), version_base=None):
                conf = compose(config_name=path.stem, overrides=hydra_overrides)
            raw = OmegaConf.to_container(conf, resolve=True)
        else:
            if overrides:
                override_conf = OmegaConf.from_dotlist(list(overrides))
                conf = OmegaConf.merge(conf, override_conf)
            raw = OmegaConf.to_container(conf, resolve=True)
    except Exception as exc:
        if isinstance(conf, DictConfig) and "defaults" in conf:
            raise RuntimeError(f"Failed to compose Hydra configuration '{path.name}' with overrides {overrides}: {exc}") from exc
        raw = _read_yaml_or_json(path)

    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

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
        p = Path(value)
        if p.is_absolute():
            return p
        cand_root = (root / p).resolve()
        if cand_root.exists():
            return cand_root
        cand_parent = (root.parent / p).resolve()
        if cand_parent.exists():
            return cand_parent
        return cand_root


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
        depformer_learning_rate=float(train["depformer_learning_rate"]) if (isinstance(train, dict) and train.get("depformer_learning_rate") is not None) else None,
        train_stage=str(train.get("stage") or train.get("train_stage") or "joint").lower() if isinstance(train, dict) else "joint",
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
        static_chunking=bool(data.get("static_chunking", False)),
        swap_roles_after_pass=bool(data.get("swap_roles_after_pass", False)),
        val_manifest_path=val_manifest_path,
        gradient_checkpointing=bool(train.get("gradient_checkpointing", False)) if isinstance(train, dict) else False,
        mixed_precision=str(train.get("mixed_precision", "bf16")) if isinstance(train, dict) else "bf16",
    )
