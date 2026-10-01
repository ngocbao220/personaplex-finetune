"""Small configuration reader with deterministic, config-relative paths and OmegaConf/Hydra support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .generation import GenerationSettings, generation_from_config


@dataclass(frozen=True)
class Config:
    path: Path
    model_root: Path
    personaplex_source: Path
    prepared_dir: Path
    output_dir: Path
    codec_cache_dir: Path | None = None
    seed: int = 42
    window_seconds: float | None = None
    shuffle: bool = False
    randomize_train: bool = False
    max_steps: int = 300
    learning_rate: float = 2e-5
    depformer_learning_rate: float | None = None
    train_stage: str = "joint"
    train_method: str = "lora"
    lora_rank: int = 16
    lora_alpha: int = 32
    qlora: bool = False
    quant_type: str = "nf4"
    device: str = "cuda"
    gradient_accumulation_steps: int = 1
    per_device_batch_size: int = 2
    num_workers: int = 4
    filter_num_workers: int = 64
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    warmup_steps: int = 0
    eval_every_steps: int = 0
    save_every_steps: int = 50
    val_ratio: float = 0.05
    prompt_aug_prob: float = 0.0
    normalize_vietnamese_diacritics: bool = False
    static_chunking: bool = False
    swap_roles_after_pass: bool = False
    val_manifest_path: Path | None = None
    test_manifest_path: Path | None = None
    gradient_checkpointing: bool = False
    mixed_precision: str = "bf16"
    duration_sec: float = 100.0
    sample_number: int | None = None
    sample_index: int | None = None
    profile_steps: bool = False
    generation_settings: GenerationSettings = GenerationSettings()
    free_running_eval_every_steps: int = 0
    free_running_eval_samples: int = 1
    free_running_eval_window_seconds: float = 30.0
    validation_max_samples: int = 32
    lora_enabled: bool = True
    lora_scaling: float = 2.0
    ft_embed: bool = False
    weight_decay: float = 0.1
    pct_start: float = 0.05
    first_codebook_weight_multiplier: float = 1.0
    text_padding_weight: float = 0.5
    log_freq: int = 1
    no_eval: bool = False
    ckpt_freq: int = 50
    eval_on_train_samples: bool = False

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

    @property
    def test_manifest(self) -> Path | None:
        if self.test_manifest_path is not None:
            return self.test_manifest_path
        test_default = self.prepared_dir / "test.jsonl"
        return test_default if test_default.is_file() else None

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
                    destination = "optim.lr" if "optim" in conf else "train.learning_rate"
                    hydra_overrides.append(f"{destination}={val}")
                elif o.startswith("stage=") or o.startswith("train_stage="):
                    val = o.split("=", 1)[1]
                    hydra_overrides.append(f"+train.stage={val}")
                elif o.startswith("freeze_depformer=") or o.startswith("freeze_tempformer="):
                    key, val = o.split("=", 1)
                    if val.lower() in {"true", "1"}:
                        st = "temporal_only" if "depformer" in key else "depth_only"
                        hydra_overrides.append(f"+train.stage={st}")
                elif o.startswith("learning_rate="):
                    val = o.split("=", 1)[1]
                    destination = "optim.lr" if "optim" in conf else "train.learning_rate"
                    hydra_overrides.append(f"{destination}={val}")
                elif o.startswith("max_steps="):
                    destination = "max_steps" if "max_steps" in conf else "train.max_steps"
                    hydra_overrides.append(f"{destination}={o.split('=', 1)[1]}")
                elif o.startswith("sample_index="):
                    hydra_overrides.append(f"++sample_index={o.split('=', 1)[1]}")
                elif o.startswith("sample_number="):
                    hydra_overrides.append(f"++sample_number={o.split('=', 1)[1]}")
                elif o.startswith("batch_size="):
                    hydra_overrides.append(f"++batch_size={o.split('=', 1)[1]}")
                elif o.startswith("output_dir="):
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
    optim = raw.get("optim", {})
    if not isinstance(model, dict) or not isinstance(data, dict):
        raise ValueError("model and data config sections must be mappings")
    if not isinstance(train, dict) or not isinstance(lora, dict) or not isinstance(optim, dict):
        raise ValueError("train, lora, and optim config sections must be mappings")
    train_method = str(train.get("method", "lora")).lower()
    if train_method not in {"lora", "full"}:
        raise ValueError("train.method must be lora or full")
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
    codec_cache_raw = data.get("codec_cache_dir")
    if codec_cache_raw is not None and (not isinstance(codec_cache_raw, str) or not codec_cache_raw.strip()):
        raise ValueError("data.codec_cache_dir must be null or a non-empty path")
    codec_cache_dir = resolve(data, "codec_cache_dir") if codec_cache_raw is not None else None
    qlora = bool(lora.get("qlora", False)) if isinstance(lora, dict) else False
    quant_type = str(lora.get("quant_type", "nf4")).lower() if isinstance(lora, dict) else "nf4"
    if quant_type not in {"nf4", "fp4"}:
        raise ValueError("lora.quant_type must be nf4 or fp4")
    sample_number_raw = raw.get("sample_number")
    sample_number = None if sample_number_raw is None else int(sample_number_raw)
    if sample_number is not None and sample_number < 1:
        raise ValueError("sample_number must be null or a positive integer")
    sample_index_raw = raw.get("sample_index", train.get("sample_index"))
    sample_index = None if sample_index_raw is None else int(sample_index_raw)
    if sample_index is not None and sample_index < 0:
        raise ValueError("sample_index must be null or a non-negative integer")
    duration_sec = float(raw.get("duration_sec", data.get("duration_sec", 100.0)))
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    free_running_eval_window_seconds = float(raw.get("free_running_eval_window_seconds", 30.0))
    if free_running_eval_window_seconds <= 0:
        raise ValueError("free_running_eval_window_seconds must be positive")
    first_codebook_weight_multiplier = float(raw.get("first_codebook_weight_multiplier", 1.0))
    text_padding_weight = float(raw.get("text_padding_weight", 0.5))
    if first_codebook_weight_multiplier < 0:
        raise ValueError("first_codebook_weight_multiplier must be non-negative")
    if not 0 <= text_padding_weight <= 1:
        raise ValueError("text_padding_weight must be within [0, 1]")
    val_manifest_raw = data.get("val_manifest")
    val_manifest_path = resolve(data, "val_manifest") if isinstance(val_manifest_raw, str) and val_manifest_raw else None
    test_manifest_key = "test_manifest_path" if data.get("test_manifest_path") else "test_manifest"
    test_manifest_raw = data.get(test_manifest_key)
    test_manifest_path = resolve(data, test_manifest_key) if isinstance(test_manifest_raw, str) and test_manifest_raw else None
    lora_rank = int(lora.get("rank", 128))
    lora_scaling = float(lora.get("scaling", 2.0))
    # Training injects LoRA with alpha = rank * scaling. Keep the recorded
    # alpha consistent with the value that is actually used at runtime.
    lora_alpha = round(lora_rank * lora_scaling)
    generation_settings = generation_from_config(raw)
    max_steps = int(raw.get("max_steps", train.get("max_steps", 300)))
    warmup_steps = int(train.get("warmup_steps", 0))
    if max_steps < 1 or warmup_steps < 0 or warmup_steps >= max_steps:
        raise ValueError("max_steps must be positive and train.warmup_steps must be in [0, max_steps)")
    configured_pct_start = optim.get("pct_start", train.get("pct_start"))
    if warmup_steps and configured_pct_start is not None:
        raise ValueError("set either train.warmup_steps or optim.pct_start, not both")
    pct_start = (
        warmup_steps / max_steps if warmup_steps
        else float(configured_pct_start if configured_pct_start is not None else 0.05)
    )
    if not 0 < pct_start < 1:
        raise ValueError("optim.pct_start must be strictly between 0 and 1")

    return Config(
        path=path,
        model_root=resolve(model, "root"),
        personaplex_source=resolve(model, "source"),
        prepared_dir=prepared_dir,
        output_dir=resolve(train if isinstance(train, dict) else {}, "output_dir", "../runs/overfit_10"),
        codec_cache_dir=codec_cache_dir,
        seed=int(raw.get("seed", 42)),
        window_seconds=(
            float(data["window_seconds"])
            if data.get("window_seconds") is not None
            else None
        ),
        shuffle=bool(data.get("shuffle", False)),
        randomize_train=bool(data.get("randomize_train", False)),
        max_steps=max_steps,
        learning_rate=float(optim.get("lr", train.get("learning_rate", 2e-5))),
        depformer_learning_rate=float(train["depformer_learning_rate"]) if (isinstance(train, dict) and train.get("depformer_learning_rate") is not None) else None,
        train_stage=str(train.get("stage") or train.get("train_stage") or "joint").lower() if isinstance(train, dict) else "joint",
        train_method=train_method,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        qlora=qlora,
        quant_type=quant_type,
        device=str(model.get("device", "cuda")),
        gradient_accumulation_steps=max(1, int(train.get("gradient_accumulation_steps", 1))) if isinstance(train, dict) else 1,
        per_device_batch_size=max(1, int(raw.get("batch_size", train.get("per_device_batch_size", 16)))),
        num_workers=max(0, int(train.get("num_workers", 4))) if isinstance(train, dict) else 4,
        filter_num_workers=max(1, int(train.get("filter_num_workers", 64))) if isinstance(train, dict) else 64,
        prefetch_factor=max(1, int(train.get("prefetch_factor", 2))) if isinstance(train, dict) else 2,
        pin_memory=bool(train.get("pin_memory", True)) if isinstance(train, dict) else True,
        persistent_workers=bool(train.get("persistent_workers", True)) if isinstance(train, dict) else True,
        warmup_steps=warmup_steps,
        eval_every_steps=max(0, int(train.get("eval_every_steps", 0))) if isinstance(train, dict) else 0,
        save_every_steps=max(1, int(train.get("save_every_steps", 50))) if isinstance(train, dict) else 50,
        val_ratio=float(data.get("val_ratio", 0.05)),
        prompt_aug_prob=float(data.get("prompt_aug_prob", 0.0)),
        normalize_vietnamese_diacritics=bool(data.get("normalize_vietnamese_diacritics", False)),
        static_chunking=bool(data.get("static_chunking", False)),
        swap_roles_after_pass=bool(data.get("swap_roles_after_pass", False)),
        val_manifest_path=val_manifest_path,
        test_manifest_path=test_manifest_path,
gradient_checkpointing=bool(raw.get("gradient_checkpointing", train.get("gradient_checkpointing", False))),
        mixed_precision=str(train.get("mixed_precision", "bf16")),
        duration_sec=duration_sec,
        sample_number=sample_number,
        sample_index=sample_index,
        profile_steps=bool(raw.get("profile_steps", False)),
        generation_settings=generation_settings,
        free_running_eval_every_steps=max(0, int(raw.get("free_running_eval_every_steps", 0))),
        free_running_eval_samples=max(1, int(raw.get("free_running_eval_samples", 1))),
        free_running_eval_window_seconds=free_running_eval_window_seconds,
        validation_max_samples=max(1, int(raw.get("validation_max_samples", 32))),
        lora_enabled=bool(lora.get("enable", True)),
        lora_scaling=lora_scaling,
        ft_embed=bool(lora.get("ft_embed", False)),
        weight_decay=float(optim.get("weight_decay", train.get("weight_decay", 0.1))),
        pct_start=pct_start,
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
        text_padding_weight=text_padding_weight,
        log_freq=max(1, int(raw.get("log_freq", 1))),
        no_eval=bool(raw.get("no_eval", True)),
        ckpt_freq=max(1, int(raw.get("ckpt_freq", train.get("save_every_steps", 50)))),
        eval_on_train_samples=bool(train.get("eval_on_train_samples", False)) if isinstance(train, dict) else False,
    )
