"""Strict configuration adapter; omitted fields retain the reference's defaults.

No training/runtime imports belong here: configuration checks must work on CPU.
The reference trains both audio streams. ``user_loss=false`` is unsupported,
as are dep_q changes, stream loss policies, split/augmentation and custom loaders.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

# Set these before even Hydra imports (and override inherited online settings).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


class ConfigError(ValueError):
    """An option cannot be represented faithfully by the reference trainer."""


def _reference_schema() -> tuple[dict[str, set[str]], set[str]]:
    """Read the actual vendored dataclasses without importing GPU dependencies."""
    source = Path(__file__).parent / "reference/moshi-finetune/finetune/args.py"
    data_source = source.parent / "data/args.py"
    if not source.is_file() or not data_source.is_file():
        raise ConfigError("Vendored TrainArgs schema missing")
    classes = {}
    annotations = {}
    for path in (source, data_source):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                fields = {}
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        # world_size is computed by reference, never user input.
                        if item.target.id != "world_size":
                            fields[item.target.id] = ast.unparse(item.annotation)
                classes[node.name] = set(fields)
                annotations[node.name] = fields
    groups = {}
    scalars = set()
    for name, annotation in annotations["TrainArgs"].items():
        typename = annotation.split(" | ")[0]
        if typename in classes:
            groups[name] = classes[typename]
        else:
            scalars.add(name)
    return groups, scalars
MAPPING = {
    "seed": "seed", "duration_sec": "duration_sec", "log_freq": "log_freq",
    "first_codebook_weight_multiplier": "first_codebook_weight_multiplier",
    "text_padding_weight": "text_padding_weight", "ckpt_freq": "ckpt_freq",
    "train.output_dir": "run_dir", "train.max_steps": "max_steps",
    "train.learning_rate": "optim.lr", "train.weight_decay": "optim.weight_decay",
    "train.pct_start": "optim.pct_start", "train.max_norm": "max_norm",
    "train.gradient_accumulation_steps": "num_microbatches",
    "train.per_device_batch_size": "batch_size",
    "train.gradient_checkpointing": "gradient_checkpointing",
    "train.save_every_steps": "ckpt_freq", "train.eval_every_steps": "eval_freq",
    "train.log_freq": "log_freq", "train.do_eval": "do_eval",
    "train.do_ckpt": "do_ckpt", "train.save_adapters": "save_adapters",
    "train.num_ckpt_keep": "num_ckpt_keep",
    "train.overwrite_run_dir": "overwrite_run_dir",
    "data.train_data": "data.train_data", "data.eval_data": "data.eval_data",
    "data.val_manifest": "data.eval_data", "data.shuffle": "data.shuffle",
    "lora.rank": "lora.rank", "lora.scaling": "lora.scaling",
    "lora.enabled": "lora.enable", "lora.enable": "lora.enable",
    "lora.ft_embed": "lora.ft_embed",
}
# Only explicitly inert options may be ignored. Unknown options always fail.
INERT = {
    "profile_steps": (False,), "free_running_eval_every_steps": (0,),
    "data.randomize_train": (False,), "data.prompt_aug_prob": (0,),
    "data.static_chunking": (False,), "data.swap_roles_after_pass": (False,),
    "data.val_ratio": (0,), "data.window_seconds": (None,),
    "data.sample_number": (None,), "data.sample_index": (None,),
    "data.vietnamese_text_mode": ("diacritics",), "lora.qlora": (False,),
    "train.depformer_learning_rate": (None,), "train.warmup_steps": (0,),
    "model.device": ("cuda",),
}


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            out.update(_flatten(item, name))
        else:
            out[name] = item
    return out


def _put(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    key = parts[-1]
    if key in target and target[key] != value:
        raise ConfigError(f"Conflicting values for reference.{dotted}")
    target[key] = value


def _path(value: Any, base: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"Expected a non-empty local path, got {value!r}")
    p = Path(value).expanduser()
    return str((p if p.is_absolute() else base / p).resolve())


def compose_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Compose actual Hydra groups/overrides, without the current Config defaults."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with initialize_config_dir(version_base=None, config_dir=str(path.parent)):
        cfg = compose(config_name=path.name, overrides=overrides or [])
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def to_train_args(config: dict[str, Any], base_dir: str | Path) -> dict[str, Any]:
    """Translate supplied values only; native ``reference`` values are an escape hatch.

    Duplicate aliases must agree. No current trainer dataclass is instantiated.
    The model config JSON is passed through unchanged (including dep_q).
    """
    base = Path(base_dir).resolve()
    native = config.get("reference", {})
    if not isinstance(native, dict):
        raise ConfigError("reference must be a mapping of native TrainArgs")
    args: dict[str, Any] = {}
    native_groups, native_scalars = _reference_schema()
    for key, value in native.items():
        if key in native_groups:
            if not isinstance(value, dict) or set(value) - native_groups[key]:
                raise ConfigError(f"Unsupported reference.{key} fields: {value!r}")
        elif key not in native_scalars:
            raise ConfigError(f"Unsupported native TrainArgs field: reference.{key}")
        _put(args, key, value.copy() if isinstance(value, dict) else value)

    flat = _flatten({k: v for k, v in config.items() if k != "reference"})
    special = {"model.root", "model.source", "model.lm_checkpoint", "model.mimi_checkpoint",
               "model.tokenizer", "model.config", "model.config_path", "data.prepared_dir",
               "lora.alpha", "train.mixed_precision", "no_eval", "user_loss"}
    for key, value in flat.items():
        if key in MAPPING:
            _put(args, MAPPING[key], value)
        elif key in INERT:
            if value not in INERT[key]:
                raise ConfigError(f"Unsupported active option {key}={value!r}")
        elif key not in special:
            raise ConfigError(f"Unsupported option {key}; use reference for native TrainArgs only")
    if "user_loss" in flat and flat["user_loss"] is not True:
        raise ConfigError("user_loss=false is unsupported: unchanged reference targets both audio streams")
    if "no_eval" in flat:
        _put(args, "do_eval", not flat["no_eval"])
    if "train.mixed_precision" in flat:
        dtype = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}.get(flat["train.mixed_precision"])
        if dtype is None:
            raise ConfigError("Unsupported train.mixed_precision")
        _put(args, "param_dtype", dtype)
    if "lora.alpha" in flat:
        rank = args.get("lora", {}).get("rank")
        if not isinstance(rank, int) or rank <= 0:
            raise ConfigError("lora.alpha requires an explicit positive lora.rank")
        _put(args, "lora.scaling", flat["lora.alpha"] / rank)
    if "data.prepared_dir" in flat:
        _put(args, "data.train_data", str(Path(_path(flat["data.prepared_dir"], base)) / "train.jsonl"))

    paths = args.setdefault("moshi_paths", {})
    if paths.get("hf_repo_id") is not None:
        raise ConfigError("reference.moshi_paths.hf_repo_id must be null; only local assets are supported")
    paths["hf_repo_id"] = None
    root = Path(_path(flat["model.root"], base)) if "model.root" in flat else None
    assets = {
        "moshi_path": ("model.lm_checkpoint", "model.safetensors"),
        "mimi_path": ("model.mimi_checkpoint", "tokenizer-e351c8d8-checkpoint125.safetensors"),
        "tokenizer_path": ("model.tokenizer", "tokenizer_spm_32k_3.model"),
        "config_path": ("model.config", "config.json"),
    }
    if "model.config_path" in flat:
        _put(paths, "config_path", _path(flat["model.config_path"], base))
    for native_key, (key, filename) in assets.items():
        if key in flat:
            _put(paths, native_key, _path(flat[key], base))
        elif native_key not in paths and root is not None:
            paths[native_key] = str(root / filename)
        if native_key not in paths:
            raise ConfigError(f"Required local asset: {key} (or model.root/reference.moshi_paths.{native_key})")
        paths[native_key] = _path(paths[native_key], base)
        if not Path(paths[native_key]).is_file():
            raise ConfigError(f"Required local asset missing: {paths[native_key]}")
    if not args.get("run_dir"):
        raise ConfigError("train.output_dir or reference.run_dir is required")
    args["run_dir"] = _path(args["run_dir"], base)
    if args.get("resume_from"):
        args["resume_from"] = _path(args["resume_from"], base)
        if not Path(args["resume_from"]).exists():
            raise ConfigError(f"Resume checkpoint missing: {args['resume_from']}")
    data = args.setdefault("data", {})
    if not data.get("train_data"):
        raise ConfigError("data.train_data/data.prepared_dir or reference.data.train_data is required")
    for name in ("train_data", "eval_data"):
        if data.get(name):
            # Native supports comma-separated weighted manifests: path[:weight].
            normalized = []
            for entry in data[name].split(","):
                path, sep, weight = entry.strip().partition(":")
                resolved = _path(path, base)
                if not Path(resolved).exists():
                    raise ConfigError(f"Manifest missing: {resolved}")
                if sep:
                    try:
                        valid = float(weight) > 0
                    except ValueError:
                        valid = False
                    if not valid:
                        raise ConfigError(f"Manifest sampling weight must be positive: {entry}")
                normalized.append(resolved + (sep + weight if sep else ""))
            data[name] = ",".join(normalized)
    if args.get("do_eval") and not data.get("eval_data"):
        raise ConfigError("do_eval requires an explicit data.eval_data manifest (no random split)")
    return args


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    return to_train_args(compose_config(path, overrides), path.parent)