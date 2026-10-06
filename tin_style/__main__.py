"""Launch the unchanged, isolated reference trainer with resolved native YAML."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from omegaconf import OmegaConf
from .config import ConfigError, load_config


def run_reference(config_path: str) -> None:
    root = Path(__file__).resolve().parent / "reference"
    trainer = root / "moshi-finetune"
    runtime = root / "personaplex" / "moshi"
    if not (trainer / "train.py").is_file() or not (runtime / "moshi").is_dir():
        raise ConfigError(f"Vendored reference trainer/runtime missing under {root}")
    # Avoid resolving the existing project's train.py, finetune or Moshi runtime.
    for name in ("moshi", "finetune"):
        if name in sys.modules:
            raise ConfigError(f"{name} already imported; launch python -m tin_style in a fresh process")
    # The reference training environment uses upstream Moshi (CheckpointInfo
    # and LoRA). Its vendored PersonaPlex runtime is inference-only.
    project = Path(__file__).resolve().parent.parent
    sys.path = [entry for entry in sys.path
                if Path(entry or ".").resolve() != project / "src"]
    sys.path.insert(0, str(trainer))
    from moshi.models import loaders
    if not hasattr(loaders, "CheckpointInfo"):
        raise ConfigError("Training requires upstream Moshi with CheckpointInfo/LoRA, not the legacy PersonaPlex runtime")
    # Reference evaluation may spawn a Python process; give it the same runtime
    # precedence rather than the current project's installed Moshi package.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(trainer), str(project)]
        + [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
           if entry and Path(entry).resolve() != project / "src"]
    )
    spec = importlib.util.spec_from_file_location("_tin_style_reference_train", trainer / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.train(config_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Hydra YAML config")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model=server train.max_steps=10")
    parser.add_argument("--check-config", action="store_true", help="Validate local assets and print native args; never import GPU/trainer code")
    parser.add_argument("--resume-from", help="Local run/checkpoint directory passed to native reference resume_from")
    cli, extra = parser.parse_known_args(argv)
    if any(item.startswith("--") for item in extra):
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    try:
        overrides = cli.overrides + extra
        if cli.resume_from:
            # Hydra ++ allows the CLI flag whether or not native field exists.
            overrides.append("++reference.resume_from=" + str(Path(cli.resume_from).expanduser().resolve()))
        args = load_config(cli.config, overrides)
        resolved = OmegaConf.to_yaml(OmegaConf.create(args), resolve=True)
        if cli.check_config:
            print(resolved, end="")
            return 0
        # Keep the file alive for the complete reference call, then remove it.
        with tempfile.TemporaryDirectory(prefix="tin-style-config-") as directory:
            path = Path(directory) / "resolved.yaml"
            path.write_text(resolved, encoding="utf-8")
            run_reference(str(path))
        return 0
    except (ConfigError, ValueError) as exc:
        parser.exit(2, f"tin_style: {exc}\n")


if __name__ == "__main__":
    main()