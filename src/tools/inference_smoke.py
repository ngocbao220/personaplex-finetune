"""
python -m your_package.smoke \
    --adapter outputs/checkpoint/adapter.pt \
    --input-file ./test.mp3 \
    --voice-prompt voice.wav
    --text-prompt ""

"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time

from personaplex_finetuning.config import load_config
from personaplex_finetuning.data import PreparedDataset
from personaplex_finetuning.inference import generation_from_config, smoke

import torch
torch.backends.cudnn.enabled = False


def create_inference_run_dir(output_root: Path) -> Path:
    """Create a unique inference run directory, matching the training layout."""
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"infer_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}"
    run_dir.mkdir(exist_ok=False)
    return run_dir


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_inference_window(sample, start: float | None, window_seconds: float):
    """Optionally replace the dataset default with an exact inference window."""
    if start is None:
        return sample
    if start < 0:
        raise ValueError(f"--start must be non-negative, got {start}")
    end = start + window_seconds
    if end > sample.audio.duration_sec:
        raise ValueError(
            f"--start {start:g} requires {window_seconds:g} seconds, but the conversation is only "
            f"{sample.audio.duration_sec:g} seconds long"
        )
    return sample.with_window(start, end)


def main() -> int:
    default_config = "configs/infer.yaml" if Path("configs/infer.yaml").is_file() else None
    parser = argparse.ArgumentParser(description="PersonaPlex inference smoke test.")
    parser.add_argument(
        "--config",
        default=default_config,
        required=default_config is None,
        help="Path to YAML/JSON configuration file (default: configs/infer.yaml).",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to fine-tuned LoRA adapter (if omitted, read from config).",
    )
    parser.add_argument("--index", type=int, default=None, help="Sample index in dataset to evaluate.")
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train",
        help="Conversation split to sample from; validation reproduces the seeded training split when no val.jsonl exists.",
    )
    parser.add_argument("--sample-id", default=None, help="Select a sample by its manifest sample_id.")
    parser.add_argument(
        "--start", type=float, default=None,
        help="Start time in seconds for one exact data.window_seconds inference window.",
    )
    parser.add_argument(
        "--window-seconds", type=float, default=None,
        help="Override data.window_seconds; defaults to the configured free-running eval window when unset.",
    )
    parser.add_argument(
        "--input-path",
        "--input-file",
        dest="input_file",
        type=Path,
        default=None,
        help="Optional WAV/MP3 input used instead of the sample conversation audio.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output root; each run is saved in a new infer_<date> subdirectory.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Read optional overrides from raw config if present
    raw_conf = {}
    try:
        from omegaconf import OmegaConf
        raw_conf = OmegaConf.load(args.config)
    except Exception:
        pass

    adapter_path = args.adapter
    if adapter_path is None and hasattr(raw_conf, "get"):
        adapter_sec = raw_conf.get("adapter", {})
        if hasattr(adapter_sec, "get"):
            adapter_path = adapter_sec.get("path")

    if not adapter_path:
        raise ValueError("LoRA adapter path must be provided via --adapter or specified in config under 'adapter.path'")

    inf_sec = raw_conf.get("inference", {}) if hasattr(raw_conf, "get") else {}
    # Sampling parameters for LMGen; unknown keys or bad values fail here, before any
    # model is loaded, so a typo cannot silently produce a different generation.
    generation = generation_from_config(raw_conf)
    sample_index = args.index if args.index is not None else int(inf_sec.get("sample_index", 0) if hasattr(inf_sec, "get") else 0)
    start_sec = args.start if args.start is not None else (float(inf_sec.get("start_sec")) if (hasattr(inf_sec, "get") and inf_sec.get("start_sec") is not None) else None)
    output_root = Path(
        args.output_dir
        if args.output_dir is not None
        else (str(inf_sec.get("output_dir", "outputs/inference")) if hasattr(inf_sec, "get") else "outputs/inference")
    ).expanduser().resolve()
    input_file = args.input_file
    if input_file is None and hasattr(inf_sec, "get") and inf_sec.get("input_file"):
        input_file = Path(inf_sec.get("input_file"))

    window_seconds = (
        args.window_seconds
        if args.window_seconds is not None
        else config.window_seconds
        if config.window_seconds is not None
        else config.free_running_eval_window_seconds
    )
    if window_seconds <= 0:
        parser.error("--window-seconds must be positive")

    output_dir = create_inference_run_dir(output_root)
    log_path = output_dir / "inference.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    logger = logging.getLogger("personaplex.inference")
    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    adapter_path = Path(adapter_path).expanduser().resolve()
    input_file = input_file.expanduser().resolve() if input_file is not None else None
    run_record = {
        "status": "running",
        "started_at": started_at,
        "config": str(Path(args.config).expanduser().resolve()),
        "adapter": str(adapter_path),
        "split": args.split,
        "input_file": str(input_file) if input_file is not None else None,
        "generation": generation.as_dict(),
        "output_dir": str(output_dir),
        "log_file": str(log_path),
    }
    write_json(output_dir / "config.json", run_record)
    logger.info("inference run started; output_dir=%s", output_dir)
    logger.info("adapter=%s", adapter_path)
    logger.info("generation=%s", generation.label())

    try:
        logger.info("loading split=%s window_seconds=%s", args.split, window_seconds)
        if args.split == "train":
            samples = PreparedDataset(config.manifest, window_seconds).load()
        elif args.split == "validation":
            if config.val_manifest is not None:
                samples = PreparedDataset(config.val_manifest, window_seconds).load()
            else:
                _, samples = PreparedDataset(config.manifest, window_seconds).split(
                    val_ratio=config.val_ratio, seed=config.seed,
                )
        elif config.test_manifest is not None:
            samples = PreparedDataset(config.test_manifest, window_seconds).load()
        else:
            raise ValueError("test split requested, but data.test_manifest is not configured")

        if args.sample_id is not None:
            matching = [sample for sample in samples if sample.sample_id == args.sample_id]
            if not matching:
                raise ValueError(f"sample_id {args.sample_id!r} is not present in the {args.split} split")
            sample = matching[0]
        else:
            if not 0 <= sample_index < len(samples):
                raise IndexError(f"sample index {sample_index} is outside the {args.split} split (size={len(samples)})")
            sample = samples[sample_index]
        sample = select_inference_window(sample, start_sec, window_seconds)
    except Exception as exc:
        run_record.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": time.perf_counter() - started,
        })
        write_json(output_dir / "run.json", run_record)
        logger.exception("inference input preparation failed")
        raise

    run_record.update({
        "sample_id": sample.sample_id,
        "window_start_sec": sample.window_start_sec,
        "window_end_sec": sample.window_end_sec,
    })
    write_json(output_dir / "config.json", run_record)
    logger.info(
        "sample=%s split=%s window=%.3f-%.3fs",
        sample.sample_id, args.split, sample.window_start_sec, sample.window_end_sec,
    )
    logger.info("voice_prompt=%s", getattr(sample, "voice_prompt_wav", "(unknown)"))
    logger.info("text_prompt=%s", getattr(sample, "text_prompt", "(unknown)"))

    try:
        quality_status = smoke(
            config=config,
            sample=sample,
            adapter=adapter_path,
            output_dir=output_dir,
            input_file=input_file,
            generation=generation,
        )
    except Exception as exc:
        run_record.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": time.perf_counter() - started,
        })
        write_json(output_dir / "run.json", run_record)
        logger.exception("inference failed after %.2fs", run_record["elapsed_sec"])
        raise

    elapsed = time.perf_counter() - started
    run_record.update({
        "status": "quality_gate_failed" if quality_status in {"empty_transcript", "does_not_improve_over_base"} else "completed",
        "quality_status": quality_status,
        "elapsed_sec": elapsed,
        "artifacts": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    })
    # smoke() writes the detailed metrics; preserve them and add run-level timing/status.
    smoke_run_path = output_dir / "run.json"
    if smoke_run_path.is_file():
        try:
            run_record.update(json.loads(smoke_run_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            logger.exception("could not read detailed smoke run report")
    run_record.update({
        "status": "quality_gate_failed" if quality_status in {"empty_transcript", "does_not_improve_over_base"} else "completed",
        "quality_status": quality_status,
        "started_at": started_at,
        "elapsed_sec": elapsed,
        "output_dir": str(output_dir),
        "log_file": str(log_path),
        "artifacts": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    })
    write_json(smoke_run_path, run_record)
    logger.info("inference finished status=%s elapsed=%.2fs", quality_status, elapsed)
    logger.info("saved outputs: %s", output_dir)
    if quality_status in {"empty_transcript", "does_not_improve_over_base"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
