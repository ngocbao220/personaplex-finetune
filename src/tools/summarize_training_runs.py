"""Summarize PersonaPlex fine-tuning trials for hardware and hyperparameter selection."""

from __future__ import annotations

import argparse
import csv
import math
import json
import statistics
import sys
from pathlib import Path


FIELDS = (
    "run",
    "world_size",
    "batch_per_device",
    "gradient_accumulation_steps",
    "global_batch_size",
    "lora_rank",
    "learning_rate",
    "max_steps",
    "completed_steps",
    "starting_generation_cer",
    "best_generation_cer",
    "best_generation_wer",
    "best_generation_step",
    "median_samples_per_second_last_20",
    "peak_gpu_gib",
    "best_inference_checkpoint",
    "inference_checkpoint_status",
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def summarize_training_run(run_dir: Path) -> dict[str, object]:
    config = _read_json(run_dir / "config.json")
    run = _read_json(run_dir / "run.json") if (run_dir / "run.json").is_file() else {}
    train_rows = _read_jsonl(run_dir / "metrics.jsonl")
    generation_rows = _read_jsonl(run_dir / "free_running_metrics.jsonl")
    baseline_rows = [
        row for row in generation_rows
        if row.get("val/generation_baseline") is True
        and isinstance(row.get("val/generation_cer"), (int, float))
    ]
    baseline_cer = baseline_rows[-1].get("val/generation_cer") if baseline_rows else run.get("starting_generation_cer")
    has_validated_baseline = (
        run.get("inference_checkpoint_status") == "validated_generation_improves_base"
        and isinstance(baseline_cer, (int, float))
        and math.isfinite(float(baseline_cer))
    )
    generation_rows = [
        row for row in generation_rows
        if has_validated_baseline
        if row.get("val/generation_baseline") is not True
        and isinstance(row.get("val/generation_cer"), (int, float))
        and row.get("val/generation_empty_samples", 0) == 0
        and row["val/generation_cer"] < baseline_cer
    ]
    best_generation = min(generation_rows, key=lambda row: row["val/generation_cer"], default={})

    rank_peaks = []
    rank_dir = run_dir / "ranks"
    if rank_dir.is_dir():
        for path in rank_dir.glob("rank_*.json"):
            info = _read_json(path)
            if isinstance(info.get("peak_gpu_bytes"), int):
                rank_peaks.append(info["peak_gpu_bytes"])
    peak_bytes = max(rank_peaks, default=run.get("peak_gpu_bytes", 0))
    throughput = [
        float(row["samples_per_second"])
        for row in train_rows[-20:]
        if isinstance(row.get("samples_per_second"), (int, float))
    ]

    return {
        "run": run_dir.name,
        "world_size": config.get("num_processes"),
        "batch_per_device": config.get("per_device_batch_size"),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
        "global_batch_size": config.get("global_batch_size"),
        "lora_rank": config.get("lora_rank"),
        "learning_rate": config.get("learning_rate"),
        "max_steps": config.get("max_steps"),
        "completed_steps": train_rows[-1].get("step") if train_rows else run.get("optimizer_step"),
        "starting_generation_cer": baseline_cer,
        "best_generation_cer": best_generation.get("val/generation_cer"),
        "best_generation_wer": best_generation.get("val/generation_wer"),
        "best_generation_step": best_generation.get("step"),
        "median_samples_per_second_last_20": statistics.median(throughput) if throughput else None,
        "peak_gpu_gib": peak_bytes / (1024**3) if peak_bytes else None,
        "best_inference_checkpoint": run.get("best_inference_checkpoint"),
        "inference_checkpoint_status": run.get("inference_checkpoint_status"),
    }


def summarize_runs(root: Path) -> list[dict[str, object]]:
    if not root.is_dir():
        raise FileNotFoundError(f"training run root does not exist: {root}")
    summaries = [
        summarize_training_run(config_path.parent)
        for config_path in root.glob("*/config.json")
    ]
    return sorted(
        summaries,
        key=lambda row: (
            row["best_generation_cer"] is None,
            row["best_generation_cer"] if row["best_generation_cer"] is not None else float("inf"),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", nargs="?", type=Path, default=Path("runs/moshi-code-style"))
    parser.add_argument("--output", type=Path, help="CSV output path; omit to print CSV to stdout")
    args = parser.parse_args()
    rows = summarize_runs(args.run_root)
    output = args.output.open("w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
