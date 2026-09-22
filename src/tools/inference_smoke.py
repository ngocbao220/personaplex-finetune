from __future__ import annotations

import argparse
from pathlib import Path

from personaplex_finetuning.config import load_config
from personaplex_finetuning.data import PreparedDataset
from personaplex_finetuning.inference import smoke


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--start", type=float, default=None,
        help="Start time in seconds for one exact data.window_seconds inference window.",
    )
    parser.add_argument("--output-dir", default="outputs/smoke")
    args = parser.parse_args()
    config = load_config(args.config)
    sample = PreparedDataset(config.manifest, config.window_seconds).load()[args.index]
    sample = select_inference_window(sample, args.start, config.window_seconds)
    print(
        f"Inference sample {sample.sample_id}: "
        f"window {sample.window_start_sec:.3f}-{sample.window_end_sec:.3f} seconds"
    )
    smoke(config, sample, Path(args.adapter).resolve(), Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
