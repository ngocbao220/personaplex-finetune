"""
python -m your_package.smoke \
    --adapter outputs/checkpoint/adapter.pt \
    --input-file ./test.mp3 \
    --voice-prompt voice.wav
    --text-prompt ""

"""

from __future__ import annotations

import argparse
from pathlib import Path

from personaplex_finetuning.config import load_config
from personaplex_finetuning.data import PreparedDataset
from personaplex_finetuning.inference import generation_from_config, smoke


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
        "--start", type=float, default=None,
        help="Start time in seconds for one exact data.window_seconds inference window.",
    )
    parser.add_argument(
        "--input-path",
        "--input-file",
        dest="input_file",
        type=Path,
        default=None,
        help="Optional WAV/MP3 input used instead of the sample conversation audio.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for generated WAV and text.")
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
    print(f"Generation settings: {generation.label()}")
    sample_index = args.index if args.index is not None else int(inf_sec.get("sample_index", 0) if hasattr(inf_sec, "get") else 0)
    start_sec = args.start if args.start is not None else (float(inf_sec.get("start_sec")) if (hasattr(inf_sec, "get") and inf_sec.get("start_sec") is not None) else None)
    output_dir = args.output_dir if args.output_dir is not None else (str(inf_sec.get("output_dir", "outputs/smoke")) if hasattr(inf_sec, "get") else "outputs/smoke")
    input_file = args.input_file
    if input_file is None and hasattr(inf_sec, "get") and inf_sec.get("input_file"):
        input_file = Path(inf_sec.get("input_file"))

    sample = PreparedDataset(config.manifest, config.window_seconds).load()[sample_index]
    sample = select_inference_window(sample, start_sec, config.window_seconds)
    print(
        f"Inference sample {sample.sample_id}: "
        f"window {sample.window_start_sec:.3f}-{sample.window_end_sec:.3f} seconds"
    )
    smoke(
        config=config,
        sample=sample,
        adapter=Path(adapter_path).resolve(),
        output_dir=Path(output_dir).resolve(),
        input_file=input_file.resolve() if input_file is not None else None,
        generation=generation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
