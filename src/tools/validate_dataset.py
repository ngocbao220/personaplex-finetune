from __future__ import annotations

import argparse
import sys

from personaplex_finetuning.data import PreparedDataset, ValidationError
from personaplex_finetuning.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate prepared PersonaPlex OtoSpeech samples.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to configuration file")
    args, unknown = parser.parse_known_args()
    overrides = [arg for arg in unknown if "=" in arg]
    try:
        config = load_config(args.config, overrides=overrides)
        samples = PreparedDataset(config.manifest, config.window_seconds).load()
    except (OSError, ValidationError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    total_seconds = sum(sample.audio.duration_sec for sample in samples)
    print(f"Dataset: {config.manifest}\nSamples: {len(samples)}\nValid: {len(samples)}\nInvalid: 0")
    print(f"Audio\n  stereo: {len(samples)}/{len(samples)}\n  total duration: {total_seconds:.1f} sec")
    print(f"Windows\n  deterministic seconds: {config.window_seconds:g}\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
