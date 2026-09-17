from __future__ import annotations

import argparse
import sys

from personaplex_finetuning.data import PreparedDataset, ValidationError


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate prepared PersonaPlex OtoSpeech samples.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    args = parser.parse_args()
    try:
        samples = PreparedDataset(args.manifest, args.window_seconds).load()
    except (OSError, ValidationError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    total_seconds = sum(sample.audio.duration_sec for sample in samples)
    print(f"Dataset: {args.manifest}\nSamples: {len(samples)}\nValid: {len(samples)}\nInvalid: 0")
    print(f"Audio\n  stereo: {len(samples)}/{len(samples)}\n  total duration: {total_seconds:.1f} sec")
    print(f"Windows\n  deterministic seconds: {args.window_seconds:g}\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
