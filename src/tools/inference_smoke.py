from __future__ import annotations

import argparse
from pathlib import Path

from personaplex_finetuning.config import load_config
from personaplex_finetuning.data import PreparedDataset
from personaplex_finetuning.inference import smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/smoke")
    args = parser.parse_args()
    config = load_config(args.config)
    sample = PreparedDataset(config.manifest, config.window_seconds).load()[args.index]
    smoke(config, sample, Path(args.adapter).resolve(), Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
