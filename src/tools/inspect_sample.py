from __future__ import annotations

import argparse

from personaplex_finetuning.config import load_config
from personaplex_finetuning.data import PreparedDataset
from personaplex_finetuning.runtime import RuntimePaths, load_runtime
from personaplex_finetuning.train import build_example


def main() -> int:
    parser = argparse.ArgumentParser(description="Print one human-readable PersonaPlex training example.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args, unknown = parser.parse_known_args()
    overrides = [arg for arg in unknown if "=" in arg]
    config = load_config(args.config, overrides=overrides)
    import torch
    device = args.device or config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    samples = PreparedDataset(config.manifest, config.window_seconds).load()
    sample = samples[args.index]
    runtime = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), device)
    example = build_example(config, sample, runtime)
    print(f"sample_id: {sample.sample_id}\nwindow: {sample.window_start_sec:.3f}-{sample.window_end_sec:.3f}\naudio sample rate: {sample.audio.sample_rate}\nMimi frame rate: {runtime.codec.frame_rate}\nhybrid prompt frames: {example.prompt_frames}\ndialogue frames: {example.dialogue_frames}\ntotal frames: {example.total_frames}")
    print(f"stream shapes: 17 x {example.total_frames}\nsupervised text positions: {sum(example.loss_mask[0])}\nsupervised audio positions: {sum(sum(stream) for stream in example.loss_mask[1:9])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
