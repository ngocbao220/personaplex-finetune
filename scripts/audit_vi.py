#!/usr/bin/env python
"""Run the native PersonaPlex audit (BATCH + NATIVE in_memory/forced_agent_audio/
forced_text/full_gt) on a Vietnamese (train_vi) adapter.

The English audit was run with hand-written overrides. For Vietnamese, the probe MUST
reuse the exact training contract of the run (text mode, duration, sample_number,
manifest, swap roles), otherwise tokenization/crop differ and every metric is invalid.
This wrapper derives those overrides from the run's recorded config.json, then calls
`audit_personaplex.py data` (no 7B allocation) and `audit_personaplex.py probe`.

Example (server):
  NO_CUDA_GRAPH=1 python scripts/audit_vi.py \
      --run-dir /home/voice/code/VDT_02/baottn/personaplex-finetune-v10/runs/train-vi/train_XXXX \
      --config configs/overfit.yaml --output-dir audit-artifacts/vi-probe-v1
Add --dry-run to only print commands.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "audit_personaplex.py"
TEXT_MODES = {"diacritics", "no_diacritics", "telex"}


def load_contract(run_dir: Path) -> dict:
    """Return the training contract recorded by the trainer (config.json or metrics.jsonl)."""
    candidates = [run_dir / "config.json"]
    for path in candidates:
        if path.is_file():
            raw = json.loads(path.read_text())
            return {**raw, **raw.get("training_contract", {})}
    metrics = run_dir / "metrics.jsonl"
    if metrics.is_file():
        for line in metrics.read_text().splitlines():
            row = json.loads(line)
            if row.get("event") == "configuration":
                return {**row, **row.get("training_contract", {})}
    raise SystemExit(f"No config.json / configuration event in {run_dir}; pass overrides manually")


def find_adapter(run_dir: Path, step: int | None) -> Path:
    found = []
    for p in run_dir.rglob("lora.safetensors"):
        m = re.search(r"checkpoint[-_](\d+)", str(p))
        found.append((int(m.group(1)) if m else -1, p))
    if not found:
        raise SystemExit(f"No lora.safetensors under {run_dir}")
    if step is not None:
        found = [f for f in found if f[0] == step]
        if not found:
            raise SystemExit(f"No checkpoint-{step} in {run_dir}")
    return max(found)[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="train_vi run directory")
    ap.add_argument("--config", type=Path, required=True, help="same base YAML used for training")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, help="default: latest checkpoint-*/lora.safetensors")
    ap.add_argument("--step", type=int, help="pick checkpoint-<step>")
    ap.add_argument("--sample-id", help="default: first retained training chunk after all filters")
    ap.add_argument("--start-sec", type=float, help="default: retained chunk start (explicit sample ID defaults to 0)")
    ap.add_argument("--end-sec", type=float, help="default: selected chunk end, bounded by actual audio duration")
    ap.add_argument("--text-mode", choices=sorted(TEXT_MODES), help="override recorded text mode (not recommended)")
    ap.add_argument("--role", choices=["left-agent", "right-agent"], default="left-agent")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--parity-frames", type=int, default=128)
    ap.add_argument("--match-user-conditioning", action="store_true")
    ap.add_argument("--override", action="append", default=[], help="extra overrides, appended last")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    c = load_contract(args.run_dir)
    text_mode = args.text_mode or c.get("vietnamese_text_mode")
    if text_mode not in TEXT_MODES:
        raise SystemExit(f"Recorded vietnamese_text_mode={text_mode!r} invalid; pass --text-mode")
    if args.role == "right-agent" and not c.get("swap_roles_after_pass"):
        raise SystemExit("right-agent audit only valid when the run trained swapped roles")
    manifest = Path(c["manifest"])
    duration = float(c.get("duration_sec", 100.0))

    overrides = [
        f"data.prepared_dir={manifest.parent}",
        f"data.vietnamese_text_mode={text_mode}",
        f"duration_sec={duration}",
    ]
    for key in ("sample_number", "sample_index", "text_padding_weight", "first_codebook_weight_multiplier", "user_loss"):
        if key in c:
            val = c[key]
            overrides.append(f"{key}={'null' if val is None else str(val).lower() if isinstance(val, bool) else val}")
    for key, config_key in (("window_seconds", "data.window_seconds"),
                            ("swap_roles_after_pass", "data.swap_roles_after_pass"),
                            ("eval_on_train_samples", "train.eval_on_train_samples")):
        if key in c:
            val = c[key]
            overrides.append(f"{config_key}={'null' if val is None else str(val).lower() if isinstance(val, bool) else val}")
    overrides += args.override

    adapter = args.adapter or find_adapter(args.run_dir, args.step)
    sample_id = args.sample_id

    common = [sys.executable, str(SCRIPT)]
    shared = ["--config", str(args.config)]
    # Missing contract means the historical trainer limited conversations first.
    selection_contract = c.get("sample_number_contract", "conversations-v1")
    shared += ["--sample-number-contract", selection_contract]
    for o in overrides:
        shared += ["--override", o]
    if sample_id is not None:
        shared += ["--sample-id", sample_id]
    if args.start_sec is not None:
        shared += ["--start-sec", str(args.start_sec)]
    if args.end_sec is not None:
        shared += ["--end-sec", str(args.end_sec)]
    shared += ["--role", args.role, "--device", args.device]

    cmds = []
    if not args.skip_data:
        cmds.append(common + ["data"] + shared + ["--output-dir", str(args.output_dir / "data")])
    probe = common + ["probe"] + shared + [
        "--adapter", str(adapter), "--parity-frames", str(args.parity_frames),
        "--forced-text", "--forced-agent-audio", "--full-gt",
        "--output-dir", str(args.output_dir / "probe")]
    if args.match_user_conditioning:
        probe.append("--match-user-conditioning")
    cmds.append(probe)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "vi_invocation.json").write_text(json.dumps(
        {"run_dir": str(args.run_dir), "adapter": str(adapter), "sample_id": sample_id,
         "sample_selection": "explicit_sample_id" if sample_id is not None else "first_retained_training_chunk",
         "sample_number_contract": selection_contract,
         "text_mode": text_mode, "overrides": overrides, "commands": cmds}, indent=2, ensure_ascii=False))
    env = {**os.environ, "NO_CUDA_GRAPH": os.environ.get("NO_CUDA_GRAPH", "1")}
    for cmd in cmds:
        print("+", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
