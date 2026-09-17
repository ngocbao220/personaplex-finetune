#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CONFIG.yaml" >&2
  echo "Example: $0 configs/server.local.yaml" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
config=$1

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi is unavailable; this smoke test requires an NVIDIA CUDA server." >&2
  exit 1
fi
nvidia-smi
python -m pip check
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Torch cannot access CUDA")
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
python -m tools.inspect_sample --config "$config" --index 0
python -m tools.train_smoke --config "$config"
