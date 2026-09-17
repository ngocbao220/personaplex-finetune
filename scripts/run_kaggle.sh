#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG.yaml [--fsdp 0,1]" >&2
  exit 2
fi

CONFIG=$1
shift || true
EXTRA_ARGS=("$@")

export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}

python -m tools.validate_dataset --config "$CONFIG"
python -m tools.train_smoke --config "$CONFIG" "${EXTRA_ARGS[@]}"
python -m personaplex_finetuning.train --config "$CONFIG" "${EXTRA_ARGS[@]}"
tensorboard --logdir runs/hf_overfit_10 --bind_all
