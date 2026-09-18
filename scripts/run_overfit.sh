#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CONFIG.yaml" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

CONFIG=$1
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}

python -m tools.validate_dataset --config "$CONFIG"
python -m tools.train_smoke --config "$CONFIG"
python -m personaplex_finetuning.train --config "$CONFIG"
tensorboard --logdir runs/hf_overfit_10 --bind_all
