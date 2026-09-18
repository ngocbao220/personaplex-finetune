#!/usr/bin/env bash
# ==============================================================================
# PersonaPlex Multi-GPU Fine-Tuning Launcher with Accelerate + DDP
#
# Usage examples:
#   bash scripts/train_gpus.sh
#   bash scripts/train_gpus.sh --num_processes 4
#   bash scripts/train_gpus.sh --device_ids 0,1,2,3
#   bash scripts/train_gpus.sh --device_ids 0,1 --config configs/train_104h.yaml train.learning_rate=1e-5
# ==============================================================================
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

NUM_PROCESSES=4
DEVICE_IDS=""
CONFIG="configs/train_104h.yaml"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device_ids|--device-ids|--gpu_ids|--gpu-ids)
      DEVICE_IDS="$2"
      shift 2
      ;;
    --num_processes|--num-processes|-n)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$DEVICE_IDS" ]]; then
  export CUDA_VISIBLE_DEVICES="$DEVICE_IDS"
  # Count comma-separated devices to automatically set NUM_PROCESSES
  IFS=',' read -ra ADDR <<< "$DEVICE_IDS"
  NUM_PROCESSES="${#ADDR[@]}"
  echo "[train_gpus.sh] Set CUDA_VISIBLE_DEVICES=$DEVICE_IDS (detected $NUM_PROCESSES GPU(s))"
fi

echo "[train_gpus.sh] Launching with Accelerate: $NUM_PROCESSES process(es), config: $CONFIG"

export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}

exec accelerate launch \
  --multi_gpu \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision bf16 \
  -m personaplex_finetuning.train \
  --config "$CONFIG" \
  "${PASSTHROUGH[@]}"
