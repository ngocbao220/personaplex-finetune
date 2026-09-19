#!/usr/bin/env bash
# ==============================================================================
# PersonaPlex Multi-GPU Fine-Tuning Launcher with Accelerate + DDP
#
# Usage examples:
#   bash scripts/train_gpus.sh gpus=0,1
#   bash scripts/train_gpus.sh gpus=0,1 data=otospeech model=server train=104h
#   bash scripts/train_gpus.sh gpus=0,1,2,3 train.learning_rate=1e-5
# ==============================================================================
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

NUM_PROCESSES=2
DEVICE_IDS=""
CONFIG="configs/config.yaml"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    gpus=*|gpu=*|devices=*|device_ids=*)
      DEVICE_IDS="${1#*=}"
      shift
      ;;
    --device_ids|--device-ids|--gpu_ids|--gpu-ids)
      DEVICE_IDS="$2"
      shift 2
      ;;
    n=*|processes=*|num_processes=*)
      NUM_PROCESSES="${1#*=}"
      shift
      ;;
    --num_processes|--num-processes|-n)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    config=*)
      CONFIG="${1#*=}"
      shift
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
