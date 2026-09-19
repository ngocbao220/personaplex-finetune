#!/usr/bin/env bash
# ==============================================================================
# Stage 0: Mimi Neural Audio Codec Fine-Tuning Launcher
#
# Usage examples:
#   bash scripts/train_mimi.sh gpus=0 data_dir=path/to/vietnamese_wavs
#   bash scripts/train_mimi.sh gpus=0 data_dir=path/to/vietnamese_wavs learning_rate=5e-5 max_steps=5000
#   bash scripts/train_mimi.sh gpus=0 data_dir=path/to/vietnamese_wavs --smoke
# ==============================================================================
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

DEVICE_ID=""
ARGS=()

for arg in "$@"; do
  case "$arg" in
    gpus=*|gpu=*|devices=*|device_ids=*)
      DEVICE_ID="${arg#*=}"
      ;;
    data_dir=*|data=*)
      ARGS+=(--data-dir "${arg#*=}")
      ;;
    model_root=*|model=*)
      ARGS+=(--model-root "${arg#*=}")
      ;;
    output_dir=*|output=*)
      ARGS+=(--output-dir "${arg#*=}")
      ;;
    learning_rate=*|lr=*)
      ARGS+=(--learning-rate "${arg#*=}")
      ;;
    max_steps=*|steps=*)
      ARGS+=(--max-steps "${arg#*=}")
      ;;
    batch_size=*|bs=*)
      ARGS+=(--batch-size "${arg#*=}")
      ;;
    device=*)
      ARGS+=(--device "${arg#*=}")
      ;;
    chunk_seconds=*|chunk=*)
      ARGS+=(--chunk-seconds "${arg#*=}")
      ;;
    save_every_steps=*|save_every=*)
      ARGS+=(--save-every-steps "${arg#*=}")
      ;;
    --freeze-encoder|freeze_encoder=true)
      ARGS+=(--freeze-encoder)
      ;;
    --smoke)
      ARGS+=(--smoke)
      ;;
    *)
      ARGS+=("$arg")
      ;;
  esac
done

if [[ -n "$DEVICE_ID" ]]; then
  # Use first GPU if comma separated
  FIRST_GPU="${DEVICE_ID%%,*}"
  export CUDA_VISIBLE_DEVICES="$FIRST_GPU"
  echo "[train_mimi.sh] Set CUDA_VISIBLE_DEVICES=$FIRST_GPU"
fi

export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
exec python -m tools.train_mimi "${ARGS[@]}"
