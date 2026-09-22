#!/usr/bin/env bash
# ==============================================================================
# PersonaPlex Live Interactive Terminal Demo Launcher
#
# Examples:
#   # Chạy với LoRA checkpoint đã train:
#   bash scripts/run_cli_demo.sh \
#     --model-root ../models \
#     --adapter ../runs/hf_overfit_10/checkpoints/checkpoint_000300 \
#     --voice-prompt ../prepared/samples/conv_0001/voice_prompt_left.wav
#
#   # Chạy với base model nguyên bản:
#   bash scripts/run_cli_demo.sh \
#     --model-root ../models \
#     --voice-prompt ../prepared/samples/conv_0001/voice_prompt_left.wav
# ==============================================================================
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}

python -m tools.interactive_cli "$@"
