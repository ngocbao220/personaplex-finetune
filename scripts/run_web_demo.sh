#!/usr/bin/env bash
# ==============================================================================
# PersonaPlex Gradio Web Demo Launcher (với Temporary Public Share URL)
#
# Examples:
#   # Chạy với file config:
#   bash scripts/run_web_demo.sh --config configs/demo.yaml
#
#   # Chạy với LoRA checkpoint & tạo temporary public URL:
#   bash scripts/run_web_demo.sh \
#     --model-root ../models \
#     --adapter ../runs/hf_overfit_10/checkpoints/checkpoint_000300 \
#     --voice-prompt ../prepared/samples/conv_0001/voice_prompt_left.wav \
#     --share
#
#   # Chạy cục bộ không mở share public:
#   bash scripts/run_web_demo.sh \
#     --model-root ../models \
#     --no-share
# ==============================================================================
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}

python -m tools.interactive_web "$@"
