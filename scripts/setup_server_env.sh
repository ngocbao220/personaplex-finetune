#!/usr/bin/env bash
# ==============================================================================
# PersonaPlex Server Environment Setup Script (Ubuntu / Linux GPU Instances)
#
# Sets up CUDA PyTorch 2.4.1, dependencies, Hugging Face CLI + hf_transfer,
# and verifies GPU availability for PersonaPlex fine-tuning & benchmarking.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== [1/5] Checking system dependencies (ffmpeg, libsox) ==="
if command -v apt-get >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update && apt-get install -y ffmpeg libsox-fmt-all git git-lfs curl tmux
    elif sudo -n true 2>/dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg libsox-fmt-all git git-lfs curl tmux
    else
        echo "Notice: Non-root user without passwordless sudo. Ensure ffmpeg is installed."
    fi
fi

echo "=== [2/5] Installing PyTorch with CUDA 12.4 ==="
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

echo "=== [3/5] Installing project requirements and hf_transfer ==="
cd "${ROOT_DIR}"
python -m pip install -r requirements.txt
python -m pip install -e ".[hub]"

echo "=== [4/5] Configuring fast Hugging Face transfer ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "HF_HUB_ENABLE_HF_TRANSFER=1 enabled in current environment."

echo "=== [5/5] Running CUDA & Environment Sanity Check ==="
python -c "
import torch
print('PyTorch Version   :', torch.__version__)
print('CUDA Available    :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Count      :', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  - GPU {i}        : {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB)')
    print('BF16 Supported    :', torch.cuda.is_bf16_supported())
    print('Flash SDP Enabled :', hasattr(torch.backends.cuda, 'enable_flash_sdp'))
else:
    print('Warning: CUDA is not available. Check your NVIDIA drivers.')
"

echo "=== Setup complete! ==="
echo "To log in to Hugging Face, run: hf auth login"
