#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CUDA_METAPACKAGE" >&2
  echo "Example after inspecting nvidia-smi: $0 12.1" >&2
  exit 2
fi

cuda_version=$1
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

command -v conda >/dev/null || { echo "Conda is required." >&2; exit 1; }

conda env create -f environment.hf.yml
eval "$(conda shell.bash hook)"
conda activate personaplex-overfit-hf
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
conda install -y -c pytorch -c nvidia "pytorch=2.4.*" "pytorch-cuda=$cuda_version"
python -m pip install --no-deps -e .
python -m pip check
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Torch cannot access CUDA; choose a compatible CUDA metapackage.")
PY
