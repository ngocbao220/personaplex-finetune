#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

command -v hf >/dev/null || { echo "Hugging Face CLI is unavailable; run setup_server_env.sh first." >&2; exit 1; }
hf auth whoami >/dev/null || { echo "Run 'hf auth login' with an account accepted for nvidia/personaplex-7b-v1." >&2; exit 1; }
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
python -m tools.download_hf_assets "$@"
