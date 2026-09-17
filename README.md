# PersonaPlex OtoSpeech overfit-10

This is the smallest offline-first training path: prepared stereo conversations →
PersonaPlex hybrid prompt sequence → LoRA → adapter reload → native conditioning
smoke. It deliberately stops after the fixed 10-sample overfit milestone.

## Data contract

The default config reads `../prepared/train.jsonl`. Every sample must provide
`conversation.wav` (stereo, LEFT agent and RIGHT user), `words.json`,
`voice_prompt.wav`, and `metadata.json.text_prompt`. Preparation, ASR and prompt
generation are out of scope; no input audio is modified.

Validate before a model run:

```bash
PYTHONPATH=src python -m tools.validate_dataset --manifest ../prepared/train.jsonl
```

## Offline CUDA server setup

Clone this code-only repository on the Linux CUDA server. Prepared data, the
PersonaPlex checkpoint, and the matching local PersonaPlex/Moshi source stay
outside Git and are referenced by absolute paths. No data/model archive or
upload workflow is used.

```bash
git clone https://github.com/ngocbao220/personaplex-finetune.git
cd personaplex-finetune
nvidia-smi
cp configs/server.example.yaml configs/server.local.yaml
```

Edit `configs/server.local.yaml` to point at the existing server assets:

```yaml
model:
  root: /mnt/models/personaplex-7b-v1
  # Directory that contains the moshi/ Python package.
  source: /opt/personaplex-source
data:
  manifest: /mnt/processed/train.jsonl
```

`server.local.yaml` is ignored. The checkpoint directory must contain exactly
these required runtime files:

```text
model.safetensors
tokenizer-e351c8d8-checkpoint125.safetensors
tokenizer_spm_32k_3.model
```

Then select the CUDA metapackage only after inspecting the server driver:

```bash
bash scripts/setup_server_env.sh 12.1
```

The setup script creates the Conda environment from `environment.yml`, installs
PyTorch 2.4 with the CUDA metapackage you selected, then checks dependencies and
CUDA visibility. Missing local assets, Python dependencies, or CUDA fail fast;
the runtime never downloads from Hugging Face. For an offline server,
pre-provision a Linux Conda package cache/wheelhouse and use the equivalent
`conda ... --offline` commands—do not transfer this macOS environment.

`requirements.txt` mirrors the direct Python package constraints for a pip-based
setup. It deliberately does not select a CUDA wheel; use a matching CUDA PyTorch
installation first, or use the server setup scripts.

## Online Hugging Face CUDA server

This repository includes the required `moshi` runtime package under `src/moshi`;
no second source checkout is needed. The online recipe downloads only prepared
data and the PersonaPlex checkpoint, then returns to local-only runtime paths.

### Publish prepared data once

On the machine containing `../prepared/`, authenticate interactively and first
inspect the upload without changing Hugging Face:

```bash
python -m pip install -e '.[hub]'
hf auth login
PYTHONPATH=src python -m tools.publish_prepared --dry-run
PYTHONPATH=src python -m tools.publish_prepared
```

The destination is the private dataset
`ngocbao220/personaplex-otospeech-prepared`. Upload validates `train.jsonl` and
excludes `.DS_Store`; it never stores an HF token in this repository.

### Clone, download, and overfit on the GPU server

The Hugging Face account used on the server must have accepted the
`nvidia/personaplex-7b-v1` license.

```bash
git clone https://github.com/ngocbao220/personaplex-finetune.git
cd personaplex-finetune
nvidia-smi
bash scripts/setup_hf_server_env.sh 12.1
hf auth login
bash scripts/download_hf_assets.sh
PYTHONPATH=src python -m tools.validate_dataset --manifest assets/prepared/train.jsonl
bash scripts/run_server_smoke.sh configs/hf_overfit_10.yaml
python -m personaplex_finetuning.train --config configs/hf_overfit_10.yaml
python -m tools.inference_smoke \
  --config configs/hf_overfit_10.yaml \
  --adapter runs/hf_overfit_10/checkpoints/checkpoint_000300/lora.safetensors
```

`download_hf_assets.sh` saves the data under `assets/prepared/`, the three
checkpoint files under `assets/personaplex-7b-v1/`, and records the resolved
dataset/model revisions in `assets/hf-assets.json`. These downloaded assets and
all run outputs are ignored by Git. The train command refuses to overwrite an
existing run directory.

## Proof order

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m tools.validate_dataset --manifest /mnt/processed/train.jsonl
python -m tools.inspect_sample --config configs/server.local.yaml --index 0
bash scripts/run_server_smoke.sh configs/server.local.yaml
python -m personaplex_finetuning.train --config configs/server.local.yaml
python -m tools.inference_smoke \
  --config configs/server.local.yaml \
  --adapter /mnt/runs/personaplex-overfit-10/checkpoints/checkpoint_000300/lora.safetensors
```

The trainer has no shuffle, augmentation, distributed mode, FSDP or full-model
fine-tuning. It logs component losses to `metrics.jsonl`, saves adapter-only
safetensors, reloads the adapter into a fresh base model, and writes
`reports/overfit_10.md`. Set `train.output_dir` to a new directory for every
attempt; runs never overwrite an existing output. Do not scale data until this
report and the inference artifacts have been reviewed.
