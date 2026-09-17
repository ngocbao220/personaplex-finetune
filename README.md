# PersonaPlex OtoSpeech LoRA fine-tuning

LoRA overfit workflow for 10 prepared stereo OtoSpeech conversations. The
repository includes the PersonaPlex/Moshi runtime; training uses a local model
directory after download and saves an adapter-only checkpoint.

## GPU server

The Hugging Face account must have accepted the `nvidia/personaplex-7b-v1`
license. The setup uses Torch 2.4.x with CUDA 12.4; do not use Torch 2.8.

```bash
git clone https://github.com/ngocbao220/personaplex-finetune.git
cd personaplex-finetune

bash scripts/setup_hf_server_env.sh 12.4
eval "$(conda shell.bash hook)"
conda activate personaplex-overfit-hf
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

hf auth login
bash scripts/download_hf_assets.sh \
  --dataset-repo ngocbao05/personaplex-otospeech-prepared

CONFIG=configs/hf_overfit_10.yaml
bash scripts/run_overfit.sh "$CONFIG"
```

## Inference smoke

```bash
python -m tools.inference_smoke \
  --config "$CONFIG" \
  --adapter runs/hf_overfit_10/checkpoints/checkpoint_000300/lora.safetensors
```

For an existing checkpoint, set `model.root` in `$CONFIG` to its directory. It
must contain `model.safetensors`,
`tokenizer-e351c8d8-checkpoint125.safetensors`, and
`tokenizer_spm_32k_3.model`.
