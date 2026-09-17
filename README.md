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

## Local-only model setup

Install this package and the matching local PersonaPlex/Moshi source in the GPU
environment. Edit `configs/overfit_10.yaml` so `model.root` contains exactly:

```text
model.safetensors
tokenizer-e351c8d8-checkpoint125.safetensors
tokenizer_spm_32k_3.model
```

`model.source` must point to the local `moshi/` source directory. Missing assets,
Python dependencies or CUDA fail immediately. The implementation never invokes a
Hugging Face download path.

## Proof order

```bash
python -m tools.inspect_sample --config configs/overfit_10.yaml --index 0
python -m tools.train_smoke --config configs/overfit_10.yaml
python -m personaplex_finetuning.train --config configs/overfit_10.yaml
python -m tools.inference_smoke \
  --config configs/overfit_10.yaml \
  --adapter runs/overfit_10/checkpoints/checkpoint_000300/lora.safetensors
```

The trainer has no shuffle, augmentation, distributed mode, FSDP or full-model
fine-tuning. It logs component losses to `metrics.jsonl`, saves adapter-only
safetensors, reloads the adapter into a fresh base model, and writes
`reports/overfit_10.md`. Do not scale data until this report and the inference
artifacts have been reviewed.
