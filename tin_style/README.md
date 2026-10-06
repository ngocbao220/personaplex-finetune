# tin-style: reference trainer with prepared inputs

The reference code is vendored from `refs/personaplex-finetune`; its exact source
commit and SHA-256 hashes are recorded in `provenance.json`. No runtime dependency
on that workspace directory is needed. Legacy code/configurations remain intact.

## Scope

- `data.py` translates prepared manifests to temporary native `path`/`duration`
  manifests for `sphn`. It reads prepared metadata/words in memory. Source data
  is never rewritten. LEFT=agent, RIGHT=user are validated.
- Only the reference dataset input and sidecar read calls are patched.
- Trainer, LoRA, optimizer, scheduler, chunking, interleaving, prompt building,
  loss, FSDP and checkpoint implementation are copied unchanged.
- Config/CLI adaptation and explicit offline model-path validation are outside
  the trainer algorithm. Unsupported configuration options fail explicitly.

Prepared sample layout (paths relative to the manifest):

```text
train.jsonl                 # {"sample_id": "one", "sample_dir": "samples/one"}
samples/one/
  conversation.wav         # stereo: left agent, right user
  voice_prompt.wav         # mono
  metadata.json            # agent_channel, user_channel, text_prompt
  words.json               # [{speaker, word, start, end}, ...]
```

Native reference manifests/sidecars remain supported too. Prepared word timestamps
must be finite, positive-duration and within the audio. Invalid samples fail;
this adapter does not repair or silently drop data.

## Environment

Use a separate CUDA environment for tin-style. The legacy environment pins
`sphn<0.2`; the reference trainer declares `sphn>=0.2,<0.3`. Install matching
CUDA torch/torchaudio first, then `python -m pip install -r tin_style/requirements.txt`.
The untouched dependency snapshots are included under `reference/`; they contain
machine-specific entries and are **not** portable install instructions.

The reference trainer uses NCCL/FSDP and must launch through `torchrun`, even on
one GPU. It is not a CPU/macOS trainer. Do not use legacy `python -m train` commands
to invoke tin-style.

Training uses the installed upstream Moshi package (`CheckpointInfo`/LoRA), as
the reference setup guide specifies. The copied PersonaPlex runtime is retained
for inference/evaluation, not inserted into the training import path. Upstream
Moshi 0.2.13 is the locally inspected API baseline; CUDA compatibility has not
been verified here. Supply a complete compatible model config with PersonaPlex
`dep_q=16` and Mimi's 8 codebooks. The reference setup guide requires extra
upstream-loader patches for sparse PersonaPlex config; this branch does not
silently apply those algorithm/runtime patches. A sparse stock config may fail
in the unchanged reference trainer. Existing local assets in this workspace
also lack `config.json`, so a training config must be provided explicitly.

## Reference semantics and limitations

This branch intentionally retains reference behavior, **not** all features of the
legacy trainer. In particular the reference targets all 16 audio codebooks
(`dep_q=16`), including user audio. Loss weighting is the reference implementation;
copying a legacy flag must not be mistaken for retaining legacy loss behavior.
This conflicts with the workspace's agent-only milestone contract. Do not claim
that milestone is complete with this experimental reference-parity pipeline.

Reference generation evaluation is preserved, including its subprocess paths,
`.venv` assumptions and external API dependencies. Offline local-model support
applies to the training load; external generation evaluation is not promised to
work offline. Keep `gen_eval.enable=false` for offline runs.

Checks without loading 7B:

```bash
python -m pytest tests/test_tin_style_data.py tests/test_tin_style_config.py tests/test_tin_style_reference.py -q
python -m tin_style configs/tin-style.yaml model=server --check-config
```

Launch (edit local model/data paths first):

```bash
torchrun --standalone --nproc-per-node=1 -m tin_style configs/tin-style.yaml model=server
```

These commands run from the repository root. Config supports YAML/Hydra defaults
and `key=value` overrides; the resulting reference args are saved by its trainer
in `run_dir/args.yaml`. Checkpoint/reload behavior follows the vendored reference.