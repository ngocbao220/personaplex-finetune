# PersonaPlex audit commands

Run from `personaplex-finetuning/`, with the interpreter of the actual training environment.
The harness reads production code and reference code; it does not modify either, retrain,
download assets, or change alignment/loss policy. Every invocation needs a fresh output directory.
Failures leave `failure.json` and child logs. A completed report can contain failed parity checks.

```sh
export PYTHONPATH=src
export NO_TORCH_COMPILE=1
python scripts/audit_personaplex.py logs --run artifact --output-dir audit-artifacts/logs-new
python scripts/audit_personaplex.py logs --run artifact --control-run /path/to/successful-10-run --output-dir audit-artifacts/logs-control
python scripts/audit_personaplex.py mini --device cpu --output-dir audit-artifacts/mini-cpu-new
python scripts/audit_personaplex.py mini --device mps --output-dir audit-artifacts/mini-mps-new
```

`mini --source-state head` replaces only the LoRA implementation with Git HEAD;
all other current dependencies remain fixed. The default tests the worktree, including
the existing QKV wrapper modification. Mini tests use synthetic inputs and native small
17-stream models, not 7B quality evaluation. They check zero-init, gradients, base freezing,
three updates, reference LM/loss/layer outputs, streaming, ring cache, greedy generation,
and fresh-process reload. Reference LM comparison holds local shared dependencies fixed.

## Prepared data and existing checkpoints on the server

Use the original YAML and **all original overrides**, not a reconstructed configuration.
Replace paths below with the actual server paths. The failed artifact used `no_diacritics`;
the effective overrides of the successful 10-sample run remain unknown. Do not assume
the current `overfit.yaml` is its exact resolved configuration.

```sh
python scripts/audit_personaplex.py data --config configs/overfit.yaml --override sample_number=100 --override data.prepared_dir=/path/to/synthetic_500h --override data.vietnamese_text_mode=no_diacritics --sample-id creativity_20079 --start-sec 0 --end-sec 43.75991666666667 --device cuda --output-dir audit-artifacts/server-data
python scripts/audit_personaplex.py probe --config configs/overfit.yaml --override sample_number=100 --override data.prepared_dir=/path/to/synthetic_500h --override data.vietnamese_text_mode=no_diacritics --sample-id creativity_20079 --start-sec 0 --end-sec 43.75991666666667 --adapter /path/to/checkpoint-500/lora.safetensors --device cuda --parity-frames 128 --output-dir audit-artifacts/server-step500
```

Repeat `probe` for step1000 and the successful 10-sample checkpoint with their own resolved
configs and fresh output directories. Preserve the exact evaluated sample, crop and role.
`--role right-agent` is permitted only when role swapping is trained. Adapter metadata must
resolve to the explicitly selected local base path; relocating assets requires deliberate
metadata/path validation. `--reference` defaults to the sibling reference checkout.

`data` uses the production loader/selection/chunk filter and real tokenizer/Mimi. It writes
retained/rejected windows, conditional sampler replay counts, token/frame alignment on both
sides, lost occurrence counts, voice codes, loss masks, asset hashes and exact source WAV.
Sampler replay is conditional on a fresh single-GPU run; it is not historical observed counts.

`probe` sequentially loads the existing adapter in memory, saves a roundtrip adapter, reloads
in a fresh process, runs the production standalone adapter inference, then base inference.
It compares masked TF logits, per-stream loss/count/accuracy, raw generated tokens, PCM/text,
and worker source/config identities. TF metrics are written both for the exact requested crop
and production padding for a single-example batch; neither is a historical step loss.
Multi-example training uses the batch maximum prompt length. The bounded parity prefix is
explicit and does not shorten primary inference. `--forced-text` adds a GT-text diagnostic,
not evidence of free-running quality. Logs include peak CUDA allocation for TF workers.

`--forced-agent-audio` adds `forced_agent_audio/`: the exact eight agent codebooks
from the requested training crop are supplied via native `LMGen.step(moshi_tokens=...)`,
while text remains free-running. System prompts are unchanged. Compare its `generated.txt`
and `tokens.json` with `in_memory/`. Recovered text supports a dependency on generated
agent audio/history, but does not prove a specific implementation defect. Its WAV is
GT-conditioned and cannot be used as evidence of generated audio quality. The independent
`--forced-text` condition does not also force audio. These probes do not retrain.

The in-memory worker already loads a saved checkpoint: it cannot establish equivalence to
the original historical training process. Capture a live training snapshot using the existing
`validation_generation.prepare_test3` / `run_test3` path to compare that boundary. Keep
the successful 10-sample logs, effective config, original generated WAV/text and adapter.

## Local legacy OtoSpeech alignment bridge

```sh
python scripts/audit_personaplex.py alignment --sample-dir ../data/conv_0001 --tokenizer ../models/tokenizer_spm_32k_3.model --start-sec 0 --end-sec 10 --encode --model-root ../models --device cpu --output-dir audit-artifacts/codec-new
```

Check the actual sample-directory path before running. Repeat `--sample-dir` for multiple
conversations. Omit `--encode` for tokenizer/alignment only; choose an end beyond each
duration to inspect all words. This explicit audit bridge accepts prepared legacy channel
metadata and prompts; it does not imply production loader compatibility. Reference alignment
is the actual AST-extracted Interleaver/tokenize source, with unique token-occurrence tracing
validated against real outputs. Voice comparison relocates only the reference CUDA literal
on CPU/MPS and records that scope.

After collecting matched evidence, test one change at a time (reload, generation settings,
text mode, role exposure, voice encoding, alignment or loss). Do not replace the objective
or retrain a larger dataset solely because reference code differs.
