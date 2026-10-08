# PersonaPlex audit commands

Run from `personaplex-finetuning/`, with the interpreter of the actual training environment.
The harness reads production code and reference code; it does not modify either, retrain,
download assets, or change alignment/loss policy. Every invocation needs a fresh output directory.
Failures leave `failure.json` and child logs. A completed report can contain failed parity checks.

For a CUDA Graph capture failure, rerun in a fresh process/output directory with
`NO_CUDA_GRAPH=1 NO_TORCH_COMPILE=1 CUDA_LAUNCH_BLOCKING=1`. Native Moshi already
supports `NO_CUDA_GRAPH`; disabling torch.compile alone does not disable graph capture.
Record these environment flags alongside results: eager execution is a diagnostic
configuration, not evidence that the original graph-enabled path passes. A capture-end
error can mask an earlier failure inside the captured function; preserve the full log.

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

`user_encoding.json` is saved before the strict equality assertion. If user tokens
differ, it contains both grids, shapes, mismatch count/fraction, first mismatch and
repeated mono/stereo results (or repeat error). Run `data` first to gather this evidence
without allocating 7B. A successful repeat can come from cache when a cache directory
is configured; the report records that directory. Do not disable the assertion or
infer causality merely from nonidentical quantized codes. Small numeric differences
between batch sizes and larger layout/window differences require separate diagnosis.

For an explicitly controlled checkpoint experiment, `probe --match-user-conditioning`
preserves the original mismatch in `user_encoding.json`, then substitutes stable
single-item inference user codes in the audit TF input and labels. The eight agent
streams, text, prompts and masks are retained. `sequence.json` is the original;
`sequence_matched.json` is the intervention. Reload reuses the saved user codes.
Reports label this as an inference-matched diagnostic, not historical training loss
or a trainer correction. Shape mismatch and unstable repeated mono encoding still
stop the probe. This option allows `--forced-agent-audio` analysis to proceed despite
a documented stereo/mono difference; default probes remain strict.

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

`--full-gt` adds `full_gt/`, forcing both GT text and GT agent audio. Every observed
generation now exports `text_logits.json` with raw text argmax, top tokens, PAD probability,
native offset, text delay and provided GT target/log-probability before native forcing.
Its summary reports CE/accuracy and PAD prediction frequency at nonpadding GT targets.
Free-text conditions have no provided GT targets, so these GT accuracy fields are null.
Compare `full_gt/text_logits.json` against the same-window text stream in
`in_memory_metrics.json`, checking token counts and masks rather than historic step losses.
The targets are taken from native delayed cache, not inferred from decoded text.
`full_gt/generated.txt` is supplied GT and does not prove prediction quality. A failure
still exports partial traces marked `completed=false`. Native logits and outputs are never
modified by observation. Keep `NO_CUDA_GRAPH=1` for the reported graph-capture failure.

## GT prefix release: isolate the beginning of free-running divergence

`probe --release-at-first-text` finds the first dialogue text token outside IDs 0/3.
`--release-at-frame N` selects an explicit dialogue frame instead (mutually exclusive).
Both create three additional diagnostic generations. For frames `t < N`, force GT
text and agent audio. Starting at `t == N`:

| Output | Text | Agent audio |
|---|---|---|
| `release_both/` | free | free |
| `release_text/` | free | GT |
| `release_audio/` | GT | free |

Prompt handling, user conditioning and model sampling are unchanged. Queued audio
tokens are not cleared at release: audio codebooks with delay 1 still consume GT
queued at `N-1` when processing step `N`. `tokens.json` records the actual provided
mask/targets, so this transition is inspectable. This is a GT-prefix intervention,
not ordinary generation quality; full-output WER includes the forced prefix.

The in-memory worker now passes a separate, observation-only reference text grid to
all conditions. `reference_target`/`reference_log_probability` remain available after
forcing stops; they do not enter model inputs. Native text-delay alignment is explicit
in `reference_dialogue_frame`. The original provided-GT metrics keep their old semantics.

Inspect `release_comparison.json` for the first nonpadding reference frame, the raw
prediction at the release boundary in each condition, its PAD/GT probability, the first
text divergence (including PAD positions), and accuracy/CE on nonpadding reference
targets after release. `first_post_release_failure` includes PAD reference positions;
`first_post_release_reference_failure` is restricted to nonpadding reference positions.
GT probability is `exp(reference_log_probability)`. Individual full traces remain
in each condition's `text_logits.json`. Check `completed` before comparing results.

To reuse the exact server request from the full-GT experiment, sync
`scripts/audit_native.py` and `scripts/audit_personaplex.py`, then run:

```sh
python - <<'PY'
import json
from pathlib import Path
request = json.loads(Path('/home/voice/code/VDT_02/baottn/personaplex-finetune-v10/outputs-full-gt-1/request.json').read_text())
request.update(output_dir='/home/voice/code/VDT_02/baottn/personaplex-finetune-v10/outputs-release-first-text-1',
               match_user_conditioning=True, forced_text=True, forced_agent_audio=True,
               full_gt=True, release_at_first_text=True, release_at_frame=None, parity_frames=0)
Path('/home/voice/code/VDT_02/baottn/personaplex-finetune-v10/outputs-release-first-text-1').mkdir(exist_ok=False)
Path('/home/voice/code/VDT_02/baottn/personaplex-finetune-v10/outputs-release-first-text-1/request.json').write_text(json.dumps(request, indent=2) + '\n')
PY

set -o pipefail
NO_CUDA_GRAPH=1 NO_TORCH_COMPILE=1 CUDA_LAUNCH_BLOCKING=1 python /home/voice/code/VDT_02/baottn/personaplex-finetune-v10/scripts/audit_personaplex.py _worker --request /home/voice/code/VDT_02/baottn/personaplex-finetune-v10/outputs-release-first-text-1/request.json --phase in_memory 2>&1 | tee /home/voice/code/VDT_02/baottn/personaplex-finetune-v10/outputs-release-first-text-1/debug.log
```

This reloads the same checkpoint once and runs four controls plus three release
conditions. It does not train or perform the other reload/standalone/parity phases.
The previous output directory is preserved. Use a new directory for each run.
To release after the first word or at a later divergence instead, create another
request/output directory with `release_at_first_text=False` and `release_at_frame=N`.

```sh
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('/home/voice/code/VDT_02/baottn/personaplex-finetune-v10/outputs-release-first-text-1/release_comparison.json').read_text())
print('Release frame:', report['release_at_frame'], 'time:', report['release_time_sec'])
for name, row in report['conditions'].items():
    print('\n==', name, '== completed:', row['completed'])
    print('Boundary:', json.dumps(row['release_boundary'], indent=2))
    print('Summary:', json.dumps(row['summary'], indent=2))
PY
```

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
