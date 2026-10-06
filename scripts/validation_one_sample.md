# Ordered one-sample validation

Entry point: `validate_one_sample.py`. No production source or prepared data is
modified. Use absolute paths and a **new empty output directory** for every run.
All child logs and failures remain on disk. A non-zero gate stops the parent.

## Server commands

Set these to actual absolute paths on the server:

```bash
PROJECT=/absolute/path/to/personaplex-finetuning
MODELS=/absolute/path/to/models
DATA=/absolute/path/to/prepared-dataset
OUT=/absolute/path/to/new-validation-run

# First gate only. Real Mimi on CPU is supported; no 7B backbone allocation.
python "$PROJECT/scripts/validate_one_sample.py" \
  --config "$PROJECT/configs/overfit-10-train-v2.yaml" \
  --override "model.root=$MODELS" --override "model.source=$PROJECT/src" \
  --override "data.prepared_dir=$DATA" \
  --output-dir "$OUT" --index 0 --chunk 0 --device cpu --through 1

# New output directory: all gates, one fixed real chunk, explicit training budget.
# Use CUDA. Start with a short chunk only if that is the intended validation
# identity; do not switch sample/length/representation between gates.
python "$PROJECT/scripts/validate_one_sample.py" \
  --config "$PROJECT/configs/overfit-10-train-v2.yaml" \
  --override "model.root=$MODELS" --override "model.source=$PROJECT/src" \
  --override "data.prepared_dir=$DATA" \
  --output-dir "${OUT}-gpu" --index 0 --chunk 0 --device cuda \
  --train-steps 300 --through 6
```

`--through 2` runs only gates 1–2. `--adapter /absolute/path/to/checkpoint`
starts from existing trained LoRA; with `--train-steps 0` it evaluates without
further optimization. Without an adapter, positive training steps are mandatory
for gates 3 onward. The helper performs single-sample LoRA diagnostic optimization,
not a faithful replacement for the production trainer's scheduler/distributed path.
Audio/context/user-loss weights are reused from production loss code/config.
CUDA memory needs depend on sequence length, LoRA prefixes and quantization;
no guaranteed VRAM minimum is claimed. No automatic CPU/QLoRA fallback.

## Gates and evidence

1. Real Mimi/tokenizer, production builder. Raw/normalized words, every subword
   occurrence, target position/label, builder and delay-effective mask in
   `test1_dump.json`; actual tensor triplet in `sample.pt`. Empty transcript,
   missing occurrences, wrong labels or masked targets fail. The analytical
   native text-delay mask is checked against model output in gate 2.
2. Same checkpoint in eval: batch `forward_train` vs native temporal streaming
   and per-frame Depformer streaming using ground-truth depth history.
   Both mapped to the original undelayed grid. Checks masks/finite logits,
   max and mean error. FP32 atol/rtol=1e-5/1e-4; FP16=1e-2/1e-3;
   BF16=5e-2/5e-3, chosen before comparison. Explicit `--atol/--rtol` overrides
   are recorded. **This tests native model streaming, not LMGen cache parity.**
3. Training child saves adapter plus native metadata and greedy baseline, then
   **exits**. Separate process reloads base+adapter+metadata through production
   inference. Exact decoded PCM and text comparison; quality is not this gate.
4. Same-sample greedy probe: require actual native sampling calls equal argmax,
   `use_sampling=false`, text/audio temperatures zero. WER <= `--max-wer` (0.05).
   This is a text overfit gate; audio fitting is deliberately assessed in gate 5.
5. Unweighted agent cb0…cb7 CE/accuracy/counts with builder AND model validity.
   Every codebook must have count>0, finite logits, accuracy>=0.99, CE<=0.1.
   Override thresholds before running with `--min-audio-accuracy/--max-audio-ce`.
6. Native free-running artifact generation only after preceding passes. Uses
   the same real sample/representation and greedy settings; WAV/text for human
   listening. No automatic claim of perceptual speech quality.

Config, sample, model/Mimi/tokenizer/audio SHA256 and Python source digests are
stored in `identity.json`, recomputed between phases. Native metadata and
adapter keys are handled by the production resolver. All gates are fresh child
processes; saved tensor evidence is reused, never silently re-encoded later.

## Mac results (2026-10-06)

Real test 1 was attempted with the local prepared dataset, CPU, index=0, chunk=0,
duration=20s, preserving configured text representation. It stopped at dataset
loading: **0 valid / 10 invalid entries**, all missing `voice_prompt_left.wav`.
Local sample folders instead contain `voice_prompt.wav`. No aliases or data
repairs were made. Final evidence:
`/tmp/personaplex-gates-mac-20261006-final/phase_1.log`,
`/tmp/personaplex-gates-mac-20261006-final/dataset_rejections.json`,
`/tmp/personaplex-gates-mac-20261006-final/invocation.json`.
This is a precondition failure, not evidence about text supervision.

The small random native FP32 CPU model unit test additionally detects a
batch/depth-streaming discrepancy at the last codebook (~0.035 max error with
fixed seed), while text logits and masks agree. This is **not** a real gate 2
result and does not identify the production failure's cause. It is retained as
a regression/detector check; no model fix or tolerance loosening was made.

Run lightweight tests with:

```bash
NO_TORCH_COMPILE=1 python -m pytest -q "$PROJECT/tests/test_validation_gates.py"
```

Result: 9 passed on Mac. Syntax parsing and CLI help passed. Full repository
suite cannot collect because existing `tests/test_hf_assets.py` imports the
missing `personaplex_finetuning.hf_assets` module; that unrelated issue was not
modified. Full-suite log: `/tmp/personaplex-validation-suite.log`.

Do not proceed to model diagnosis until the chosen real prepared sample meets
the existing loader contract. Obtain approval for one minimal correction, then
rerun the first gate and subsequent affected gates.