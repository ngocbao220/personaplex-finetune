# tin-style implementation verification

Branch: `tin-style`, based on `moshi-code-style`. Source revision is recorded in
`provenance.json`. Legacy trainer/config files are untouched; README adds a link.

## Verified locally

- `python -m pytest tests/test_tin_style_data.py tests/test_tin_style_config.py tests/test_tin_style_reference.py -q`: **28 passed**.
- `python -m compileall -q tin_style`: passed.
- `git diff --check`: passed.
- Prepared OtoSpeech train manifest: all 10 entries validated successfully.
- Real `sphn.dataset_jsonl(...duration_sec=2...)` sample: `(2, 48000)` at 24 kHz,
  start time 0, 48000 unpadded samples; LEFT/RIGHT layout retained.
- Native reader integration test: two 0.5-second stereo chunks, original timeline.
- Hash tests verify reference code is unchanged except two data-boundary calls.
- CLI composition/overrides, local path validation, offline settings and native
  resume forwarding covered by CPU tests. These tests use dummy asset files;
  they do not assert that dummy checkpoints contain valid model weights.
- Real CLI with `model.root=../../models --check-config`: fails fast on missing
  `models/config.json`, as intended. No downloads performed.

## Broader legacy suite

`PYTHONPATH="$PWD/src:$PWD" python -m pytest tests -q` fails collection because
the existing `tests/test_hf_assets.py` imports absent
`personaplex_finetuning.hf_assets`.

With `--ignore=tests/test_hf_assets.py`: 284 passed, 4 skipped, 4 failed (at the
time of that run). Failures are existing config B200/default validation tests,
standalone inference checkpoint CLI test and native train audio contract test.
No legacy implementation/config files were modified to fix unrelated failures.

## Not verified

No CUDA/NCCL training, 7B forward/backward, loss decrease, adapter save/reload or
inference smoke test was performed on this macOS environment. GPU memory and
training losses: not measured. Missing local model config and reference training
dependencies also prevent a real training launch here.

Reference training depends on upstream Moshi with LoRA/CheckpointInfo, not its
vendored inference runtime. Its documented sparse-config loader patches are not
included in the reference source tree. Supply a compatible complete config or
the reference's prepared training environment; do not assume the upstream package
with a sparse stock config works without those patches.

Reference loss/user-stream behavior intentionally differs from workspace milestone
rules; see README. This is a reference-parity branch, not a completed overfit-10
milestone. Changes are left uncommitted for review; no push performed.