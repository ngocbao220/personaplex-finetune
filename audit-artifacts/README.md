# Audit findings — 2026-10-07

The audit harness is implemented. No production training/inference policy was changed.
The pre-existing `src/personaplex_finetuning/lora.py` QKV edit and supplied artifacts remain
untouched. Synthetic_500h data and 7B checkpoints are server-only, so the root cause of poor
generation remains unproven. The successful 10-sample result is in-training generation;
its fresh checkpoint inference and effective config are not available here.

## Recorded failed run

[Log report](failing-run/README.md): 100 requested conversations, 57 retained chunks, 1,000 steps.
Base CER 0.7888; steps100–300 produce empty text; best step500 CER 0.7125/WER 0.8361;
step1000 CER 0.8957/WER 0.9754. No checkpoint reload checks were recorded.
Mean total loss is 3.8155 for steps1–100, 1.3093 for steps501–600, and 2.4938 for
steps901–1000. Real text loss changes 3.2064 → 0.3734 → 0.8214. This evidence does not
show uniformly improving losses through the entire run. Pooled role means cannot identify
a causal role difference without matching crop identities and exposures.

## Locally executed probes

| Probe | Evidence | Result and limit |
|---|---|---|
| Native miniature MPS, Torch2.8 | [report](mini-worktree-mps-torch280/report.json) | Zero-init exact; frozen base unchanged; 52/84 adapter tensors have positive gradients after three updates; fresh-process logits and 22 generated frames exactly match. No MPS CPU fallback. Synthetic inputs, not 7B quality. |
| Reference LM source and layers | Same mini report | Five observed layer boundaries and native outputs match with shared local dependencies held fixed. Local/reference objectives intentionally differ. |
| Streaming and ring cache | Same mini report | Only final user acoustic codebook16 (audio index15) differs materially. Cache capacity16 gives positions `[16,1,...,15]` at boundary, from shared local/reference `delta <= 0`. Agent streams match; failed run has user_loss=false. This shared defect is not established as the generation cause. |
| All ten local OtoSpeech conversations | [report](otospeech-alignment-all-v2/report.json) | Local loses0 tokens, reference loses1,054 occurrences; token/frame outputs and individual losses exported. These are not the failing synthetic corpus. |
| Real tokenizer/Mimi, conv_0001 0–10s | [report](mimi-real-cpu-torch280/report.json) | 17 streams;111 prompt/125 dialogue frames; prompt loss masked; user train/infer codes equal. Local voice equals native LMGen; raw reference batch voice differs480/600 overlapping tokens. Alignment local lost0/reference lost2. |

Reference overlap loss is reproducible and should be evaluated on synthetic_500h before an
alignment change. A reference implementation can replace tokens already waiting for placement;
its lower token count is not by itself proof of better supervision.

## Environment and verification

Active interpreter: `/opt/anaconda3/envs/personaplex/bin/python`.
Torch and torchaudio upgraded together to2.8.0. Requirements already allow `<2.9`.
`pip check` still reports stale editable package metadata requiring torch<2.5; the current
checkout has no packaging file. Site-packages metadata was not edited manually.

Commands executed from the repository:

```sh
/opt/anaconda3/envs/personaplex/bin/python -m pip install --upgrade torch==2.8.0 torchaudio==2.8.0
PYTHONPATH=src NO_TORCH_COMPILE=1 /opt/anaconda3/envs/personaplex/bin/python scripts/audit_personaplex.py mini --device mps --output-dir audit-artifacts/mini-worktree-mps-torch280
PYTHONPATH=src NO_TORCH_COMPILE=1 /opt/anaconda3/envs/personaplex/bin/python -m unittest discover -s tests -p test_audit_harness.py -q
PYTHONPATH=src NO_TORCH_COMPILE=1 /opt/anaconda3/envs/personaplex/bin/python -m unittest discover -s tests -p 'test_*.py' -q
```

Focused audit suite:11 tests pass. Full suite:256 tests,3 failures,4 errors,4 skips.
Existing failing areas are default generation settings, user-loss default expectations,
padding token expectations, missing B200 preset, incomplete CLI fixture, missing hf_assets
module, and legacy text_prompt fixture. The full suite is not green; these were not changed
under this audit. Failed earlier MPS/legacy bridge attempts remain as diagnostic artifacts.

## Remaining proof

Follow [server command guide](../scripts/audit_personaplex.md) to run the real production
data/checkpoint/inference probes for the failed step500/1000 and successful10 run. Preserve
exact config/overrides, sample/crop/role and local asset hashes. The new full-checkpoint padded
TF and worker identity checks have not run against server7B assets. Compare live training
snapshot with reload separately; an existing-checkpoint probe cannot recreate historical state.
No retraining, distributed execution or dataset scaling was performed.
