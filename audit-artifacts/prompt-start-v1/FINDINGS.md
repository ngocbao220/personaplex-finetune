# Native hybrid prompt start audit

CPU synthetic reproduction, 2026-10-08. No production source changed.

`reproduce.py` executes native `step_system_prompts`, `step`,
`prepare_step_input`, and `process_transformer_output`. The codec emits known
tokens; neural depth is replaced with fully provided ground-truth targets.
Streaming state is installed directly, as native streaming entry does, without
an implicit initial step. This establishes token/cache history, not 7B logits.

| Measurement | Current unprimed call | Explicit BOS control |
|---|---:|---:|
| Canonical prompt frames | 61 | 61 |
| Temporal forwards during prompt | 60 | 61 |
| Whole-sequence temporal forwards (training: 99) | 98 | 99 |
| Same-index input mismatches in overlapping history | 653 | 0 |
| First dialogue input token-vector mismatches | 0 | 0 |

Training's first agent CB0 inputs are `[2048, 0, 1, 2]`; current native
inputs are `[2048, 1, 2, 948]`. At offset zero, the first supplied zero-delay
tokens are replaced with initial tokens. Delayed streams retain their queued
first frame, so the beginning is not merely a uniform frame deletion.
The 653 count includes shifted frames and is not an estimate of corrupted
dialogue tokens. At the dialogue boundary the immediate token vectors match,
but prior prompt history and temporal step count differ.

Existing `test_all_streams_match_fully_forced_lmgen_before_forward` primes
with `prepare_step_input()` before supplying the sequence. It passes, but
does not exercise the unprimed production prompt entry.

AST comparison found `prepare_step_input` and `step_system_prompts` identical
to both `refs/personaplex-original` and `refs/personaplex-finetune`.
This is a train/native contract discrepancy, not a demonstrated local-only
regression or a proven cause of empty generation. A controlled GPU comparison
with the same checkpoint, same codes and explicit initial BOS is needed to
measure its effect; do not alter production policy based on this CPU result.

Reproduction (local):

```sh
PYTHONPATH=src NO_TORCH_COMPILE=1 NO_CUDA_GRAPH=1 /opt/anaconda3/envs/personaplex/bin/python audit-artifacts/prompt-start-v1/reproduce.py
```

The reproduction assertions pass. The existing five-test native-contract
module has three passes, one failure (expects user-stream weight 1, current
default is 0), and one error (fixture metadata missing `text_prompt`). Those
tests were not edited; the module does not fully pass.

Full token vectors and source hashes are in `report.json`.
