# Implementation Plan: Skip PersonaPlex Text-Overflow Chunks

## Overview

Remove the temporary behavior that places text tokens before their word timestamp. After conversation-level train/validation/test separation and fixed-duration chunking, check every chunk with the training tokenizer and Mimi frame grid. Exclude chunks whose text target cannot fit without changing alignment, then use that filtered chunk list consistently for sampling, evaluation, resume validation, and reporting.

## Architecture Decisions

- Keep `PreparedDataset.load()` responsible for reading and validating prepared conversation rows. It has no tokenizer or Mimi frame-rate context, so filtering belongs after `duration_chunks()`.
- Keep conversation splitting and `sample_number` selection before chunk filtering. Filtering must not move a conversation or chunk between data splits.
- Use `align_dialogue_text_targets()` as the only capacity rule. Text tokens remain at or after their aligned timestamps; insufficient capacity rejects the chunk.
- If role swapping is enabled, reject a training chunk if either role view overflows. This keeps both directions trained on the same audio intervals. Continue validating both role prompt assets.
- Filter train, validation, and test chunks before sample counts, run config, resume checks, and samplers. Record each rejection's role, time range, first overflowing word, and reason.
- At startup, report invalid/out-of-bounds manifest rows separately from out-of-bounds chunks and text-overflow chunks rejected after chunking.
- Pass the already chunked and filtered list into the training iterator. Do not recreate chunks each epoch.
- Use the runtime Mimi frame rate and fixed-duration padded frame count for text-only preflight. Assert actual encoded frames match the expected count when examples are built. This avoids an extra full-dataset Mimi encode; the standalone checker remains a real-Mimi audit.
- Bind the filter algorithm version and ordered kept-chunk identity into the resume contract. Older checkpoints without verifiable filter state must not silently resume against a different dataset.

## Task List

### Phase 1: Restore strict alignment

#### Task 1: Keep every text target at or after its timestamp

**Description:** Remove backward frame borrowing. The alignment helper reports overflow, and training fails clearly if an unfiltered overflowing chunk reaches the builder.

**Acceptance criteria:**
- [ ] No token is placed before its word timestamp.
- [ ] Overflow remains explicit and identifies the first overflowing word.
- [ ] In-capacity token placement is unchanged.

**Verification:** Focused sequence and capacity-checker tests cover ordinary placement and late-word overflow.

**Files likely touched:** `src/personaplex_finetuning/sequence.py`, `src/tools/check_text_chunk_capacity.py`, `tests/test_sequence.py`.

**Estimated scope:** Small.

### Phase 2: Filter chunks before training

#### Task 2: Classify chunk capacity deterministically

**Description:** Add a focused helper that returns accepted/rejected status and per-role details using actual chunk words, the configured normalization, and the fixed Mimi frame grid.

**Acceptance criteria:**
- [ ] Classification is deterministic for the same sample, bounds, tokenizer, normalization setting, and role policy.
- [ ] With role swapping enabled, overflow in either view rejects the interval for both views.
- [ ] Rejection details identify sample, bounds, role, and first overflowing word.

**Verification:** Unit tests cover left-only overflow, right-only overflow, both passing, and normalization on/off.

**Files likely touched:** `src/personaplex_finetuning/chunk_filter.py`, `tests/test_chunk_filter.py`.

**Estimated scope:** Small.

#### Task 3: Apply one filtered chunk list throughout training

**Description:** Filter train/validation/test chunks after conversation-level splitting and duration chunking. Use the resulting lists for counts, rank partitioning, evaluation, and training. Write kept/rejected counts and per-chunk reasons to the run directory.

**Acceptance criteria:**
- [ ] Conversation splits and `sample_number` selection happen before filtering.
- [ ] Rejected chunks never reach Mimi batch encoding or the LM.
- [ ] All ranks receive the same ordered kept list before rank-stride partitioning.
- [ ] Empty or too-small datasets fail before optimizer startup with clear counts.
- [ ] Training iterator consumes prefiltered chunks without resplitting conversations.
- [ ] Startup reports sample-level out-of-bounds/invalid rows separately from chunk-level out-of-bounds and text-overflow skips.

**Verification:** Training iterator tests cover filtering, role symmetry, no forwarding of rejected chunks, and equal distributed batch counts.

**Files likely touched:** `src/personaplex_finetuning/train.py`, `tests/test_train.py`.

**Estimated scope:** Medium.

### Checkpoint: Filtering contract

- [ ] No token is shifted before its aligned timestamp and no token is dropped from a kept chunk.
- [ ] Training and evaluation do not later encounter chunks already classified as overflow.

### Phase 3: Resume and preflight integration

#### Task 4: Make the filtering decision reproducible and auditable

**Description:** Bind filter version and ordered kept-chunk fingerprint into resume validation; expose the rejection report and document the workflow.

**Acceptance criteria:**
- [ ] Resume rejects checkpoints with a different or unverifiable filtered chunk set.
- [ ] Actual encoded frame count is asserted against the fixed-duration frame count.
- [ ] The checker distinguishes text overflow from frame mismatch and agrees with training's filter decision.
- [ ] README explains skip counts, report location, and the preflight command.
- [ ] Logs distinguish sample-level out-of-bounds skips from chunk-level text overflow skips.

**Verification:** Focused resume, training, sequence, and checker tests pass; `git diff --check` and checker `--help` pass.

**Files likely touched:** `src/personaplex_finetuning/train.py`, `src/tools/check_text_chunk_capacity.py`, `tests/test_train.py`, `README.md`.

**Estimated scope:** Medium.

## Optimization Notes

- Capacity preflight tokenizes transcript words on the expected fixed Mimi grid. It avoids an extra audio decode/Mimi pass and avoids DataLoader/LM work on rejected chunks.
- The standalone checker continues to encode with real Mimi as an audit. Its cache can be reused when `codec_cache_dir` is configured.
- Report rejected counts and examples; do not optimize by moving timestamps or dropping individual text tokens.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Expected frame count differs from actual Mimi output | A chunk could pass preflight then fail in training | Assert actual frame count before using targets; retain real-Mimi checker. |
| One role overflows and the other does not | Bidirectional training sees different audio intervals | Reject the chunk interval for both training role views. |
| Filtered data order/count changes on resume | Training trajectory changes | Check the ordered kept-chunk fingerprint and algorithm version. |
| Many chunks are rejected | Less training data than expected | Report counts and identities; fail early when too few batches remain. |

## Open Questions

- None. The plan rejects a whole chunk interval if any configured training role overflows.
