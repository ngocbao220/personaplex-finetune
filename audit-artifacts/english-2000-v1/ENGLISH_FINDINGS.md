# English training-collapse audit — 2026-10-08

Input: `bug-eng/config.json`, `metrics.jsonl`, `free_running_metrics.jsonl` (hashes in `english_evidence.json`). No production sources changed.

## Confirmed generation trajectory

Same three sample windows at every evaluation: conv_0001/2/3, 0–100 seconds. Base: 0/3 empty; step1000: 2/3 empty (conv_0002: "Good how are you doing?"); step2000: 3/3 empty. This means all evaluated windows are empty, not proven all training windows.

## Training block means

| Metric | Steps1–100 | Steps901–1000 | Steps1901–2000 |
|---|---:|---:|---:|
| loss/total | 4.172378 | 2.296876 | 1.786981 |
| loss/text_nonpadding | 2.553330 | 0.614928 | 0.081582 |
| accuracy/text_nonpad | 0.451482 | 0.867734 | 0.987963 |
| target_pad_pct | 87.946220 | 88.023380 | 88.101792 |
| predicted_pad_pct | 88.260183 | 85.506513 | 86.490062 |
| probability/pad_given_nonpad_target | 0.096663 | 0.003539 | 0.001435 |
| loss/audio_cb0 | 1.533032 | 1.355783 | 1.310591 |
| accuracy/audio_cb0 | 0.602268 | 0.627129 | 0.636351 |
| accuracy/audio_cb7 | 0.350832 | 0.418276 | 0.425545 |
| loss/audio_cb8 | 2.338940 | 2.056212 | 1.970447 |
| accuracy/audio_cb8 | 0.468843 | 0.511522 | 0.521778 |

Training metrics are teacher-forced, pooled across different windows. `probability/pad_given_nonpad_target` is a frequency of argmax PAD predictions, not mean softmax probability in current source. At the end it is 0.001435 (~0.14%), while real-text accuracy is 98.80%. Thus the logs do not support simple all-PAD collapse under ground-truth histories. Free-running raw token IDs are unavailable: do not assert that its empty decoded text proves every generated token is PAD.

## Effective run contract

- 10 loaded conversations, 97 candidate chunks, 91 retained 100-second chunks; requested sample_number=100 is not 100 actual conversations.
- One GPU, batch1, accumulation2, 2000 updates, shuffle=true, role swapping=false, no prompt augmentation.
- LoRA128/alpha256, joint temporal+depth training; temporal LR2e-6, depth LR5e-6; BF16.
- text_padding_weight=0.03, user_loss=true, diacritics mode.
- use_sampling=false: generation is greedy despite nonzero stored temperatures.
- No NaN/Inf found in logged numeric values; logs do not show full parameter/logit finiteness.
- Historical source path is personaplex-finetune-v10/src; no commit or source hash supplied.

## Interpretation

The text objective is fitting well with ground-truth text and audio history, while agent audio CB0 accuracy only changes ~60.23% to63.64% and CB7 ~35.08% to42.55%. Joint autoregressive memorization is not established. The source temporal embedding sums text plus audio embeddings; inference replaces ground-truth agent history with generated history. This is a concrete dependency that can expose a history mismatch; it is not proof of which history component triggers collapse. Native teacher forcing itself is expected, not automatically a code bug.

user_loss=true additionally trains user outputs, beyond the agent-only milestone contract. With the current objective, user audio contributes to a shared weighted audio denominator; changing it requires a controlled ablation, not assuming it is the cause. Low PAD weight already weakens the claim that padding weight alone explains the failure.

## Next discriminating test

Use step1000/2000 adapters on the same three windows, keeping all effective settings. Record raw token IDs plus teacher-forced metrics for those exact windows. Compare native streaming under: (A) free text/free agent audio, (B) free text with GT agent audio, (C) GT text with free agent audio, (D) full GT history. Condition B isolates whether generated agent audio/history drives text suppression; condition C is diagnostic and cannot establish free-running text quality. Existing probe supports A, C and bounded full-GT parity; B requires an audit-only GT-agent-audio hook or native LMGen moshi_tokens call. Do not claim B has already been run.

Base emits text and degradation occurs inside live training, so checkpoint save/load is not necessary to explain this symptom; independent checkpoint auditing remains useful for distinct reload issues. No additional training or policy change performed.
