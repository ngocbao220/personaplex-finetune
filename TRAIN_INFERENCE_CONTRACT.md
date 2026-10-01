# PersonaPlex train–inference contract audit

## Native source of truth

The checked-out upstream source is `../refs/personaplex-original/moshi/moshi/`.
`models/loaders.py:121,187` sets 17 delays to
`[0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1]` and overrides `dep_q` to 16.
`models/lm.py:356-424` defines audio initial token as `card=2048`, text initial
token as `text_card=32000`, text PAD as 3, END_PAD as 0, zero token as -1,
and audio offset as 1. Streams are text 0, agent Mimi codebooks 0–7 at 1–8,
and user Mimi codebooks 0–7 at 9–16.

`LMModel.forward_train` (`models/lm.py:531-550`) accepts canonical `[B,17,T]`
codes, applies the native delay, prepends the native initial frame, uses
`delayed[:, :, :-1]` as the LM input, and uses `delayed[:, :, 1:]` for the
depth transformer. Its text/audio logits are undelayed back to canonical
time, so CE targets are the original codes without another shift.
`LMGen.prepare_step_input` (`models/lm.py:727-813`) performs the same initial
frame and stream delays in streaming inference. Full teacher forcing requires
GT text, GT agent Mimi, and GT user Mimi at every subsequent call after the
initialization call. Providing only user audio leaves agent history sampled.

`LMGen.step_system_prompts` (`models/lm.py:1010-1128`) orders native voice
frames, silence, system text, silence. Its voice frames use
`encode_from_sphn(_iterate_audio(..., pad=True))` at 80 ms per frame; the
Mimi encoder runs in a fresh streaming state for that prompt. The
agent audio is voice/silence, the user audio is native sine tokens, and text
is PAD 3 except the `<system> ... <system>` tokens. The training builder
uses the same order and masks prompt frames from loss. LEFT channel is agent;
RIGHT channel is user.

## Verified gates

Run from `personaplex-finetuning/`:

```text
PYTHONPATH=src /opt/anaconda3/envs/personaplex/bin/python -m unittest tests.test_native_train_contract -v
PYTHONPATH=src /opt/anaconda3/envs/personaplex/bin/python -m unittest tests.test_sequence tests.test_objective tests.test_batching tests.test_native_train_contract -q
PYTHONPATH=src /opt/anaconda3/envs/personaplex/bin/python -m unittest tests.test_golden_sample -q
PYTHONPATH=src /opt/anaconda3/envs/personaplex/bin/python -m unittest tests.test_train tests.test_lora -q
```

The real prepared `conv_0001` 3.04 s window and a controlled 3.04 s sample
both pass the instrumented comparison immediately before native LM forward:

```text
Initial token mismatch: 0
Delay mismatch:         0
Input mismatch:         0
Target mismatch:        0
17/17 streams verified
PASS: TRAIN–INFERENCE CONTRACT EQUIVALENT
```

Focused results on this host: 42 sequence/objective/batching/contract/golden
tests passed; 39 trainer/LoRA tests passed with one skipped; 45 inference
tests passed. The trainer/LoRA group ran outside the sandbox because its
DataLoader and fresh-process test need OpenMP shared memory.

The real voice prompt initially differed from native LMGen at 507 Mimi
tokens because training encoded the whole file at once and without Mimi's
streaming state. Training now calls the native frame encoder inside a fresh
Mimi streaming state; the real-Mimi equality test passes. A real 3.04 s
user audio window also matches native streaming Mimi input exactly. Zero LoRA
matches base logits to `atol=1e-7`; a fresh process loads saved adapter
weights and matches in-memory logits and effective parameter delta to the
same tolerance. The fresh-process test needs shared memory available for
OpenMP.

The loss uses native logits aligned to canonical targets. Prompt frames have
zero loss. PAD and END_PAD have weight 0.5; the first audio codebook of each
speaker has full weight, with other audio codebooks at 0.02. The user audio
stream is now also a supervised target as requested for this audit. Training
logs text/nonpadding text loss and per-codebook losses for all 16 audio
codebooks.

The intended first GPU gate uses the checked-in overfit config with an
explicit local model and prepared data root. Run one fixed window before any
larger dataset:

```text
python -m train configs/overfit.yaml model.root=/absolute/path/to/personaplex-7b-v1 data.prepared_dir=/absolute/path/to/otospeech-prepared sample_number=1 sample_index=0 duration_sec=3.04 data.swap_roles_after_pass=false data.shuffle=false data.prompt_aug_prob=0.0
```

This command is recorded for the GPU environment; it has not been executed.
Its resulting adapter must be reloaded for native LMGen inference in both
greedy and sampling modes before increasing `sample_number` to 10.

## Remaining gate

This macOS host reports `torch.cuda.is_available() == False` and
`torch.backends.mps.is_available() == False`. No one-sample overfit or native
LMGen greedy/sampling audio inference was run here. Consequently there are
no training losses, GPU memory measurements, saved new overfit adapter,
or intelligibility results to report. Do not scale to 10 samples or more
until the one-sample GPU overfit and both inference modes pass.
