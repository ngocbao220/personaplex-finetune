# Prepared-data invalid chunk filter

Prepared conversations are retained even when finite word timestamps are invalid.
The adapter uses the actual reference `chunk_step_sec` (including prompt budget),
not a separately configured approximation. Before tokenization, windows touching
invalid words are skipped. Metadata, speaker, prompt and unreadable/non-finite
timestamp errors still fail fast.

Valid word bounds are `0 <= start < end <= audio_duration + 0.05`, matching
the legacy reader's 50 ms endpoint tolerance. No timestamp is clipped or repaired.
For invalid words, the interval between start and end determines affected windows;
words entirely outside the WAV reject the nearest edge window. A word spanning
several windows rejects each overlapping window. Windows are half-open, so a word
beginning exactly on a boundary affects the next window.

Preflight logs each rejected path, window and word index/timestamps, prints kept
and dropped counts, and fails if a manifest has no valid chunks. Logs should be
saved with the training run. The input files and reference training algorithms
are unchanged. Native reference sidecars do not opt into this prepared-data filter.

Deploy both files together:

- `tin_style/data.py`
- `tin_style/reference/moshi-finetune/finetune/data/dataset.py`

`native_manifest()` alone validates assets but does not filter chunks: chunk size
is only known when the reference tokenizer is constructed. The normal reference
iterator performs preflight and filtering automatically, before Mimi/tokenization.

The iterator also filters actual unpadded audio lengths using the loaded
`mimi.encoder.hop_length`. Empty chunks or lengths not divisible by this hop are
logged and skipped before Mimi encoding, without padding or cropping. A missing
or invalid hop fails explicitly rather than guessing from codec frame rate.
If a rank yields no usable chunks across all source manifests in an epoch
(after both filters), iteration fails instead of spinning through empty epochs.
This checks encoder stride compatibility only; other Mimi assertions remain
errors with path, start time and audio shape diagnostics.

Verification: run `python3 -m pytest tests/test_tin_style_data.py
tests/test_tin_style_config.py tests/test_tin_style_reference.py -q` from the
repository root. Tests include real sphn chunk loading, retained audio shapes,
boundary and multi-window rejection, empty-manifest protection, unchanged source
files, and reference provenance guards. GPU training remains a server check.
Also run `tests/test_tin_style_audio_filter.py` for audio-length filtering and
execution of the actual reference iterator with CPU test doubles.