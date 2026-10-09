"""Benchmark Vietnamese text tokenization exactly as the trainer places agent text.

Per tokenizer x vietnamese_text_mode: round-trip fidelity, tokens per syllable,
SentencePiece byte-fallback, and chunks lost to text overflow on the Mimi grid.
CPU only; loads neither the LM nor Mimi.

    PYTHONPATH=src python -m tools.benchmark_vi_tokenizer --config configs/train_vi_synthetic.yaml \
        data.prepared_dir=../synthetic_samples --tokenizer ../models/tokenizer_spm_32k_3.model
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from personaplex_finetuning.chunk_filter import expected_mimi_frames, filter_text_capacity_chunks
from personaplex_finetuning.config import load_config
from personaplex_finetuning.data import PreparedDataset, duration_chunks, limit_conversations
from personaplex_finetuning.runtime import SentencePieceTokenizer
from personaplex_finetuning.sequence import align_dialogue_text_targets
from personaplex_finetuning.text_normalization import normalize_vietnamese_text
from personaplex_finetuning.text_vocab import TranslatedTokenizer

MIMI_FRAME_RATE = 12.5
MODES = ("diacritics", "no_diacritics", "telex")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _is_syllable(word: str) -> bool:
    return any(character.isalpha() for character in word)


def _percentile(values: list[int | float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))])


class WordTokenizer:
    """Memoized per-word encoding, mirroring align_dialogue_text_targets."""

    def __init__(self, tokenizer, mode: str):
        self.tokenizer = tokenizer
        self.mode = mode
        self._cache: dict[str, tuple[str, tuple[int, ...]]] = {}
        self._is_byte = getattr(tokenizer, "is_byte", lambda _id: False)
        self._is_unk = getattr(tokenizer, "is_unk", lambda _id: False)

    def encode(self, word: str) -> tuple[str, tuple[int, ...]]:
        if word not in self._cache:
            target = normalize_vietnamese_text(word, self.mode)
            self._cache[word] = (target, tuple(self.tokenizer.encode(target)))
        return self._cache[word]

    def is_byte(self, token: int) -> bool:
        return bool(self._is_byte(token))

    def is_unk(self, token: int) -> bool:
        return bool(self._is_unk(token))


def roundtrip_stats(texts: list[str], words: WordTokenizer, examples: int = 10) -> dict:
    """Utterance-level decode(encode(x)) == x after NFC, x already mode-normalized."""
    exact = unk = replacement = 0
    failures = []
    for text in texts:
        target = _nfc(normalize_vietnamese_text(text, words.mode))
        ids = words.tokenizer.encode(target)
        decoded = _nfc(words.tokenizer.decode(ids))
        unk += any(words.is_unk(token) for token in ids)
        replacement += "�" in decoded
        if decoded.strip() == target.strip():
            exact += 1
        elif len(failures) < examples:
            failures.append({"expected": target, "decoded": decoded})
    total = max(1, len(texts))
    return {
        "utterances": len(texts), "exact_rate": exact / total,
        "with_unk_rate": unk / total, "with_replacement_char_rate": replacement / total,
        "failures": failures,
    }


def syllable_stats(word_list: list[str], words: WordTokenizer, agent_seconds: float) -> dict:
    counts = [len(words.encode(word)[1]) for word in word_list if _is_syllable(word)]
    histogram = Counter(min(count, 4) for count in counts)
    total_tokens = sum(len(words.encode(word)[1]) for word in word_list)
    return {
        "syllables": len(counts),
        "mean": statistics.fmean(counts) if counts else 0.0,
        "p50": _percentile(counts, 0.5), "p90": _percentile(counts, 0.9),
        "p99": _percentile(counts, 0.99), "max": max(counts, default=0),
        "histogram": {("4+" if key == 4 else str(key)): histogram[key] for key in sorted(histogram)},
        "agent_tokens_per_speech_sec": total_tokens / agent_seconds if agent_seconds > 0 else 0.0,
        "frame_budget_per_sec": MIMI_FRAME_RATE,
    }


def byte_fallback_stats(word_list: list[str], words: WordTokenizer, top: int = 20) -> dict:
    tokens = byte_tokens = byte_words = 0
    offenders: Counter[str] = Counter()
    for word in word_list:
        target, ids = words.encode(word)
        hits = sum(words.is_byte(token) for token in ids)
        tokens += len(ids)
        byte_tokens += hits
        if hits:
            byte_words += 1
            offenders[target] += 1
    return {
        "token_rate": byte_tokens / max(1, tokens),
        "word_rate": byte_words / max(1, len(word_list)),
        "top_words": offenders.most_common(top),
    }


def overflow_stats(chunks, tokenizer, mode: str, swap_roles: bool, num_workers: int) -> dict:
    result = filter_text_capacity_chunks(
        chunks, tokenizer, MIMI_FRAME_RATE, swap_roles=swap_roles, num_workers=num_workers,
        vietnamese_text_mode=mode,
    )
    densities = []
    for chunk in chunks:
        duration = chunk.window_end_sec - chunk.window_start_sec
        frames = expected_mimi_frames(duration, MIMI_FRAME_RATE)
        targets = align_dialogue_text_targets(chunk, frames, MIMI_FRAME_RATE, tokenizer, mode)
        densities.append(targets.required_tokens / frames)
    total_sec = sum(chunk.window_end_sec - chunk.window_start_sec for chunk in chunks)
    kept_sec = sum(chunk.window_end_sec - chunk.window_start_sec for chunk in result.kept)
    return {
        "chunks": len(chunks), "kept": len(result.kept),
        "text_overflow": result.skipped_text_overflow,
        "out_of_bounds": result.skipped_out_of_bounds,
        "lost_rate": 1 - len(result.kept) / max(1, len(chunks)),
        "lost_hours": (total_sec - kept_sec) / 3600,
        "token_density_mean": statistics.fmean(densities) if densities else 0.0,
        "token_density_p99": _percentile(densities, 0.99),
        "token_density_max": max(densities, default=0.0),
    }


def agent_corpus(conversations, speakers: str) -> tuple[list[str], list[str], float]:
    """Words and utterances (consecutive same-speaker words) plus agent speech seconds."""
    word_list: list[str] = []
    utterances: list[str] = []
    seconds = 0.0
    for sample in conversations:
        current_speaker, current = None, []
        for word in sample.words:
            if speakers == "agent" and word.speaker != "agent":
                if current:
                    utterances.append(" ".join(current))
                current_speaker, current = None, []
                continue
            word_list.append(word.word)
            if word.speaker == "agent":
                seconds += max(0.0, word.end - word.start)
            if word.speaker != current_speaker and current:
                utterances.append(" ".join(current))
                current = []
            current_speaker = word.speaker
            current.append(word.word)
        if current:
            utterances.append(" ".join(current))
    return word_list, utterances, seconds


def benchmark(config, conversations, tokenizers: dict[str, object], modes, speakers: str, num_workers: int) -> dict:
    word_list, utterances, agent_seconds = agent_corpus(conversations, speakers)
    chunks = duration_chunks(conversations, config.duration_sec)
    rows = []
    for name, tokenizer in tokenizers.items():
        for mode in modes:
            words = WordTokenizer(tokenizer, mode)
            rows.append({
                "tokenizer": name, "mode": mode,
                "roundtrip": roundtrip_stats(utterances, words),
                "tokens_per_syllable": syllable_stats(word_list, words, agent_seconds),
                "byte_fallback": byte_fallback_stats(word_list, words),
                "overflow": overflow_stats(chunks, tokenizer, mode, config.swap_roles_after_pass, num_workers),
            })
    return {
        "conversations": len(conversations), "words": len(word_list), "utterances": len(utterances),
        "duration_sec": config.duration_sec, "speakers": speakers, "results": rows,
    }


def print_table(report: dict) -> None:
    print(
        f"conversations={report['conversations']} words={report['words']} "
        f"utterances={report['utterances']} chunk={report['duration_sec']:g}s speakers={report['speakers']}"
    )
    header = (f"{'tokenizer':<28} {'mode':<14} {'roundtrip':>9} {'tok/syl':>7} {'p99':>4} "
              f"{'tok/s':>6} {'byte%':>6} {'overflow':>12} {'oob':>4} {'lost_h':>7} {'dens_max':>8}")
    print(header)
    print("-" * len(header))
    for row in report["results"]:
        syl, over = row["tokens_per_syllable"], row["overflow"]
        print(
            f"{row['tokenizer'][:28]:<28} {row['mode']:<14} {row['roundtrip']['exact_rate']:>8.1%} "
            f"{syl['mean']:>7.2f} {syl['p99']:>4.0f} {syl['agent_tokens_per_speech_sec']:>6.2f} "
            f"{row['byte_fallback']['token_rate']:>6.2%} "
            f"{over['text_overflow']:>5}/{over['chunks']:<6} {over['out_of_bounds']:>4} {over['lost_hours']:>7.2f} "
            f"{over['token_density_max']:>8.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Vietnamese tokenization for PersonaPlex training.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tokenizer", action="append", default=None,
                        help="SentencePiece .model; repeat to compare. Default: <model.root>/tokenizer_spm_32k_3.model")
    parser.add_argument("--translated", action="append", default=[],
                        help="SentencePiece .model loaded through the PersonaPlex ID translation layer "
                             "(model.text_tokenizer=vit5); repeat to compare.")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--speakers", choices=("agent", "all"), default="agent",
                        help="Word statistics over agent text (what is trained) or all speakers.")
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("outputs/tokenizer_benchmark/report.json"))
    args, unknown = parser.parse_known_args()
    config = load_config(args.config, overrides=[value for value in unknown if "=" in value])

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for mode in modes:
        normalize_vietnamese_text("", mode)  # Fail fast on an unknown mode.
    paths = [Path(path) for path in (args.tokenizer or [Path(config.model_root) / "tokenizer_spm_32k_3.model"])]
    for path in paths:
        if not path.is_file():
            print(f"FAIL: tokenizer not found: {path}", file=sys.stderr)
            return 1
    tokenizers = {path.name: SentencePieceTokenizer(path) for path in paths}
    for path in map(Path, args.translated):
        tokenizers[f"{path.name}[translated]"] = TranslatedTokenizer(path)

    conversations = PreparedDataset(config.manifest, config.window_seconds).load()
    if args.max_conversations is not None:
        conversations = limit_conversations(conversations, sample_number=args.max_conversations)
    report = benchmark(config, conversations, tokenizers, modes, args.speakers, args.num_workers)
    report["manifest"] = str(config.manifest)
    print_table(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
