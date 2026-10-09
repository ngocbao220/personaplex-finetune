"""Versioned JSONL cache for validated prepared samples and filtered chunks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .data import AudioInfo, PreparedSample, Word


FILTER_CACHE_SCHEMA = 1


def _stat_record(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


_ASSET_CACHE: dict[tuple, list[dict[str, Any]]] = {}


def _manifest_assets(manifest: Path) -> list[dict[str, Any]]:
    """Return inexpensive file fingerprints for assets referenced by JSONL.

    Memoized per process on the manifest's own stat: one startup fingerprints
    the dataset and each chunk split, and re-stating every WAV on network
    storage dominated startup time.
    """
    if not manifest.is_file():
        return []
    manifest_stat = manifest.stat()
    memo_key = (str(manifest.resolve()), manifest_stat.st_size, manifest_stat.st_mtime_ns)
    if memo_key not in _ASSET_CACHE:
        _ASSET_CACHE[memo_key] = _scan_manifest_assets(manifest)
    return _ASSET_CACHE[memo_key]


def _scan_manifest_assets(manifest: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    root_dir = manifest.parent.resolve()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # The manifest content hash still detects this malformed input.
        sample_dir = entry.get("sample_dir") if isinstance(entry, dict) else None
        if not isinstance(sample_dir, str) or Path(sample_dir).is_absolute():
            records.append({"line": line_number, "invalid_sample_dir": sample_dir})
            continue
        sample_root = (root_dir / sample_dir).resolve()
        try:
            sample_root.relative_to(root_dir)
        except ValueError:
            records.append({"line": line_number, "invalid_sample_dir": sample_dir})
            continue
        records.append({
            "line": line_number,
            "assets": [
                _stat_record(sample_root / name)
                for name in (
                    "conversation.wav",
                    "voice_prompt_left.wav",
                    "voice_prompt_right.wav",
                    "words.json",
                    "metadata.json",
                )
            ],
        })
    return records


def filter_fingerprint(
    manifests: list[Path], options: dict[str, Any], tokenizer_path: Path | None = None,
) -> str:
    sources = []
    for manifest in manifests:
        manifest = Path(manifest).expanduser().resolve()
        try:
            content_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        except OSError:
            content_hash = None
        sources.append({
            "manifest": str(manifest),
            "content_sha256": content_hash,
            "manifest_stat": _stat_record(manifest),
            "assets": _manifest_assets(manifest),
        })
    tokenizer = _stat_record(tokenizer_path) if tokenizer_path is not None else None
    payload = {
        "schema": FILTER_CACHE_SCHEMA,
        "sources": sources,
        "options": options,
        "tokenizer": tokenizer,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _encode_sample(sample: PreparedSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "conversation_wav": str(sample.conversation_wav),
        "voice_prompt_wav": str(sample.voice_prompt_wav),
        "voice_prompt_right_wav": (
            str(sample.voice_prompt_right_wav) if sample.voice_prompt_right_wav is not None else None
        ),
        # Columnar words: ~2x faster to parse and ~40% smaller than one dict per word.
        "words_columns": {
            "speaker": [word.speaker for word in sample.words],
            "word": [word.word for word in sample.words],
            "start": [word.start for word in sample.words],
            "end": [word.end for word in sample.words],
        },
        "text_prompt": sample.text_prompt,
        "text_prompt_right": sample.text_prompt_right,
        "metadata": sample.metadata,
        "audio": {
            "sample_rate": sample.audio.sample_rate,
            "channels": sample.audio.channels,
            "duration_sec": sample.audio.duration_sec,
        },
        "window_start_sec": sample.window_start_sec,
        "window_end_sec": sample.window_end_sec,
        "agent_channel": sample.agent_channel,
        "user_channel": sample.user_channel,
    }


def _decode_words(raw: dict[str, Any]) -> tuple[Word, ...]:
    columns = raw.get("words_columns")
    if columns is not None:
        return tuple(map(Word, columns["speaker"], columns["word"], columns["start"], columns["end"]))
    # Caches written before the columnar layout.
    return tuple(
        Word(item["speaker"], item["word"], float(item["start"]), float(item["end"]))
        for item in raw["words"]
    )


def _decode_sample(raw: dict[str, Any]) -> PreparedSample:
    audio = raw["audio"]
    prompt_right = raw.get("voice_prompt_right_wav")
    return PreparedSample(
        sample_id=raw["sample_id"],
        conversation_wav=Path(raw["conversation_wav"]),
        voice_prompt_wav=Path(raw["voice_prompt_wav"]),
        voice_prompt_right_wav=Path(prompt_right) if prompt_right else None,
        words=_decode_words(raw),
        text_prompt=raw["text_prompt"],
        text_prompt_right=raw.get("text_prompt_right"),
        metadata=raw["metadata"],
        audio=AudioInfo(
            int(audio["sample_rate"]), int(audio["channels"]), float(audio["duration_sec"]),
        ),
        window_start_sec=float(raw["window_start_sec"]),
        window_end_sec=float(raw["window_end_sec"]),
        agent_channel=int(raw["agent_channel"]),
        user_channel=int(raw["user_channel"]),
    )


def load_filter_manifest(path: Path, fingerprint: str) -> dict[str, Any] | None:
    """Load a complete matching cache, returning None for stale or damaged files."""
    try:
        with Path(path).open(encoding="utf-8") as source:
            first = json.loads(next(source))
            if not isinstance(first, dict):
                return None
            if (
                first.get("record") != "header"
                or first.get("schema") != FILTER_CACHE_SCHEMA
                or first.get("fingerprint") != fingerprint
            ):
                return None
            payload: dict[str, Any] = {}
            complete = False
            for line in source:
                record = json.loads(line)
                if not isinstance(record, dict):
                    return None
                kind = record.get("record")
                if kind == "samples":
                    payload[record["collection"]] = []
                elif kind == "sample":
                    payload.setdefault(record["collection"], []).append(_decode_sample(record["value"]))
                elif kind == "value":
                    payload[record["key"]] = record["value"]
                elif kind == "complete":
                    complete = True
                    if any(rest.strip() for rest in source):
                        return None
                    break
                else:
                    return None
            return payload if complete else None
    except (OSError, StopIteration, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
        return None


def save_filter_manifest(path: Path, fingerprint: str, payload: dict[str, Any]) -> None:
    """Atomically replace a JSONL cache only after all records are written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(json.dumps({
                "record": "header", "schema": FILTER_CACHE_SCHEMA, "fingerprint": fingerprint,
            }, ensure_ascii=False) + "\n")
            for key, value in payload.items():
                if isinstance(value, (list, tuple)) and all(isinstance(item, PreparedSample) for item in value):
                    output.write(json.dumps({"record": "samples", "collection": key}) + "\n")
                    for sample in value:
                        output.write(json.dumps({
                            "record": "sample", "collection": key, "value": _encode_sample(sample),
                        }, ensure_ascii=False) + "\n")
                else:
                    output.write(json.dumps({
                        "record": "value", "key": key, "value": value,
                    }, ensure_ascii=False) + "\n")
            output.write('{"record":"complete"}\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
