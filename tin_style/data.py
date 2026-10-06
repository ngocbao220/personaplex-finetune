"""Translate prepared OtoSpeech data at the reference loader boundary only.

No alignment, cropping, tokenization, audio encoding or prompt construction
is performed here. Those operations remain owned by the reference pipeline.
"""

from __future__ import annotations

import json
import math
import logging
import tempfile
import wave
from numbers import Integral
from functools import lru_cache
from pathlib import Path


def _audio_info(path: Path, channels: int) -> float:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != channels:
            raise ValueError(f"{path}: expected {channels} audio channels")
        duration = audio.getnframes() / audio.getframerate()
    if duration <= 0:
        raise ValueError(f"{path}: empty audio")
    return duration


def prepared_sidecar(directory: Path) -> dict:
    """Read a prepared sample into the reference sidecar schema in memory."""
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    # Current flat prepared format defines roles by channel, like the legacy
    # reader. Older explicit-role samples must still declare their mapping.
    flat = metadata.get("layout") == "flat"
    agent_channel = metadata.get("agent_channel", "left" if flat else None)
    user_channel = metadata.get("user_channel", "right" if flat else None)
    if agent_channel != "left" or user_channel != "right":
        raise ValueError(f"{directory}: require LEFT=agent and RIGHT=user")
    prompt = metadata.get("text_prompt_left" if flat else "text_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{directory}: metadata.text_prompt must be non-empty")
    duration = _audio_info(directory / "conversation.wav", 2)
    voice_path = directory / ("voice_prompt_left.wav" if flat else "voice_prompt.wav")
    _audio_info(voice_path, 1)
    words = json.loads((directory / "words.json").read_text(encoding="utf-8"))
    if not isinstance(words, list) or not words:
        raise ValueError(f"{directory}: words.json must be a non-empty list")
    speakers = {"agent": "SPEAKER_BROKER", "user": "SPEAKER_CLIENT"}
    alignments = []
    invalid_words = []
    for index, word in enumerate(words):
        if word.get("speaker") not in speakers:
            raise ValueError(f"{directory}: invalid speaker at word {index}")
        text = word.get("word")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{directory}: empty word at index {index}")
        start, end = float(word["start"]), float(word["end"])
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"{directory}: invalid timestamps at word {index}: {start}, {end}")
        if not 0 <= start < end <= duration + 0.05:
            invalid_words.append({"index": index, "start": start, "end": end})
        alignments.append([text, [start, end], speakers[word["speaker"]]])
    # The reference binary search expects timestamp-sorted alignments.
    alignments.sort(key=lambda item: item[1][0])
    return {
        "text_prompt": prompt,
        "voice_prompt": str(voice_path.resolve()),
        "alignments": alignments,
        "invalid_words": invalid_words,
        "duration": duration,
    }


@lru_cache(maxsize=1024)
def load_sidecar(wav_path: str) -> dict:
    path = Path(wav_path)
    if path.name == "conversation.wav" and (path.parent / "metadata.json").is_file():
        return prepared_sidecar(path.parent)
    # Native reference datasets remain supported without semantic changes.
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


# Temporary native manifests live for the process lifetime, not in the dataset.
_MANIFEST_DIRECTORY = tempfile.TemporaryDirectory(prefix="tin-style-data-")


def filter_audio_chunks(dataset, mimi):
    """Check actual unpadded sphn lengths against the loaded SEANet encoder."""
    hop = getattr(getattr(mimi, "encoder", None), "hop_length", None)
    if isinstance(hop, bool) or not isinstance(hop, Integral) or hop <= 0:
        raise ValueError("Mimi encoder must expose a positive integer hop_length")
    for sample in dataset:
        length = sample["data"][..., : sample["unpadded_len"]].shape[-1]
        if length <= 0 or length % hop:
            logging.getLogger(__name__).warning(
                "Skipping invalid audio-length chunk: path=%s start_sec=%.6f "
                "samples=%d encoder_hop=%d remainder=%d",
                sample["path"], sample["start_time_sec"], length, hop, length % hop,
            )
            continue
        yield sample


def chunk_rejections(wav_path: str, start: float, step: float) -> list[dict]:
    """Reject windows touching invalid words; never repair transcript timestamps.

    Out-of-audio endpoints are mapped to the edge chunk for filtering only.
    Native reference sidecars without invalid_words remain unchanged.
    """
    if not math.isfinite(step) or step <= 0:
        raise ValueError("chunk step must be finite and positive")
    data = load_sidecar(wav_path)
    rejected = []
    for word in data.get("invalid_words", []):
        duration = data["duration"]
        low = max(0.0, min(duration, min(word["start"], word["end"])))
        high = max(0.0, min(duration, max(word["start"], word["end"])))
        if low == high:
            overlaps = start <= low < start + step or (low == duration and start < duration <= start + step)
        else:
            overlaps = low < start + step and high > start
        if overlaps:
            rejected.append(word)
    return rejected


@lru_cache(maxsize=128)
def validate_chunk_manifest(manifest: str, step: float) -> None:
    """Preflight every window, log exclusions, fail instead of empty infinite epochs."""
    if not math.isfinite(step) or step <= 0:
        raise ValueError("chunk step must be finite and positive")
    kept = dropped = 0
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        start = 0.0
        while start < row["duration"]:
            rejected = chunk_rejections(row["path"], start, step)
            if rejected:
                dropped += 1
                logging.getLogger(__name__).warning(
                    "Skipping invalid chunk: path=%s window=[%.6f, %.6f) words=%s",
                    row["path"], start, start + step, rejected,
                )
            else:
                kept += 1
            start += step
    print(f"[tin-style chunks] kept={kept} dropped={dropped} step={step}", flush=True)
    if not kept:
        raise ValueError(f"{manifest}: no valid chunks remain")


@lru_cache(maxsize=128)
def native_manifest(manifest: str) -> str:
    """Supply sphn with its native path/duration schema; leave chunking intact."""
    path = Path(manifest).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    if all("path" in row and "duration" in row for row in rows):
        return str(path)
    converted = []
    for row in rows:
        if not isinstance(row.get("sample_dir"), str) or not row["sample_dir"]:
            raise ValueError(f"{path}: prepared entry requires sample_dir")
        directory = (path.parent / row["sample_dir"]).resolve()
        # Validate all prepared assets before the reference loader starts.
        sidecar = load_sidecar(str(directory / "conversation.wav"))
        converted.append({
            "path": str(directory / "conversation.wav"),
            "duration": _audio_info(directory / "conversation.wav", 2),
        })
        if len(converted) == 1:
            print(f"[tin-style data] sample={row.get('sample_id', directory.name)} "
                  f"LEFT=agent RIGHT=user words={len(sidecar['alignments'])} "
                  f"voice_prompt={sidecar['voice_prompt']}", flush=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".jsonl",
                                     dir=_MANIFEST_DIRECTORY.name, delete=False) as output:
        for row in converted:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        return output.name