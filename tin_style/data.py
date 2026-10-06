"""Translate prepared OtoSpeech data at the reference loader boundary only.

No alignment, cropping, tokenization, audio encoding or prompt construction
is performed here. Those operations remain owned by the reference pipeline.
"""

from __future__ import annotations

import json
import math
import tempfile
import wave
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
    for index, word in enumerate(words):
        if word.get("speaker") not in speakers:
            raise ValueError(f"{directory}: invalid speaker at word {index}")
        text = word.get("word")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{directory}: empty word at index {index}")
        start, end = float(word["start"]), float(word["end"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            raise ValueError(f"{directory}: invalid timestamps at word {index}: {start}, {end}")
        alignments.append([text, [start, end], speakers[word["speaker"]]])
    # The reference binary search expects timestamp-sorted alignments.
    alignments.sort(key=lambda item: item[1][0])
    return {
        "text_prompt": prompt,
        "voice_prompt": str(voice_path.resolve()),
        "alignments": alignments,
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