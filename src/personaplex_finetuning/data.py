"""Strict reader for externally prepared OtoSpeech conversations."""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


class ValidationError(ValueError):
    """Raised when a prepared sample does not meet the training contract."""


Speaker = Literal["agent", "user"]
_SPEAKER_ALIASES = {
    "agent": "agent", "left": "agent", "a": "agent",
    "user": "user", "right": "user", "b": "user",
}


@dataclass(frozen=True)
class Word:
    speaker: Speaker
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    duration_sec: float


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    conversation_wav: Path
    voice_prompt_wav: Path
    words: tuple[Word, ...]
    text_prompt: str
    metadata: dict[str, Any]
    audio: AudioInfo
    window_start_sec: float
    window_end_sec: float
    agent_channel: int = 0
    user_channel: int = 1


def read_wav_info(path: Path) -> AudioInfo:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
    except (OSError, wave.Error) as exc:
        raise ValidationError(f"cannot read WAV {path}: {exc}") from exc
    if sample_rate <= 0 or frames <= 0:
        raise ValidationError(f"WAV has no audio frames: {path}")
    return AudioInfo(sample_rate, channels, frames / sample_rate)


class PreparedDataset:
    def __init__(self, manifest: str | Path, window_seconds: float = 30.0) -> None:
        self.manifest = Path(manifest).resolve()
        self.window_seconds = window_seconds
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

    def load(self) -> list[PreparedSample]:
        if not self.manifest.is_file():
            raise ValidationError(f"manifest does not exist: {self.manifest}")
        samples: list[PreparedSample] = []
        for line_number, line in enumerate(self.manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid JSONL at line {line_number}: {exc}") from exc
            samples.append(self._load_entry(entry, line_number))
        if not samples:
            raise ValidationError(f"manifest has no samples: {self.manifest}")
        ids = [sample.sample_id for sample in samples]
        if len(ids) != len(set(ids)):
            raise ValidationError("manifest contains duplicate sample_id values")
        return samples

    def _load_entry(self, entry: dict[str, Any], line_number: int) -> PreparedSample:
        sample_id = str(entry.get("sample_id", "")).strip()
        sample_dir = entry.get("sample_dir")
        if not sample_id or not isinstance(sample_dir, str):
            raise ValidationError(f"line {line_number}: sample_id and sample_dir are required")
        sample_dir_path = Path(sample_dir)
        if sample_dir_path.is_absolute():
            raise ValidationError(f"{sample_id}: sample_dir must be relative to the prepared directory")
        root = (self.manifest.parent / sample_dir_path).resolve()
        try:
            root.relative_to(self.manifest.parent)
        except ValueError as exc:
            raise ValidationError(f"{sample_id}: sample_dir must stay within the prepared directory") from exc
        conversation = root / "conversation.wav"
        voice_prompt = root / "voice_prompt.wav"
        words_path = root / "words.json"
        metadata_path = root / "metadata.json"
        for path in (conversation, voice_prompt, words_path, metadata_path):
            if not path.is_file():
                raise ValidationError(f"{sample_id}: required file missing: {path}")
        audio = read_wav_info(conversation)
        if audio.channels != 2:
            raise ValidationError(f"{sample_id}: conversation.wav must have exactly 2 channels")
        prompt_audio = read_wav_info(voice_prompt)
        if prompt_audio.duration_sec <= 0:
            raise ValidationError(f"{sample_id}: voice_prompt.wav is empty")
        metadata = self._read_object(metadata_path, sample_id)
        if metadata.get("agent_channel", "left").lower() != "left":
            raise ValidationError(f"{sample_id}: agent_channel must be left")
        if metadata.get("user_channel", "right").lower() != "right":
            raise ValidationError(f"{sample_id}: user_channel must be right")
        text_prompt = str(metadata.get("text_prompt", "")).strip()
        if not text_prompt:
            raise ValidationError(f"{sample_id}: metadata.text_prompt is required")
        words = self._read_words(words_path, sample_id, audio.duration_sec)
        agent_words = [word for word in words if word.speaker == "agent"]
        if not agent_words:
            raise ValidationError(f"{sample_id}: no agent words")
        start = agent_words[0].start
        end = min(audio.duration_sec, start + self.window_seconds)
        return PreparedSample(
            sample_id=sample_id,
            conversation_wav=conversation,
            voice_prompt_wav=voice_prompt,
            words=tuple(words),
            text_prompt=text_prompt,
            metadata=metadata,
            audio=audio,
            window_start_sec=start,
            window_end_sec=end,
        )

    @staticmethod
    def _read_object(path: Path, sample_id: str) -> dict[str, Any]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{sample_id}: invalid {path.name}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValidationError(f"{sample_id}: {path.name} must be a JSON object")
        return parsed

    @staticmethod
    def _read_words(path: Path, sample_id: str, duration_sec: float) -> list[Word]:
        try:
            raw_words = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{sample_id}: invalid words.json: {exc}") from exc
        if not isinstance(raw_words, list):
            raise ValidationError(f"{sample_id}: words.json must be a JSON array")
        words: list[Word] = []
        last_start = -1.0
        for index, raw in enumerate(raw_words):
            if not isinstance(raw, dict):
                raise ValidationError(f"{sample_id}: word {index} is not an object")
            speaker = _SPEAKER_ALIASES.get(str(raw.get("speaker", "")).lower())
            word = str(raw.get("word", "")).strip()
            try:
                start, end = float(raw["start"]), float(raw["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(f"{sample_id}: word {index} has invalid timestamps") from exc
            if speaker is None or not word:
                raise ValidationError(f"{sample_id}: word {index} has invalid speaker or text")
            if start < 0 or end <= start or end > duration_sec + 0.05:
                raise ValidationError(f"{sample_id}: word {index} is outside audio bounds")
            if start < last_start:
                raise ValidationError(f"{sample_id}: words are not sorted by start")
            last_start = start
            words.append(Word(speaker, word, start, end))
        return words
