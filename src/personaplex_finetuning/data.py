"""Strict reader for externally prepared OtoSpeech conversations."""

from __future__ import annotations

import json
import wave
import dataclasses
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
    voice_prompt_right_wav: Path | None = None
    text_prompt_right: str | None = None

    def with_window(self, start_sec: float, end_sec: float, text_prompt: str | None = None) -> PreparedSample:
        return dataclasses.replace(
            self,
            window_start_sec=start_sec,
            window_end_sec=end_sec,
            text_prompt=self.text_prompt if text_prompt is None else text_prompt,
        )

    def swapped_roles(self) -> PreparedSample:
        """Use the original right speaker as the logical PersonaPlex agent."""
        if self.voice_prompt_right_wav is None:
            raise ValidationError(
                f"{self.sample_id}: role-swapped training requires voice_prompt_right.wav"
            )
        if not self.text_prompt_right:
            raise ValidationError(
                f"{self.sample_id}: role-swapped training requires metadata.text_prompt_right"
            )
        swapped_words = tuple(
            Word("user" if word.speaker == "agent" else "agent", word.word, word.start, word.end)
            for word in self.words
        )
        return dataclasses.replace(
            self,
            voice_prompt_wav=self.voice_prompt_right_wav,
            text_prompt=self.text_prompt_right,
            words=swapped_words,
            agent_channel=self.user_channel,
            user_channel=self.agent_channel,
        )

    def sample_window(self, window_seconds: float, random_crop: bool = False, rng=None) -> tuple[float, float]:
        """Crop window. If random_crop=True, picks a speech-centered random window across the full audio."""
        agent_words = [word for word in self.words if word.speaker == "agent"]
        if not agent_words:
            start = 0.0
            end = min(self.audio.duration_sec, start + window_seconds)
            return start, end

        if not random_crop or self.audio.duration_sec <= window_seconds:
            start = agent_words[0].start
            end = min(self.audio.duration_sec, start + window_seconds)
            return start, end

        if rng is None:
            import random
            rng = random

        # Pick a random agent word so the window contains active agent dialogue
        target_word = rng.choice(agent_words)
        max_start = max(0.0, self.audio.duration_sec - window_seconds)
        offset = rng.uniform(0.0, min(window_seconds * 0.8, target_word.start))
        start = max(0.0, min(max_start, target_word.start - offset))
        end = min(self.audio.duration_sec, start + window_seconds)
        return start, end

    def get_augmented_prompt(self, prompt_aug_prob: float = 0.3, rng=None) -> str:
        """Sample text prompt from multiple granularity levels in English or Vietnamese."""
        if prompt_aug_prob <= 0.0:
            return self.text_prompt
        if rng is None:
            import random
            rng = random
        if rng.random() > prompt_aug_prob:
            return self.text_prompt

        first_sentence = self.text_prompt.split(". ")[0].strip()
        if not first_sentence.endswith("."):
            first_sentence += "."

        # Detect language (Vietnamese vs English)
        lang = str(self.metadata.get("language", "")).lower()
        vi_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
                       "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ")
        is_vietnamese = lang in ("vi", "vietnamese") or any(c in vi_chars for c in self.text_prompt)

        if is_vietnamese:
            candidates = [
                first_sentence,
                "Bạn thích trò chuyện một cách cởi mở và tự nhiên.",
                "Bạn là một người bạn trò chuyện thân thiện và tự nhiên.",
                "Bạn là một trợ lý trò chuyện thân thiện, luôn lắng nghe và phản hồi tích cực.",
                "Bạn là một người bạn đồng hành trò chuyện hòa nhã và chu đáo.",
                "Bạn đang trò chuyện tự nhiên cùng một người bạn.",
            ]
        else:
            candidates = [
                first_sentence,
                "You enjoy having a good conversation.",
                "You are having a casual and natural conversation.",
                "You are a friendly and engaging conversational partner.",
                "You are a helpful and polite conversational assistant.",
                "You are talking with a partner.",
            ]
        return str(rng.choice(candidates))

    def dynamic_sample(self, window_seconds: float, random_crop: bool = True, prompt_aug_prob: float = 0.3, rng=None) -> PreparedSample:
        """Return a copy of this sample with dynamic window slicing and optional prompt augmentation."""
        start, end = self.sample_window(window_seconds, random_crop=random_crop, rng=rng)
        prompt = self.get_augmented_prompt(prompt_aug_prob=prompt_aug_prob, rng=rng) if random_crop else self.text_prompt
        return self.with_window(start, end, prompt)


def contiguous_chunks(samples: list[PreparedSample], window_seconds: float) -> list[PreparedSample]:
    """Split conversations into fixed consecutive windows, padding the final window at encode time."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    chunks: list[PreparedSample] = []
    for sample in samples:
        start = 0.0
        while start < sample.audio.duration_sec:
            chunks.append(sample.with_window(start, start + window_seconds))
            start += window_seconds
    if not chunks:
        raise ValidationError("no contiguous chunks were created")
    return chunks


def sample_for_training_position(
    chunks: list[PreparedSample], position: int, seed: int, shuffle: bool, swap_roles: bool
) -> PreparedSample:
    """Return one chunk from a deterministic role pass; the second pass swaps speakers."""
    if not chunks or position < 0:
        raise ValueError("chunks must be non-empty and position must be non-negative")
    import random

    chunk_count = len(chunks)
    pass_index, index_within_pass = divmod(position, chunk_count)
    indices = list(range(chunk_count))
    if shuffle:
        random.Random(seed + pass_index).shuffle(indices)
    sample = chunks[indices[index_within_pass]]
    return sample.swapped_roles() if swap_roles and pass_index % 2 else sample


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

    def split(self, val_ratio: float = 0.05, seed: int = 42) -> tuple[list[PreparedSample], list[PreparedSample]]:
        """Split samples into train and validation sets deterministically."""
        samples = self.load()
        if len(samples) <= 1 or val_ratio <= 0.0:
            return samples, []
        import random
        rng = random.Random(seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        val_size = max(1, int(len(samples) * val_ratio))
        val_set = shuffled[:val_size]
        train_set = shuffled[val_size:]
        return train_set, val_set

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
        voice_prompt_right = root / "voice_prompt_right.wav"
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
            voice_prompt_right_wav=voice_prompt_right if voice_prompt_right.is_file() else None,
            text_prompt_right=str(metadata.get("text_prompt_right", "")).strip() or None,
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
