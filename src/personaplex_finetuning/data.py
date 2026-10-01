"""Strict reader for externally prepared OtoSpeech conversations."""

from __future__ import annotations

import json
import logging
import multiprocessing
import wave
import dataclasses
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal


class ValidationError(ValueError):
    """Raised when a prepared sample does not meet the training contract."""


class OutOfBoundsError(ValidationError):
    """Raised when a word timestamp extends beyond its source audio."""


@dataclass(frozen=True)
class DatasetLoadReport:
    manifest_entries: int = 0
    loaded_samples: int = 0
    skipped_out_of_bounds: int = 0
    skipped_invalid: int = 0
    rejected_entries: tuple[str, ...] = ()


logger = logging.getLogger(__name__)


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

    def sample_window(self, window_seconds: float) -> tuple[float, float]:
        """Return the deterministic window beginning at the first agent word."""
        agent_words = [word for word in self.words if word.speaker == "agent"]
        if not agent_words:
            start = 0.0
            end = min(self.audio.duration_sec, start + window_seconds)
            return start, end

        start = agent_words[0].start
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

    def dynamic_sample(self, window_seconds: float, prompt_aug_prob: float = 0.3, rng=None) -> PreparedSample:
        """Return a deterministically sliced sample with optional prompt augmentation."""
        start, end = self.sample_window(window_seconds)
        prompt = self.get_augmented_prompt(prompt_aug_prob=prompt_aug_prob, rng=rng)
        return self.with_window(start, end, prompt)


def conversation_group_keys(sample: PreparedSample) -> tuple[str, ...]:
    conversation_id = str(sample.metadata.get("conversation_id", "")).strip()
    keys = [f"wav:{sample.conversation_wav.resolve()}"]
    if conversation_id:
        keys.append(f"id:{conversation_id}")
    return tuple(keys)


def limit_conversations(
    samples: list[PreparedSample],
    sample_number: int | None = None,
    sample_index: int | None = None,
) -> list[PreparedSample]:
    """Select unique conversations by sample_index or first N samples, before chunking."""
    if sample_number is not None and sample_number < 1:
        raise ValueError("sample_number must be positive or None")
    if sample_index is not None and sample_index < 0:
        raise ValueError("sample_index must be non-negative or None")

    # Deduplicate conversations while preserving order
    unique_conversations: list[PreparedSample] = []
    seen_keys: set[str] = set()
    for sample in samples:
        keys = conversation_group_keys(sample)
        if any(key in seen_keys for key in keys):
            continue
        seen_keys.update(keys)
        unique_conversations.append(sample)

    if sample_index is not None:
        if sample_index >= len(unique_conversations):
            raise IndexError(
                f"sample_index={sample_index} is out of bounds for dataset with {len(unique_conversations)} unique conversations"
            )
        return [unique_conversations[sample_index]]

    if sample_number is not None:
        return unique_conversations[:sample_number]

    return unique_conversations


def turn_aware_chunks(
    samples: list[PreparedSample], target_seconds: float = 25.0,
    min_seconds: float = 10.0, max_seconds: float = 30.0,
    rng=None, randomize: bool = False,
) -> list[PreparedSample]:
    """Partition each conversation at turn, sentence, then word boundaries."""
    if not 0 < min_seconds <= target_seconds <= max_seconds:
        raise ValueError("chunk durations must satisfy 0 < min <= target <= max")
    import random
    rng = rng or random.Random(0)
    chunks: list[PreparedSample] = []
    for sample in samples:
        start = 0.0
        duration = sample.audio.duration_sec
        words = sorted(sample.words, key=lambda item: (item.start, item.end))
        word_boundaries = sorted({min(duration, word.end) for word in words if start < word.end < duration})
        sentence_boundaries = sorted({
            min(duration, word.end) for word in words
            if word.end < duration and word.word.rstrip().endswith((".", "?", "!", "…"))
        })
        turn_boundaries = sorted({
            min(duration, word.end) for index, word in enumerate(words[:-1])
            if words[index + 1].speaker != word.speaker and word.end < duration
        })
        while start < duration:
            remaining = duration - start
            if remaining <= max_seconds:
                end = duration
            else:
                target = rng.uniform(max(min_seconds, target_seconds - 3), min(max_seconds, target_seconds + 3)) if randomize else target_seconds
                end = None
                for boundaries in (turn_boundaries, sentence_boundaries, word_boundaries):
                    candidates = [
                        point for point in boundaries
                        if start + max(min_seconds, target - 3) <= point <= start + min(max_seconds, target + 3)
                    ]
                    if candidates:
                        distance = min(abs(point - (start + target)) for point in candidates)
                        nearest = [point for point in candidates if abs(point - (start + target)) <= distance + 1.0]
                        end = rng.choice(nearest) if randomize else min(candidates, key=lambda point: (abs(point - start - target), -point))
                        break
                if end is None:
                    # No preferred boundary is close to target; use the nearest
                    # word boundary within limits, then the hard duration cap.
                    candidates = [point for point in word_boundaries if start + min_seconds <= point <= start + max_seconds]
                    if candidates:
                        end = min(candidates, key=lambda point: (abs(point - start - target), point))
                    else:
                        end = min(duration, start + max_seconds)
            if end <= start:
                raise AssertionError("chunk boundary did not advance")
            chunks.append(sample.with_window(start, end))
            start = end
    if not chunks:
        raise ValidationError("no contiguous chunks were created")
    return chunks


def fixed_chunks(samples: list[PreparedSample], window_seconds: float) -> list[PreparedSample]:
    """Fixed, clipped chunks for deterministic debugging only."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    chunks = []
    for sample in samples:
        start = 0.0
        while start < sample.audio.duration_sec:
            end = min(sample.audio.duration_sec, start + window_seconds)
            chunks.append(sample.with_window(start, end))
            start = end
    if not chunks:
        raise ValidationError("no chunks were created")
    return chunks


def duration_chunks(samples: list[PreparedSample], duration_sec: float) -> list[PreparedSample]:
    """Split each full conversation into contiguous fixed-duration windows.

    The final window retains its target end beyond the source audio duration so
    the audio reader can zero-pad it to the same Mimi frame length as other
    chunks. Valid-audio masks are derived from the source duration downstream.
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    chunks: list[PreparedSample] = []
    for sample in samples:
        if sample.audio.duration_sec <= 0:
            raise ValidationError(f"{sample.sample_id}: conversation audio is empty")
        start = 0.0
        while start < sample.audio.duration_sec:
            chunks.append(sample.with_window(start, start + duration_sec))
            start += duration_sec
    if not chunks:
        raise ValidationError("no fixed-duration chunks were created")
    return chunks


def contiguous_chunks(samples: list[PreparedSample], window_seconds: float) -> list[PreparedSample]:
    """Compatibility name for fixed debug chunking; windows never exceed real audio."""
    return fixed_chunks(samples, window_seconds)


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
    return sample_for_role_pass(sample, pass_index, swap_roles)


def sample_for_role_pass(sample: PreparedSample, pass_index: int, swap_roles: bool) -> PreparedSample:
    """Use LEFT as agent on even passes and RIGHT as agent on odd passes."""
    if pass_index < 0:
        raise ValueError("pass_index must be non-negative")
    return sample.swapped_roles() if swap_roles and pass_index % 2 else sample


def samples_for_role_pass(
    samples: list[PreparedSample], pass_index: int, swap_roles: bool
) -> list[PreparedSample]:
    """Validate and materialize one pass's agent/user interpretation."""
    if pass_index < 0:
        raise ValueError("pass_index must be non-negative")
    if not swap_roles or pass_index % 2 == 0:
        return samples
    return [sample_for_role_pass(sample, pass_index, True) for sample in samples]


def read_wav_info(path: Path) -> AudioInfo:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            if frames > 0:
                wav.setpos(frames - 1)
                if len(wav.readframes(1)) != channels * wav.getsampwidth():
                    raise ValidationError(
                        f"WAV header declares {frames} frames but audio payload is truncated: {path}"
                    )
    except (OSError, wave.Error) as exc:
        raise ValidationError(f"cannot read WAV {path}: {exc}") from exc
    if sample_rate <= 0 or frames <= 0:
        raise ValidationError(f"WAV has no audio frames: {path}")
    return AudioInfo(sample_rate, channels, frames / sample_rate)


def read_stereo_window(path: Path, start_sec: float, end_sec: float, sample_rate: int, sample_id: str):
    """Decode one real stereo window; never turn a failed seek into padded silence."""
    import numpy as np
    import sphn

    duration_sec = end_sec - start_sec
    if start_sec < 0 or duration_sec <= 0:
        raise ValueError(f"{sample_id}: invalid audio window {start_sec:.6f}-{end_sec:.6f}s")

    def channels_first(value):
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 2 and value.shape[0] != 2 and value.shape[1] == 2:
            value = value.T
        return value

    audio, decoded_rate = sphn.read(
        str(path), start_sec=start_sec, duration_sec=duration_sec, sample_rate=sample_rate,
    )
    audio = channels_first(audio)
    if audio.ndim == 2 and audio.shape == (2, 0):
        # sphn can return an empty time slice near a WAV boundary even when
        # the complete file decodes. Retry by slicing the decoded waveform.
        full_audio, decoded_rate = sphn.read(str(path), sample_rate=sample_rate)
        full_audio = channels_first(full_audio)
        if decoded_rate <= 0 or full_audio.ndim != 2 or full_audio.shape[0] != 2:
            raise ValueError(f"{sample_id}: cannot decode stereo WAV {path}; got {full_audio.shape}")
        start_frame = round(start_sec * decoded_rate)
        end_frame = start_frame + round(duration_sec * decoded_rate)
        audio = full_audio[:, start_frame:end_frame]
        if audio.shape[-1] == 0:
            raise ValueError(
                f"{sample_id}: requested WAV window {start_sec:.6f}-{end_sec:.6f}s "
                f"lies beyond decoded audio ({full_audio.shape[-1] / decoded_rate:.6f}s): {path}"
            )
    if decoded_rate != sample_rate or audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[-1] == 0:
        raise ValueError(
            f"{sample_id}: decoded audio must be stereo [2,T] at {sample_rate} Hz; "
            f"got {audio.shape} at {decoded_rate} Hz: {path}"
        )
    return audio


_PREPARED_SAMPLE_WORKER = None


def _prepared_sample_rejection(line_number: int, sample_id: str | None, error: ValidationError) -> str:
    detail = f"line {line_number}"
    if sample_id:
        detail += f" ({sample_id})"
    return f"{detail}: {error}"


def _initialize_prepared_sample_worker(manifest: str, window_seconds: float | None) -> None:
    global _PREPARED_SAMPLE_WORKER
    _PREPARED_SAMPLE_WORKER = PreparedDataset(manifest, window_seconds)


def _load_prepared_sample_worker(item: tuple[int, dict[str, Any]]):
    line_number, entry = item
    sample_id = entry.get("sample_id") if isinstance(entry, dict) else None
    try:
        return _PREPARED_SAMPLE_WORKER._load_entry(entry, line_number), None, False
    except ValidationError as exc:
        return (
            None,
            _prepared_sample_rejection(line_number, sample_id, exc),
            isinstance(exc, OutOfBoundsError),
        )


class PreparedDataset:
    def __init__(
        self,
        manifest: str | Path,
        window_seconds: float | None = None,
        filter_num_workers: int = 1,
        force_filter: bool = False,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        self.window_seconds = window_seconds
        if filter_num_workers < 1:
            raise ValueError("filter_num_workers must be positive")
        self.filter_num_workers = filter_num_workers
        self.force_filter = force_filter
        self.load_report = DatasetLoadReport()
        if window_seconds is not None and window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

    def load(self) -> list[PreparedSample]:
        if not self.manifest.is_file():
            raise ValidationError(f"manifest does not exist: {self.manifest}")
        from .filter_cache import filter_fingerprint, load_filter_manifest, save_filter_manifest

        cache_path = self.manifest.parent / ".filter-cache" / f"prepared-{self.manifest.stem}.jsonl"
        fingerprint = filter_fingerprint(
            [self.manifest], {"kind": "prepared-samples", "window_seconds": self.window_seconds},
        )
        if not self.force_filter:
            cached = load_filter_manifest(cache_path, fingerprint)
            if cached is not None and "samples" in cached and "report" in cached:
                raw_report = cached["report"]
                self.load_report = DatasetLoadReport(
                    manifest_entries=raw_report["manifest_entries"],
                    loaded_samples=raw_report["loaded_samples"],
                    skipped_out_of_bounds=raw_report["skipped_out_of_bounds"],
                    skipped_invalid=raw_report["skipped_invalid"],
                    rejected_entries=tuple(raw_report["rejected_entries"]),
                )
                print(
                    f"[Prepared sample filter workers] manifest={self.manifest} "
                    f"configured={self.filter_num_workers} active=0 mode=cache",
                    flush=True,
                )
                print(
                    f"[Prepared sample filter cache] hit path={cache_path} samples={len(cached['samples'])}",
                    flush=True,
                )
                if not cached["samples"]:
                    raise ValidationError(
                        f"no valid samples in {self.manifest}; skipped "
                        f"{self.load_report.skipped_out_of_bounds} out-of-bounds and "
                        f"{self.load_report.skipped_invalid} invalid entries"
                    )
                return cached["samples"]
        samples: list[PreparedSample] = []
        rejected: list[str] = []
        skipped_out_of_bounds = 0
        skipped_invalid = 0
        manifest_entries = 0
        entries: list[tuple[int, dict[str, Any]]] = []
        for line_number, line in enumerate(self.manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            manifest_entries += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid JSONL at line {line_number}: {exc}") from exc
            entries.append((line_number, entry))

        worker_chunksize = 8
        task_count = math.ceil(len(entries) / worker_chunksize)
        use_process_pool = self.filter_num_workers > 1 and task_count >= 2
        active_workers = min(self.filter_num_workers, task_count) if use_process_pool else 1
        print(
            f"[Prepared sample filter workers] manifest={self.manifest} entries={len(entries)} "
            f"configured={self.filter_num_workers} active={active_workers} "
            f"mode={'processes' if use_process_pool else 'sequential'}",
            flush=True,
        )
        if use_process_pool:
            with ProcessPoolExecutor(
                max_workers=active_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_prepared_sample_worker,
                initargs=(str(self.manifest), self.window_seconds),
            ) as executor:
                outcomes = executor.map(
                    _load_prepared_sample_worker, entries, chunksize=worker_chunksize,
                )
                for outcome in outcomes:
                    sample, detail, out_of_bounds = outcome
                    if sample is not None:
                        samples.append(sample)
                    else:
                        rejected.append(detail)
                        if out_of_bounds:
                            skipped_out_of_bounds += 1
                        else:
                            skipped_invalid += 1
                        logger.warning("Skipping invalid prepared sample: %s", detail)
        else:
            for line_number, entry in entries:
                sample_id = entry.get("sample_id") if isinstance(entry, dict) else None
                try:
                    samples.append(self._load_entry(entry, line_number))
                except ValidationError as exc:
                    detail = _prepared_sample_rejection(line_number, sample_id, exc)
                    rejected.append(detail)
                    if isinstance(exc, OutOfBoundsError):
                        skipped_out_of_bounds += 1
                    else:
                        skipped_invalid += 1
                    logger.warning("Skipping invalid prepared sample: %s", detail)
        self.load_report = DatasetLoadReport(
            manifest_entries=manifest_entries,
            loaded_samples=len(samples),
            skipped_out_of_bounds=skipped_out_of_bounds,
            skipped_invalid=skipped_invalid,
            rejected_entries=tuple(rejected),
        )
        ids = [sample.sample_id for sample in samples]
        if len(ids) != len(set(ids)):
            raise ValidationError("manifest contains duplicate sample_id values")
        save_filter_manifest(cache_path, fingerprint, {
            "samples": samples,
            "report": {
                "manifest_entries": manifest_entries,
                "loaded_samples": len(samples),
                "skipped_out_of_bounds": skipped_out_of_bounds,
                "skipped_invalid": skipped_invalid,
                "rejected_entries": rejected,
            },
        })
        print(
            f"[Prepared sample filter cache] {'forced rebuild' if self.force_filter else 'built'} "
            f"path={cache_path} samples={len(samples)}", flush=True,
        )
        if not samples:
            if rejected:
                raise ValidationError(
                    f"manifest has no valid samples: {self.manifest}; "
                    f"rejected entries: {'; '.join(rejected)}"
                )
            raise ValidationError(f"manifest has no samples: {self.manifest}")
        return samples

    def split(self, val_ratio: float = 0.05, seed: int = 42) -> tuple[list[PreparedSample], list[PreparedSample]]:
        """Split by conversation before any temporal chunking."""
        samples = self.load()
        parents = list(range(len(samples)))

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        owners: dict[str, int] = {}
        for index, sample in enumerate(samples):
            for key in conversation_group_keys(sample):
                previous = owners.setdefault(key, index)
                left, right = find(index), find(previous)
                parents[left] = right
        groups: dict[int, list[PreparedSample]] = {}
        for index, sample in enumerate(samples):
            groups.setdefault(find(index), []).append(sample)
        if len(groups) <= 1 or val_ratio <= 0.0:
            return samples, []
        import random
        rng = random.Random(seed)
        shuffled = list(groups)
        rng.shuffle(shuffled)
        val_size = max(1, min(len(shuffled) - 1, round(len(shuffled) * val_ratio)))
        val_groups = set(shuffled[:val_size])
        train_set = [sample for key, group in groups.items() if key not in val_groups for sample in group]
        val_set = [sample for key, group in groups.items() if key in val_groups for sample in group]
        return train_set, val_set

    def _load_entry(self, entry: dict[str, Any], line_number: int) -> PreparedSample:
        if not isinstance(entry, dict):
            raise ValidationError(f"line {line_number}: manifest entry must be a JSON object")
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
        voice_prompt = root / "voice_prompt_left.wav"
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
            raise ValidationError(f"{sample_id}: voice_prompt_left.wav is empty")
        metadata = self._read_object(metadata_path, sample_id)
        if metadata.get("agent_channel", "left").lower() != "left":
            raise ValidationError(f"{sample_id}: agent_channel must be left")
        if metadata.get("user_channel", "right").lower() != "right":
            raise ValidationError(f"{sample_id}: user_channel must be right")
        text_prompt = str(metadata.get("text_prompt_left", "")).strip()
        if not text_prompt:
            raise ValidationError(f"{sample_id}: metadata.text_prompt_left is required")
        words = self._read_words(words_path, sample_id, audio.duration_sec)
        agent_words = [word for word in words if word.speaker == "agent"]
        if not agent_words:
            raise ValidationError(f"{sample_id}: no agent words")
        # Full-conversation loading is required before duration-based chunking.
        # An explicit window remains available for small/debug dataset reads.
        start = 0.0 if self.window_seconds is None else agent_words[0].start
        end = (
            audio.duration_sec
            if self.window_seconds is None
            else min(audio.duration_sec, start + self.window_seconds)
        )
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
                raise OutOfBoundsError(f"{sample_id}: word {index} is outside audio bounds")
            if start < last_start:
                raise ValidationError(f"{sample_id}: words are not sorted by start")
            last_start = start
            words.append(Word(speaker, word, start, end))
        return words
