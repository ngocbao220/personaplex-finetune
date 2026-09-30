"""Deterministically exclude chunks whose text targets cannot be aligned."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data import PreparedSample, Word
from .sequence import Tokenizer, align_dialogue_text_targets


@dataclass(frozen=True)
class RejectedChunk:
    sample_id: str
    window_start_sec: float
    window_end_sec: float
    reason: str
    roles: tuple[str, ...]
    word: Word


@dataclass(frozen=True)
class ChunkFilterResult:
    kept: tuple[PreparedSample, ...]
    rejected: tuple[RejectedChunk, ...]

    @property
    def skipped_out_of_bounds(self) -> int:
        return sum(item.reason == "out_of_bounds" for item in self.rejected)

    @property
    def skipped_text_overflow(self) -> int:
        return sum(item.reason == "text_overflow" for item in self.rejected)


def expected_mimi_frames(duration_sec: float, frame_rate: float) -> int:
    if duration_sec <= 0 or frame_rate <= 0:
        raise ValueError("duration_sec and Mimi frame_rate must be positive")
    return math.ceil(duration_sec * frame_rate - 1e-6)


def filter_text_capacity_chunks(
    chunks: list[PreparedSample],
    tokenizer: Tokenizer,
    frame_rate: float,
    normalize_vietnamese_diacritics: bool = False,
    swap_roles: bool = False,
) -> ChunkFilterResult:
    """Keep chunks whose configured agent role views fit without retiming text."""
    if frame_rate <= 0:
        raise ValueError("Mimi frame_rate must be positive")

    kept: list[PreparedSample] = []
    rejected: list[RejectedChunk] = []
    for chunk in chunks:
        duration = chunk.window_end_sec - chunk.window_start_sec
        if duration <= 0:
            raise ValueError(f"{chunk.sample_id}: invalid chunk duration {duration}")
        frames = expected_mimi_frames(duration, frame_rate)
        roles = [("left-agent", chunk)]
        if swap_roles:
            roles.append(("right-agent", chunk.swapped_roles()))

        out_of_bounds = [
            word for word in chunk.words
            if chunk.window_start_sec <= word.start < chunk.window_end_sec
            and (word.start >= chunk.audio.duration_sec or word.end > chunk.audio.duration_sec)
        ]
        if out_of_bounds:
            rejected.append(RejectedChunk(
                chunk.sample_id, chunk.window_start_sec, chunk.window_end_sec,
                "out_of_bounds", tuple(role for role, _ in roles), out_of_bounds[0],
            ))
            continue

        overflow_roles: list[tuple[str, Word]] = []
        for role, role_chunk in roles:
            targets = align_dialogue_text_targets(
                role_chunk, frames, frame_rate, tokenizer,
                normalize_vietnamese_diacritics,
            )
            if targets.overflow_word is not None:
                overflow_roles.append((role, targets.overflow_word))
        if overflow_roles:
            rejected.append(RejectedChunk(
                chunk.sample_id, chunk.window_start_sec, chunk.window_end_sec,
                "text_overflow", tuple(role for role, _ in overflow_roles),
                overflow_roles[0][1],
            ))
        else:
            kept.append(chunk)
    return ChunkFilterResult(tuple(kept), tuple(rejected))
