"""Deterministically exclude chunks whose text targets cannot be aligned."""

from __future__ import annotations

import math
import multiprocessing
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

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


def _filter_chunk(
    chunk: PreparedSample,
    tokenizer: Tokenizer,
    frame_rate: float,
    normalize_vietnamese_diacritics: bool,
    swap_roles: bool,
) -> RejectedChunk | None:
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
        return RejectedChunk(
            chunk.sample_id, chunk.window_start_sec, chunk.window_end_sec,
            "out_of_bounds", tuple(role for role, _ in roles), out_of_bounds[0],
        )

    overflow_roles: list[tuple[str, Word]] = []
    for role, role_chunk in roles:
        targets = align_dialogue_text_targets(
            role_chunk, frames, frame_rate, tokenizer,
            normalize_vietnamese_diacritics,
        )
        if targets.overflow_word is not None:
            overflow_roles.append((role, targets.overflow_word))
    if overflow_roles:
        return RejectedChunk(
            chunk.sample_id, chunk.window_start_sec, chunk.window_end_sec,
            "text_overflow", tuple(role for role, _ in overflow_roles),
            overflow_roles[0][1],
        )
    return None


_WORKER_FILTER_ARGS = None


def _initialize_filter_worker(tokenizer, frame_rate, normalize_vietnamese_diacritics, swap_roles):
    global _WORKER_FILTER_ARGS
    _WORKER_FILTER_ARGS = (
        tokenizer, frame_rate, normalize_vietnamese_diacritics, swap_roles,
    )


def _filter_chunk_worker(chunk: PreparedSample) -> RejectedChunk | None:
    tokenizer, frame_rate, normalize_vietnamese_diacritics, swap_roles = _WORKER_FILTER_ARGS
    return _filter_chunk(
        chunk, tokenizer, frame_rate, normalize_vietnamese_diacritics, swap_roles,
    )


def _compact_chunks_for_workers(chunks: list[PreparedSample]) -> list[PreparedSample]:
    """Send each worker only words in its window, avoiding repeated full transcripts."""
    starts_by_words: dict[int, tuple[tuple[Word, ...], list[float]]] = {}
    compact: list[PreparedSample] = []
    for chunk in chunks:
        key = id(chunk.words)
        cached = starts_by_words.get(key)
        if cached is None:
            starts = [word.start for word in chunk.words]
            starts_by_words[key] = (chunk.words, starts)
        else:
            starts = cached[1]
        left = bisect_left(starts, chunk.window_start_sec)
        right = bisect_left(starts, chunk.window_end_sec)
        compact.append(replace(chunk, words=chunk.words[left:right]))
    return compact


def filter_text_capacity_chunks(
    chunks: list[PreparedSample],
    tokenizer: Tokenizer,
    frame_rate: float,
    normalize_vietnamese_diacritics: bool = False,
    swap_roles: bool = False,
    num_workers: int = 1,
) -> ChunkFilterResult:
    """Keep chunks whose configured agent role views fit without retiming text."""
    if frame_rate <= 0:
        raise ValueError("Mimi frame_rate must be positive")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")

    use_process_pool = num_workers > 1 and len(chunks) >= 64
    active_workers = min(num_workers, len(chunks)) if use_process_pool else 1
    print(
        f"[Chunk filter workers] candidates={len(chunks)} configured={num_workers} "
        f"active={active_workers} mode={'processes' if use_process_pool else 'sequential'}",
        flush=True,
    )

    if use_process_pool:
        worker_chunks = _compact_chunks_for_workers(chunks)
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_filter_worker,
            initargs=(tokenizer, frame_rate, normalize_vietnamese_diacritics, swap_roles),
        ) as executor:
            decisions = list(executor.map(_filter_chunk_worker, worker_chunks, chunksize=16))
    else:
        decisions = [
            _filter_chunk(chunk, tokenizer, frame_rate, normalize_vietnamese_diacritics, swap_roles)
            for chunk in chunks
        ]

    kept = [chunk for chunk, rejection in zip(chunks, decisions, strict=True) if rejection is None]
    rejected = [rejection for rejection in decisions if rejection is not None]
    return ChunkFilterResult(tuple(kept), tuple(rejected))
