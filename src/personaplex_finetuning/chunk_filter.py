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
    swap_roles: bool,
    vietnamese_text_mode: str,
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
            vietnamese_text_mode,
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


def _initialize_filter_worker(tokenizer, frame_rate, swap_roles, vietnamese_text_mode):
    global _WORKER_FILTER_ARGS
    _WORKER_FILTER_ARGS = (
        tokenizer, frame_rate, swap_roles, vietnamese_text_mode,
    )


def _filter_chunk_worker(chunk: PreparedSample) -> RejectedChunk | None:
    tokenizer, frame_rate, swap_roles, vietnamese_text_mode = _WORKER_FILTER_ARGS
    return _filter_chunk(
        chunk, tokenizer, frame_rate, swap_roles, vietnamese_text_mode,
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
    swap_roles: bool = False,
    num_workers: int = 1,
    cache_path=None,
    cache_fingerprint: str | None = None,
    force_filter: bool = False,
    vietnamese_text_mode: str = "diacritics",
    max_kept: int | None = None,
) -> ChunkFilterResult:
    """Keep the first max_kept valid chunks, or all when no quota is set."""
    if frame_rate <= 0:
        raise ValueError("Mimi frame_rate must be positive")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    if max_kept is not None and (type(max_kept) is not int or max_kept < 1):
        raise ValueError("max_kept must be a positive integer or None")
    if cache_path is not None:
        if not cache_fingerprint:
            raise ValueError("cache_fingerprint is required when cache_path is set")
        if max_kept is not None:
            cache_fingerprint = f"{cache_fingerprint}:valid-chunk-quota={max_kept}"
        from .filter_cache import load_filter_manifest, save_filter_manifest

        if not force_filter:
            cached = load_filter_manifest(cache_path, cache_fingerprint)
            if cached is not None and "kept" in cached and "rejected" in cached:
                print(
                    f"[Chunk filter workers] candidates={len(chunks)} configured={num_workers} "
                    f"active=0 mode=cache", flush=True,
                )
                print(
                    f"[Chunk filter cache] hit path={cache_path} candidates={len(chunks)} kept={len(cached['kept'])}",
                    flush=True,
                )
                rejected_cached = tuple(
                    RejectedChunk(
                        item["sample_id"], float(item["window_start_sec"]),
                        float(item["window_end_sec"]), item["reason"], tuple(item["roles"]),
                        Word(item.get("word_speaker", "agent"), item["word"], float(item["word_start_sec"]),
                             float(item.get("word_end_sec", item["word_start_sec"]))),
                    ) for item in cached["rejected"]
                )
                return ChunkFilterResult(tuple(cached["kept"]), rejected_cached)

    tasks_per_worker = 8
    task_count = math.ceil(len(chunks) / tasks_per_worker)
    use_process_pool = num_workers > 1 and task_count >= 2
    active_workers = min(num_workers, task_count) if use_process_pool else 1
    print(
        f"[Chunk filter workers] candidates={len(chunks)} configured={num_workers} "
        f"active={active_workers} mode={'processes' if use_process_pool else 'sequential'}",
        flush=True,
    )

    kept = []
    rejected = []

    def retain(batch, decisions):
        for chunk, rejection in zip(batch, decisions, strict=True):
            if rejection is None:
                kept.append(chunk)
            else:
                rejected.append(rejection)

    if use_process_pool:
        with ProcessPoolExecutor(
            max_workers=active_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_filter_worker,
            initargs=(tokenizer, frame_rate, swap_roles, vietnamese_text_mode),
        ) as executor:
            cursor = 0
            while cursor < len(chunks) and (max_kept is None or len(kept) < max_kept):
                # Never schedule more than the remaining quota: even if all
                # are valid, no worker evaluates a chunk past the stopping point.
                size = active_workers * tasks_per_worker
                if max_kept is not None:
                    size = min(size, max_kept - len(kept))
                batch = chunks[cursor:cursor + size]
                decisions = list(executor.map(
                    _filter_chunk_worker, _compact_chunks_for_workers(batch),
                    chunksize=tasks_per_worker,
                ))
                retain(batch, decisions)
                cursor += len(batch)
    else:
        for chunk in chunks:
            retain([chunk], [_filter_chunk(chunk, tokenizer, frame_rate, swap_roles, vietnamese_text_mode)])
            if max_kept is not None and len(kept) >= max_kept:
                break
    result = ChunkFilterResult(tuple(kept), tuple(rejected))
    if cache_path is not None:
        save_filter_manifest(cache_path, cache_fingerprint, {
            "kept": result.kept,
            "rejected": [{
                "sample_id": item.sample_id,
                "window_start_sec": item.window_start_sec,
                "window_end_sec": item.window_end_sec,
                "reason": item.reason,
                "roles": list(item.roles),
                "word": item.word.word,
                "word_speaker": item.word.speaker,
                "word_start_sec": item.word.start,
                "word_end_sec": item.word.end,
            } for item in result.rejected],
        })
        print(
            f"[Chunk filter cache] {'forced rebuild' if force_filter else 'built'} "
            f"path={cache_path} candidates={len(chunks)} kept={len(result.kept)}", flush=True,
        )
    return result
