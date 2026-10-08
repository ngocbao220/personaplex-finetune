from dataclasses import replace
from pathlib import Path
import unittest
import tempfile

from personaplex_finetuning.chunk_filter import _filter_chunk, expected_mimi_frames, filter_text_capacity_chunks
from personaplex_finetuning.data import AudioInfo, PreparedSample, Word


class FakeTokenizer:
    padding_id = 3
    end_padding_id = 0

    def encode(self, _text):
        return [11, 12]


def _chunk(words, duration=1.0):
    return PreparedSample(
        sample_id="conversation",
        conversation_wav=Path("conversation.wav"),
        voice_prompt_wav=Path("left.wav"),
        words=tuple(words),
        text_prompt="left prompt",
        metadata={},
        audio=AudioInfo(24000, 2, duration),
        window_start_sec=0.0,
        window_end_sec=1.0,
        voice_prompt_right_wav=Path("right.wav"),
        text_prompt_right="right prompt",
    )


def test_role_swap_rejects_interval_when_only_left_agent_overflows():
    chunk = _chunk([Word("agent", "late", 0.9, 0.95)])

    result = filter_text_capacity_chunks(
        [chunk], FakeTokenizer(), frame_rate=10.0, swap_roles=True,
    )

    assert result.kept == ()
    assert result.skipped_text_overflow == 1
    assert result.rejected[0].roles == ("left-agent",)


def test_role_swap_rejects_interval_when_only_right_agent_overflows():
    chunk = _chunk([Word("user", "late", 0.9, 0.95)])

    result = filter_text_capacity_chunks(
        [chunk], FakeTokenizer(), frame_rate=10.0, swap_roles=True,
    )

    assert result.kept == ()
    assert result.skipped_text_overflow == 1
    assert result.rejected[0].roles == ("right-agent",)


def test_out_of_bounds_is_reported_separately_from_text_overflow():
    chunk = _chunk([Word("agent", "late", 0.9, 0.99)], duration=0.95)

    result = filter_text_capacity_chunks(
        [chunk], FakeTokenizer(), frame_rate=10.0, swap_roles=False,
    )

    assert result.kept == ()
    assert result.skipped_out_of_bounds == 1
    assert result.skipped_text_overflow == 0
    assert result.rejected[0].reason == "out_of_bounds"


def test_chunk_fits_when_agent_words_fit_on_or_after_their_timestamps():
    chunk = _chunk([Word("agent", "early", 0.1, 0.3)])

    result = filter_text_capacity_chunks(
        [chunk], FakeTokenizer(), frame_rate=10.0, swap_roles=False,
    )

    assert result.kept == (chunk,)
    assert result.skipped_out_of_bounds == 0
    assert result.skipped_text_overflow == 0


class ParallelChunkFilterTest(unittest.TestCase):
    def test_multiple_cpu_workers_match_single_worker_filter_results(self):
        chunks = []
        for index in range(64):
            if index % 3 == 0:
                chunks.append(_chunk([Word("agent", "late", 0.9, 0.95)]))
            elif index % 3 == 1:
                chunks.append(_chunk([Word("agent", "late", 0.9, 0.99)], duration=0.95))
            else:
                chunks.append(_chunk([Word("agent", "early", 0.1, 0.3)]))

        sequential = filter_text_capacity_chunks(
            chunks, FakeTokenizer(), frame_rate=10.0, swap_roles=True,
        )
        parallel = filter_text_capacity_chunks(
            chunks, FakeTokenizer(), frame_rate=10.0, swap_roles=True, num_workers=2,
        )

        self.assertEqual(parallel, sequential)

    def test_chunk_filter_cache_reuses_result_and_force_rebuilds(self):
        from unittest.mock import patch

        chunk = _chunk([Word("agent", "early", 0.1, 0.3)])
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "chunks.jsonl"
            kwargs = {"cache_path": cache, "cache_fingerprint": "fingerprint"}
            first = filter_text_capacity_chunks([chunk], FakeTokenizer(), 10.0, **kwargs)
            with patch("personaplex_finetuning.chunk_filter._filter_chunk", side_effect=AssertionError("cache miss")):
                hit = filter_text_capacity_chunks([chunk], FakeTokenizer(), 10.0, **kwargs)
            self.assertEqual(hit, first)

            with patch("personaplex_finetuning.chunk_filter._filter_chunk", wraps=_filter_chunk) as filter_one:
                rebuilt = filter_text_capacity_chunks(
                    [chunk], FakeTokenizer(), 10.0, force_filter=True, **kwargs,
                )
            self.assertEqual(rebuilt, first)
            filter_one.assert_called_once()

    def test_chunk_filter_cache_round_trips_rejection_details(self):
        chunk = _chunk([Word("user", "late", 0.9, 0.95)])
        with tempfile.TemporaryDirectory() as directory:
            kwargs = {
                "cache_path": Path(directory) / "chunks.jsonl",
                "cache_fingerprint": "reject-fingerprint",
            }
            first = filter_text_capacity_chunks(
                [chunk], FakeTokenizer(), 10.0, swap_roles=True, **kwargs,
            )
            cached = filter_text_capacity_chunks(
                [chunk], FakeTokenizer(), 10.0, swap_roles=True, **kwargs,
            )
            self.assertEqual(first, cached)


def test_expected_fixed_duration_mimi_grid_uses_ceil_for_partial_frames():
    assert expected_mimi_frames(100.0, 12.5) == 1250
    assert expected_mimi_frames(100.01, 12.5) == 1251


class ChunkQuotaTest(unittest.TestCase):
    def chunks(self, count=40):
        return [replace(_chunk([Word("agent", "word", .9 if i % 3 == 0 else .1,
                                    .95 if i % 3 == 0 else .3)]), sample_id=str(i))
                for i in range(count)]

    def test_quota_counts_valid_chunks_and_stops_at_required_prefix(self):
        from unittest.mock import patch
        chunks = self.chunks()
        with patch("personaplex_finetuning.chunk_filter._filter_chunk", wraps=_filter_chunk) as check:
            result = filter_text_capacity_chunks(chunks, FakeTokenizer(), 10, max_kept=3)
        self.assertEqual([s.sample_id for s in result.kept], ["1", "2", "4"])
        self.assertEqual(check.call_count, 5)
        self.assertEqual(len(result.rejected), 2)

    def test_exhausted_source_returns_available_valid_chunks(self):
        result = filter_text_capacity_chunks(self.chunks(3), FakeTokenizer(), 10, max_kept=100)
        self.assertEqual(len(result.kept), 2)
        self.assertEqual(len(result.rejected), 1)

    def test_parallel_quota_matches_sequential_prefix(self):
        chunks = self.chunks()
        sequential = filter_text_capacity_chunks(chunks, FakeTokenizer(), 10, max_kept=11)
        parallel = filter_text_capacity_chunks(chunks, FakeTokenizer(), 10, max_kept=11, num_workers=2)
        self.assertEqual(parallel, sequential)
        self.assertEqual(len(parallel.kept), 11)

    def test_cache_is_separate_for_each_quota_and_unlimited(self):
        from unittest.mock import patch
        chunks = self.chunks(6)
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(cache_path=Path(directory) / "quota.jsonl", cache_fingerprint="same-source")
            one = filter_text_capacity_chunks(chunks, FakeTokenizer(), 10, max_kept=1, **kwargs)
            with patch("personaplex_finetuning.chunk_filter._filter_chunk", side_effect=AssertionError("cache miss")):
                self.assertEqual(filter_text_capacity_chunks(chunks, FakeTokenizer(), 10, max_kept=1, **kwargs), one)
            two = filter_text_capacity_chunks(chunks, FakeTokenizer(), 10, max_kept=2, **kwargs)
            all_chunks = filter_text_capacity_chunks(chunks, FakeTokenizer(), 10, **kwargs)
        self.assertEqual(len(two.kept), 2)
        self.assertEqual(len(all_chunks.kept), 4)

    def test_invalid_quota_fails_before_filtering(self):
        for quota in (0, -1, True, 1.5):
            with self.subTest(quota=quota), self.assertRaises(ValueError):
                filter_text_capacity_chunks(self.chunks(), FakeTokenizer(), 10, max_kept=quota)
