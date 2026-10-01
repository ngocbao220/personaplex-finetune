from dataclasses import replace
from pathlib import Path
import unittest

from personaplex_finetuning.chunk_filter import expected_mimi_frames, filter_text_capacity_chunks
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


def test_expected_fixed_duration_mimi_grid_uses_ceil_for_partial_frames():
    assert expected_mimi_frames(100.0, 12.5) == 1250
    assert expected_mimi_frames(100.01, 12.5) == 1251
