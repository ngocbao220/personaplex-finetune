import unittest
from types import SimpleNamespace

from personaplex_finetuning.chunk_filter import filter_text_capacity_chunks
from personaplex_finetuning.data import AudioInfo, PreparedSample, Word
from pathlib import Path
from tools import benchmark_vi_tokenizer as bench


class CharTokenizer:
    """One token per character; non-ASCII characters count as byte fallback."""
    padding_id = 3
    end_padding_id = 0

    def encode(self, text):
        return [1000 + ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i - 1000) for i in ids)


def sample(words, duration=4.0):
    return PreparedSample(
        sample_id="s", conversation_wav=Path("c.wav"), voice_prompt_wav=Path("v.wav"),
        voice_prompt_right_wav=None, words=tuple(words), text_prompt="p", text_prompt_right=None,
        metadata={}, audio=AudioInfo(24_000, 2, duration), window_start_sec=0.0,
        window_end_sec=duration, agent_channel=0, user_channel=1,
    )


class TokenizerBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.words = bench.WordTokenizer(CharTokenizer(), "no_diacritics")
        self.words._is_byte = lambda token: token >= 1128

    def test_word_metrics_follow_mode_normalization(self):
        stats = bench.syllable_stats(["bạn", "ừ", ","], self.words, agent_seconds=1.0)
        self.assertEqual(stats["syllables"], 2)          # "," is not a syllable
        self.assertEqual(stats["mean"], 2.0)             # "ban"=3, "u"=1
        self.assertEqual(stats["agent_tokens_per_speech_sec"], 5.0)
        raw = bench.WordTokenizer(CharTokenizer(), "diacritics")
        raw._is_byte = lambda token: token >= 1128
        self.assertEqual(bench.byte_fallback_stats(["bạn", "ok"], raw)["word_rate"], 0.5)
        self.assertEqual(bench.byte_fallback_stats(["bạn"], self.words)["token_rate"], 0.0)
        self.assertEqual(bench.roundtrip_stats(["Bạn ơi"], self.words)["exact_rate"], 1.0)

    def test_overflow_matches_trainer_filter(self):
        dense = [Word("agent", "x" * 40, 0.0, 0.5)]   # 40 + 20 tokens > 50 frames (4s at 12.5 fps)
        dense.append(Word("agent", "y" * 20, 1.0, 1.5))
        chunks = [sample(dense), sample([Word("agent", "ok", 0.0, 0.2)])]
        tokenizer = CharTokenizer()
        expected = filter_text_capacity_chunks(chunks, tokenizer, 12.5, vietnamese_text_mode="diacritics")
        stats = bench.overflow_stats(chunks, tokenizer, "diacritics", False, 1)
        self.assertEqual(stats["text_overflow"], expected.skipped_text_overflow)
        self.assertEqual(stats["kept"], len(expected.kept))
        self.assertEqual(stats["text_overflow"], 1)


if __name__ == "__main__":
    unittest.main()
