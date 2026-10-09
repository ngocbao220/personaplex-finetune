import json
import tempfile
import unittest
from pathlib import Path

from personaplex_finetuning.data import AudioInfo, PreparedDataset, PreparedSample, Word
from personaplex_finetuning.filter_cache import _decode_sample, _encode_sample


def sample():
    return PreparedSample(
        sample_id="c1", conversation_wav=Path("/d/c.wav"), voice_prompt_wav=Path("/d/v.wav"),
        words=(Word("agent", "xin", 0.1, 0.3), Word("user", "chào", 0.4, 0.6)),
        text_prompt="Bạn là trợ lý.", metadata={"duration_sec": 1.0}, audio=AudioInfo(24000, 2, 1.0),
        window_start_sec=0.0, window_end_sec=1.0,
    )


class FilterCacheFormatTest(unittest.TestCase):
    def test_columnar_words_round_trip(self):
        encoded = _encode_sample(sample())
        self.assertNotIn("words", encoded)
        self.assertEqual(encoded["words_columns"]["word"], ["xin", "chào"])
        self.assertEqual(_decode_sample(json.loads(json.dumps(encoded))), sample())

    def test_reads_caches_written_with_one_dict_per_word(self):
        encoded = _encode_sample(sample())
        columns = encoded.pop("words_columns")
        encoded["words"] = [
            {"speaker": s, "word": w, "start": a, "end": b}
            for s, w, a, b in zip(columns["speaker"], columns["word"], columns["start"], columns["end"])
        ]
        self.assertEqual(_decode_sample(encoded), sample())

    def test_cache_only_load_returns_none_on_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "train.jsonl"
            manifest.write_text('{"sample_id":"c1","sample_dir":"c1"}\n')
            dataset = PreparedDataset(manifest, None)
            self.assertIsNone(dataset.load(cache_only=True))
            self.assertIsNone(dataset.split(cache_only=True))
            self.assertFalse((Path(tmp) / ".filter-cache").exists())


if __name__ == "__main__":
    unittest.main()
