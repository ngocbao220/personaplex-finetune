import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from personaplex_finetuning.chunk_filter import filter_language_chunks, load_language_report
from personaplex_finetuning.data import AudioInfo, PreparedSample, Word, duration_chunks


def conversation(sample_id: str, seconds: float) -> PreparedSample:
    return PreparedSample(
        sample_id, Path(f"{sample_id}.wav"), Path("voice.wav"), (Word("agent", "xin", 0.0, 0.2),),
        "prompt", {}, AudioInfo(24000, 2, seconds), 0.0, seconds,
    )


class LanguageFilterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "train.jsonl"
        self.manifest.write_text('{"sample_id": "a", "sample_dir": "samples/a"}\n')

    def tearDown(self):
        self.tmp.cleanup()

    def write_report(self, rows, duration=100.0, threshold=0.5, manifest_sha=None):
        path = self.root / "lid.json"
        path.write_text(json.dumps({
            "threshold": threshold, "duration_sec": duration,
            "source_manifest_sha256": manifest_sha or hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "chunks": [
                {"sample_id": sid, "window_start_sec": start, "window_end_sec": start + duration, "vi_probability": p}
                for sid, start, p in rows
            ],
        }))
        return path

    def test_filters_scored_chunks_and_passes_unscored_conversations(self):
        report = load_language_report(
            self.write_report([("a", 0.0, 0.9), ("a", 100.0, 0.1), ("a", 200.0, 0.5)]), self.manifest, 100.0,
        )
        chunks = duration_chunks([conversation("a", 250.0), conversation("val_only", 150.0)], 100.0)
        kept, rejected, unscored = filter_language_chunks(chunks, report)
        self.assertEqual([(c.sample_id, c.window_start_sec) for c in kept],
                         [("a", 0.0), ("a", 200.0), ("val_only", 0.0), ("val_only", 100.0)])
        self.assertEqual((rejected, unscored), (1, 2))

    def test_non_integer_duration_matches_by_chunk_index(self):
        duration = 0.1 * 3  # accumulated float starts differ from index * duration
        rows = [("a", index * duration, 0.9) for index in range(10)]
        report = load_language_report(self.write_report(rows, duration=duration), self.manifest, duration)
        kept, rejected, _ = filter_language_chunks(duration_chunks([conversation("a", 3.0)], duration), report)
        self.assertEqual((len(kept), rejected), (10, 0))

    def test_rejects_stale_or_invalid_reports(self):
        with self.assertRaisesRegex(ValueError, "missing chunk"):
            report = load_language_report(self.write_report([("a", 0.0, 0.9)]), self.manifest, 100.0)
            filter_language_chunks(duration_chunks([conversation("a", 150.0)], 100.0), report)
        with self.assertRaisesRegex(ValueError, "different manifest"):
            load_language_report(self.write_report([], manifest_sha="0" * 64), self.manifest, 100.0)
        with self.assertRaisesRegex(ValueError, "duration"):
            load_language_report(self.write_report([]), self.manifest, 60.0)
        for threshold in (None, 1.5, True):
            with self.assertRaisesRegex(ValueError, "threshold"):
                load_language_report(self.write_report([], threshold=threshold), self.manifest, 100.0)


if __name__ == "__main__":
    unittest.main()
