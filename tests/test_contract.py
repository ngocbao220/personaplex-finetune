import json
import tempfile
import unittest
import wave
from pathlib import Path

from personaplex_finetuning.data import PreparedDataset, ValidationError


def write_stereo_wav(path: Path, frames: int = 24000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\0\0\0\0" * frames)


class PreparedDatasetTest(unittest.TestCase):
    def test_loads_prompt_from_metadata_and_selects_first_agent_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "samples" / "conv_0001"
            sample_dir.mkdir(parents=True)
            write_stereo_wav(sample_dir / "conversation.wav", frames=24000 * 90)
            write_stereo_wav(sample_dir / "voice_prompt.wav", frames=24000 * 6)
            (sample_dir / "metadata.json").write_text(
                json.dumps({
                    "sample_id": "conv_0001",
                    "agent_channel": "left",
                    "user_channel": "right",
                    "text_prompt": "Be helpful.",
                })
            )
            (sample_dir / "words.json").write_text(json.dumps([
                {"speaker": "user", "word": "Hi", "start": 2.0, "end": 2.2},
                {"speaker": "agent", "word": "Hello", "start": 52.0, "end": 52.4},
            ]))
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps({"sample_id": "conv_0001", "sample_dir": "samples/conv_0001"}) + "\n")

            sample = PreparedDataset(manifest, window_seconds=30).load()[0]

            self.assertEqual(sample.text_prompt, "Be helpful.")
            self.assertEqual(sample.window_start_sec, 52.0)
            self.assertEqual(sample.window_end_sec, 82.0)
            self.assertEqual(sample.agent_channel, 0)
            self.assertEqual(sample.user_channel, 1)

    def test_rejects_non_stereo_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "samples" / "conv_0001"
            sample_dir.mkdir(parents=True)
            with wave.open(str(sample_dir / "conversation.wav"), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                output.writeframes(b"\0\0" * 24000)
            write_stereo_wav(sample_dir / "voice_prompt.wav")
            (sample_dir / "metadata.json").write_text(json.dumps({"text_prompt": "x"}))
            (sample_dir / "words.json").write_text(json.dumps([
                {"speaker": "agent", "word": "Hi", "start": 0.0, "end": 0.2},
            ]))
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps({"sample_id": "conv_0001", "sample_dir": "samples/conv_0001"}) + "\n")

            with self.assertRaisesRegex(ValidationError, "exactly 2 channels"):
                PreparedDataset(manifest).load()
