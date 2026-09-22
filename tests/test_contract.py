import json
import tempfile
import unittest
import wave
from pathlib import Path

from personaplex_finetuning.data import (
    PreparedDataset,
    ValidationError,
    contiguous_chunks,
    sample_for_training_position,
)
from personaplex_finetuning.runtime import _pad_audio_window


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
            write_stereo_wav(sample_dir / "voice_prompt_left.wav", frames=24000 * 6)
            (sample_dir / "metadata.json").write_text(
                json.dumps({
                    "sample_id": "conv_0001",
                    "agent_channel": "left",
                    "user_channel": "right",
                    "text_prompt_left": "Be helpful.",
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
            write_stereo_wav(sample_dir / "voice_prompt_left.wav")
            (sample_dir / "metadata.json").write_text(json.dumps({"text_prompt_left": "x"}))
            (sample_dir / "words.json").write_text(json.dumps([
                {"speaker": "agent", "word": "Hi", "start": 0.0, "end": 0.2},
            ]))
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps({"sample_id": "conv_0001", "sample_dir": "samples/conv_0001"}) + "\n")

            with self.assertRaisesRegex(ValidationError, "exactly 2 channels"):
                PreparedDataset(manifest).load()

    def test_rejects_absolute_sample_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "train.jsonl"
            manifest.write_text(
                json.dumps({"sample_id": "conv_0001", "sample_dir": "/data/samples/conv_0001"}) + "\n"
            )

            with self.assertRaisesRegex(ValidationError, "relative"):
                PreparedDataset(manifest).load()

    def test_contiguous_chunks_cover_audio_and_pad_the_final_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "samples" / "conv_0001"
            sample_dir.mkdir(parents=True)
            write_stereo_wav(sample_dir / "conversation.wav", frames=24000 * 95)
            write_stereo_wav(sample_dir / "voice_prompt_left.wav", frames=24000)
            (sample_dir / "metadata.json").write_text(json.dumps({"text_prompt_left": "x"}))
            (sample_dir / "words.json").write_text(json.dumps([
                {"speaker": "agent", "word": "Hi", "start": 0.0, "end": 0.2},
            ]))
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps({"sample_id": "conv_0001", "sample_dir": "samples/conv_0001"}) + "\n")
            chunks = contiguous_chunks(PreparedDataset(manifest).load(), 30.0)

        self.assertEqual([(chunk.window_start_sec, chunk.window_end_sec) for chunk in chunks], [
            (0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 120.0),
        ])

    def test_right_role_pass_uses_right_voice_prompt_and_inverts_channels_and_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "samples" / "conv_0001"
            sample_dir.mkdir(parents=True)
            write_stereo_wav(sample_dir / "conversation.wav", frames=24000 * 60)
            write_stereo_wav(sample_dir / "voice_prompt_left.wav", frames=24000)
            write_stereo_wav(sample_dir / "voice_prompt_right.wav", frames=24000)
            (sample_dir / "metadata.json").write_text(json.dumps({
                "text_prompt_left": "left persona", "text_prompt_right": "right persona",
            }))
            (sample_dir / "words.json").write_text(json.dumps([
                {"speaker": "agent", "word": "Left", "start": 1.0, "end": 1.2},
                {"speaker": "user", "word": "Right", "start": 2.0, "end": 2.2},
            ]))
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps({"sample_id": "conv_0001", "sample_dir": "samples/conv_0001"}) + "\n")
            chunks = contiguous_chunks(PreparedDataset(manifest).load(), 30.0)
            first_right_pass = sample_for_training_position(chunks, position=2, seed=42, shuffle=False, swap_roles=True)

        self.assertEqual(first_right_pass.voice_prompt_wav.name, "voice_prompt_right.wav")
        self.assertEqual(first_right_pass.text_prompt, "right persona")
        self.assertEqual((first_right_pass.agent_channel, first_right_pass.user_channel), (1, 0))
        self.assertEqual([word.speaker for word in first_right_pass.words], ["user", "agent"])

    def test_right_role_pass_rejects_missing_right_voice_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "samples" / "conv_0001"
            sample_dir.mkdir(parents=True)
            write_stereo_wav(sample_dir / "conversation.wav", frames=24000 * 30)
            write_stereo_wav(sample_dir / "voice_prompt_left.wav", frames=24000)
            (sample_dir / "metadata.json").write_text(json.dumps({"text_prompt_left": "x"}))
            (sample_dir / "words.json").write_text(json.dumps([
                {"speaker": "agent", "word": "Hi", "start": 0.0, "end": 0.2},
            ]))
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps({"sample_id": "conv_0001", "sample_dir": "samples/conv_0001"}) + "\n")
            chunk = contiguous_chunks(PreparedDataset(manifest).load(), 30.0)[0]

            with self.assertRaisesRegex(ValidationError, "voice_prompt_right"):
                sample_for_training_position([chunk], position=1, seed=42, shuffle=False, swap_roles=True)

    def test_right_role_pass_rejects_missing_right_text_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "samples" / "conv_0001"
            sample_dir.mkdir(parents=True)
            write_stereo_wav(sample_dir / "conversation.wav", frames=24000 * 30)
            write_stereo_wav(sample_dir / "voice_prompt_left.wav", frames=24000)
            write_stereo_wav(sample_dir / "voice_prompt_right.wav", frames=24000)
            (sample_dir / "metadata.json").write_text(json.dumps({"text_prompt_left": "left persona"}))
            (sample_dir / "words.json").write_text(json.dumps([
                {"speaker": "agent", "word": "Hi", "start": 0.0, "end": 0.2},
            ]))
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps({"sample_id": "conv_0001", "sample_dir": "samples/conv_0001"}) + "\n")
            chunk = contiguous_chunks(PreparedDataset(manifest).load(), 30.0)[0]

            with self.assertRaisesRegex(ValidationError, "text_prompt_right"):
                sample_for_training_position([chunk], position=1, seed=42, shuffle=False, swap_roles=True)

    def test_final_contiguous_chunk_is_zero_padded_to_the_requested_duration(self) -> None:
        import numpy as np

        padded = _pad_audio_window(np.ones((2, 3), dtype=np.float32), sample_rate=10, duration_sec=0.5)

        self.assertEqual(padded.shape, (2, 5))
        self.assertTrue(np.array_equal(padded[:, :3], np.ones((2, 3), dtype=np.float32)))
        self.assertTrue(np.array_equal(padded[:, 3:], np.zeros((2, 2), dtype=np.float32)))
