import json
import tempfile
import unittest
import wave
from pathlib import Path

from personaplex_finetuning.data import (
    AudioInfo,
    PreparedSample,
    PreparedDataset,
    ValidationError,
    contiguous_chunks,
    sample_for_training_position,
    turn_aware_chunks,
    Word,
)
from personaplex_finetuning.runtime import _pad_audio_window


def write_stereo_wav(path: Path, frames: int = 24000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\0\0\0\0" * frames)


class PreparedDatasetTest(unittest.TestCase):
    def test_turn_aware_chunks_use_safe_boundaries_and_clip_to_real_audio(self) -> None:
        sample = PreparedSample(
            "conversation", Path("conversation.wav"), Path("voice.wav"),
            tuple(
                Word("agent" if index % 2 == 0 else "user", f"word{index}.", index * 2.0, index * 2.0 + 1.0)
                for index in range(20)
            ), "prompt", {}, AudioInfo(24000, 2, 40.0), 0.0, 40.0,
        )
        chunks = turn_aware_chunks([sample], 25, 10, 30, randomize=False)
        bounds = [(item.window_start_sec, item.window_end_sec) for item in chunks]
        self.assertEqual(bounds[0][0], 0.0)
        self.assertEqual(bounds[-1][1], 40.0)
        self.assertTrue(all(0 < end - start <= 30 for start, end in bounds))
        self.assertTrue(all(left[1] == right[0] for left, right in zip(bounds, bounds[1:])))
        safe_ends = {word.end for word in sample.words}
        self.assertTrue(all(end == 40.0 or end in safe_ends for _, end in bounds))

    def test_training_chunk_boundaries_are_seeded_but_can_vary_by_epoch(self) -> None:
        sample = PreparedSample(
            "conversation", Path("conversation.wav"), Path("voice.wav"),
            tuple(Word("user", "word", index * 2.0, index * 2.0 + 1.0) for index in range(30)),
            "prompt", {}, AudioInfo(24000, 2, 60.0), 0.0, 60.0,
        )
        from random import Random
        first = turn_aware_chunks([sample], 25, 10, 30, Random(1), randomize=True)
        same = turn_aware_chunks([sample], 25, 10, 30, Random(1), randomize=True)
        other = turn_aware_chunks([sample], 25, 10, 30, Random(2), randomize=True)
        self.assertEqual([item.window_end_sec for item in first], [item.window_end_sec for item in same])
        self.assertNotEqual([item.window_end_sec for item in first], [item.window_end_sec for item in other])

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

    def test_skips_invalid_sample_and_loads_remaining_manifest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for sample_id in ("invalid", "valid"):
                sample_dir = root / "samples" / sample_id
                sample_dir.mkdir(parents=True)
                write_stereo_wav(sample_dir / "conversation.wav")
                write_stereo_wav(sample_dir / "voice_prompt_left.wav")
                (sample_dir / "metadata.json").write_text(json.dumps({"text_prompt_left": "Be helpful."}))
                end = 2.0 if sample_id == "invalid" else 0.2
                (sample_dir / "words.json").write_text(json.dumps([
                    {"speaker": "agent", "word": "Hello", "start": 0.0, "end": end},
                ]))
                entries.append({"sample_id": sample_id, "sample_dir": f"samples/{sample_id}"})
            manifest = root / "train.jsonl"
            manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))

            with self.assertLogs("personaplex_finetuning.data", level="WARNING") as logs:
                samples = PreparedDataset(manifest).load()

        self.assertEqual([sample.sample_id for sample in samples], ["valid"])
        self.assertIn("line 1", logs.output[0])
        self.assertIn("invalid", logs.output[0])
        self.assertIn("word 0 is outside audio bounds", logs.output[0])

    def test_reports_validation_reason_when_all_manifest_samples_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "samples" / "invalid"
            sample_dir.mkdir(parents=True)
            write_stereo_wav(sample_dir / "conversation.wav")
            write_stereo_wav(sample_dir / "voice_prompt_left.wav")
            (sample_dir / "metadata.json").write_text(json.dumps({"text_prompt_left": "Be helpful."}))
            (sample_dir / "words.json").write_text(json.dumps([
                {"speaker": "agent", "word": "Hello", "start": 0.0, "end": 2.0},
            ]))
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps({"sample_id": "invalid", "sample_dir": "samples/invalid"}) + "\n")

            with self.assertRaisesRegex(ValidationError, "invalid.*word 0 is outside audio bounds"):
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

    def test_contiguous_chunks_cover_audio_without_extending_final_window(self) -> None:
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
            (0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 95.0),
        ])

    def test_split_keeps_entries_from_one_conversation_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for index, group in enumerate(("shared", "shared", "other", "third")):
                sample_dir = root / "samples" / str(index)
                sample_dir.mkdir(parents=True)
                write_stereo_wav(sample_dir / "conversation.wav")
                write_stereo_wav(sample_dir / "voice_prompt_left.wav")
                (sample_dir / "metadata.json").write_text(json.dumps({
                    "conversation_id": group, "text_prompt_left": "x",
                }))
                (sample_dir / "words.json").write_text(json.dumps([
                    {"speaker": "agent", "word": "Hi", "start": 0.0, "end": 0.2},
                ]))
                entries.append({"sample_id": str(index), "sample_dir": f"samples/{index}"})
            manifest = root / "train.jsonl"
            manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
            train, val = PreparedDataset(manifest).split(val_ratio=0.5, seed=42)
        train_ids, val_ids = {sample.sample_id for sample in train}, {sample.sample_id for sample in val}
        self.assertTrue({"0", "1"} <= train_ids or {"0", "1"} <= val_ids)
        self.assertFalse(train_ids & val_ids)

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
