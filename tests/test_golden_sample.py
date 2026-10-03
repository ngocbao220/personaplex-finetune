from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from personaplex_finetuning.data import AudioInfo, PreparedSample, Word
from personaplex_finetuning.objective import stream_weights_torch
from personaplex_finetuning.runtime import MimiCodec, RuntimePaths, SentencePieceTokenizer
from personaplex_finetuning.sequence import PersonaPlexTrainingExampleBuilder
from tools import inspect_sample
from tools.inspect_sample import build_debug_payload, write_debug_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
MODEL_ROOT = WORKSPACE_ROOT / "models"


def _write_pcm16(path: Path, channels: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(np.rint(channels * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(pcm.shape[0])
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.T.tobytes())


def _tone(sample_rate: int, duration: float, frequency: int, start: float, end: float) -> np.ndarray:
    count = round(sample_rate * duration)
    samples = np.zeros(count, dtype=np.float32)
    first, last = round(start * sample_rate), round(end * sample_rate)
    time = np.arange(last - first, dtype=np.float64) / sample_rate
    samples[first:last] = (0.25 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    return samples


def _dominant_frequency(audio: np.ndarray, sample_rate: int) -> int:
    spectrum = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(audio.size, 1.0 / sample_rate)
    return int(round(frequencies[int(np.argmax(spectrum))]))


class CapturingMimi:
    def __init__(self, mimi) -> None:
        self.mimi = mimi
        self.received = None

    def encode(self, audio):
        self.received = audio.detach().cpu().numpy().copy()
        return self.mimi.encode(audio)

    def parameters(self):
        return self.mimi.parameters()

    def streaming(self, batch_size):
        return self.mimi.streaming(batch_size)


class GoldenSampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [
            str(MODEL_ROOT / name)
            for name in (
                "model.safetensors",
                "tokenizer-e351c8d8-checkpoint125.safetensors",
                "tokenizer_spm_32k_3.model",
            )
            if not (MODEL_ROOT / name).is_file()
        ]
        if missing:
            raise unittest.SkipTest("local PersonaPlex assets missing: " + ", ".join(missing))
        try:
            import torch
            from moshi.models import loaders, lm
        except ImportError as exc:
            raise unittest.SkipTest(f"Mimi runtime dependencies unavailable: {exc}")
        cls.torch = torch
        cls.loaders = loaders
        cls.lm = lm
        cls.paths = RuntimePaths(MODEL_ROOT, PROJECT_ROOT / "src").validate()

    def test_real_mimi_golden_sequence_and_debug_artifacts(self) -> None:
        sample_rate = 24000
        duration = 5.0
        user = _tone(sample_rate, duration, 880, 0.0, 1.0)
        agent = _tone(sample_rate, duration, 440, 2.0, 3.0)
        original = np.stack([agent, user])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            conversation = root / "conversation.wav"
            voice_prompt = root / "voice_prompt.wav"
            _write_pcm16(conversation, original, sample_rate)
            _write_pcm16(voice_prompt, _tone(sample_rate, 1.0, 440, 0.0, 1.0)[None], sample_rate)
            sample = PreparedSample(
                sample_id="golden_001",
                conversation_wav=conversation,
                voice_prompt_wav=voice_prompt,
                words=(
                    Word("user", "xin", 0.08, 0.48),
                    Word("user", "chào", 0.56, 0.96),
                    Word("agent", "chào", 2.08, 2.48),
                    Word("agent", "bạn", 2.56, 2.96),
                ),
                text_prompt="Be helpful.",
                metadata={},
                audio=AudioInfo(sample_rate, 2, duration),
                window_start_sec=0.0,
                window_end_sec=duration,
            )

            mimi = self.loaders.get_mimi(self.paths.mimi_weight, device="cpu")
            capture = CapturingMimi(mimi)
            codec = MimiCodec(capture, mimi.sample_rate, mimi.frame_rate, "cpu", self.lm)
            tokenizer = SentencePieceTokenizer(self.paths.tokenizer)
            self.assertEqual(inspect_sample._token_piece(tokenizer, 32000), "SPECIAL_32000")
            agent_codes, user_codes = codec.encode_conversation_stereo(conversation, 0, 1, 0.0, duration)

            expected_pcm = np.rint(original * 32767.0).astype(np.int16).astype(np.float32) / 32768.0
            self.assertEqual(capture.received.shape, (2, 1, sample_rate * 5))
            np.testing.assert_array_equal(capture.received[0, 0], expected_pcm[0])
            np.testing.assert_array_equal(capture.received[1, 0], expected_pcm[1])
            self.assertEqual(_dominant_frequency(capture.received[0, 0, 2 * sample_rate : 3 * sample_rate], sample_rate), 440)
            self.assertEqual(_dominant_frequency(capture.received[1, 0, :sample_rate], sample_rate), 880)
            self.assertTrue(np.all(capture.received[0, 0, 4 * sample_rate :] == 0))
            self.assertTrue(np.all(capture.received[1, 0, 4 * sample_rate :] == 0))
            self.assertEqual(agent_codes, self._encode_one(mimi, expected_pcm[0]))
            self.assertEqual(user_codes, self._encode_one(mimi, expected_pcm[1]))

            delays = list(self.loaders._lm_kwargs["delays"])
            builder = PersonaPlexTrainingExampleBuilder(
                codec, tokenizer, initial_tokens=tuple(range(17)), zero_token=-1
            )
            base = builder.build(sample)
            delayed = inspect_sample.native_debug_delay(base, builder.initial_tokens, delays, builder.zero_token)

            self.assertEqual(len(base.input_codes), 17)
            self.assertEqual(sample.agent_channel, 0)
            self.assertEqual(sample.user_channel, 1)
            self.assertEqual(base.word_alignments[0].speaker, "user")
            self.assertEqual(base.word_alignments[2].speaker, "agent")
            self.assertEqual(
                [(item.speaker, item.word, item.start_frame) for item in base.word_alignments],
                [("user", "xin", 1), ("user", "chào", 7), ("agent", "chào", 26), ("agent", "bạn", 32)],
            )
            prompt_start = base.voice_prompt_frames + builder.pause_frames
            prompt_end = prompt_start + base.text_prompt_frames
            dialogue_start = base.prompt_frames
            self.assertLess(base.voice_prompt_frames, prompt_start)
            self.assertLess(prompt_end, dialogue_start)
            prompt_ids = tokenizer.encode("<system> Be helpful. <system>")
            self.assertEqual(prompt_ids, [607, 4831, 578, 1451, 3850, 263, 607, 4831, 578])
            self.assertEqual(
                base.labels[0][prompt_start:prompt_end], tuple(prompt_ids)
            )
            self.assertTrue(any(base.input_codes[1][: base.voice_prompt_frames]))
            expected_word_tokens = {
                "chào": [5617, 8260, 423],
                "bạn": [720, 229, 190, 165, 320],
            }
            for alignment in base.word_alignments:
                if alignment.speaker != "agent":
                    self.assertEqual(alignment.token_frames, ())
                    continue
                token_ids = tokenizer.encode(alignment.word)
                self.assertEqual(token_ids, expected_word_tokens[alignment.word])
                self.assertEqual(len(token_ids), len(alignment.token_frames))
                for token_id, frame in zip(token_ids, alignment.token_frames, strict=True):
                    self.assertEqual(base.labels[0][dialogue_start + frame], token_id)

            # Verify exact objective weights, including delay and appended batch padding.
            labels = self.torch.tensor(delayed.labels, dtype=self.torch.long)
            mask = self.torch.tensor(delayed.loss_mask, dtype=self.torch.bool)
            labels = self.torch.nn.functional.pad(labels, (0, 2), value=-1)
            mask = self.torch.nn.functional.pad(mask, (0, 2), value=False)
            weights = stream_weights_torch(labels, mask, (tokenizer.padding_id, tokenizer.end_padding_id))
            self.assertTrue(self.torch.all(weights[:, 0] == 0).item())
            self.assertTrue(self.torch.all(weights[:, -2:] == 0).item())
            for stream, delay in enumerate(delays):
                prompt_begin = 1 + delay
                prompt_end = prompt_begin + base.prompt_frames
                dialogue_begin = prompt_end
                dialogue_end = dialogue_begin + base.dialogue_frames
                self.assertTrue(self.torch.all(weights[stream, prompt_begin:prompt_end] == 0).item())
                self.assertTrue(self.torch.all(weights[stream, 1 : 1 + delay] == 0).item())
                delay_tail_start = 1 + delay + base.prompt_frames + base.dialogue_frames
                self.assertTrue(self.torch.all(weights[stream, delay_tail_start : delayed.total_frames] == 0).item())
            text_dialogue_begin = 1 + delays[0] + base.prompt_frames
            text_dialogue_end = text_dialogue_begin + base.dialogue_frames
            dialog_text = labels[0, text_dialogue_begin:text_dialogue_end]
            dialog_text_weights = weights[0, text_dialogue_begin:text_dialogue_end]
            self.assertTrue(self.torch.any((dialog_text == tokenizer.padding_id) & (dialog_text_weights == 0.3)).item())
            self.assertTrue(self.torch.any((dialog_text == tokenizer.end_padding_id) & (dialog_text_weights == 0.3)).item())
            for stream in range(1, 9):
                dialogue_begin = 1 + delays[stream] + base.prompt_frames
                dialogue_end = dialogue_begin + base.dialogue_frames
                silent_start = dialogue_begin + int(4.0 * mimi.frame_rate)
                silent_weights = weights[stream, silent_start:dialogue_end]
                expected_weight = 1.0 if stream == 1 else 0.02
                self.assertTrue(self.torch.allclose(
                    silent_weights,
                    self.torch.full_like(silent_weights, expected_weight),
                ))
            self.assertTrue(self.torch.all(weights[9, dialogue_start + delays[9] + 1:dialogue_start + delays[9] + 2] == 1).item())
            self.assertTrue(self.torch.all(weights[10, dialogue_start + delays[10] + 1:dialogue_start + delays[10] + 2] == 0.02).item())
            self.assertEqual(float(weights[1, 1 + delays[1] + base.prompt_frames]), 1.0)
            self.assertAlmostEqual(float(weights[2, 1 + delays[2] + base.prompt_frames]), 0.02, places=7)

            payload = build_debug_payload(sample, delayed, tokenizer, mimi.frame_rate, delays)
            for stream_index, stream_dump in enumerate(payload["streams"]):
                np.testing.assert_array_equal(
                    stream_dump["mask"], delayed.loss_mask[stream_index]
                )
                np.testing.assert_allclose(
                    stream_dump["weights"],
                    weights[stream_index, : delayed.total_frames].cpu().tolist(),
                )
            agent_only_payload = build_debug_payload(
                sample, delayed, tokenizer, mimi.frame_rate, delays, user_loss=False,
            )
            self.assertEqual(agent_only_payload["channel_map"]["right"], "user conditioning")
            for stream_dump in agent_only_payload["streams"][9:17]:
                self.assertTrue(all(weight == 0.0 for weight in stream_dump["weights"]))
            with self.subTest(debug_files=True):
                json_path, text_path = write_debug_artifacts(root / "debug", payload)
                decoded = json.loads(json_path.read_text(encoding="utf-8"))
                self.assertEqual(decoded["sample_id"], "golden_001")
                self.assertEqual(len(decoded["streams"]), 17)
                self.assertIn("token_piece", decoded["frames"][0])
                self.assertIn("weights", decoded["streams"][0])
                dialogue_region = decoded["regions"]["dialogue"]
                first_agent_word = next(event for event in decoded["word_events"] if event["word"] == "chào" and event["speaker"] == "agent")
                first_agent_frame = dialogue_region["start"] + 26
                self.assertAlmostEqual(decoded["frames"][first_agent_frame]["source_time_sec"], 2.08)
                self.assertEqual(first_agent_word["token_frames"], list(range(first_agent_frame, first_agent_frame + 3)))
                text = text_path.read_text(encoding="utf-8")
                self.assertIn("Text frame visualization", text)
                self.assertIn("Actual loss masks and weights", text)

    def _encode_one(self, mimi, audio: np.ndarray):
        torch_audio = self.torch.as_tensor(audio, dtype=self.torch.float32).reshape(1, 1, -1)
        with self.torch.no_grad():
            codes = mimi.encode(torch_audio)[0]
        return tuple(tuple(int(token) for token in stream.tolist()) for stream in codes)

    def test_official_moshi_delay_round_trip_except_masked_boundary(self) -> None:
        delays = list(self.loaders._lm_kwargs["delays"])
        original = self.torch.arange(17 * 12, dtype=self.torch.long).reshape(1, 17, 12)
        padding = self.torch.full((1, 17, 1), -1, dtype=self.torch.long)
        delayed = self.lm._delay_sequence(delays, original, padding)
        recovered, valid = self.lm._undelay_sequence(delays, delayed, fill_value=-1)
        for stream, delay in enumerate(delays):
            if delay:
                self.assertTrue(self.torch.all(delayed[0, stream, :delay] == -1).item())
                self.assertTrue(self.torch.equal(recovered[0, stream, :-delay], original[0, stream, :-delay]))
                self.assertTrue(self.torch.all(recovered[0, stream, -delay:] == -1).item())
                self.assertFalse(self.torch.any(valid[0, stream, -delay:]).item())
            else:
                self.assertTrue(self.torch.equal(recovered[0, stream], original[0, stream]))
                self.assertTrue(self.torch.all(valid[0, stream]).item())

    def test_cli_selects_sample_id_and_writes_both_debug_files(self) -> None:
        class Codec:
            frame_rate = 12.5
            codebooks = 8

            def encode_conversation_stereo(self, *_args):
                return tuple(tuple(range(4)) for _ in range(8)), tuple(tuple(range(4)) for _ in range(8))

            def encode_voice_prompt(self, _path):
                return tuple(tuple(range(2)) for _ in range(8))

            def sine(self, frames):
                return tuple(tuple([20 + index] * frames) for index in range(8))

            def silence(self, frames):
                return tuple(tuple([30 + index] * frames) for index in range(8))

        class Tokenizer:
            padding_id = 3
            end_padding_id = 0

            def encode(self, text):
                return [6] if text == " hello" else [4, 5]

            class Processor:
                @staticmethod
                def id_to_piece(token_id):
                    return {4: "▁friendly", 5: "▁assistant", 6: "▁hello"}.get(token_id, str(token_id))

            _processor = Processor()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = PreparedSample(
                sample_id="golden_cli",
                conversation_wav=root / "conversation.wav",
                voice_prompt_wav=root / "voice.wav",
                words=(Word("agent", "hello", 0.16, 0.56),),
                text_prompt="friendly assistant",
                metadata={},
                audio=AudioInfo(24000, 2, 5.0),
                window_start_sec=0.0,
                window_end_sec=5.0,
            )
            runtime = SimpleNamespace(
                codec=Codec(), tokenizer=Tokenizer(), initial_tokens=(1,) * 17,
                zero_token=-1, delays=(0,) * 17,
            )
            config = SimpleNamespace(
                device="cpu", model_root=MODEL_ROOT, personaplex_source=PROJECT_ROOT / "src",
                manifest=root / "train.jsonl", window_seconds=5.0,
                vietnamese_text_mode="diacritics",
            )
            output_dir = root / "debug"
            with patch.object(inspect_sample, "load_config", return_value=config), \
                    patch.object(inspect_sample, "PreparedDataset") as dataset, \
                    patch.object(inspect_sample, "load_runtime", return_value=runtime), \
                    patch("sys.argv", ["inspect_sample", "golden_cli", "--output-dir", str(output_dir)]):
                dataset.return_value.load.return_value = [sample]
                self.assertEqual(inspect_sample.main(), 0)
            self.assertTrue((output_dir / "sequence_debug.json").is_file())
            self.assertTrue((output_dir / "sequence_debug.txt").is_file())
            report = json.loads((output_dir / "sequence_debug.json").read_text(encoding="utf-8"))
            self.assertEqual(report["sample_id"], "golden_cli")


if __name__ == "__main__":
    unittest.main()
