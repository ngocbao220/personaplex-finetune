import contextlib
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from personaplex_finetuning import inference
from tools import inference_smoke
from tools.inference_smoke import select_inference_window


class _UserTokens:
    shape = (1, 8, 2)

    def unsqueeze(self, _dimension):
        return self

    def __getitem__(self, _index):
        return self


class _GeneratedTokens:
    def __getitem__(self, index):
        if isinstance(index, tuple) and index == (0, 0, 0):
            return 7
        return self


class _DecodedAudio:
    def squeeze(self):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.zeros(1920, dtype=np.float32)


class _Mimi:
    def __init__(self):
        self.streaming_calls = []
        self.decode_calls = 0

    def streaming(self, batch_size):
        self.streaming_calls.append(batch_size)
        return contextlib.nullcontext()

    def decode(self, _tokens):
        self.decode_calls += 1
        return _DecodedAudio()


class _Generator:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.streaming_calls = []
        self.text_prompt_tokens = None
        type(self).instances.append(self)

    def streaming(self, batch_size):
        self.streaming_calls.append(batch_size)
        return contextlib.nullcontext()

    def load_voice_prompt(self, _path):
        pass

    def step_system_prompts(self, _mimi):
        pass

    def step(self, **_kwargs):
        return _GeneratedTokens()


class InferenceStreamingTest(unittest.TestCase):
    def test_generate_keeps_mimi_decoder_streaming_across_generated_frames(self):
        mimi = _Mimi()
        runtime = SimpleNamespace(
            model=SimpleNamespace(eval=lambda: None),
            codec=SimpleNamespace(
                mimi=mimi,
                sample_rate=24000,
                frame_rate=12.5,
                encode_conversation=lambda *_args: ((1, 2),) * 8,
            ),
            tokenizer=SimpleNamespace(
                padding_id=3,
                encode=lambda _text: [1],
                _processor=SimpleNamespace(id_to_piece=lambda _token: "hello"),
            ),
        )
        fake_torch = types.SimpleNamespace(
            no_grad=contextlib.nullcontext,
            tensor=lambda *_args, **_kwargs: _UserTokens(),
        )
        fake_sphn = types.SimpleNamespace(
            write_wav=lambda path, _audio, _sample_rate: Path(path).write_bytes(b"wav"),
        )
        fake_lm = SimpleNamespace(LMGen=_Generator)
        config = SimpleNamespace(model_root="model", personaplex_source="source", device="cpu", qlora=False, quant_type="nf4")
        sample = SimpleNamespace(
            voice_prompt_wav=Path("voice.wav"), text_prompt="Helpful", conversation_wav=Path("conversation.wav"),
            user_channel=1, window_start_sec=0.0, window_end_sec=0.16,
        )

        _Generator.instances = []
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(sys.modules, {"torch": fake_torch, "sphn": fake_sphn}), \
             patch.object(inference, "load_runtime", return_value=runtime), \
             patch.object(importlib, "import_module", return_value=fake_lm):
            inference.generate(config, sample, Path(directory) / "agent.wav", Path(directory) / "agent.txt", None)

        self.assertEqual(mimi.streaming_calls, [1])
        self.assertEqual(_Generator.instances[0].streaming_calls, [1])
        self.assertEqual(mimi.decode_calls, 2)


class InferenceOutputWarningTest(unittest.TestCase):
    def test_missing_outputs_warn_and_write_run_report_instead_of_raising(self):
        sample = SimpleNamespace(
            sample_id="sample-0",
            voice_prompt_wav=Path("voice.wav"),
            text_prompt="Helpful",
            conversation_wav=Path("conversation.wav"),
            user_channel=1,
            window_start_sec=0.0,
            window_end_sec=0.16,
            words=(),
        )
        config = SimpleNamespace(model_root=Path("model"))
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch.object(inference, "_export_context"), \
                 patch.object(inference, "generate"), \
                 patch.dict(sys.modules, {"sphn": types.SimpleNamespace(read=lambda _path: (_ for _ in ()).throw(AssertionError("missing output should not be read")))}), \
                 self.assertLogs(inference.__name__, level="WARNING") as logs:
                inference.smoke(config, sample, Path("adapter.safetensors"), output_dir)

            report = __import__("json").loads((output_dir / "run.json").read_text())

        self.assertTrue(any("missing or empty" in message for message in logs.output))
        self.assertGreater(len(report["warnings"]), 0)


class InferenceWindowSelectionTest(unittest.TestCase):
    def test_start_selects_an_exact_configured_window(self):
        sample = SimpleNamespace(
            audio=SimpleNamespace(duration_sec=90.0),
            with_window=lambda start, end: (start, end),
        )

        self.assertEqual(select_inference_window(sample, start=42.5, window_seconds=30.0), (42.5, 72.5))

    def test_start_rejects_a_window_past_the_end_of_the_conversation(self):
        sample = SimpleNamespace(audio=SimpleNamespace(duration_sec=60.0))

        with self.assertRaisesRegex(ValueError, "requires 30"):
            select_inference_window(sample, start=30.1, window_seconds=30.0)

    def test_start_rejects_a_negative_offset(self):
        sample = SimpleNamespace(audio=SimpleNamespace(duration_sec=60.0))

        with self.assertRaisesRegex(ValueError, "non-negative"):
            select_inference_window(sample, start=-0.1, window_seconds=30.0)


class InferenceCliTest(unittest.TestCase):
    def test_input_path_alias_is_forwarded_as_the_external_input(self):
        sample = SimpleNamespace(
            sample_id="sample-0",
            window_start_sec=0.0,
            window_end_sec=30.0,
            audio=SimpleNamespace(duration_sec=30.0),
        )
        with patch.object(sys, "argv", [
            "inference_smoke.py",
            "--config", "config.yaml",
            "--adapter", "adapter.safetensors",
            "--input-path", "external.wav",
        ]), patch.object(inference_smoke, "load_config", return_value=SimpleNamespace(
            manifest="manifest.jsonl", window_seconds=30.0,
        )), patch.object(inference_smoke, "PreparedDataset", return_value=SimpleNamespace(
            load=lambda: [sample],
        )), patch.object(inference_smoke, "smoke", autospec=True) as smoke:
            self.assertEqual(inference_smoke.main(), 0)

        self.assertEqual(smoke.call_args.kwargs["input_file"], Path("external.wav").resolve())
        self.assertIsInstance(smoke.call_args.kwargs["adapter"], Path)
        self.assertIsInstance(smoke.call_args.kwargs["output_dir"], Path)
