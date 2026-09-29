import contextlib
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from personaplex_finetuning import inference
from tools import inference_smoke
from tools.inference_smoke import select_inference_window


def _write_lora_weights(path: Path, rank: int) -> None:
    save_file({
        "transformer.layers.0.linear.lora_a.weight": torch.zeros((rank, 3)),
        "transformer.layers.0.linear.lora_b.weight": torch.zeros((2, rank)),
    }, str(path))


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

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
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
    @contextlib.contextmanager
    def _generate_with_fakes(self, generation=None, seed=1234, adapter=None):
        mimi = _Mimi()
        seeded = []
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
             patch.dict(sys.modules, {"sphn": fake_sphn}), \
             patch.object(torch, "manual_seed", side_effect=seeded.append), \
             patch.object(inference, "load_runtime", return_value=runtime), \
             patch.object(inference, "inject_lora") as inject_lora, \
             patch.object(inference, "load_adapter") as load_adapter, \
             patch.object(importlib, "import_module", return_value=fake_lm):
            inference.generate(
                config, sample, Path(directory) / "agent.wav", Path(directory) / "agent.txt", adapter,
                generation=generation, seed=seed,
            )
            yield SimpleNamespace(
                mimi=mimi, config=config, seeded=seeded, generator=_Generator.instances[0],
                model=runtime.model, inject_lora=inject_lora, load_adapter=load_adapter,
            )

    def test_generate_keeps_mimi_decoder_streaming_across_generated_frames(self):
        with self._generate_with_fakes() as fakes:
            self.assertEqual(fakes.mimi.streaming_calls, [1])
            self.assertEqual(fakes.generator.streaming_calls, [1])
            self.assertEqual(fakes.mimi.decode_calls, 2)

    def test_generate_disables_moshi_compile_before_loading_runtime(self):
        with patch.dict(os.environ):
            os.environ.pop("NO_TORCH_COMPILE", None)
            with self._generate_with_fakes():
                self.assertEqual(os.environ.get("NO_TORCH_COMPILE"), "1")

    def test_generate_reads_lora_rank_and_alpha_from_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_000123"
            checkpoint.mkdir()
            adapter_file = checkpoint / "lora.safetensors"
            _write_lora_weights(adapter_file, rank=64)
            (checkpoint / "adapter.json").write_text(
                json.dumps({"rank": 64, "alpha": 128}), encoding="utf-8",
            )

            with self._generate_with_fakes(adapter=checkpoint) as fakes:
                fakes.inject_lora.assert_called_once_with(fakes.model, 64, 128)
                fakes.load_adapter.assert_called_once_with(fakes.model, adapter_file)

    def test_resolve_adapter_requires_valid_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            adapter_file = checkpoint / "lora.safetensors"
            _write_lora_weights(adapter_file, rank=8)
            (checkpoint / "adapter.json").write_text(
                json.dumps({"rank": 8, "alpha": 16}), encoding="utf-8",
            )

            self.assertEqual(
                inference.resolve_adapter_checkpoint(checkpoint),
                (adapter_file, 8, 16),
            )
            (checkpoint / "adapter.json").write_text(
                json.dumps({"rank": 8}), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "alpha"):
                inference.resolve_adapter_checkpoint(adapter_file)

    def test_resolve_adapter_recovers_legacy_alpha_from_run_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            checkpoint = run_dir / "checkpoints" / "checkpoint_000123"
            checkpoint.mkdir(parents=True)
            adapter_file = checkpoint / "lora.safetensors"
            _write_lora_weights(adapter_file, rank=128)
            (checkpoint / "adapter.json").write_text(
                json.dumps({"rank": 128, "alpha": 32}), encoding="utf-8",
            )
            (run_dir / "config.json").write_text(
                json.dumps({"lora_rank": 128, "lora_scaling": 2.0}), encoding="utf-8",
            )

            self.assertEqual(
                inference.resolve_adapter_checkpoint(checkpoint),
                (adapter_file, 128, 256),
            )
            (checkpoint / "adapter.json").write_text(
                json.dumps({"rank": 128, "alpha": 32, "scaling": 2.0}), encoding="utf-8",
            )
            self.assertEqual(
                inference.resolve_adapter_checkpoint(checkpoint),
                (adapter_file, 128, 256),
            )

    def test_resolve_adapter_uses_tensor_rank_when_metadata_is_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            adapter_file = checkpoint / "lora.safetensors"
            _write_lora_weights(adapter_file, rank=128)
            (checkpoint / "adapter.json").write_text(
                json.dumps({"rank": 16, "alpha": 32, "scaling": 2.0}), encoding="utf-8",
            )

            with self.assertLogs(inference.logger, level="WARNING") as logged:
                resolved = inference.resolve_adapter_checkpoint(checkpoint)

        self.assertEqual(resolved, (adapter_file, 128, 256))
        self.assertIn("using tensor rank", logged.output[0])

    def test_generate_without_settings_keeps_the_previous_hard_coded_values(self):
        # Regression guard for the milestone: an absent generation block must not change
        # what LMGen receives compared to the pre-config behaviour.
        with self._generate_with_fakes(generation=None) as fakes:
            self.assertEqual(fakes.generator.init_kwargs["use_sampling"], True)
            self.assertEqual(fakes.generator.init_kwargs["temp"], 0.8)
            self.assertEqual(fakes.generator.init_kwargs["temp_text"], 0.7)
            self.assertEqual(fakes.generator.init_kwargs["top_k"], 250)
            self.assertEqual(fakes.generator.init_kwargs["top_k_text"], 25)
            self.assertEqual(fakes.generator.init_kwargs["audio_silence_frame_cnt"], 6)
            self.assertEqual(fakes.seeded, [1234])

    def test_generate_forwards_the_configured_generation_settings_to_lmgen(self):
        settings = inference.GenerationSettings(
            use_sampling=True, temp=0.5, temp_text=0.2, top_k=7, top_k_text=3, audio_silence_frame_cnt=2,
        )

        with self._generate_with_fakes(generation=settings, seed=99) as fakes:
            for name, value in settings.as_dict().items():
                self.assertEqual(fakes.generator.init_kwargs[name], value, name)
            self.assertEqual(fakes.seeded, [99])

    def test_generate_skips_rng_seeding_when_sampling_is_disabled(self):
        # Greedy decoding is deterministic already, so seeding would add nothing.
        with self._generate_with_fakes(generation=inference.GenerationSettings(use_sampling=False)) as fakes:
            self.assertEqual(fakes.generator.init_kwargs["use_sampling"], False)
            self.assertEqual(fakes.seeded, [])


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


class OriginalDialogueExportTest(unittest.TestCase):
    def test_exports_original_window_with_original_channel_order(self):
        original = np.array([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]], dtype=np.float32)
        written = {}
        fake_sphn = types.SimpleNamespace(
            read=lambda _path: (original, 24000),
            write_wav=lambda path, audio, sample_rate: written.update(
                {Path(path).name: (np.array(audio), sample_rate)}
            ),
        )
        sample = SimpleNamespace(
            conversation_wav=Path("conversation.wav"),
            user_channel=1,
            window_start_sec=2 / 24000,
            window_end_sec=5 / 24000,
            words=(),
            text_prompt="Prompt",
            voice_prompt_wav=Path("missing.wav"),
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"sphn": fake_sphn}):
            inference._export_context(sample, Path(directory))

        self.assertIn("dialogue_original.wav", written)
        np.testing.assert_array_equal(written["dialogue_original.wav"][0], original[:, 2:5])
        self.assertEqual(written["dialogue_original.wav"][1], 24000)


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


class GenerationSettingsTest(unittest.TestCase):
    def test_defaults_match_the_moshi_lmgen_reference(self):
        self.assertEqual(inference.GenerationSettings().as_dict(), {
            "use_sampling": True, "temp": 0.8, "temp_text": 0.7,
            "top_k": 250, "top_k_text": 25, "audio_silence_frame_cnt": 6,
        })

    def test_missing_or_empty_generation_block_falls_back_to_defaults(self):
        for raw in (None, {}, {"generation": None}, {"generation": {}}):
            self.assertEqual(inference.generation_from_config(raw), inference.GenerationSettings(), str(raw))

    def test_partial_block_only_overrides_the_given_keys(self):
        settings = inference.generation_from_config({"generation": {"temp": 0.9}})

        self.assertEqual(settings.temp, 0.9)
        self.assertEqual(settings.temp_text, 0.7)
        self.assertEqual(settings.top_k, 250)

    def test_hydra_style_section_is_accepted(self):
        # tools.inference_smoke passes an OmegaConf DictConfig, not a plain dict.
        try:
            from omegaconf import OmegaConf
        except ImportError:  # pragma: no cover - omegaconf is a package dependency
            self.skipTest("omegaconf is not installed")

        settings = inference.generation_from_config(
            OmegaConf.create({"generation": {"temp": 0.6, "use_sampling": False}})
        )

        self.assertEqual(settings.temp, 0.6)
        self.assertFalse(settings.use_sampling)

    def test_unknown_key_is_rejected_instead_of_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "unknown generation config keys"):
            inference.generation_from_config({"generation": {"temperature": 0.9}})

    def test_invalid_values_are_rejected(self):
        for section in (
            {"temp": -0.1},
            {"temp_text": "warm"},
            {"top_k": -1},
            {"top_k_text": 1.5},
            {"use_sampling": "maybe"},
            {"audio_silence_frame_cnt": -1},
        ):
            with self.assertRaises(ValueError, msg=str(section)):
                inference.generation_from_config({"generation": section})

    def test_zero_temperature_is_allowed_and_means_greedy_for_that_stream(self):
        settings = inference.generation_from_config({"generation": {"temp": 0.0}})

        self.assertEqual(settings.temp, 0.0)
        self.assertIn("audio=greedy", settings.label())
        self.assertIn("text=sampling", settings.label())

    def test_label_describes_the_mechanism_actually_used(self):
        # Regression guard for the wrong "greedy native LMGen" label once written to run.json.
        self.assertNotIn("greedy", inference.GenerationSettings().label())
        self.assertIn("audio=sampling temp=0.8", inference.GenerationSettings().label())
        self.assertIn("text=sampling temp=0.7", inference.GenerationSettings().label())
        self.assertIn("audio=greedy", inference.GenerationSettings(use_sampling=False).label())
        self.assertIn("top_k=250", inference.GenerationSettings().label())


class SmokeGenerationReportTest(unittest.TestCase):
    def test_run_report_records_the_settings_and_seed_used_by_both_runs(self):
        sample = SimpleNamespace(
            sample_id="sample-0",
            voice_prompt_wav=Path("voice.wav"),
            text_prompt="Helpful",
            conversation_wav=Path("conversation.wav"),
            user_channel=1,
            window_start_sec=0.0,
            window_end_sec=0.1,
            words=(),
        )
        config = SimpleNamespace(model_root=Path("model"), seed=1234)
        settings = inference.GenerationSettings(use_sampling=False, audio_silence_frame_cnt=2)
        fake_sphn = types.SimpleNamespace(
            read=lambda _path: (_ for _ in ()).throw(AssertionError("generate is patched, nothing should be read"))
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch.object(inference, "_export_context"), \
                 patch.object(inference, "generate") as generate, \
                 patch.dict(sys.modules, {"sphn": fake_sphn}):
                inference.smoke(config, sample, Path("adapter.safetensors"), output_dir, generation=settings)
            report = json.loads((output_dir / "run.json").read_text())

        self.assertEqual(report["generation_settings"], settings.as_dict())
        self.assertEqual(report["generation"], settings.label())
        self.assertIn("greedy", report["generation"])
        self.assertEqual(report["seed"], 1234)
        # Both the base and the fine-tuned run must use identical settings and seed.
        self.assertEqual(len(generate.call_args_list), 2)
        for call in generate.call_args_list:
            self.assertEqual(call.kwargs["generation"], settings)
            self.assertEqual(call.kwargs["seed"], 1234)


class InferenceCliGenerationTest(unittest.TestCase):
    def _run_cli(self, config_text):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "infer.yaml"
            config_path.write_text(config_text, encoding="utf-8")
            sample = SimpleNamespace(
                sample_id="sample-0", window_start_sec=0.0, window_end_sec=1.0,
                audio=SimpleNamespace(duration_sec=1.0),
            )
            with patch.object(sys, "argv", ["inference_smoke.py", "--config", str(config_path)]), \
                 patch.object(inference_smoke, "load_config", return_value=SimpleNamespace(
                     manifest="manifest.jsonl", window_seconds=30.0,
                 )), \
                 patch.object(inference_smoke, "PreparedDataset", return_value=SimpleNamespace(
                     load=lambda: [sample],
                 )), \
                 patch.object(inference_smoke, "smoke", autospec=True) as smoke:
                status = inference_smoke.main()
            return status, smoke

    def test_generation_block_from_config_reaches_smoke(self):
        status, smoke = self._run_cli(
            "adapter:\n  path: adapter.safetensors\n"
            "generation:\n  use_sampling: false\n  temp: 0.9\n  top_k: 17\n"
            "inference:\n  sample_index: 0\n"
        )

        self.assertEqual(status, 0)
        self.assertEqual(
            smoke.call_args.kwargs["generation"],
            inference.GenerationSettings(use_sampling=False, temp=0.9, top_k=17),
        )

    def test_invalid_generation_block_fails_before_any_model_is_loaded(self):
        with self.assertRaisesRegex(ValueError, "top_k_text"):
            self._run_cli(
                "adapter:\n  path: adapter.safetensors\n"
                "generation:\n  top_k_text: -3\n"
                "inference:\n  sample_index: 0\n"
            )
