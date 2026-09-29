import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from personaplex_finetuning.runtime import MimiCodec, RuntimePaths


class RuntimePathsTest(unittest.TestCase):
    def test_stereo_mimi_batch_matches_per_sample_channels_and_uses_disk_cache(self) -> None:
        import numpy as np
        import torch

        class CodecModel:
            def __init__(self):
                self.calls = 0

            def encode(self, audio):
                self.calls += 1
                values = audio.mean(dim=(1, 2)).round().long()
                return values[:, None, None].expand(-1, 8, 2)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "first.wav", root / "second.wav"
            first.touch()
            second.touch()
            source = {
                str(first): np.stack([np.ones(16), np.full(16, 2.0)]).astype(np.float32),
                str(second): np.stack([np.full(16, 3.0), np.full(16, 4.0)]).astype(np.float32),
            }
            fake_sphn = types.SimpleNamespace(
                read=lambda path, **_kwargs: (source[path], 24_000),
            )
            model = CodecModel()
            codec = MimiCodec(
                model, 24_000, 12.5, "cpu", object(), cache_dir=root / "cache",
                cache_namespace="test-mimi",
            )
            windows = [
                (first, 0, 1, 0.0, 16 / 24_000),
                (second, 1, 0, 0.0, 16 / 24_000),
            ]
            raw_audio = {
                "waveforms": torch.as_tensor(np.stack([source[str(first)], source[str(second)]])),
                "valid_samples": torch.tensor([16, 16]),
            }
            with patch.dict("sys.modules", {"sphn": fake_sphn}):
                actual = codec.encode_conversation_stereo_batch(windows, raw_audio=raw_audio)
                self.assertEqual(model.calls, 1)
                expected = [
                    (tuple((1,) * 2 for _ in range(8)), tuple((2,) * 2 for _ in range(8))),
                    (tuple((4,) * 2 for _ in range(8)), tuple((3,) * 2 for _ in range(8))),
                ]
                self.assertEqual(actual, expected)
                self.assertEqual(codec.encode_conversation_stereo_batch(windows), expected)
                self.assertEqual(model.calls, 1)

    def test_training_prompt_frames_match_personaplex_native_tokens_without_mimi_encode(self) -> None:
        class Helpers:
            SINE_TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]
            SILENCE_TOKENS = [11, 12, 13, 14, 15, 16, 17, 18]

        class CodecModel:
            def encode(self, *_args):
                raise AssertionError("native prompt token frames must not be Mimi re-encoded")

        codec = MimiCodec(CodecModel(), 24_000, 12.5, "cpu", Helpers())

        self.assertEqual(codec.sine(3), tuple((token,) * 3 for token in Helpers.SINE_TOKENS))
        self.assertEqual(codec.silence(2), tuple((token,) * 2 for token in Helpers.SILENCE_TOKENS))

    def test_requires_vendored_moshi_loader_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "moshi").mkdir(parents=True)
            model = root / "model"
            model.mkdir()
            for name in ("model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer_spm_32k_3.model"):
                (model / name).write_bytes(b"x")

            with self.assertRaisesRegex(FileNotFoundError, "moshi/models/loaders.py"):
                RuntimePaths(model, source).validate()

    def test_requires_all_local_personaplex_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "moshi" / "models").mkdir(parents=True)
            (source / "moshi" / "models" / "loaders.py").touch()
            model = root / "model"
            model.mkdir()
            for name in ("model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer_spm_32k_3.model"):
                (model / name).write_bytes(b"x")

            paths = RuntimePaths(model, source)

            self.assertEqual(paths.validate().moshi_weight.name, "model.safetensors")
            (model / "model.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "model.safetensors"):
                paths.validate()
