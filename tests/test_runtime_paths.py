import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from personaplex_finetuning.data import read_stereo_window
from personaplex_finetuning.runtime import MimiCodec, RuntimePaths


class RuntimePathsTest(unittest.TestCase):
    def test_training_codes_match_inference_with_batch_sensitive_mimi(self) -> None:
        import torch

        class BatchSensitiveMimi:
            def __init__(self):
                self.shapes = []
                self.autocast = []

            def encode(self, audio):
                self.shapes.append(tuple(audio.shape))
                self.autocast.append(torch.is_autocast_enabled("cpu"))
                value = audio.mean(dim=(1, 2)).round().long() + 10 * audio.shape[0]
                return value[:, None, None].expand(-1, 8, 2)

        sources = {"first.wav": np.full((2, 16), [[1], [2]], dtype=np.float32),
                   "second.wav": np.full((2, 16), [[3], [4]], dtype=np.float32)}
        fake_sphn = types.SimpleNamespace(read=lambda path, **_kwargs: (sources[path], 24_000))
        model = BatchSensitiveMimi()
        codec = MimiCodec(model, 24_000, 12.5, "cpu", object())
        windows = [(Path("first.wav"), 0, 1, 0, 16 / 24_000),
                   (Path("second.wav"), 1, 0, 0, 16 / 24_000)]
        raw = {"waveforms": torch.from_numpy(np.stack(list(sources.values()))),
               "valid_samples": torch.tensor([16, 16])}
        with patch.dict("sys.modules", {"sphn": fake_sphn}):
            inference = [(codec.encode_conversation(p, a, s, e), codec.encode_conversation(p, u, s, e))
                         for p, a, u, s, e in windows]
            with torch.autocast("cpu", dtype=torch.bfloat16):
                training = codec.encode_conversation_stereo_batch(windows, raw_audio=raw)
                self.assertEqual(training, inference)
                unpadded = codec.encode_stereo_waveform(raw["waveforms"][1], 1, 0)
            self.assertEqual(training, inference)
            self.assertEqual(unpadded, inference[1])
            self.assertTrue(all(shape == (1, 1, 16) for shape in model.shapes))
            self.assertFalse(any(model.autocast))

    def test_legacy_stereo_batch_cache_is_not_reused(self) -> None:
        import hashlib
        import json
        import torch

        class MonoMimi:
            def __init__(self): self.calls = 0
            def encode(self, audio):
                self.calls += 1
                return torch.full((audio.shape[0], 8, 2), 7, dtype=torch.long)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "audio.wav"
            path.touch()
            model = MonoMimi()
            codec = MimiCodec(model, 24_000, 12.5, "cpu", object(), cache_dir=root / "cache")
            identity, _ = codec._conversation_cache_info(path, 0, 16 / 24_000)
            legacy = json.loads(identity)
            legacy["format_version"] = 1
            legacy.pop("encoding_contract", None)
            legacy.pop("encoding_device", None)
            old_identity = json.dumps(legacy, sort_keys=True, separators=(",", ":"))
            old_file = root / "cache" / (hashlib.sha256(old_identity.encode()).hexdigest() + ".pt")
            old_file.parent.mkdir()
            old_codes = tuple((99, 99) for _ in range(8))
            torch.save({"identity": old_identity, "agent": old_codes, "user": old_codes}, old_file)
            fake_sphn = types.SimpleNamespace(read=lambda *_args, **_kwargs: (np.ones((2, 16), dtype=np.float32), 24_000))
            with patch.dict("sys.modules", {"sphn": fake_sphn}):
                pair = codec.encode_conversation_stereo_cached(path, 0, 1, 0, 16 / 24_000)
                self.assertEqual(pair, (tuple((7, 7) for _ in range(8)),) * 2)
                self.assertEqual(codec.encode_conversation_stereo_cached(path, 0, 1, 0, 16 / 24_000), pair)
            self.assertEqual(model.calls, 2)
            self.assertTrue(old_file.exists())

    def test_cache_is_shared_across_role_swap_and_ddp_ranks(self) -> None:
        import torch

        class ChannelMimi:
            def __init__(self): self.calls = 0
            def encode(self, audio):
                self.calls += 1
                # Code value identifies the physical channel (LEFT=1, RIGHT=2).
                return torch.full((audio.shape[0], 8, 2), int(audio.mean().item()), dtype=torch.long)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "audio.wav"
            path.touch()
            stereo = np.vstack((np.ones(16, dtype=np.float32), np.full(16, 2, dtype=np.float32)))
            fake_sphn = types.SimpleNamespace(read=lambda *_args, **_kwargs: (stereo, 24_000))
            left, right = tuple((1, 1) for _ in range(8)), tuple((2, 2) for _ in range(8))
            model = ChannelMimi()
            first = MimiCodec(model, 24_000, 12.5, "cpu", object(), cache_dir=root / "cache")
            with patch.dict("sys.modules", {"sphn": fake_sphn}):
                self.assertEqual(first.encode_conversation_stereo_cached(path, 0, 1, 0, 16 / 24_000), (left, right))
                self.assertEqual(model.calls, 2)
                # Second role pass swaps agent/user: same physical channels, no re-encode.
                self.assertEqual(first.encode_conversation_stereo_cached(path, 1, 0, 0, 16 / 24_000), (right, left))
                self.assertEqual(model.calls, 2)
            rank0 = MimiCodec(model, 24_000, 12.5, "cuda:0", object(), cache_dir=root / "cache")
            rank1 = MimiCodec(model, 24_000, 12.5, "cuda:1", object(), cache_dir=root / "cache")
            self.assertEqual(rank0._conversation_cache_info(path, 0, 1), rank1._conversation_cache_info(path, 0, 1))
            self.assertNotEqual(rank0._conversation_cache_info(path, 0, 1), first._conversation_cache_info(path, 0, 1))

    def test_inference_encodes_mono_channel_zero_and_preserves_stereo_channel_selection(self) -> None:
        import torch  # Load before patch.dict(sys.modules) restores the module table.

        mono = np.ones((1, 8), dtype=np.float32)
        stereo = np.vstack((mono, mono * 2))
        sources = {"mono.wav": mono[0], "mono2d.wav": mono, "stereo.wav": stereo}
        fake_sphn = types.SimpleNamespace(read=lambda path, **_kwargs: (sources[path], 24_000))
        codec = MimiCodec(object(), 24_000, 12.5, "cpu", object())

        with patch.dict("sys.modules", {"sphn": fake_sphn}), patch.object(codec, "_encode", side_effect=lambda audio, _torch: audio.copy()):
            np.testing.assert_array_equal(codec.encode_conversation(Path("mono.wav"), 0, 0, 8 / 24_000), mono)
            np.testing.assert_array_equal(codec.encode_conversation(Path("mono2d.wav"), 0, 0, 8 / 24_000), mono)
            np.testing.assert_array_equal(codec.encode_conversation(Path("stereo.wav"), 1, 0, 8 / 24_000), stereo[1:2])
            with self.assertRaisesRegex(ValueError, "channel 1 is unavailable"):
                codec.encode_conversation(Path("mono.wav"), 1, 0, 8 / 24_000)
            with self.assertRaisesRegex(ValueError, "stereo \\[2,T\\]"):
                read_stereo_window(Path("mono.wav"), 0, 8 / 24_000, 24_000, "training")

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
                self.assertEqual(model.calls, 4)
                expected = [
                    (tuple((1,) * 2 for _ in range(8)), tuple((2,) * 2 for _ in range(8))),
                    (tuple((4,) * 2 for _ in range(8)), tuple((3,) * 2 for _ in range(8))),
                ]
                self.assertEqual(actual, expected)
                self.assertEqual(codec.encode_conversation_stereo_batch(windows), expected)
                self.assertEqual(model.calls, 4)

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
