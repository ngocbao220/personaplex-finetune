import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from personaplex_finetuning import train
from personaplex_finetuning.runtime import MimiCodec


class CountingMimi:
    def __init__(self):
        self.calls = 0

    def encode(self, audio):
        self.calls += 1
        return torch.full((audio.shape[0], 8, 2), int(audio.mean().item()), dtype=torch.long)


def chunk(path, start, agent=0):
    return SimpleNamespace(conversation_wav=path, agent_channel=agent, user_channel=1 - agent,
                           window_start_sec=start, window_end_sec=start + 16 / 24_000)


class CodecCachePrecomputeTest(unittest.TestCase):
    def test_precompute_fills_rank_share_and_training_reads_it_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "audio.wav"
            path.touch()
            stereo = np.vstack((np.ones(16, dtype=np.float32), np.full(16, 2, dtype=np.float32)))
            fake_sphn = types.SimpleNamespace(read=lambda *_a, **_k: (stereo, 24_000))
            model = CountingMimi()
            codec = MimiCodec(model, 24_000, 12.5, "cpu", object(), cache_dir=root / "cache")
            config = SimpleNamespace(codec_cache_dir=root / "cache", model_root=root, personaplex_source=root)
            chunks = [chunk(path, 0.0), chunk(path, 1.0), chunk(path, 2.0)]

            with patch.dict("sys.modules", {"sphn": fake_sphn}), \
                    patch.object(train, "load_mimi_codec", return_value=codec):
                rank1 = train.precompute_codec_cache(config, chunks, "cpu", rank=1, world_size=2)
                self.assertEqual(rank1, {"encoded": 1, "already_cached": 0})
                rank0 = train.precompute_codec_cache(config, chunks, "cpu", rank=0, world_size=2)
                self.assertEqual(rank0, {"encoded": 2, "already_cached": 0})
                self.assertEqual(model.calls, 6)  # 3 windows x 2 channels
                # Re-run and role-swapped training reads are pure cache hits.
                again = train.precompute_codec_cache(config, chunks, "cpu", rank=0, world_size=1)
                self.assertEqual(again, {"encoded": 0, "already_cached": 3})
                swapped = codec.encode_conversation_stereo_batch([
                    (path, 1, 0, 1.0, 1.0 + 16 / 24_000),
                ])
                self.assertEqual(model.calls, 6)
            self.assertEqual(swapped[0][0], tuple((2, 2) for _ in range(8)))

    def test_requires_cache_directory(self):
        with self.assertRaisesRegex(ValueError, "codec_cache_dir"):
            train.precompute_codec_cache(SimpleNamespace(codec_cache_dir=None), [], "cpu", 0, 1)


if __name__ == "__main__":
    unittest.main()
