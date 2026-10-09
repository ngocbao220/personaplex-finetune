"""Bounded first-batch diagnostics before any DDP collective."""
import unittest
from contextlib import redirect_stdout
from io import StringIO
import json
from types import SimpleNamespace
from unittest.mock import patch

from personaplex_finetuning.train import iter_training_batches


class BatchStartupTest(unittest.TestCase):
    def config(self, workers):
        return SimpleNamespace(per_device_batch_size=1, seed=42, shuffle=False,
                               num_workers=workers, pin_memory=False,
                               persistent_workers=False, prefetch_factor=2)

    def test_worker_loader_has_bounded_timeout_and_forkserver_context(self):
        runtime = SimpleNamespace(codec=SimpleNamespace(
            sample_rate=24000, encode_conversation_stereo_batch=lambda *a, **k: None))
        with patch("personaplex_finetuning.train.torch.utils.data.DataLoader",
                   side_effect=RuntimeError("loader inspected")) as loader:
            with self.assertRaisesRegex(RuntimeError, "loader inspected"):
                next(iter_training_batches(self.config(2), [SimpleNamespace(sample_id="sample", conversation_wav="audio.wav", window_start_sec=0, window_end_sec=1, audio=SimpleNamespace(duration_sec=1))] * 2, runtime, "cpu", 0, 2, False))
        self.assertEqual(loader.call_args.kwargs["multiprocessing_context"], "forkserver")
        self.assertGreater(loader.call_args.kwargs["timeout"], 0)
        self.assertLess(loader.call_args.kwargs["timeout"], 600)

    def test_no_worker_loader_uses_zero_timeout(self):
        runtime = SimpleNamespace(codec=SimpleNamespace(
            sample_rate=24000, encode_conversation_stereo_batch=lambda *a, **k: None))
        with patch("personaplex_finetuning.train.torch.utils.data.DataLoader",
                   side_effect=RuntimeError("loader inspected")) as loader:
            with self.assertRaisesRegex(RuntimeError, "loader inspected"):
                next(iter_training_batches(self.config(0), [SimpleNamespace(sample_id="sample", conversation_wav="audio.wav", window_start_sec=0, window_end_sec=1, audio=SimpleNamespace(duration_sec=1))] * 2, runtime, "cpu", 0, 2, False))
        self.assertEqual(loader.call_args.kwargs["timeout"], 0)

    def test_batch_watchdog_cancels_even_when_preparation_raises(self):
        from personaplex_finetuning.train import trace_batch_preparation
        stream = StringIO()
        with patch("personaplex_finetuning.train.faulthandler.dump_traceback_later") as arm, \
             patch("personaplex_finetuning.train.faulthandler.cancel_dump_traceback_later") as cancel, \
             redirect_stdout(stream):
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                with trace_batch_preparation(0, 0, enabled=True):
                    raise RuntimeError("decode failed")
        arm.assert_called_once()
        cancel.assert_called_once()
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(rows[0]["phase"], "batch_prepare")
        self.assertEqual(rows[0]["rank"], 0)
        self.assertEqual(rows[-1]["status"], "error")
        self.assertIn("decode failed", rows[-1]["error"])

    def test_success_and_disabled_trace_clean_up_without_changing_result(self):
        from personaplex_finetuning.train import trace_batch_preparation
        stream = StringIO()
        with patch("personaplex_finetuning.train.faulthandler.dump_traceback_later") as arm, \
             patch("personaplex_finetuning.train.faulthandler.cancel_dump_traceback_later") as cancel, \
             redirect_stdout(stream):
            with trace_batch_preparation(1, 4, enabled=True):
                value = "batch"
            with trace_batch_preparation(1, 5, enabled=False):
                self.assertEqual(value, "batch")
        arm.assert_called_once()
        cancel.assert_called_once()
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["status"], "complete")
        self.assertEqual(rows[-1]["micro_step"], 4)

    def test_worker_indices_resolve_original_parent_samples_in_rank_order(self):
        import numpy as np
        import torch
        from pathlib import Path
        from personaplex_finetuning.batching import RawAudioItem, collate_raw_audio
        from personaplex_finetuning.data import AudioInfo, PreparedSample
        samples = [PreparedSample(str(i), Path(f"{i}.wav"), Path("voice.wav"), (),
                                  f"parent prompt {i}", {"original": i}, AudioInfo(24000, 2, 1), 0, 1)
                   for i in range(2)]
        codec = SimpleNamespace(sample_rate=24000, frame_rate=10,
                                encode_conversation_stereo_batch=lambda windows, raw_audio: ["codes"])
        runtime = SimpleNamespace(codec=codec, tokenizer=SimpleNamespace(padding_id=3), zero_token=-1)
        config = self.config(0)
        config.duration_sec = 1
        config.prompt_aug_prob = 0
        example = SimpleNamespace(prompt_frames=0, dialogue_frames=10,
                                  input_codes=((0,) * 10,) * 17, loss_mask=((True,) * 10,) * 17)

        def loader(**options):
            window = options["dataset"].windows[1]
            return [collate_raw_audio([RawAudioItem(window, torch.from_numpy(np.zeros((2, 24000), dtype=np.float32)), 24000)])]

        with patch("personaplex_finetuning.train.torch.utils.data.DataLoader", side_effect=loader), \
             patch("personaplex_finetuning.train.build_example", return_value=example) as build, \
             patch("personaplex_finetuning.train.pad_training_example", side_effect=lambda item, *args: item), \
             patch("personaplex_finetuning.train.post_encode_collate", return_value={"codes": "batch"}):
            iterator = iter_training_batches(config, samples, runtime, "cpu", 1, 2, False)
            try:
                result = next(iterator)
            finally:
                iterator.close()
        self.assertIs(build.call_args.args[1], samples[1])
        self.assertIs(result[-1][0], samples[1])
        self.assertEqual(build.call_args.args[1].text_prompt, "parent prompt 1")
