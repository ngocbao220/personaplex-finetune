import unittest
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from personaplex_finetuning.batching import (
    DistributedBucketBatchSampler,
    RawAudioDataset,
    RawAudioItem,
    collate_raw_audio,
    post_encode_collate,
)
from personaplex_finetuning.data import AudioInfo, PreparedSample
from personaplex_finetuning.objective import stream_weights_torch
from personaplex_finetuning.runtime import MimiCodec
from personaplex_finetuning.sequence import PersonaPlexTrainingExampleBuilder, TrainingExample
from personaplex_finetuning.train import loss_components


def example(length, padding_id=3):
    streams = [[0] * length for _ in range(17)]
    masks = [[False] * length for _ in range(17)]
    streams[0] = [11] * length
    streams[0][1] = padding_id  # Semantic text PAD within the valid dialogue timeline.
    for stream in range(9):
        masks[stream] = [True] * length
    masks[0][0] = False  # Prompt/delay position already masked before batching.
    return TrainingExample(
        tuple(tuple(row) for row in streams), tuple(tuple(row) for row in streams),
        tuple(tuple(row) for row in masks), tuple(str(i) for i in range(17)),
        1, length - 1,
    )


class BatchContractTest(unittest.TestCase):
    def test_raw_dataset_decodes_only_requested_real_audio_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "conversation.wav"
            with wave.open(str(audio_path), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(24000)
                output.writeframes(b"\x01\x00\x02\x00" * 24000)
            sample = PreparedSample(
                "sample", audio_path, audio_path, (), "prompt", {},
                AudioInfo(24000, 2, 1.0), 0.25, 0.75,
            )
            item = RawAudioDataset([sample], 24000)[0]
        self.assertEqual(tuple(item.waveform.shape), (2, 12000))
        self.assertEqual(item.valid_samples, 12000)

    def test_raw_dataset_accepts_stereo_decoder_output_with_channels_last_layout(self):
        audio_path = Path("conversation.wav")
        sample = PreparedSample(
            "sample", audio_path, audio_path, (), "prompt", {},
            AudioInfo(24000, 2, 1.0), 0.0, 1.0,
        )
        channels_last = np.array([[1.0, 7.0], [2.0, 8.0], [3.0, 9.0]], dtype=np.float32)
        fake_sphn = SimpleNamespace(read=lambda *_args, **_kwargs: (channels_last, 24000))
        with patch.dict("sys.modules", {"sphn": fake_sphn}):
            item = RawAudioDataset([sample], 24000)[0]
        self.assertEqual(tuple(item.waveform.shape), (2, 3))
        self.assertTrue(torch.equal(item.waveform, torch.tensor([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])))

    def test_raw_dataset_retries_empty_time_slice_by_decoding_and_cropping_full_audio(self):
        audio_path = Path("conversation.wav")
        sample = PreparedSample(
            "sample", audio_path, audio_path, (), "prompt", {},
            AudioInfo(100, 2, 1.0), 0.25, 0.5,
        )
        full_audio = np.stack([np.arange(100), 100 + np.arange(100)]).astype(np.float32)

        def read_audio(_path, **kwargs):
            if "start_sec" in kwargs:
                return np.empty((2, 0), dtype=np.float32), 100
            return full_audio, 100

        fake_sphn = SimpleNamespace(read=read_audio)
        with patch.dict("sys.modules", {"sphn": fake_sphn}):
            item = RawAudioDataset([sample], 100)[0]
        expected = torch.from_numpy(full_audio[:, 25:50].copy())
        self.assertTrue(torch.equal(item.waveform, expected))

    def test_mimi_receives_each_unpadded_stereo_item_with_requested_channel_order(self):
        class FakeMimi:
            def encode(self, audio):
                self.audio = audio.clone()
                return torch.zeros((1, 2, 8, 2), dtype=torch.long)

        mimi = FakeMimi()
        codec = MimiCodec(mimi, 24000, 12.5, "cpu", None)
        waveform = torch.tensor([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
        agent, user = codec.encode_stereo_waveform(waveform, 1, 0)
        self.assertEqual(mimi.audio.shape, (1, 2, 3))
        self.assertTrue(torch.equal(mimi.audio[0], waveform.flip(0)))
        self.assertEqual(len(agent), 8)
        self.assertEqual(len(user), 8)

    def test_raw_wave_collate_keeps_length_separate_from_transport_padding(self):
        batch = collate_raw_audio([
            RawAudioItem("short", torch.ones(2, 3), 3),
            RawAudioItem("long", torch.ones(2, 5), 5),
        ])
        self.assertEqual(batch["waveforms"].shape, (2, 2, 5))
        self.assertEqual(batch["valid_samples"].tolist(), [3, 5])
        self.assertTrue(torch.equal(batch["waveforms"][0, :, 3:], torch.zeros(2, 2)))

    def test_post_delay_masks_distinguish_text_pad_from_batch_padding(self):
        batch = post_encode_collate([example(5), example(3)], 3, -1, "cpu")
        self.assertEqual(batch["valid_frame_mask"].tolist(), [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ])
        weights = stream_weights_torch(batch["labels"], batch["loss_mask"], 3)
        self.assertAlmostEqual(float(weights[0, 0, 1]), 0.3)
        self.assertAlmostEqual(float(weights[1, 0, 1]), 0.3)
        self.assertEqual(float(weights[1, 0, 3]), 0.0)
        self.assertEqual(float(weights[1, 1, 2]), 1.0)
        self.assertEqual(float(weights[1, 1, 3]), 0.0)
        self.assertEqual(float(weights[1, 9, 1]), 0.0)

    def test_stream_masks_are_preserved_per_codebook_after_delay(self):
        delayed = example(6)
        masks = [list(row) for row in delayed.loss_mask]
        masks[1] = [False, True, True, True, False, False]
        masks[2] = [False, False, True, True, True, False]
        delayed = TrainingExample(
            delayed.input_codes, delayed.labels, tuple(tuple(row) for row in masks),
            delayed.stream_names, delayed.prompt_frames, delayed.dialogue_frames,
        )
        batch = post_encode_collate([delayed], 3, -1, "cpu")
        self.assertEqual(batch["audio_loss_mask"][0, 0].tolist(), masks[1])
        self.assertEqual(batch["audio_loss_mask"][0, 1].tolist(), masks[2])

    def test_batched_objective_runs_forward_and_backward_with_padded_targets(self):
        batch = post_encode_collate([example(5), example(3)], 3, -1, "cpu")
        text_logits = torch.randn(2, 1, 5, 16, requires_grad=True)
        audio_logits = torch.randn(2, 16, 5, 8, requires_grad=True)
        output = SimpleNamespace(
            text_logits=text_logits,
            logits=audio_logits,
            text_mask=torch.ones((2, 1, 5), dtype=torch.bool),
            mask=torch.ones((2, 16, 5), dtype=torch.bool),
        )
        total, components = loss_components(output, batch["codes"], batch, 3, torch)
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(text_logits.grad).all())
        self.assertTrue(torch.isfinite(audio_logits.grad).all())
        self.assertGreater(float(components["audio_semantic"]), 0.0)

    def test_actual_delay_transform_masks_each_stream_boundary_before_batch_padding(self):
        class Codec:
            codebooks = 8

        class Tokenizer:
            padding_id = 3
            end_padding_id = 0

        builder = PersonaPlexTrainingExampleBuilder(Codec(), Tokenizer(), [0] * 17, -1)
        delays = tuple(index % 4 for index in range(17))
        delayed = builder.apply_delays(example(6), delays)
        batch = post_encode_collate([delayed, example(4)], 3, -1, "cpu")
        for stream in range(17):
            self.assertEqual(batch["loss_mask"][0, stream, :delayed.total_frames].tolist(), list(delayed.loss_mask[stream]))
        self.assertFalse(batch["loss_mask"][1, :, 4:].any())
        self.assertTrue(batch["valid_frame_mask"][0].all())
        self.assertFalse(batch["valid_frame_mask"][1, 4:].any())

    def test_distributed_bucket_sampler_has_disjoint_equal_rank_batches(self):
        durations = [8, 9, 10, 11, 18, 19, 20, 21, 28, 29, 30, 30]
        rank_batches = [list(DistributedBucketBatchSampler(
            durations, batch_size=2, rank=rank, world_size=2, shuffle=False,
        )) for rank in range(2)]
        self.assertEqual(len(rank_batches[0]), len(rank_batches[1]))
        left = {item for batch in rank_batches[0] for item in batch}
        right = {item for batch in rank_batches[1] for item in batch}
        self.assertFalse(left & right)
        self.assertTrue(all(len(batch) == 2 for batches in rank_batches for batch in batches))


if __name__ == "__main__":
    unittest.main()
