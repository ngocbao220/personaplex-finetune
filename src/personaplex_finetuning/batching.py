"""CPU audio loading and variable-length training batch helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import read_stereo_window


@dataclass(frozen=True)
class RawAudioItem:
    sample: object
    waveform: torch.Tensor
    valid_samples: int


class RawAudioDataset(torch.utils.data.Dataset):
    """Decode only the requested stereo window in a DataLoader worker."""

    def __init__(self, samples, sample_rate: int) -> None:
        self.samples = samples
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        audio = read_stereo_window(
            sample.conversation_wav, sample.window_start_sec,
            min(sample.window_end_sec, sample.audio.duration_sec), self.sample_rate, sample.sample_id,
        )
        waveform = torch.from_numpy(audio.copy())
        return RawAudioItem(sample, waveform, waveform.shape[-1])


def collate_raw_audio(items: list[RawAudioItem]) -> dict:
    """Pad raw waveforms for transport only; lengths preserve real audio extent."""
    if not items:
        raise ValueError("cannot collate an empty raw audio batch")
    max_samples = max(item.valid_samples for item in items)
    waveforms = torch.zeros((len(items), 2, max_samples), dtype=torch.float32)
    for index, item in enumerate(items):
        if item.waveform.shape[0] != 2 or item.waveform.shape[-1] != item.valid_samples:
            raise ValueError("raw waveform shape/length mismatch")
        waveforms[index, :, :item.valid_samples] = item.waveform
    return {
        "samples": [item.sample for item in items],
        "waveforms": waveforms,
        "valid_samples": torch.tensor([item.valid_samples for item in items], dtype=torch.long),
    }


def post_encode_collate(examples, text_padding_id: int, zero_token: int, device):
    """Pad encoded PersonaPlex streams and masks after per-item Mimi encoding."""
    if not examples:
        raise ValueError("cannot collate an empty encoded batch")
    lengths = [example.total_frames for example in examples]
    max_frames = max(lengths)
    batch = len(examples)
    codes = torch.zeros((batch, 17, max_frames), dtype=torch.long, device=device)
    codes[:, 0] = zero_token
    labels = torch.full_like(codes, zero_token)
    loss_mask = torch.zeros((batch, 17, max_frames), dtype=torch.bool, device=device)
    valid_frame_mask = torch.zeros((batch, max_frames), dtype=torch.bool, device=device)
    for index, (example, length) in enumerate(zip(examples, lengths, strict=True)):
        stream_codes = torch.tensor(example.input_codes, dtype=torch.long, device=device)
        stream_labels = torch.tensor(example.labels, dtype=torch.long, device=device)
        stream_mask = torch.tensor(example.loss_mask, dtype=torch.bool, device=device)
        if stream_codes.shape != (17, length) or stream_labels.shape != (17, length) or stream_mask.shape != (17, length):
            raise ValueError(f"encoded sample {index} has inconsistent stream layout")
        codes[index, :, :length] = stream_codes
        labels[index, :, :length] = stream_labels
        loss_mask[index, :, :length] = stream_mask
        valid_frame_mask[index, :length] = True
    return {
        "codes": codes,
        "labels": labels,
        "valid_frame_mask": valid_frame_mask,
        "text_loss_mask": loss_mask[:, 0],
        "audio_loss_mask": loss_mask[:, 1:17],
        "loss_mask": loss_mask & valid_frame_mask[:, None, :],
    }


class DistributedBucketBatchSampler(torch.utils.data.Sampler):
    """Length-bucketed local batches with one rank-aware sampler."""

    def __init__(self, durations, batch_size: int, rank: int = 0, world_size: int = 1,
                 seed: int = 42, shuffle: bool = True) -> None:
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed bucket sampler dimensions")
        self.durations = list(durations)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self):
        import random

        rng = random.Random(self.seed + self.epoch)
        ordered = sorted(range(len(self.durations)), key=lambda i: self.durations[i])
        bucket_size = self.batch_size * self.world_size * 8
        pools = [ordered[i:i + bucket_size] for i in range(0, len(ordered), bucket_size)]
        batches = []
        for pool in pools:
            if self.shuffle:
                rng.shuffle(pool)
            for start in range(0, len(pool), self.batch_size):
                batch = pool[start:start + self.batch_size]
                if self.world_size == 1 or len(batch) == self.batch_size:
                    batches.append(batch)
        # Keep all ranks on the same number of optimizer micro-steps.
        usable = len(batches) - len(batches) % self.world_size
        batches = batches[:usable]
        if not batches and self.world_size > 1 and len(self.durations) > 0:
            raise ValueError("not enough samples to make one batch on every distributed rank")
        if self.shuffle:
            rng.shuffle(batches)
        return batches[self.rank::self.world_size]

    def __iter__(self):
        yield from self._batches()

    def __len__(self):
        return len(self._batches())


class RankStrideBatchSampler(torch.utils.data.Sampler):
    """Yield fixed local batches after Moshi-style global rank-stride partitioning."""

    def __init__(self, sample_count: int, batch_size: int, rank: int, world_size: int,
                 seed: int = 42, shuffle: bool = False) -> None:
        if sample_count < 0 or batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid rank-stride batch sampler dimensions")
        self.sample_count = sample_count
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _indices(self) -> list[int]:
        indices = list(range(self.sample_count))
        if self.shuffle:
            import random
            random.Random(self.seed + self.epoch).shuffle(indices)
        return indices[self.rank::self.world_size]

    def __iter__(self):
        rank_indices = self._indices()
        # All ranks must execute the same number of DDP forwards. Drop only the
        # incomplete tail, matching the previous in-process rank-stride iterator.
        common_batches = min(
            len(list(range(worker, self.sample_count, self.world_size))) // self.batch_size
            for worker in range(self.world_size)
        )
        usable = common_batches * self.batch_size
        for start in range(0, usable, self.batch_size):
            yield rank_indices[start:start + self.batch_size]

    def __len__(self) -> int:
        return min(
            len(list(range(worker, self.sample_count, self.world_size))) // self.batch_size
            for worker in range(self.world_size)
        )
