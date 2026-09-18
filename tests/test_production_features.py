import random
import unittest
from pathlib import Path

from personaplex_finetuning.data import AudioInfo, PreparedSample, Word
from personaplex_finetuning.train import get_cosine_schedule_with_warmup


class ProductionFeaturesTest(unittest.TestCase):
    def setUp(self):
        words = (
            Word("user", "hello", 0.5, 1.0),
            Word("agent", "hi", 1.2, 1.8),
            Word("user", "how", 2.0, 2.5),
            Word("agent", "there", 3.0, 3.5),
            Word("agent", "friend", 50.0, 50.5),
        )
        self.sample = PreparedSample(
            sample_id="test_001",
            conversation_wav=Path("/dummy/conv.wav"),
            voice_prompt_wav=Path("/dummy/voice.wav"),
            words=words,
            text_prompt="You are a helpful assistant. You enjoy chatting about sports and science.",
            metadata={"sample_id": "test_001"},
            audio=AudioInfo(sample_rate=24000, channels=2, duration_sec=100.0),
            window_start_sec=1.2,
            window_end_sec=31.2,
        )

    def test_deterministic_sample_window(self):
        start, end = self.sample.sample_window(window_seconds=10.0, random_crop=False)
        self.assertEqual(start, 1.2)
        self.assertEqual(end, 11.2)

    def test_random_sample_window_bounds(self):
        rng = random.Random(42)
        for _ in range(20):
            start, end = self.sample.sample_window(window_seconds=15.0, random_crop=True, rng=rng)
            self.assertGreaterEqual(start, 0.0)
            self.assertLessEqual(end, 100.0)
            self.assertAlmostEqual(end - start, 15.0, places=4)

    def test_augmented_prompt_presets(self):
        rng = random.Random(42)
        # With prob=0.0, returns exact original prompt
        self.assertEqual(self.sample.get_augmented_prompt(prompt_aug_prob=0.0, rng=rng), self.sample.text_prompt)

        # With prob=1.0, returns one of the augmented versions
        augmented = self.sample.get_augmented_prompt(prompt_aug_prob=1.0, rng=rng)
        self.assertIsInstance(augmented, str)
        self.assertGreater(len(augmented), 5)

    def test_vietnamese_augmented_prompt_presets(self):
        vi_sample = self.sample.with_window(
            0.0, 10.0, text_prompt="Bạn là một trợ lý ảo thông minh và thân thiện."
        )
        rng = random.Random(42)
        augmented = vi_sample.get_augmented_prompt(prompt_aug_prob=1.0, rng=rng)
        self.assertIsInstance(augmented, str)
        self.assertTrue(augmented.startswith("Bạn") or "trò chuyện" in augmented)

    def test_cosine_schedule_with_warmup(self):
        import torch
        param = torch.nn.Parameter(torch.tensor([1.0]))
        opt = torch.optim.SGD([param], lr=1.0)
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=100)

        # Step 0: lr is 0.0
        self.assertAlmostEqual(sched.get_last_lr()[0], 0.0)

        # Warmup phase: step 5 -> lr is 0.5
        for _ in range(5):
            sched.step()
        self.assertAlmostEqual(sched.get_last_lr()[0], 0.5, places=4)

        # Warmup peak: step 10 -> lr is 1.0
        for _ in range(5):
            sched.step()
        self.assertAlmostEqual(sched.get_last_lr()[0], 1.0, places=4)

        # Decay phase: step 100 -> lr approaches 0.0
        for _ in range(90):
            sched.step()
        self.assertAlmostEqual(sched.get_last_lr()[0], 0.0, places=3)
