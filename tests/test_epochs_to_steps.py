import tempfile
import unittest
from pathlib import Path

from personaplex_finetuning.config import load_config
from personaplex_finetuning.train import steps_for_epochs


class EpochsToStepsTest(unittest.TestCase):
    def test_matches_rank_stride_batches_and_accumulation(self):
        # 17_100 chunks, 2 ranks x batch 4 -> 2_137 micro batches per rank per epoch.
        self.assertEqual(steps_for_epochs(1, 17_100, 4, 2, 4), 535)
        self.assertEqual(steps_for_epochs(2, 17_100, 4, 2, 4), 1069)
        self.assertEqual(steps_for_epochs(0.5, 17_100, 4, 2, 4), 268)
        self.assertEqual(steps_for_epochs(1, 10, 1, 1, 1), 10)

    def test_rejects_dataset_without_one_full_batch(self):
        with self.assertRaisesRegex(ValueError, "no full training batch"):
            steps_for_epochs(1, 3, 4, 2, 1)

    def test_config_reads_epochs_and_rejects_non_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.json"
            base = '{"model":{"root":"/m","source":"/s"},"data":{"prepared_dir":"/d"}'
            config.write_text(base + "}")
            self.assertIsNone(load_config(config).epochs)
            config.write_text(base + ',"epochs":2}')
            self.assertEqual(load_config(config).epochs, 2.0)
            config.write_text(base + ',"train":{"epochs":1.5}}')
            self.assertEqual(load_config(config).epochs, 1.5)
            config.write_text(base + ',"epochs":0}')
            with self.assertRaisesRegex(ValueError, "epochs"):
                load_config(config)


if __name__ == "__main__":
    unittest.main()
