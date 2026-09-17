import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import validate_dataset


class ValidateDatasetTest(unittest.TestCase):
    def test_reads_manifest_and_window_from_config(self) -> None:
        loaded = SimpleNamespace(manifest=Path("/prepared/train.jsonl"), window_seconds=17)
        samples = [SimpleNamespace(audio=SimpleNamespace(duration_sec=3.5))]
        dataset = SimpleNamespace(load=lambda: samples)

        with patch("sys.argv", ["validate_dataset", "--config", "configs/test.yaml"]), \
             patch.object(validate_dataset, "load_config", return_value=loaded), \
             patch.object(validate_dataset, "PreparedDataset", return_value=dataset) as prepared:
            self.assertEqual(validate_dataset.main(), 0)

        prepared.assert_called_once_with(Path("/prepared/train.jsonl"), 17)
