import tempfile
import unittest
from pathlib import Path

from personaplex_finetuning.config import load_config


class ConfigTest(unittest.TestCase):
    def test_resolves_paths_relative_to_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "configs" / "test.yaml"
            config.parent.mkdir()
            config.write_text('{"model": {"root": "../models/personaplex", "source": "../source/moshi"}, "data": {"manifest": "../prepared/train.jsonl"}}')

            loaded = load_config(config)

            self.assertEqual(loaded.model_root, (root / "models/personaplex").resolve())
            self.assertEqual(loaded.manifest, (root / "prepared/train.jsonl").resolve())
            self.assertFalse(loaded.shuffle)
