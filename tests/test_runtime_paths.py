import tempfile
import unittest
from pathlib import Path

from personaplex_finetuning.runtime import RuntimePaths


class RuntimePathsTest(unittest.TestCase):
    def test_requires_all_local_personaplex_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "moshi").mkdir(parents=True)
            model = root / "model"
            model.mkdir()
            for name in ("model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer_spm_32k_3.model"):
                (model / name).write_bytes(b"x")

            paths = RuntimePaths(model, source)

            self.assertEqual(paths.validate().moshi_weight.name, "model.safetensors")
            (model / "model.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "model.safetensors"):
                paths.validate()
