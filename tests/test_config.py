import tempfile
import unittest
from pathlib import Path

from personaplex_finetuning.config import load_config


class ConfigTest(unittest.TestCase):
    def test_resolves_prepared_directory_and_derives_local_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "configs" / "test.yaml"
            config.parent.mkdir()
            config.write_text(
                '{"model": {"root": "../models/personaplex", "source": "../source/moshi"}, '
                '"data": {"prepared_dir": "../prepared"}}'
            )

            loaded = load_config(config)

            self.assertEqual(loaded.prepared_dir, (root / "prepared").resolve())
            self.assertEqual(loaded.manifest, (root / "prepared/train.jsonl").resolve())

    def test_reads_qlora_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "qlora.yaml"
            config.write_text(
                '{"model": {"root": "/models/personaplex", "source": "/source"}, '
                '"data": {"prepared_dir": "/prepared"}, '
                '"lora": {"qlora": true, "quant_type": "nf4"}}'
            )

            loaded = load_config(config)

            self.assertTrue(loaded.qlora)
            self.assertEqual(loaded.quant_type, "nf4")

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

    def test_preserves_explicit_absolute_server_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "server.yaml"
            config.write_text(
                '{"model": {"root": "/mnt/models/personaplex", '
                '"source": "/opt/personaplex-source"}, '
                '"data": {"manifest": "/mnt/processed/train.jsonl"}}'
            )

            loaded = load_config(config)

            self.assertEqual(loaded.model_root, Path("/mnt/models/personaplex"))
            self.assertEqual(loaded.personaplex_source, Path("/opt/personaplex-source"))
            self.assertEqual(loaded.manifest, Path("/mnt/processed/train.jsonl"))

    def test_applies_dotlist_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "test.yaml"
            config.write_text(
                '{"model": {"root": "/models/personaplex", "source": "/source"}, '
                '"data": {"prepared_dir": "/prepared"}, '
                '"train": {"learning_rate": 2.0e-5, "max_steps": 100}}'
            )
            loaded = load_config(
                config,
                overrides=["train.learning_rate=1e-4", "train.max_steps=500", "train.gradient_checkpointing=true"],
            )
            self.assertEqual(loaded.learning_rate, 1e-4)
            self.assertEqual(loaded.max_steps, 500)
            self.assertTrue(loaded.gradient_checkpointing)
            self.assertEqual(loaded.mixed_precision, "bf16")

