import tempfile
import unittest
from pathlib import Path

from personaplex_finetuning.config import load_config


class ConfigTest(unittest.TestCase):
    def test_reads_moshi_duration_sample_limit_and_optimizer_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.json"
            config.write_text(
                '{"model":{"root":"/models/base","source":"/source"},'
                '"data":{"prepared_dir":"/data"},"duration_sec":100,'
                '"sample_number":10,"batch_size":16,"max_steps":2000,'
                '"lora":{"enable":true,"rank":128,"scaling":2.0,"ft_embed":false},'
                '"optim":{"lr":2e-6,"weight_decay":0.1,"pct_start":0.05}}'
            )

            loaded = load_config(config)

            self.assertEqual(loaded.duration_sec, 100)
            self.assertEqual(loaded.sample_number, 10)
            self.assertEqual(loaded.per_device_batch_size, 16)
            self.assertEqual(loaded.max_steps, 2000)
            self.assertEqual(loaded.lora_rank, 128)
            self.assertEqual(loaded.lora_alpha, 256)
            self.assertEqual(loaded.lora_scaling, 2.0)
            self.assertFalse(loaded.ft_embed)
            self.assertAlmostEqual(loaded.learning_rate, 2e-6)
            self.assertEqual(loaded.weight_decay, 0.1)
            self.assertEqual(loaded.pct_start, 0.05)

    def test_reads_moshi_loss_and_logging_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.json"
            config.write_text(
                '{"model":{"root":"/models/base","source":"/source"},'
                '"data":{"prepared_dir":"/data"},"first_codebook_weight_multiplier":2.0,'
                '"text_padding_weight":0.25,"log_freq":10,"ckpt_freq":100,"no_eval":true}'
            )

            loaded = load_config(config)

            self.assertEqual(loaded.first_codebook_weight_multiplier, 2.0)
            self.assertEqual(loaded.text_padding_weight, 0.25)
            self.assertEqual(loaded.log_freq, 10)
            self.assertEqual(loaded.ckpt_freq, 100)
            self.assertTrue(loaded.no_eval)

    def test_null_sample_number_means_no_training_sample_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.json"
            config.write_text(
                '{"model":{"root":"/models/base","source":"/source"},'
                '"data":{"prepared_dir":"/data"},"sample_number":null}'
            )

            self.assertIsNone(load_config(config).sample_number)

    def test_optional_codec_cache_path_resolves_from_the_data_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.json"
            config.write_text(
                '{"model":{"root":"/models/base","source":"/source"},'
                '"data":{"prepared_dir":"/data","codec_cache_dir":"/nvme/mimi-codes"}}'
            )

            self.assertEqual(load_config(config).codec_cache_dir, Path("/nvme/mimi-codes"))

    def test_profile_steps_is_an_explicit_training_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.yaml"
            config.write_text(
                "model:\n  root: /models/base\n  source: /source\n"
                "data:\n  prepared_dir: /data\n"
                "train:\n  output_dir: /runs\nprofile_steps: true\n",
                encoding="utf-8",
            )

            self.assertTrue(load_config(config).profile_steps)

    def test_hydra_moshi_overrides_target_the_effective_top_level_parameters(self) -> None:
        config = Path(__file__).resolve().parents[1] / "configs" / "moshi_code_style.yaml"

        loaded = load_config(
            config,
            overrides=[
                "model=server", "max_steps=20", "learning_rate=1e-5", "profile_steps=true",
                "batch_size=4", "train.gradient_accumulation_steps=2", "lora.rank=64",
                "no_eval=false", "train.eval_every_steps=100",
                "free_running_eval_every_steps=200", "validation_max_samples=8",
                "generation.audio_silence_frame_cnt=4",
            ],
        )

        self.assertEqual(loaded.max_steps, 20)
        self.assertEqual(loaded.learning_rate, 1e-5)
        self.assertTrue(loaded.profile_steps)
        self.assertEqual(loaded.per_device_batch_size, 4)
        self.assertEqual(loaded.gradient_accumulation_steps, 2)
        self.assertEqual(loaded.lora_rank, 64)
        self.assertEqual(loaded.free_running_eval_every_steps, 200)
        self.assertEqual(loaded.validation_max_samples, 8)
        self.assertEqual(loaded.free_running_eval_samples, 3)
        self.assertEqual(loaded.generation_settings.audio_silence_frame_cnt, 4)

    def test_default_training_config_runs_heldout_generation_validation(self) -> None:
        config = Path(__file__).resolve().parents[1] / "configs" / "moshi_code_style.yaml"
        loaded = load_config(config)

        self.assertFalse(loaded.no_eval)
        self.assertEqual(loaded.eval_every_steps, 500)
        self.assertEqual(loaded.free_running_eval_every_steps, 500)
        self.assertEqual(loaded.validation_max_samples, 32)
        self.assertEqual(loaded.num_workers, 4)
        self.assertEqual(loaded.prefetch_factor, 2)
        self.assertTrue(loaded.pin_memory)
        self.assertTrue(loaded.persistent_workers)
        inference_config = load_config(Path(__file__).resolve().parents[1] / "configs" / "infer.yaml")
        self.assertEqual(loaded.generation_settings, inference_config.generation_settings)

    def test_ten_sample_overfit_is_deterministic_and_monitors_generation_on_train_set(self) -> None:
        config = Path(__file__).resolve().parents[1] / "configs" / "moshi_overfit_10.yaml"
        loaded = load_config(config)

        self.assertEqual(loaded.sample_number, 10)
        self.assertFalse(loaded.shuffle)
        self.assertFalse(loaded.no_eval)
        self.assertTrue(loaded.eval_on_train_samples)
        self.assertEqual(loaded.free_running_eval_every_steps, 100)

    def test_ten_conversation_overfit_config_disables_dataset_shuffling(self) -> None:
        config = Path(__file__).resolve().parents[1] / "configs" / "moshi_overfit_10.yaml"

        loaded = load_config(config)

        self.assertEqual(loaded.sample_number, 10)
        self.assertFalse(loaded.shuffle)

    def test_full_standalone_config_loads_every_feature_setting(self) -> None:
        root = Path(__file__).resolve().parents[1]
        loaded = load_config(root / "configs" / "config.full.yaml")

        self.assertEqual(loaded.model_root, Path("/storage-voice/voice/vdt/baottn/personaplex-7b-v1"))
        self.assertEqual(loaded.prepared_dir, Path("/storage-voice/voice/vdt/baottn/personaplex-otospeech-prepared"))
        self.assertEqual(loaded.target_window_seconds, 25)
        self.assertEqual(loaded.min_window_seconds, 10)
        self.assertEqual(loaded.max_window_seconds, 30)
        self.assertEqual(loaded.per_device_batch_size, 2)
        self.assertEqual(loaded.num_workers, 4)
        self.assertTrue(loaded.randomize_train)
        self.assertTrue(loaded.persistent_workers)
        self.assertEqual(loaded.max_steps, 10_000)

    def test_full_server_hydra_preset_preserves_production_settings(self) -> None:
        root = Path(__file__).resolve().parents[1]
        loaded = load_config(
            root / "configs" / "config.yaml",
            overrides=["data=otospeech", "model=server", "train=full"],
        )

        self.assertEqual(loaded.max_steps, 10_000)
        self.assertEqual(loaded.gradient_accumulation_steps, 8)
        self.assertTrue(loaded.gradient_checkpointing)
        self.assertEqual(loaded.prompt_aug_prob, 0.0)
        self.assertFalse(loaded.static_chunking)
        self.assertTrue(loaded.swap_roles_after_pass)
        self.assertEqual(loaded.per_device_batch_size, 2)

    def test_b200_preset_uses_large_batches_and_prefetch(self) -> None:
        root = Path(__file__).resolve().parents[1]
        loaded = load_config(
            root / "configs" / "config.yaml",
            overrides=["data=otospeech", "model=server", "train=b200"],
        )

        self.assertEqual(loaded.per_device_batch_size, 1)
        self.assertEqual(loaded.gradient_accumulation_steps, 16)
        self.assertEqual(loaded.num_workers, 8)
        self.assertEqual(loaded.prefetch_factor, 4)
        self.assertTrue(loaded.gradient_checkpointing)

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

    def test_reads_static_chunking_and_role_swap_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "static_chunks.yaml"
            config.write_text(
                '{"model": {"root": "/models/personaplex", "source": "/source"}, '
                '"data": {"prepared_dir": "/prepared", "static_chunking": true, '
                '"swap_roles_after_pass": true}}'
            )

            loaded = load_config(config)

            self.assertTrue(loaded.static_chunking)
            self.assertTrue(loaded.swap_roles_after_pass)

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

    def test_loads_explicit_test_manifest_and_batching_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "configs" / "test.yaml"
            config.parent.mkdir()
            config.write_text(
                '{"model": {"root": "/models", "source": "/source"}, '
                '"data": {"prepared_dir": "/data", "test_manifest_path": "/data/test.jsonl"}, '
                '"train": {"per_device_batch_size": 3, "num_workers": 0, "pin_memory": false}}'
            )
            loaded = load_config(config)
        self.assertEqual(loaded.test_manifest, Path("/data/test.jsonl"))
        self.assertEqual(loaded.per_device_batch_size, 3)
        self.assertEqual(loaded.num_workers, 0)
        self.assertFalse(loaded.pin_memory)

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
