import tempfile
import unittest
import json
import os
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from personaplex_finetuning.train import (
    create_run_dir,
    effective_global_batch_size,
    evenly_spaced_validation_samples,
    evaluate_free_running,
    generation_checkpoint_score,
    inference_config_snapshot,
    load_training_state,
    model_forward_train,
    resolve_training_device,
    device_memory_bytes,
    lora_prefixes_for_stage,
    inspect_training_sample,
    iter_training_batches,
    text_supervision_counts,
    text_target_token_loss_stats,
    pack_text_training_stats,
    unwrap_parallel_model,
    rank_stride_indices,
    reduce_distributed_loss,
    run,
    sample_index_for_rank,
    step_optimizer_if_ready,
    validation_selection_loss,
    save_training_state,
    save_adapter_state,
    write_rank_info,
    validate_resume_step,
    validate_resume_checkpoint,
    training_contract,
    dataset_load_summary,
    chunk_filter_payload,
    chunk_filter_summary,
    write_tensorboard_scalars,
    verify_reloaded_adapter,
)


class TrainTest(unittest.TestCase):
    def test_training_disables_moshi_cuda_graphs_before_runtime_setup(self) -> None:
        config = SimpleNamespace(
            train_method="lora", ft_embed=False, qlora=False, train_stage="joint",
            randomize_train=False, mixed_precision="bf16", shuffle=True,
        )
        with patch.dict("os.environ"):
            os.environ.pop("NO_CUDA_GRAPH", None)
            with self.assertRaisesRegex(ValueError, "smoke configuration"):
                run(config, smoke=True)
            self.assertEqual(os.environ.get("NO_CUDA_GRAPH"), "1")

    def test_text_training_stats_pack_prediction_diagnostics_as_three_scalars(self) -> None:
        packed = pack_text_training_stats(
            torch.tensor(11), torch.tensor(2), torch.tensor(4.5), torch.tensor(9),
            torch.tensor(7), torch.tensor([3, 5, 1]),
        )

        self.assertEqual(packed.shape, (8,))
        self.assertEqual(packed.tolist(), [11.0, 2.0, 4.5, 9.0, 7.0, 3.0, 5.0, 1.0])

    def test_training_device_honors_explicit_mps_in_single_process(self) -> None:
        with patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(resolve_training_device("mps", 0, 1), torch.device("mps"))
        with patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "MPS"):
                resolve_training_device("mps", 0, 1)

    def test_mps_memory_report_reads_driver_allocation(self) -> None:
        with patch("torch.mps.driver_allocated_memory", return_value=123456):
            self.assertEqual(device_memory_bytes(torch.device("mps")), 123456)

    def test_startup_summaries_distinguish_out_of_bounds_from_text_overflow(self) -> None:
        from personaplex_finetuning.chunk_filter import ChunkFilterResult, RejectedChunk
        from personaplex_finetuning.data import DatasetLoadReport, Word

        load = DatasetLoadReport(10, 8, 1, 1)
        self.assertIn("skipped_out_of_bounds_samples=1", dataset_load_summary("train", load))
        self.assertIn("skipped_invalid_samples=1", dataset_load_summary("train", load))

        rejected = RejectedChunk(
            "conv", 0.0, 100.0, "text_overflow", ("left-agent",),
            Word("agent", "hello", 99.9, 100.0),
        )
        payload = chunk_filter_payload("train", 4, ChunkFilterResult((), (rejected,)))
        summary = chunk_filter_summary(payload)
        self.assertIn("skipped_out_of_bounds_chunks=0", summary)
        self.assertIn("skipped_text_overflow_chunks=1", summary)
        self.assertEqual(payload["rejected"][0]["word"], "hello")

    def test_resume_requires_complete_matching_checkpoint(self) -> None:
        from safetensors.torch import save_file
        from personaplex_finetuning.config import Config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "train.jsonl").write_text('{"sample_id":"one"}\n')
            config = Config(
                path=root / "config.yaml", model_root=Path("/models/personaplex"),
                personaplex_source=root / "src", prepared_dir=prepared, output_dir=root,
                lora_rank=2, lora_alpha=4,
            )
            checkpoint = root / "run" / "checkpoints" / "checkpoint_000007"
            checkpoint.mkdir(parents=True)
            save_file({
                "transformer.projection.lora_a.weight": torch.zeros(2, 3),
                "transformer.projection.lora_b.weight": torch.zeros(4, 2),
            }, str(checkpoint / "lora.safetensors"))
            (checkpoint / "adapter.json").write_text(json.dumps({
                "step": 7, "rank": 2, "alpha": 4,
                "model_root": "/models/personaplex",
            }))
            with self.assertRaisesRegex(RuntimeError, "training_state.pt"):
                validate_resume_checkpoint(config, str(checkpoint), ("transformer",), [])

            (checkpoint / "training_state.pt").write_bytes(b"placeholder")
            run_config = root / "run" / "config.json"
            run_config.write_text(json.dumps({"training_contract": training_contract(config, [])}))
            adapter, step = validate_resume_checkpoint(config, str(checkpoint), ("transformer",), [])
            self.assertEqual((adapter, step), (checkpoint / "lora.safetensors", 7))
            with self.assertRaisesRegex(RuntimeError, "LoRA configuration differs"):
                validate_resume_checkpoint(config, str(checkpoint), ("depformer",), [])
            with self.assertRaisesRegex(RuntimeError, "base model differs"):
                validate_resume_checkpoint(
                    config.replace(model_root=Path("/other")),
                    str(checkpoint), ("transformer",), [],
                )
            (prepared / "train.jsonl").write_text('{"sample_id":"changed"}\n')
            with self.assertRaisesRegex(RuntimeError, "manifest_sha256"):
                validate_resume_checkpoint(config, str(checkpoint), ("transformer",), [])

    def test_training_contract_detects_changed_prepared_text(self) -> None:
        from personaplex_finetuning.config import Config
        from personaplex_finetuning.data import AudioInfo, PreparedSample, Word

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.jsonl").write_text("{}\n")
            (root / "conversation.wav").write_bytes(b"audio")
            (root / "voice.wav").write_bytes(b"prompt")
            config = Config(root / "config.yaml", root, root, root, root)
            sample = PreparedSample(
                "sample", root / "conversation.wav", root / "voice.wav",
                (Word("agent", "xin", 0.0, 0.2),), "Trò chuyện tiếng Việt.", {},
                AudioInfo(24000, 2, 1.0), 0.0, 1.0,
            )
            original = training_contract(config, [sample])
            without_user_loss = training_contract(config.replace(user_loss=False), [sample])
            changed = training_contract(config, [sample.with_window(0.0, 1.0, "Speak English.")])
            self.assertNotEqual(original["prepared_sources_sha256"], changed["prepared_sources_sha256"])
            self.assertTrue(original["user_loss"])
            self.assertFalse(without_user_loss["user_loss"])

            kept = training_contract(config, [sample], [sample.with_window(0.0, 1.0)])
            rejected = training_contract(config, [sample], [])
            self.assertNotEqual(kept["kept_train_chunks_sha256"], rejected["kept_train_chunks_sha256"])
            self.assertEqual(kept["kept_train_chunk_count"], 1)
            self.assertEqual(rejected["kept_train_chunk_count"], 0)

    def test_training_iterator_switches_voice_text_and_channels_on_role_epoch(self) -> None:
        from personaplex_finetuning.data import AudioInfo, PreparedSample
        from personaplex_finetuning.train import iter_training_batches

        class FakeCodec:
            frame_rate = 12.5
            codebooks = 8

            def encode_conversation(self, *_args):
                return ((1,) * 25,) * 8

            def encode_voice_prompt(self, *_args):
                return ((1,),) * 8

            def sine(self, frames):
                return ((1,) * frames,) * 8

            def silence(self, frames):
                return ((0,) * frames,) * 8

        sample = PreparedSample(
            sample_id="swap", conversation_wav=Path("conversation.wav"),
            voice_prompt_wav=Path("voice_prompt_left.wav"), words=(),
            text_prompt="left prompt", metadata={}, audio=AudioInfo(24_000, 2, 4.0),
            window_start_sec=0.0, window_end_sec=4.0,
            voice_prompt_right_wav=Path("voice_prompt_right.wav"),
            text_prompt_right="right prompt",
        )
        runtime = SimpleNamespace(
            codec=FakeCodec(), tokenizer=SimpleNamespace(padding_id=3),
            initial_tokens=[0] * 17, zero_token=-1,
        )
        config = SimpleNamespace(
            duration_sec=2.0, per_device_batch_size=1, shuffle=False, seed=42,
            prompt_aug_prob=0.0, swap_roles_after_pass=True,
        )
        example = SimpleNamespace(
            prompt_frames=0, dialogue_frames=25,
            input_codes=tuple((0,) * 25 for _ in range(17)),
            loss_mask=tuple((False,) * 25 for _ in range(17)),
        )
        with patch("personaplex_finetuning.train.build_example", return_value=example), \
             patch("personaplex_finetuning.train.pad_training_example", side_effect=lambda item, *_args: item), \
             patch("personaplex_finetuning.train.post_encode_collate", return_value={"codes": "batch"}):
            from personaplex_finetuning.data import duration_chunks
            batches = iter_training_batches(
                config, duration_chunks([sample], config.duration_sec),
                runtime, "cpu", 0, 1, smoke=True,
            )
            left_pass = [next(batches)[-1][0] for _ in range(2)]
            right_pass = next(batches)[-1][0]

        self.assertTrue(all(item.voice_prompt_wav.name == "voice_prompt_left.wav" for item in left_pass))
        self.assertEqual(right_pass.voice_prompt_wav.name, "voice_prompt_right.wav")
        self.assertEqual(right_pass.text_prompt, "right prompt")
        self.assertEqual((right_pass.agent_channel, right_pass.user_channel), (1, 0))

    def test_training_iterator_prefetches_audio_and_batches_mimi_in_loader_order(self) -> None:
        from personaplex_finetuning.data import AudioInfo, PreparedSample

        class FakeCodec:
            sample_rate = 24_000
            frame_rate = 12.5

            def __init__(self):
                self.raw_audio = None

            def encode_conversation_stereo_batch(self, windows, raw_audio=None):
                self.raw_audio = raw_audio
                self.windows = windows
                return [(((1,),) * 8, ((2,),) * 8) for _ in windows]

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "conversation.wav"
            with wave.open(str(wav_path), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(24_000)
                output.writeframes(b"\x00\x00\x00\x00" * 48_000)
            sample = PreparedSample(
                sample_id="prefetch-1", conversation_wav=wav_path,
                voice_prompt_wav=Path(tmp) / "voice.wav", words=(), text_prompt="prompt",
                metadata={}, audio=AudioInfo(24_000, 2, 2.0),
                window_start_sec=0.0, window_end_sec=2.0,
            )
            codec = FakeCodec()
            runtime = SimpleNamespace(codec=codec, tokenizer=SimpleNamespace(padding_id=3), zero_token=-1)
            config = SimpleNamespace(
                duration_sec=2.0, per_device_batch_size=1, shuffle=False, seed=42,
                num_workers=1, prefetch_factor=2, pin_memory=False,
                persistent_workers=False, prompt_aug_prob=0.0,
            )
            fake_example = SimpleNamespace(
                prompt_frames=0, dialogue_frames=25,
                input_codes=tuple((0,) * 25 for _ in range(17)),
                loss_mask=tuple((False,) * 25 for _ in range(17)),
            )
            sentinel_batch = {"codes": "collated"}
            with patch("personaplex_finetuning.train.build_example", return_value=fake_example), \
                 patch("personaplex_finetuning.train.pad_training_example", side_effect=lambda example, *_args: example), \
                 patch("personaplex_finetuning.train.post_encode_collate", return_value=sentinel_batch):
                batches = iter_training_batches(config, [sample], runtime, "cpu", 0, 1, smoke=False)
                try:
                    batch, _epoch, count, _seconds, _frames, loaded_samples = next(batches)
                finally:
                    batches.close()

        self.assertIs(batch, sentinel_batch)
        self.assertEqual(count, 1)
        self.assertEqual([item.sample_id for item in loaded_samples], ["prefetch-1"])
        self.assertEqual(codec.raw_audio["waveforms"].shape, (1, 2, 48_000))
        self.assertEqual(codec.raw_audio["valid_samples"].tolist(), [48_000])

    def test_moshi_rank_stride_partition_is_disjoint_and_complete(self) -> None:
        partitions = [rank_stride_indices(11, rank, 4) for rank in range(4)]
        self.assertEqual(partitions, [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7]])
        self.assertEqual(sorted(index for partition in partitions for index in partition), list(range(11)))

    def test_global_batch_multiplies_processes_and_accumulation(self) -> None:
        self.assertEqual(effective_global_batch_size(1, 4, 2), 8)

    def test_validation_subset_is_deterministically_spread_across_dataset(self) -> None:
        samples = list(range(101))
        self.assertEqual(evenly_spaced_validation_samples(samples, 3), [0, 50, 100])
        self.assertEqual(evenly_spaced_validation_samples(samples, 1), [50])
        with self.assertRaisesRegex(ValueError, "must be positive"):
            evenly_spaced_validation_samples(samples, 0)

    def test_b200_and_a100_pilot_candidates_hold_global_batch_constant(self) -> None:
        candidates = ((2, 1, 8), (2, 2, 4), (2, 4, 2), (4, 1, 4), (4, 2, 2), (4, 4, 1))
        self.assertEqual(
            {effective_global_batch_size(batch, world_size, accumulation)
             for world_size, batch, accumulation in candidates},
            {16},
        )

    def test_best_validation_checkpoint_uses_nonpadding_text_and_audio_losses(self) -> None:
        metrics = {
            "val/loss_total": 0.01,
            "val/loss_text_real": 1.2,
            "val/loss_agent_semantic": 0.2,
            "val/loss_agent_acoustic": 0.01,
            "val/loss_user_semantic": 0.1,
            "val/loss_user_acoustic": 0.01,
        }
        self.assertAlmostEqual(validation_selection_loss(metrics), 1.52)
        self.assertEqual(validation_selection_loss({"val/loss_total": 0.001}), float("inf"))

    def test_empty_free_running_generation_cannot_win_checkpoint_selection(self) -> None:
        self.assertEqual(
            generation_checkpoint_score({"val/generation_cer": 0.01, "val/generation_empty_samples": 1}),
            float("inf"),
        )
        self.assertEqual(
            generation_checkpoint_score({"val/generation_cer": 0.2, "val/generation_empty_samples": 0}),
            0.2,
        )
        self.assertEqual(
            generation_checkpoint_score(
                {"val/generation_cer": 0.2, "val/generation_empty_samples": 0}, baseline_cer=0.2,
            ),
            float("inf"),
        )
        self.assertEqual(
            generation_checkpoint_score(
                {"val/generation_cer": 0.1, "val/generation_empty_samples": 0}, baseline_cer=0.2,
            ),
            0.1,
        )

    def test_inference_config_snapshot_keeps_training_dataset_model_and_generation(self) -> None:
        from personaplex_finetuning.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_config = root / "train.json"
            source_config.write_text(json.dumps({
                "model": {"root": str(root / "base"), "source": str(root / "moshi"), "device": "cuda"},
                "data": {
                    "prepared_dir": str(root / "prepared"),
                    "val_manifest": str(root / "custom-val.jsonl"),
                },
                "lora": {"qlora": True, "quant_type": "fp4"},
                "seed": 77,
                "generation": {"use_sampling": False, "top_k_text": 9},
            }), encoding="utf-8")
            training_config = load_config(source_config)
            snapshot = inference_config_snapshot(
                training_config, root / "adapter" / "lora.safetensors", root / "outputs",
            )
            snapshot_path = root / "inference.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

            inference_config = load_config(snapshot_path)

        self.assertEqual(inference_config.model_root, training_config.model_root)
        self.assertEqual(inference_config.personaplex_source, training_config.personaplex_source)
        self.assertEqual(inference_config.manifest, training_config.manifest)
        self.assertEqual(inference_config.val_manifest, training_config.val_manifest)
        self.assertEqual(inference_config.device, training_config.device)
        self.assertEqual(inference_config.qlora, training_config.qlora)
        self.assertEqual(inference_config.generation_settings, training_config.generation_settings)

    def test_free_running_validation_scores_generated_text_on_the_requested_window(self) -> None:
        from personaplex_finetuning.data import AudioInfo, PreparedSample, Word

        sample = PreparedSample(
            sample_id="heldout-1", conversation_wav=Path("conversation.wav"),
            voice_prompt_wav=Path("voice.wav"), words=(Word("agent", "Xin chào.", 1.0, 1.5),),
            text_prompt="Trò chuyện bằng tiếng Việt.", metadata={},
            audio=AudioInfo(24_000, 2, 60.0), window_start_sec=0.0, window_end_sec=60.0,
        )
        config = SimpleNamespace(
            seed=42, free_running_eval_samples=1, free_running_eval_window_seconds=30.0,
        )

        with patch(
            "personaplex_finetuning.train.generate_text_with_runtime",
            side_effect=lambda _runtime, window, **_kwargs: (
                "Xin chào." if (window.window_start_sec, window.window_end_sec) == (1.0, 31.0) else "wrong window"
            ),
        ):
            metrics = evaluate_free_running(object(), [sample], config)

        self.assertEqual(metrics["val/generation_samples"], 1)
        self.assertEqual(metrics["val/generation_cer"], 0.0)
        self.assertEqual(metrics["val/generation_wer"], 0.0)
        self.assertEqual(metrics["samples"][0]["sample_id"], "heldout-1")
        self.assertEqual(metrics["samples"][0]["window_start_sec"], 1.0)
        self.assertEqual(metrics["samples"][0]["source_duration_sec"], 60.0)

    def test_free_running_validation_exports_audio_and_records_its_path(self) -> None:
        from personaplex_finetuning.data import AudioInfo, PreparedSample, Word

        sample = PreparedSample(
            sample_id="heldout-audio", conversation_wav=Path("conversation.wav"),
            voice_prompt_wav=Path("voice.wav"), words=(Word("agent", "Xin chào.", 1.0, 1.5),),
            text_prompt="Trò chuyện bằng tiếng Việt.", metadata={},
            audio=AudioInfo(24_000, 2, 60.0), window_start_sec=0.0, window_end_sec=60.0,
        )
        config = SimpleNamespace(
            seed=42, free_running_eval_samples=1, free_running_eval_window_seconds=30.0,
        )

        def generate(_runtime, _window, *, output_wav, **_kwargs):
            output_wav.parent.mkdir(parents=True, exist_ok=True)
            output_wav.write_bytes(b"wav")
            return "Xin chào."

        def export_original(_window, output_wav):
            output_wav.parent.mkdir(parents=True, exist_ok=True)
            output_wav.write_bytes(b"original")
            return output_wav

        with tempfile.TemporaryDirectory() as directory, \
             patch("personaplex_finetuning.train.generate_text_with_runtime", side_effect=generate), \
             patch("personaplex_finetuning.train.export_original_audio_window", side_effect=export_original):
            metrics = evaluate_free_running(
                object(), [sample], config, audio_output_dir=Path(directory) / "step_000100",
            )

            audio_path = Path(metrics["samples"][0]["audio_path"])
            original_audio_path = Path(metrics["samples"][0]["original_audio_path"])
            self.assertEqual(audio_path.name, "sample_000.wav")
            self.assertEqual(audio_path.read_bytes(), b"wav")
            self.assertEqual(original_audio_path.name, "sample_000_original.wav")
            self.assertEqual(original_audio_path.read_bytes(), b"original")

    def test_free_running_metrics_log_reference_in_configured_text_mode(self) -> None:
        from personaplex_finetuning.data import AudioInfo, PreparedSample, Word

        sample = PreparedSample(
            sample_id="telex-1", conversation_wav=Path("conversation.wav"),
            voice_prompt_wav=Path("voice.wav"), words=(Word("agent", "tương", 1.0, 1.5),),
            text_prompt="Trò chuyện bằng tiếng Việt.", metadata={},
            audio=AudioInfo(24_000, 2, 60.0), window_start_sec=0.0, window_end_sec=60.0,
        )
        config = SimpleNamespace(
            seed=42, free_running_eval_samples=1, free_running_eval_window_seconds=30.0,
            vietnamese_text_mode="telex",
        )

        with patch("personaplex_finetuning.train.generate_text_with_runtime", return_value="tuowng"):
            metrics = evaluate_free_running(object(), [sample], config)

        self.assertEqual(metrics["samples"][0]["reference"], "tuowng")
        self.assertEqual(metrics["samples"][0]["hypothesis"], "tuowng")
        self.assertEqual(metrics["samples"][0]["cer"], 0.0)
        self.assertEqual(metrics["samples"][0]["wer"], 0.0)

    def test_free_running_validation_counts_empty_hypotheses(self) -> None:
        from personaplex_finetuning.data import AudioInfo, PreparedSample, Word

        sample = PreparedSample(
            sample_id="heldout-empty", conversation_wav=Path("conversation.wav"),
            voice_prompt_wav=Path("voice.wav"), words=(Word("agent", "Xin chào.", 1.0, 1.5),),
            text_prompt="Trò chuyện bằng tiếng Việt.", metadata={},
            audio=AudioInfo(24_000, 2, 60.0), window_start_sec=0.0, window_end_sec=60.0,
        )
        config = SimpleNamespace(
            seed=42, free_running_eval_samples=1, free_running_eval_window_seconds=30.0,
        )
        with patch("personaplex_finetuning.train.generate_text_with_runtime", return_value="  "):
            metrics = evaluate_free_running(object(), [sample], config)
        self.assertEqual(metrics["val/generation_empty_samples"], 1)
        self.assertEqual(generation_checkpoint_score(metrics, baseline_cer=0.5), float("inf"))

    def test_distributed_loss_reduces_components_in_one_collective_and_preserves_gradient(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        local_stats = {
            "text": (parameter * 4, torch.tensor(2.0)),
            "audio": (parameter * 3, torch.tensor(1.0)),
        }
        remote_values = torch.tensor([2.0, 3.0, 6.0, 3.0])
        calls = []

        def all_reduce(value):
            calls.append(value.clone())
            value.add_(remote_values)
            return value

        components, total = reduce_distributed_loss(
            local_stats, all_reduce, preserve_grad=True, world_size=2,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].numel(), 4)
        self.assertAlmostEqual(float(components["text"]), 2.0)
        self.assertAlmostEqual(float(components["audio"]), 3.0)
        self.assertAlmostEqual(float(total), 5.0)
        total.backward()
        self.assertAlmostEqual(float(parameter.grad), 3.1, places=6)

    def test_each_rank_receives_a_different_sample_before_dataset_wraparound(self) -> None:
        indices = {sample_index_for_rank(3, rank, 4, 393) for rank in range(4)}
        self.assertEqual(indices, {12, 13, 14, 15})

    def test_rank_info_records_physical_gpu_and_sample_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_rank_info(Path(tmp), 1, 4, "cuda:1", 393)
            info = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(info["rank"], 1)
        self.assertEqual(info["world_size"], 4)
        self.assertEqual(info["first_sample_index"], 1)

    def test_optimizer_only_steps_at_accumulation_sync_boundary(self) -> None:
        class SyncState:
            def __init__(self, sync_gradients: bool) -> None:
                self.sync_gradients = sync_gradients

        class Optimizer:
            def __init__(self) -> None:
                self.steps = 0
                self.zeroes = 0

            def step(self) -> None:
                self.steps += 1

            def zero_grad(self, set_to_none: bool) -> None:
                self.zeroes += 1

        class Scheduler:
            def __init__(self) -> None:
                self.steps = 0

            def step(self) -> None:
                self.steps += 1

        optimizer = Optimizer()
        scheduler = Scheduler()
        unsynced = SyncState(sync_gradients=False)
        synced = SyncState(sync_gradients=True)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        parameter.grad = torch.tensor(0.75)
        trainable = [parameter]

        self.assertEqual(step_optimizer_if_ready(unsynced, optimizer, scheduler, trainable), 0.0)
        self.assertEqual((optimizer.steps, optimizer.zeroes, scheduler.steps), (0, 0, 0))
        self.assertAlmostEqual(step_optimizer_if_ready(synced, optimizer, scheduler, trainable), 0.75)
        self.assertEqual((optimizer.steps, optimizer.zeroes, scheduler.steps), (1, 1, 1))

    def test_nonfinite_gradient_aborts_before_optimizer_update(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        parameter.grad = torch.tensor(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "non-finite gradient norm"):
            step_optimizer_if_ready(SimpleNamespace(sync_gradients=True), optimizer, None, [parameter])
        self.assertEqual(float(parameter.detach()), 1.0)

    def test_training_state_restores_optimizer_scheduler_and_step(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        parameter.grad = torch.tensor(1.0)
        optimizer.step()
        scheduler.step()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            save_training_state(checkpoint, optimizer, scheduler, 12, 2, 4)

            restored_parameter = torch.nn.Parameter(torch.tensor(1.0))
            restored_optimizer = torch.optim.AdamW([restored_parameter], lr=0.1)
            restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
            restored_step = load_training_state(checkpoint, restored_optimizer, restored_scheduler, 2, 4)

            self.assertEqual(restored_step, 12)
            self.assertEqual(restored_scheduler.last_epoch, scheduler.last_epoch)
            self.assertTrue(restored_optimizer.state_dict()["state"])

    def test_resume_rejects_a_changed_max_step_onecycle_schedule(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.1, total_steps=8, pct_start=0.25,
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            save_training_state(checkpoint, optimizer, scheduler, 4, 1, 1)

            with self.assertRaisesRegex(RuntimeError, r"max_steps \(8\).*\(12\)"):
                load_training_state(checkpoint, optimizer, scheduler, 1, 1, total_steps=12)

    def test_resume_rejects_a_checkpoint_already_at_max_steps(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "already complete"):
            validate_resume_step(start_step=2000, max_steps=2000)
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            validate_resume_step(start_step=2001, max_steps=2000)

    def test_adapter_metadata_records_effective_lora_alpha_and_scaling(self) -> None:
        config = SimpleNamespace(
            model_root=Path("/models/personaplex"), lora_rank=128,
            lora_alpha=256, lora_scaling=2.0, per_device_batch_size=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            adapter = save_adapter_state(
                Path(tmp), {"lora.weight": torch.ones(1)}, config, step=7,
            )
            metadata = json.loads((adapter.parent / "adapter.json").read_text())

        self.assertEqual((metadata["rank"], metadata["alpha"], metadata["scaling"]), (128, 256, 2.0))

    def test_resume_rejects_changed_ddp_topology(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            save_training_state(checkpoint, optimizer, None, 12, 2, 4)

            with self.assertRaisesRegex(RuntimeError, "num_processes"):
                load_training_state(checkpoint, optimizer, None, 2, 1)

            with self.assertRaisesRegex(RuntimeError, "gradient_accumulation_steps"):
                load_training_state(checkpoint, optimizer, None, 1, 4)

            save_training_state(checkpoint, optimizer, None, 12, 2, 4, per_device_batch_size=2)
            with self.assertRaisesRegex(RuntimeError, "per_device_batch_size"):
                load_training_state(checkpoint, optimizer, None, 2, 4, per_device_batch_size=1)

    def test_creates_distinct_timestamped_run_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"

            smoke = create_run_dir(root, smoke=True)
            train = create_run_dir(root, smoke=False)

            self.assertTrue(smoke.is_dir())
            self.assertTrue(train.is_dir())
            self.assertNotEqual(smoke, train)
            self.assertTrue(smoke.name.startswith("smoke_"))
            self.assertTrue(train.name.startswith("train_"))

    def test_uses_lmmodel_forward_train_instead_of_module_forward(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.codes = None

            def forward_train(self, codes):
                self.codes = codes
                return "lm-output"

            def forward(self, _codes):
                raise AssertionError("nn.Module.forward must not be called")

        model = Model()

        self.assertEqual(model_forward_train(model, "codes"), "lm-output")
        self.assertEqual(model.codes, "codes")

    def test_ddp_forward_uses_wrapper_call_for_gradient_reduction(self) -> None:
        class FakeDDP:
            def __init__(self):
                self.called = False

            def __call__(self, codes):
                self.called = True
                return "ddp-output"

            def forward_train(self, _codes):
                raise AssertionError("direct forward_train bypasses DDP hooks")

        model = FakeDDP()
        with patch("torch.nn.parallel.DistributedDataParallel", FakeDDP):
            output = model_forward_train(model, "codes")

        self.assertEqual(output, "ddp-output")
        self.assertTrue(model.called)

    def test_unwrap_parallel_model_removes_ddp_module_prefix_before_saving(self) -> None:
        base = torch.nn.Linear(2, 2)

        class FakeDDP:
            def __init__(self, module):
                self.module = module

        wrapped = FakeDDP(base)
        self.assertIs(unwrap_parallel_model(wrapped), base)

    def test_joint_stage_targets_both_temporal_and_depth_lora_without_dual_lr(self) -> None:
        config = SimpleNamespace(train_stage="joint", depformer_learning_rate=None)
        self.assertEqual(lora_prefixes_for_stage(config), ("transformer", "depformer"))

    def test_reload_verification_reuses_loaded_base_instead_of_loading_a_second_7b_copy(self) -> None:
        from personaplex_finetuning.lora import inject_lora

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.projection = torch.nn.Linear(2, 2)

        model = Model()
        inject_lora(model, rank=2, alpha=4)
        saved = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters() if "lora_" in name
        }
        runtime = SimpleNamespace(model=model, codec=object(), tokenizer=object())
        config = SimpleNamespace(device="cpu")

        def reload_checkpoint(target, _adapter):
            with torch.no_grad():
                for name, parameter in target.named_parameters():
                    if name in saved:
                        parameter.copy_(saved[name])

        def teacher_forced_loss(_config, target_runtime, _example, _optimizer=None):
            return sum(
                parameter.sum() for name, parameter in target_runtime.model.named_parameters()
                if "lora_" in name
            ), {}, 0.0

        with patch("personaplex_finetuning.train.load_runtime", side_effect=AssertionError("must not load a second base")), \
             patch("personaplex_finetuning.train.build_example", return_value=object()), \
             patch("personaplex_finetuning.train.load_adapter", side_effect=reload_checkpoint), \
             patch("personaplex_finetuning.train.one_step", side_effect=teacher_forced_loss):
            loss = verify_reloaded_adapter(config, object(), Path("adapter.safetensors"), runtime)

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        for name, parameter in model.named_parameters():
            if name in saved:
                self.assertTrue(torch.equal(parameter, saved[name]))

    def test_stage_specific_lora_prefixes_are_shared_by_train_and_reload(self) -> None:
        self.assertEqual(
            lora_prefixes_for_stage(SimpleNamespace(train_stage="temporal_only")),
            ("transformer",),
        )
        self.assertEqual(
            lora_prefixes_for_stage(SimpleNamespace(train_stage="depth_only")),
            ("depformer",),
        )

    def test_training_sample_inspection_checks_target_roundtrip_and_channels(self) -> None:
        sample = SimpleNamespace(
            sample_id="vi-1", agent_channel=0, user_channel=1,
            window_start_sec=0.0, window_end_sec=2.0,
            text_prompt="Hãy trả lời bằng tiếng Việt.",
            words=[SimpleNamespace(speaker="agent", word="Xin chào", start=0.2, end=1.0)],
        )

        class Tokenizer:
            def encode(self, text):
                self.text = text
                return [1, 2]

            def decode(self, _tokens):
                return self.text

        report = inspect_training_sample(sample, Tokenizer())
        self.assertEqual(report["agent_text"], "Xin chào")
        self.assertEqual(report["agent_channel"], 0)
        self.assertEqual(report["user_channel"], 1)
        self.assertEqual(report["token_count"], 2)
        self.assertEqual(report["text_prompt"], "Hãy trả lời bằng tiếng Việt.")

    def test_text_supervision_counts_separate_real_targets_from_padding(self) -> None:
        batch = {
            "labels": torch.tensor([[[4, 3, 0, 3]] + [[0, 0, 0, 0]] * 16]),
            "loss_mask": torch.tensor([[[True, True, True, False]] + [[False, False, False, False]] * 16]),
        }
        output = SimpleNamespace(text_mask=torch.tensor([[[True, True, True, True]]]))
        tokens, padding = text_supervision_counts(batch, output, padding_id=(3, 0))
        self.assertEqual((int(tokens), int(padding)), (1, 2))

    def test_nonpadding_text_loss_measures_only_valid_target_tokens(self) -> None:
        logits = torch.tensor([[[[0.0, 2.0, 0.0, 0.0, 0.0], [100.0] * 5, [0.0] * 5]]])
        batch = {
            "labels": torch.tensor([[[1, 3, 0]] + [[0, 0, 0]] * 16]),
            "loss_mask": torch.tensor([[[True, True, False]] + [[False, False, False]] * 16]),
        }
        output = SimpleNamespace(
            text_logits=logits,
            text_mask=torch.tensor([[[True, True, True]]]),
        )

        loss_sum, token_count = text_target_token_loss_stats(batch, output, padding_id=(3, 0))
        expected = torch.nn.functional.cross_entropy(logits[0, 0, 0].reshape(1, -1), torch.tensor([1]), reduction="sum")
        self.assertEqual(int(token_count), 1)
        self.assertTrue(torch.allclose(loss_sum, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for BF16 autocast regression")
    def test_cuda_forward_autocasts_fp32_activations_for_bf16_projection(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = torch.nn.Linear(3, 2, dtype=torch.bfloat16)

            def forward_train(self, codes):
                return self.projection(codes)

        model = Model().cuda()
        output = model_forward_train(model, torch.randn(4, 3, device="cuda", dtype=torch.float32))

        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().sum().backward()
        self.assertIsNotNone(model.projection.weight.grad)

    def test_writes_losses_and_training_parameters_to_tensorboard(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.scalars = []

            def add_scalar(self, name, value, step) -> None:
                self.scalars.append((name, value, step))

        writer = Writer()
        write_tensorboard_scalars(
            writer,
            {"step": 2, "loss/total": 1.0, "loss/text": 0.2, "loss/text_real": 0.25,
             "loss/agent_semantic": 0.2, "loss/agent_acoustic": 0.3,
             "loss/user_semantic": 0.1, "loss/user_acoustic": 0.1,
             "loss/audio_semantic": 0.3,
             "loss/text_nonpadding": 1.5, "loss/audio_nonsemantic": 0.4,
             "lr": 2e-5, "grad_norm": 0.5, "gpu_peak_bytes": 1024,
             "timing/data_sec": 1.0, "timing/forward_loss_sec": 2.0,
             "timing/backward_sec": 3.0, "timing/optimizer_sec": 0.5,
             "timing/profiled_phase_sum_sec": 6.5},
            trainable_parameters=42,
            cpu_threads=1,
        )

        self.assertEqual({name for name, _, _ in writer.scalars}, {
            "loss/total", "loss/text", "loss/text_real", "loss/text_nonpadding", "loss/audio_total",
            "loss/agent_semantic", "loss/agent_acoustic", "loss/user_semantic", "loss/user_acoustic",
            "loss/audio_semantic", "loss/audio_nonsemantic", "accuracy/text", "accuracy/audio_total",
            "accuracy/text_nonpad", "accuracy/text_pad", "target_pad_pct", "predicted_pad_pct",
            "probability/pad_given_nonpad_target", "probability/correct_given_nonpad_target",
            "train/valid_token_pct",
            "train/learning_rate", "train/gradient_norm", "system/gpu_peak_bytes",
            "system/trainable_parameters", "system/cpu_threads", "timing/data_sec",
            "timing/forward_loss_sec", "timing/backward_sec", "timing/optimizer_sec",
            "timing/profiled_phase_sum_sec",
        })
