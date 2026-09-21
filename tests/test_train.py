import tempfile
import unittest
import json
from pathlib import Path

import torch

from personaplex_finetuning.train import (
    create_run_dir,
    effective_global_batch_size,
    load_training_state,
    model_forward_train,
    sample_index_for_rank,
    step_optimizer_if_ready,
    save_training_state,
    write_rank_info,
    write_tensorboard_scalars,
)


class TrainTest(unittest.TestCase):
    def test_global_batch_multiplies_processes_and_accumulation(self) -> None:
        self.assertEqual(effective_global_batch_size(1, 4, 2), 8)

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
        class Accelerator:
            def __init__(self, sync_gradients: bool) -> None:
                self.sync_gradients = sync_gradients
                self.clip_calls = 0

            def clip_grad_norm_(self, _parameters, _max_norm):
                self.clip_calls += 1
                return 0.75

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
        unsynced = Accelerator(sync_gradients=False)
        synced = Accelerator(sync_gradients=True)

        self.assertEqual(step_optimizer_if_ready(unsynced, optimizer, scheduler, []), 0.0)
        self.assertEqual((optimizer.steps, optimizer.zeroes, scheduler.steps, unsynced.clip_calls), (0, 0, 0, 0))
        self.assertEqual(step_optimizer_if_ready(synced, optimizer, scheduler, []), 0.75)
        self.assertEqual((optimizer.steps, optimizer.zeroes, scheduler.steps, synced.clip_calls), (1, 1, 1, 1))

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

    def test_writes_losses_and_training_parameters_to_tensorboard(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.scalars = []

            def add_scalar(self, name, value, step) -> None:
                self.scalars.append((name, value, step))

        writer = Writer()
        write_tensorboard_scalars(
            writer,
            {"step": 2, "loss/total": 1.0, "loss/text": 0.2, "loss/audio_semantic": 0.3,
             "loss/audio_nonsemantic": 0.4, "lr": 2e-5, "grad_norm": 0.5, "gpu_peak_bytes": 1024},
            trainable_parameters=42,
            cpu_threads=1,
        )

        self.assertEqual({name for name, _, _ in writer.scalars}, {
            "loss/total", "loss/text", "loss/audio_semantic", "loss/audio_nonsemantic",
            "train/learning_rate", "train/gradient_norm", "system/gpu_peak_bytes",
            "system/trainable_parameters", "system/cpu_threads",
        })
