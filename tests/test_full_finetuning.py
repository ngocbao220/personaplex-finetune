import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from personaplex_finetuning.config import Config, load_config
from personaplex_finetuning.full_checkpoint import load_full_weights, resolve_full_checkpoint
from personaplex_finetuning.inference import checkpoint_model_root
from personaplex_finetuning.train import (
    configure_trainable_parameters,
    full_attention_no_decay_ids,
    inference_config_snapshot,
    load_training_state,
    save_full_checkpoint,
    training_contract,
    validate_resume_checkpoint,
    run,
)
from tools import inference_smoke


class FullFinetuningTest(unittest.TestCase):
    def test_checkpoint_cli_alias_reaches_standalone_inference_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "infer.yaml"
            config_path.write_text("{}\n", encoding="utf-8")
            checkpoint = root / "checkpoint_000001"
            with patch.object(sys, "argv", [
                "inference_smoke.py", "--config", str(config_path), "--checkpoint", str(checkpoint),
                "--input-file", str(root / "user.wav"), "--voice-prompt", str(root / "voice.wav"),
                "--text-prompt", "Trò chuyện", "--output-dir", str(root / "outputs"),
            ]), patch.object(inference_smoke, "load_config", return_value=SimpleNamespace(
                window_seconds=30.0,
            )), patch.object(inference_smoke, "PreparedDataset") as dataset, \
                 patch.object(inference_smoke, "smoke", autospec=True) as smoke:
                smoke.return_value = "reference_unavailable"
                self.assertEqual(inference_smoke.main(), 0)
            dataset.assert_not_called()
            self.assertEqual(smoke.call_args.kwargs["adapter"], checkpoint.resolve())
            self.assertIsNone(smoke.call_args.kwargs["sample"])

    def test_full_rejects_qlora_and_partial_training_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config(
                path=root / "config.yaml", model_root=root, personaplex_source=root,
                prepared_dir=root, output_dir=root, train_method="full", qlora=True,
            )
            with self.assertRaisesRegex(ValueError, "cannot use lora.qlora"):
                run(config)
            with self.assertRaisesRegex(ValueError, "requires train.stage=joint"):
                run(config.replace(qlora=False, train_stage="temporal_only"))

    def test_full_preset_and_trainable_lm_parameters(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / "full-finetuning.yaml")
        self.assertEqual(config.train_method, "full")
        self.assertEqual(config.per_device_batch_size, 1)
        self.assertFalse(config.shuffle)

        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        mimi = torch.nn.Linear(2, 2)
        model.requires_grad_(False)
        mimi.requires_grad_(False)
        self.assertEqual(configure_trainable_parameters(model, config, ("transformer",)), [])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model(torch.ones(1, 2)).sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in mimi.parameters()))
        optimizer.step()

    def test_full_freezes_user_output_only_layers_but_trains_user_input(self):
        class SmallPersonaPlex(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dep_q = 16
                self.depformer_multi_linear = True
                self.depformer_weights_per_step_schedule = None
                self.emb = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(16))
                self.depformer_in = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(16))
                self.depformer_emb = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(15))
                self.linears = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(16))
                layer = torch.nn.Module()
                layer.gating = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(16))
                layer.self_attn = torch.nn.Module()
                layer.self_attn.weights_per_step = 16
                layer.self_attn.in_proj_weight = torch.nn.Parameter(torch.randn(32, 2))
                layer.self_attn.out_proj = torch.nn.Linear(2, 32, bias=False)
                self.depformer = torch.nn.Module()
                self.depformer.layers = torch.nn.ModuleList([layer])

        model = SmallPersonaPlex()
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / "full-finetuning.yaml")
        configure_trainable_parameters(model, config, ())
        self.assertTrue(all(layer.weight.requires_grad for layer in model.emb))
        self.assertTrue(all(layer.weight.requires_grad for layer in model.linears[:8]))
        for layers, start in ((model.depformer_in, 8), (model.depformer_emb, 7),
                              (model.linears, 8), (model.depformer.layers[0].gating, 8)):
            self.assertTrue(all(not parameter.requires_grad for layer in layers[start:] for parameter in layer.parameters()))

        frozen_before = model.linears[8].weight.detach().clone()
        packed_before = model.depformer.layers[0].self_attn.in_proj_weight.detach().clone()
        no_decay = full_attention_no_decay_ids(model)
        self.assertEqual(len(no_decay), 2)
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in no_decay], "weight_decay": 0.1},
            {"params": [p for p in model.parameters() if p.requires_grad and id(p) in no_decay], "weight_decay": 0.0},
        ], lr=0.1)
        value = model.linears[0](model.depformer_in[0](model.emb[8](torch.ones(1, 2)))).sum()
        value = value + torch.nn.functional.linear(
            torch.ones(1, 2), model.depformer.layers[0].self_attn.in_proj_weight[:16]
        ).sum()
        value.backward()
        optimizer.step()
        torch.testing.assert_close(model.linears[8].weight, frozen_before)
        torch.testing.assert_close(model.depformer.layers[0].self_attn.in_proj_weight[16:], packed_before[16:])
        self.assertIsNotNone(model.emb[8].weight.grad)

    def test_complete_checkpoint_roundtrip_resume_and_inference_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "train.jsonl").write_text("{}\n", encoding="utf-8")
            config = Config(
                path=root / "config.yaml", model_root=root / "base", personaplex_source=root / "src",
                prepared_dir=prepared, output_dir=root, train_method="full", per_device_batch_size=1,
            )
            run = root / "run"
            run.mkdir()
            (run / "config.json").write_text(
                json.dumps({"training_contract": training_contract(config, [])}), encoding="utf-8",
            )
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            expected = model.weight.detach().clone()
            checkpoint = save_full_checkpoint(run, model, config, 1, optimizer, None, 1, 1)

            weights, metadata = resolve_full_checkpoint(checkpoint)
            self.assertEqual(weights, checkpoint / "model.safetensors")
            self.assertEqual(metadata["method"], "full")
            self.assertEqual(checkpoint_model_root(checkpoint), config.model_root.resolve())
            model.weight.data.zero_()
            load_full_weights(model, checkpoint)
            torch.testing.assert_close(model.weight, expected)
            self.assertEqual(validate_resume_checkpoint(config, str(checkpoint), (), []), (weights, 1))
            restored = torch.optim.AdamW(model.parameters(), lr=0.01)
            self.assertEqual(load_training_state(checkpoint, restored, None, 1, 1, 1), 1)
            self.assertTrue(restored.state_dict()["state"])

            snapshot = inference_config_snapshot(config, checkpoint, root / "inference")
            self.assertEqual(snapshot["checkpoint"]["path"], str(checkpoint))
            self.assertNotIn("adapter", snapshot)

            (checkpoint / "checkpoint.json").unlink()
            with self.assertRaises(FileNotFoundError):
                resolve_full_checkpoint(checkpoint)
