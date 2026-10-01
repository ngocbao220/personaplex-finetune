import unittest
import tempfile
import json
import os
import subprocess
import sys
from pathlib import Path


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised on documentation-only environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for LoRA module tests")
class LoRATest(unittest.TestCase):
    def test_zero_adapter_and_fresh_process_reload_match_logits_and_delta(self) -> None:
        from safetensors.torch import save_file
        from personaplex_finetuning.lora import adapter_state_dict, inject_lora

        def make_model():
            torch.manual_seed(71)
            model = torch.nn.Module()
            model.transformer = torch.nn.Module()
            model.transformer.projection = torch.nn.Linear(3, 2)
            return model

        x = torch.tensor([[0.2, -0.4, 0.8]])
        base = make_model().transformer.projection(x).detach()
        model = make_model()
        inject_lora(model, rank=2, alpha=4)
        projection = model.transformer.projection
        self.assertTrue(torch.allclose(base, projection(x), rtol=0, atol=1e-7))
        self.assertFalse(any(parameter.requires_grad for parameter in projection.base.parameters()))
        with torch.no_grad():
            projection.lora_b.weight.fill_(0.25)
        expected_logits = projection(x).detach()
        expected_delta = (projection.weight - projection.base.weight).detach()
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter.safetensors"
            save_file(adapter_state_dict(model), str(adapter))
            program = """
import json, sys, torch
from personaplex_finetuning.lora import inject_lora, load_adapter
torch.manual_seed(71)
model = torch.nn.Module()
model.transformer = torch.nn.Module()
model.transformer.projection = torch.nn.Linear(3, 2)
inject_lora(model, rank=2, alpha=4)
load_adapter(model, sys.argv[1])
projection = model.transformer.projection
x = torch.tensor([[0.2, -0.4, 0.8]])
print(json.dumps({"logits": projection(x).detach().tolist(),
                  "delta": (projection.weight - projection.base.weight).detach().tolist()}))
"""
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            environment["OMP_NUM_THREADS"] = "1"
            environment["MKL_NUM_THREADS"] = "1"
            result = subprocess.run(
                [sys.executable, "-c", program, str(adapter)],
                capture_output=True, text=True, env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            reloaded = json.loads(result.stdout)
            self.assertTrue(torch.allclose(torch.tensor(reloaded["logits"]), expected_logits, rtol=0, atol=1e-7))
            self.assertTrue(torch.allclose(torch.tensor(reloaded["delta"]), expected_delta, rtol=0, atol=1e-7))

    def test_adapter_load_rejects_extra_or_missing_lora_weights(self) -> None:
        from safetensors.torch import save_file
        from personaplex_finetuning.lora import adapter_state_dict, inject_lora, load_adapter

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.projection = torch.nn.Linear(3, 2)

        model = Model()
        inject_lora(model, rank=2, alpha=4)
        state = adapter_state_dict(model)
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "lora.safetensors"
            save_file(state, str(adapter))
            load_adapter(model, adapter)

            save_file({**state, "depformer.extra.lora_a.weight": torch.zeros(2, 2)}, str(adapter))
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                load_adapter(model, adapter)

            save_file({next(iter(state)): next(iter(state.values()))}, str(adapter))
            with self.assertRaisesRegex(RuntimeError, "missing"):
                load_adapter(model, adapter)

    def test_replacement_accepts_bitsandbytes_4bit_linear(self) -> None:
        from personaplex_finetuning.lora import inject_lora

        class Linear4bit(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_features = 3
                self.out_features = 2
                self.compute_dtype = torch.bfloat16
                self.weight = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16), requires_grad=False)
                self.bias = None

            def forward(self, value):
                return torch.nn.functional.linear(value, self.weight, self.bias)

        Linear4bit.__module__ = "bitsandbytes.nn.modules"

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.projection = Linear4bit()

        model = Model()
        inject_lora(model, rank=2, alpha=4)
        projection = model.transformer.projection
        self.assertEqual(projection.base.__class__.__name__, "Linear4bit")
        self.assertEqual(projection.lora_a.weight.dtype, torch.float32)

    def test_replacement_exposes_effective_linear_weight_for_direct_access(self) -> None:
        from personaplex_finetuning.lora import inject_lora

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.projection = torch.nn.Linear(3, 2)

        model = Model()
        inject_lora(model, rank=2, alpha=4)
        projection = model.transformer.projection
        with torch.no_grad():
            projection.lora_b.weight.fill_(0.25)

        expected = projection.base.weight + (
            projection.lora_b.weight @ projection.lora_a.weight
        ) * projection.scale
        self.assertTrue(torch.allclose(projection.weight, expected))
        self.assertIs(projection.bias, projection.base.bias)
        self.assertEqual(projection.in_features, 3)
        self.assertEqual(projection.out_features, 2)

        projection.weight.square().sum().backward()
        self.assertIsNotNone(projection.lora_a.weight.grad)
        self.assertIsNotNone(projection.lora_b.weight.grad)
