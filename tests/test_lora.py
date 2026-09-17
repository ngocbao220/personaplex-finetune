import unittest


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised on documentation-only environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for LoRA module tests")
class LoRATest(unittest.TestCase):
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
