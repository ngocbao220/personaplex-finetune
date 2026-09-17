import unittest


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - documentation-only environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for gating tests")
class ActivationGatingTest(unittest.TestCase):
    def test_uses_lora_linear_forwards_without_materializing_effective_weights(self) -> None:
        from moshi.modules.gating import ActivationGating

        class LoRALikeLinear(torch.nn.Module):
            def __init__(self, in_features: int, out_features: int) -> None:
                super().__init__()
                self.lora_a = torch.nn.Linear(in_features, 1, bias=False)
                self.projection = torch.nn.Linear(in_features, out_features, bias=False)

            def forward(self, value):
                return self.projection(value)

        gating = ActivationGating(2, 4, torch.nn.functional.silu)
        gating.linear_in = LoRALikeLinear(2, 4)
        gating.linear_out = LoRALikeLinear(2, 2)
        value = torch.ones(1, 3, 2)

        result = gating(value)

        self.assertEqual(tuple(result.shape), (1, 3, 2))
