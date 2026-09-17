"""Minimal LoRA injection for PersonaPlex transformer Linear modules."""

from __future__ import annotations


def inject_lora(model, rank: int, alpha: float, dropout: float = 0.0) -> list[str]:
    import torch

    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")

    class LoRALinear(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.scale = alpha / rank
            self.dropout = torch.nn.Dropout(dropout)
            self.lora_a = torch.nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=base.weight.dtype)
            self.lora_b = torch.nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype)
            torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
            torch.nn.init.zeros_(self.lora_b.weight)
            for parameter in self.base.parameters():
                parameter.requires_grad = False

        def forward(self, value):
            return self.base(value) + self.lora_b(self.lora_a(self.dropout(value))) * self.scale

        @property
        def weight(self):
            """Effective weight for Moshi modules that access ``Linear.weight`` directly."""
            return self.base.weight + (self.lora_b.weight @ self.lora_a.weight) * self.scale

        @property
        def bias(self):
            return self.base.bias

        @property
        def in_features(self):
            return self.base.in_features

        @property
        def out_features(self):
            return self.base.out_features

    targets: list[tuple[str, object, str]] = []
    for prefix in ("transformer", "depformer"):
        root = getattr(model, prefix, None)
        if root is None:
            continue
        for name, module in root.named_modules():
            if isinstance(module, torch.nn.Linear):
                parent_name, _, attribute = name.rpartition(".")
                parent = root.get_submodule(parent_name) if parent_name else root
                targets.append((f"{prefix}.{name}", parent, attribute))
    if not targets:
        raise RuntimeError("no PersonaPlex transformer Linear modules found for LoRA")
    for name, parent, attribute in targets:
        setattr(parent, attribute, LoRALinear(getattr(parent, attribute)))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_a." in name or ".lora_b." in name
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise AssertionError("only LoRA parameters may be trainable")
    return [name for name, _, _ in targets]


def adapter_state_dict(model):
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if ".lora_a." in name or ".lora_b." in name}


def load_adapter(model, path) -> None:
    from safetensors.torch import load_file
    state = load_file(str(path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any("lora_" in name for name in missing):
        raise RuntimeError(f"adapter mismatch; missing={missing}, unexpected={unexpected}")
