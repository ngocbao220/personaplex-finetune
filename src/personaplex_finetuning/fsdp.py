"""Small FSDP helpers aligned with the Moshi LoRA wrapping policy."""

from __future__ import annotations

import socket


def parse_gpu_ids(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) < 2 or any(not part.isdigit() for part in parts):
        raise ValueError("GPU ids must be a comma-separated list containing at least two ids")
    ids = [int(part) for part in parts]
    if len(set(ids)) != len(ids):
        raise ValueError("GPU ids must be unique")
    return ids


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def materialize_meta_module(module, device) -> None:
    """Materialize only this module's meta parameters without calling reset_parameters."""
    module.to_empty(device=device, recurse=False)


def get_fsdp_policy(is_lora: bool):
    """Wrap transformer blocks and isolate trainable LoRA groups."""
    import torch.distributed.fsdp.wrap as wrap
    from moshi.modules.transformer import StreamingTransformerLayer

    transformer_policy = wrap.transformer_auto_wrap_policy
    import functools

    block_policy = functools.partial(
        transformer_policy, transformer_layer_cls=(StreamingTransformerLayer,)
    )
    if not is_lora:
        return block_policy

    def lora_policy_fn(module):
        parameters = list(module.parameters())
        return bool(parameters) and all(parameter.requires_grad for parameter in parameters)

    lora_policy = functools.partial(
        wrap.lambda_auto_wrap_policy,
        lambda_fn=lora_policy_fn,
    )
    return functools.partial(wrap._or_policy, policies=[lora_policy, block_policy])


def wrap_model_fsdp(model, strategy: str = "full_shard", *, device=None, param_init_fn=None):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import BackwardPrefetch
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.api import ShardingStrategy

    if not torch.distributed.is_initialized():
        raise RuntimeError("initialize torch.distributed before wrapping the model with FSDP")
    if param_init_fn is None:
        target_device = device or torch.cuda.current_device()

        def param_init_fn(module):
            # Moshi modules such as StreamingMultiheadAttention do not define
            # reset_parameters(); materialize their meta tensors before FSDP sync.
            materialize_meta_module(module, target_device)
    strategies = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    try:
        sharding_strategy = strategies[strategy.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported FSDP sharding strategy: {strategy}") from exc

    return FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=get_fsdp_policy(is_lora=True),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=device or torch.cuda.current_device(),
        sync_module_states=True,
        param_init_fn=param_init_fn,
        use_orig_params=True,
        limit_all_gathers=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )


def fsdp_adapter_state_dict(model) -> dict[str, object]:
    """Collect just LoRA weights while all ranks participate in full-param gather."""
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    def collect():
        return {
            name.removeprefix("_fsdp_wrapped_module."): parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if "lora_" in name
        }

    if isinstance(model, FSDP):
        with FSDP.summon_full_params(model, recurse=True, writeback=False, rank0_only=True):
            if torch.distributed.get_rank() == 0:
                return collect()
        return {}
    return collect()
