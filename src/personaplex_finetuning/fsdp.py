"""FSDP (Fully Sharded Data Parallel) utilities for PersonaPlex LoRA training."""

from __future__ import annotations

import functools
import os
import socket
from typing import Callable, Iterable

import torch
import torch.distributed as dist
import torch.distributed.fsdp.wrap as torch_wrap
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def parse_gpu_ids(fsdp_str: str) -> list[int]:
    """Parse comma-separated GPU IDs string e.g. '0,1' -> [0, 1]."""
    parts = [p.strip() for p in fsdp_str.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid --fsdp argument: '{fsdp_str}'. Expected format like '0,1'.")
    gpu_ids = [int(p) for p in parts]
    if len(gpu_ids) < 2:
        raise ValueError(f"FSDP requires at least 2 GPUs, got: {gpu_ids}")
    return gpu_ids


def get_fsdp_policy(is_lora: bool = True) -> Callable[[torch.nn.Module], bool]:
    """Create FSDP wrapping policy adhering to Moshi fine-tuning architecture.
    
    Each StreamingTransformerLayer becomes an FSDP unit.
    When LoRA is enabled, trainable LoRA parameters and frozen parameters are
    segregated into separate FSDP groups as required by PyTorch mixed requires_grad FSDP.
    """
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        transformer_layer_cls = (StreamingTransformerLayer,)
    except ImportError:
        transformer_layer_cls = ()

    transformer_block_wrap_policy = functools.partial(
        torch_wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )

    if not is_lora:
        return transformer_block_wrap_policy

    def fsdp_lora_policy_fn(module: torch.nn.Module) -> bool:
        # Check if all parameters in this module are trainable
        params = list(module.parameters())
        return len(params) > 0 and all(p.requires_grad for p in params)

    fsdp_lora_policy = functools.partial(
        torch_wrap.lambda_auto_wrap_policy, lambda_fn=fsdp_lora_policy_fn
    )

    policies = [fsdp_lora_policy, transformer_block_wrap_policy]
    return functools.partial(torch_wrap._or_policy, policies=policies)


def wrap_model_fsdp(
    model: torch.nn.Module,
    device_id: int | torch.device | None = None,
    strategy: str | ShardingStrategy = "shard_grad_op",
    mixed_precision: bool = True,
) -> FullyShardedDataParallel:
    """Wrap model with FSDP.

    Defaults to SHARD_GRAD_OP (ZeRO-2) with bf16 MixedPrecision for ~2× throughput.
    All-reduce communication is kept in fp32 to avoid gradient underflow.
    Set ``mixed_precision=False`` to disable (e.g. on GPUs that don't support bf16).
    """
    if isinstance(strategy, str):
        strat_map = {
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "full_shard": ShardingStrategy.FULL_SHARD,
            "no_shard": ShardingStrategy.NO_SHARD,
            "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        }
        sharding_strategy = strat_map.get(strategy.lower(), ShardingStrategy.SHARD_GRAD_OP)
    else:
        sharding_strategy = strategy

    auto_wrap_policy = get_fsdp_policy(is_lora=True)

    # BF16 MixedPrecision: params/buffers in bf16, gradients reduced in fp32.
    # This matches the Kyutai official fine-tuning approach and gives ~2× throughput
    # on Ampere+ GPUs (A100, H100, RTX 3090+) at no accuracy cost for LoRA.
    mp_policy: MixedPrecision | None = None
    if mixed_precision and torch.cuda.is_bf16_supported():
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,   # fp32 allreduce prevents gradient underflow
            buffer_dtype=torch.bfloat16,
        )

    kwargs = {
        "sharding_strategy": sharding_strategy,
        "auto_wrap_policy": auto_wrap_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "limit_all_gathers": True,
        "sync_module_states": True,
        "use_orig_params": True,
    }
    if device_id is not None:
        kwargs["device_id"] = device_id
    if mp_policy is not None:
        kwargs["mixed_precision"] = mp_policy
    return FullyShardedDataParallel(model, **kwargs)


# ---------------------------------------------------------------------------
# Mixed-precision optimizer helpers (adapted from kyutai-labs/moshi-finetune)
# ---------------------------------------------------------------------------
# These allow LoRA adapter params to be stored in bf16 during forward/backward
# but updated by AdamW in fp32 master copies – preventing precision loss in
# the optimizer state while keeping memory and compute cost low.

def mp_prepare(
    params: Iterable[torch.nn.Parameter],
    param_dtype: torch.dtype = torch.bfloat16,
    optim_dtype: torch.dtype = torch.float32,
) -> None:
    """Attach fp32 master copies to every trainable parameter."""
    with torch.no_grad():
        for p in params:
            if p.requires_grad:
                p._mp_param = torch.empty_like(p, dtype=optim_dtype)  # type: ignore[attr-defined]
                p._mp_param.copy_(p.to(optim_dtype))  # type: ignore[attr-defined]
            p.data = p.data.to(param_dtype)


def mp_upcast(
    params: Iterable[torch.nn.Parameter],
    optim_dtype: torch.dtype = torch.float32,
) -> None:
    """Swap parameter data to fp32 master copy before optimizer.step()."""
    with torch.no_grad():
        for p in params:
            if p.requires_grad and p.grad is not None:
                p._temp = p.data  # type: ignore[attr-defined]
                p.data = p._mp_param  # type: ignore[attr-defined]
                p.grad = p.grad.to(optim_dtype)


def mp_downcast(
    params: Iterable[torch.nn.Parameter],
    param_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Copy updated fp32 weights back to bf16 storage after optimizer.step()."""
    with torch.no_grad():
        for p in params:
            if p.requires_grad and p.grad is not None:
                p._temp.copy_(p.data)  # type: ignore[attr-defined]
                p.data = p._temp  # type: ignore[attr-defined]
                p.grad = p.grad.to(param_dtype)



def fsdp_adapter_state_dict(model: torch.nn.Module | FullyShardedDataParallel) -> dict[str, torch.Tensor]:
    """Retrieve full un-sharded LoRA parameters for checkpointing."""
    if not isinstance(model, FullyShardedDataParallel):
        from .lora import adapter_state_dict
        return adapter_state_dict(model)

    state_dict: dict[str, torch.Tensor] = {}
    is_distributed = dist.is_initialized() and dist.get_world_size() > 1

    # Traverse modules with trainable weights
    def is_trainable_fsdp(module: torch.nn.Module | FullyShardedDataParallel) -> bool:
        params = list(module.parameters())
        return len(params) > 0 and all(p.requires_grad for p in params)

    modules = {k: m for k, m in model.named_modules() if is_trainable_fsdp(m)}
    for key, module in modules.items():
        parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
            "_checkpoint_wrapped_module.", ""
        )
        if is_distributed and isinstance(module, FullyShardedDataParallel):
            with module.summon_full_params(module, writeback=False, offload_to_cpu=True):
                for k, v in module.state_dict().items():
                    name = f"{parent_prefix}.{k}" if parent_prefix else k
                    if ".lora_a." in name or ".lora_b." in name:
                        state_dict[name] = v.detach().cpu().clone()
        else:
            for k, v in module.state_dict().items():
                name = f"{parent_prefix}.{k}" if parent_prefix else k
                if ".lora_a." in name or ".lora_b." in name:
                    state_dict[name] = v.detach().cpu().clone()

    if not state_dict:
        # Fallback summon over the root FSDP module
        if is_distributed:
            with model.summon_full_params(model, writeback=False, offload_to_cpu=True):
                for k, v in model.state_dict().items():
                    clean_k = k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
                    if ".lora_a." in clean_k or ".lora_b." in clean_k:
                        state_dict[clean_k] = v.detach().cpu().clone()
        else:
            for k, v in model.state_dict().items():
                clean_k = k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
                if ".lora_a." in clean_k or ".lora_b." in clean_k:
                    state_dict[clean_k] = v.detach().cpu().clone()

    return state_dict
