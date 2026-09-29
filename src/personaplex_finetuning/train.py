"""PersonaPlex LoRA training path supporting Accelerate, DDP, BF16, and single-GPU execution."""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm

from .config import Config, load_config
from .data import PreparedDataset, conversation_group_keys, duration_chunks, limit_conversations
from .batching import (
    RankStrideBatchSampler, RawAudioDataset, collate_raw_audio, post_encode_collate,
)
from .lora import adapter_state_dict, inject_lora, load_adapter
from .generation import GenerationSettings
from .inference import generate_text_with_runtime, text_error_metrics
from .objective import stream_weights_torch, torch_weighted_cross_entropy, torch_weighted_cross_entropy_stats
from .runtime import RuntimePaths, load_runtime
from .sequence import PersonaPlexTrainingExampleBuilder, pad_training_example


def limit_cpu_threads(torch_module) -> int:
    """Keep training processes from consuming every CPU core."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    torch_module.set_num_threads(1)
    torch_module.set_num_interop_threads(1)
    return 1


def seed_everything(seed: int, torch_module) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def create_run_dir(output_root: Path, smoke: bool) -> Path:
    """Create a unique, timestamped run directory below the configured root."""
    label = "smoke" if smoke else "train"
    run_dir = output_root / f"{label}_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_example(config: Config, sample, runtime, random_crop: bool = False, rng=None, dialogue_codes=None):
    pause_frames = getattr(getattr(config, "generation_settings", None), "audio_silence_frame_cnt", 6)
    builder = PersonaPlexTrainingExampleBuilder(
        runtime.codec, runtime.tokenizer, runtime.initial_tokens, runtime.zero_token,
        pause_frames=pause_frames,
    )
    # PersonaPlex LMModel.forward_train applies its native per-stream delays.
    return builder.build(sample, dialogue_codes=dialogue_codes)


def effective_global_batch_size(per_process_batch_size: int, num_processes: int, gradient_accumulation_steps: int) -> int:
    """Return the number of samples contributing to one optimizer update."""
    if min(per_process_batch_size, num_processes, gradient_accumulation_steps) < 1:
        raise ValueError("batch size, process count, and accumulation steps must all be positive")
    return per_process_batch_size * num_processes * gradient_accumulation_steps


def rank_stride_indices(sample_count: int, rank: int, world_size: int) -> list[int]:
    if sample_count < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid sample count or distributed rank")
    return list(range(rank, sample_count, world_size))


def lora_prefixes_for_stage(config: Config) -> tuple[str, ...]:
    """Select the same LoRA target modules for training and adapter reload."""
    stage = getattr(config, "train_stage", "joint").lower()
    if stage in {"temporal_only", "freeze_depformer"}:
        return ("transformer",)
    if stage in {"depth_only", "freeze_tempformer"}:
        return ("depformer",)
    return ("transformer", "depformer")


def unwrap_parallel_model(model):
    """Return the underlying model so checkpoints use inference-compatible keys."""
    while hasattr(model, "module"):
        model = model.module
    return model


def inspect_training_sample(sample, tokenizer) -> dict[str, object]:
    """Validate and describe the actual agent-text target before optimization."""
    if sample.agent_channel != 0 or sample.user_channel != 1:
        raise ValueError(f"{sample.sample_id}: expected LEFT=agent and RIGHT=user")
    words = [
        word.word for word in sample.words
        if word.speaker == "agent" and sample.window_start_sec <= word.start < sample.window_end_sec
    ]
    if not words:
        raise ValueError(f"{sample.sample_id}: training window has no agent text target")
    text = " ".join(words)
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    normalize = lambda value: " ".join(value.split())
    if normalize(decoded) != normalize(text):
        raise ValueError(
            f"{sample.sample_id}: tokenizer round-trip changed agent text; "
            f"source={text!r}, decoded={decoded!r}"
        )
    return {
        "sample_id": sample.sample_id,
        "window_start_sec": sample.window_start_sec,
        "window_end_sec": sample.window_end_sec,
        "agent_channel": sample.agent_channel,
        "user_channel": sample.user_channel,
        "agent_word_count": len(words),
        "token_count": len(tokens),
        "text_prompt": sample.text_prompt,
        "agent_text": text,
        "tokenizer_round_trip": decoded,
    }


def text_supervision_counts(batch, model_output, padding_id: int):
    """Count valid non-padding text targets and weighted padding positions."""
    labels = batch["labels"][:, 0, :]
    valid = batch["loss_mask"][:, 0, :] & model_output.text_mask[:, 0]
    return ((labels != padding_id) & valid).sum(), ((labels == padding_id) & valid).sum()


def text_target_token_loss_stats(batch, model_output, padding_id: int):
    """Return diagnostic CE over valid non-padding text tokens only."""
    labels = batch["labels"][:, 0, :]
    valid_targets = (
        batch["loss_mask"][:, 0, :]
        & model_output.text_mask[:, 0]
        & (labels != padding_id)
    )
    with torch.no_grad():
        selected_logits = model_output.text_logits[:, 0][valid_targets].float()
        selected_targets = labels[valid_targets]
        loss_sum = torch.nn.functional.cross_entropy(
            selected_logits, selected_targets, reduction="sum"
        )
    return loss_sum, valid_targets.sum()


def step_optimizer_if_ready(sync_state, optimizer, scheduler, trainable, model=None, max_norm=1.0) -> float:
    if not sync_state.sync_gradients:
        return 0.0

    if model is not None and hasattr(model, "clip_grad_norm_"):
        # Distributed wrappers may provide a model-aware global norm operation.
        grad_norm = model.clip_grad_norm_(max_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm)

    grad_norm = float(grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm

def deterministic_crop_rng(seed: int, process_index: int, micro_step: int) -> random.Random:
    """Make dynamic crops reproducible across resume without saving Python RNG state."""
    return random.Random(seed + (process_index * 1_000_003) + micro_step)


def _initialize_audio_worker(_worker_id: int) -> None:
    """Keep each spawned audio decoder worker from oversubscribing CPU threads."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)


def iter_training_batches(config, conversations, runtime, device, rank: int, world_size: int, smoke: bool, skip_batches: int = 0):
    """Moshi-style iterator: fixed conversation chunks, rank stride, then small batches."""
    epoch = 0
    samples = duration_chunks(conversations, config.duration_sec)
    use_prefetch = hasattr(runtime.codec, "encode_conversation_stereo_batch")
    loader = sampler = None
    if use_prefetch:
        sampler = RankStrideBatchSampler(
            len(samples), config.per_device_batch_size, rank, world_size,
            seed=config.seed, shuffle=config.shuffle and not smoke,
        )
        dataset = RawAudioDataset(samples, runtime.codec.sample_rate)
        worker_count = 0 if smoke else config.num_workers
        loader_options = {
            "dataset": dataset,
            "batch_sampler": sampler,
            "collate_fn": collate_raw_audio,
            "num_workers": worker_count,
            "pin_memory": config.pin_memory and str(device).startswith("cuda"),
            "persistent_workers": config.persistent_workers and worker_count > 0,
            "worker_init_fn": _initialize_audio_worker if worker_count > 0 else None,
        }
        if worker_count > 0:
            loader_options["prefetch_factor"] = config.prefetch_factor
            # The model has already initialized CUDA in the parent. Spawn keeps
            # decoder workers isolated from that CUDA runtime state.
            loader_options["multiprocessing_context"] = "spawn"
        loader = torch.utils.data.DataLoader(**loader_options)

    while True:
        if use_prefetch:
            sampler.set_epoch(epoch)
            batches = enumerate(loader)
        else:
            ordered_samples = list(samples)
            if config.shuffle and not smoke:
                random.Random(config.seed + epoch).shuffle(ordered_samples)
            rank_samples = [ordered_samples[index] for index in rank_stride_indices(len(ordered_samples), rank, world_size)]
            common_batch_count = min(
                len(rank_stride_indices(len(ordered_samples), worker, world_size)) // config.per_device_batch_size
                for worker in range(world_size)
            )
            batches = enumerate(
                rank_samples[index * config.per_device_batch_size:(index + 1) * config.per_device_batch_size]
                for index in range(common_batch_count)
            )

        common_batch_count = len(loader) if use_prefetch else common_batch_count
        if common_batch_count == 0:
            raise ValueError(
                f"only {len(samples)} duration chunks for world_size={world_size} and "
                f"batch_size_per_gpu={config.per_device_batch_size}; use sample_number=null/full data "
                "or run a one-GPU overfit config with batch_size=1"
            )
        for batch_index, batch_value in batches:
            if skip_batches:
                skip_batches -= 1
                continue
            if use_prefetch:
                raw_audio = batch_value
                batch_samples = raw_audio["samples"]
            else:
                raw_audio = None
                batch_samples = batch_value
            prepared_batch = []
            for sample_index, sample in enumerate(batch_samples):
                if config.prompt_aug_prob and not smoke:
                    sample = sample.with_window(
                        sample.window_start_sec, sample.window_end_sec,
                        sample.get_augmented_prompt(
                            config.prompt_aug_prob,
                            rng=random.Random(
                                config.seed + epoch * 1_000_003
                                + batch_index * config.per_device_batch_size + sample_index
                            ),
                        ),
                    )
                prepared_batch.append(sample)
            if hasattr(runtime.codec, "encode_conversation_stereo_batch"):
                dialogue_codes = runtime.codec.encode_conversation_stereo_batch([
                    (
                        sample.conversation_wav, sample.agent_channel, sample.user_channel,
                        sample.window_start_sec, sample.window_end_sec,
                    )
                    for sample in prepared_batch
                ], raw_audio=raw_audio)
                examples = [
                    build_example(config, sample, runtime, dialogue_codes=codes)
                    for sample, codes in zip(prepared_batch, dialogue_codes, strict=True)
                ]
            else:
                examples = [build_example(config, sample, runtime) for sample in prepared_batch]
            fixed_frames = round(config.duration_sec * runtime.codec.frame_rate) + max(
                (example.prompt_frames for example in examples), default=0
            )
            examples = [
                pad_training_example(example, fixed_frames, runtime.tokenizer.padding_id, runtime.zero_token)
                for example in examples
            ]
            if any(
                len(example.input_codes) != 17
                or any(len(stream) != fixed_frames for stream in example.input_codes)
                or any(len(stream) != fixed_frames for stream in example.loss_mask)
                for example in examples
            ):
                raise AssertionError("batch padding produced inconsistent PersonaPlex stream shapes")
            batch = post_encode_collate(
                examples, runtime.tokenizer.padding_id, runtime.zero_token, device
            )
            audio_seconds = sum(
                max(0.0, min(item.window_end_sec, item.audio.duration_sec) - item.window_start_sec)
                for item in batch_samples
            )
            audio_frames = sum(item.dialogue_frames for item in examples)
            yield batch, epoch, len(examples), audio_seconds, audio_frames, batch_samples
        epoch += 1


def sample_index_for_rank(micro_step: int, process_index: int, num_processes: int, sample_count: int) -> int:
    """Select a disjoint rank-local sample before wrapping around the dataset."""
    if micro_step < 0 or process_index < 0 or process_index >= num_processes or sample_count < 1:
        raise ValueError("invalid distributed sample selection inputs")
    return (micro_step * num_processes + process_index) % sample_count


def write_rank_info(run_dir: Path, process_index: int, num_processes: int, device, sample_count: int, peak_gpu_bytes: int | None = None) -> Path:
    """Emit one non-contended runtime record per DDP rank for GPU verification."""
    visible_devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    physical_gpu = visible_devices[process_index] if process_index < len(visible_devices) else None
    info = {
        "rank": process_index,
        "world_size": num_processes,
        "device": str(device),
        "physical_gpu": physical_gpu,
        "first_sample_index": sample_index_for_rank(0, process_index, num_processes, sample_count),
    }
    if peak_gpu_bytes is not None:
        info["peak_gpu_bytes"] = peak_gpu_bytes
    path = run_dir / "ranks" / f"rank_{process_index:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return path


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    import math
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Enable activation checkpointing on StreamingTransformer layers without breaking streaming inference."""
    import torch.utils.checkpoint
    for module in model.modules():
        if module.__class__.__name__ == "StreamingTransformer":
            orig_forward = module.forward
            def make_checkpointed_forward(mod, orig_fwd):
                def checkpointed_forward(x: torch.Tensor, *args, **kwargs):
                    if not mod.training:
                        return orig_fwd(x, *args, **kwargs)
                    B, T, C = x.shape
                    state = mod._streaming_state
                    if state is None:
                        offset = torch.zeros(1, dtype=torch.long, device=x.device)
                    else:
                        offset = state.offset
                    if mod.positional_embedding in {"sin", "sin_rope"}:
                        from moshi.modules.transformer import create_sin_embedding
                        positions = torch.arange(T, device=x.device).view(1, -1, 1)
                        positions = positions + offset.view(-1, 1, 1)
                        pos_emb = create_sin_embedding(
                            positions, C, max_period=mod.max_period, dtype=x.dtype
                        )
                        x = x + mod.positional_scale * pos_emb
                    for layer in mod.layers:
                        x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
                    if state is not None:
                        state.offset.add_(T)
                    return x
                return checkpointed_forward
            module.forward = make_checkpointed_forward(module, orig_forward)


def model_forward_train(model, codes: torch.Tensor):
    """Run Moshi's training forward with the runtime's CUDA BF16 precision."""
    from torch.nn.parallel import DistributedDataParallel

    # Some Moshi activations are produced in FP32 while the PersonaPlex model
    # weights are BF16. Autocast keeps all projections dtype-compatible.
    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if getattr(getattr(codes, "device", None), "type", None) == "cuda"
        else nullcontext()
    )
    with precision_context:
        if isinstance(model, DistributedDataParallel):
            # Calling forward_train directly bypasses distributed wrapper hooks.
            output = model(codes)
        elif hasattr(model, "forward_train"):
            output = model.forward_train(codes)
        else:
            output = model(codes)
    if hasattr(output, "logits") and hasattr(output, "text_logits"):
        if output.logits.shape[0] != codes.shape[0] or output.logits.shape[2] != codes.shape[2]:
            raise AssertionError(f"audio logits must align to [B,K,T] codes; got {output.logits.shape} vs {codes.shape}")
        if output.text_logits.shape[0] != codes.shape[0] or output.text_logits.shape[2] != codes.shape[2]:
            raise AssertionError(f"text logits must align to [B,1,T] codes; got {output.text_logits.shape} vs {codes.shape}")
    return output


def write_tensorboard_scalars(writer, record: dict[str, float | int], trainable_parameters: int, cpu_threads: int) -> None:
    step = int(record["step"])
    for name, value in (
        ("loss/total", record["loss/total"]),
        ("loss/text", record["loss/text"]),
        ("loss/text_nonpadding", record["loss/text_nonpadding"]),
        ("loss/audio_semantic", record["loss/audio_semantic"]),
        ("loss/audio_nonsemantic", record["loss/audio_nonsemantic"]),
        ("train/learning_rate", record["lr"]),
        ("train/gradient_norm", record["grad_norm"]),
        ("system/gpu_peak_bytes", record["gpu_peak_bytes"]),
        ("system/trainable_parameters", trainable_parameters),
        ("system/cpu_threads", cpu_threads),
        *((name, value) for name, value in record.items() if name.startswith("timing/")),
    ):
        writer.add_scalar(name, value, step)


def loss_components(model_output, codes, example, text_padding_id, torch_module, first_codebook_weight_multiplier=1.0, text_padding_weight=0.3, *, distributed=False):
    """Compute per-stream losses using GPU-native vectorized weights."""
    if isinstance(example, dict):
        labels_tensor = example["labels"]
        mask_tensor = example["loss_mask"]
    else:
        labels_tensor = torch_module.tensor(example.labels, dtype=torch_module.long, device=codes.device).unsqueeze(0)
        mask_tensor = torch_module.tensor(example.loss_mask, dtype=torch_module.bool, device=codes.device).unsqueeze(0)
    weights = stream_weights_torch(
        labels_tensor, mask_tensor, text_padding_id,
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
        text_padding_weight=text_padding_weight,
    )

    text_target = labels_tensor[:, 0, :]
    text_weight = weights[:, 0, :] * model_output.text_mask[:, 0].to(weights.dtype)
    cross_entropy = torch_weighted_cross_entropy_stats if distributed else torch_weighted_cross_entropy
    text_loss = cross_entropy(
        model_output.text_logits.reshape(-1, model_output.text_logits.shape[-1]),
        text_target.reshape(-1),
        text_weight.reshape(-1),
    )

    audio_target = labels_tensor[:, 1:17, :]
    audio_weights = weights[:, 1:17, :] * model_output.mask.to(weights.dtype)

    semantic = cross_entropy(
        model_output.logits[:, 0].reshape(-1, model_output.logits.shape[-1]),
        audio_target[:, 0].reshape(-1),
        audio_weights[:, 0].reshape(-1),
    )
    nonsemantic = cross_entropy(
        model_output.logits[:, 1:8].reshape(-1, model_output.logits.shape[-1]),
        audio_target[:, 1:8].reshape(-1),
        audio_weights[:, 1:8].reshape(-1),
    )
    components = {"text": text_loss, "audio_semantic": semantic, "audio_nonsemantic": nonsemantic}
    if distributed:
        return components["text"] + components["audio_semantic"] + components["audio_nonsemantic"], components
    return sum(components.values()), components


def reduce_distributed_loss(stats, all_reduce_sum, *, preserve_grad: bool = False, world_size: int = 1):
    """Reduce weighted CE sums/weights in one collective and return global means."""
    names = tuple(stats)
    packed = torch.stack([
        value
        for name in names
        for value in stats[name]
    ]).detach()
    global_values = all_reduce_sum(packed)
    components = {}
    total = None
    for index, name in enumerate(names):
        numerator, denominator = stats[name]
        global_numerator = global_values[index * 2]
        global_denominator = global_values[index * 2 + 1].clamp_min(1e-12)
        global_mean = global_numerator / global_denominator
        if preserve_grad:
            # DDP averages gradients across ranks; scale each local numerator
            # so the averaged gradient equals the global weighted-mean gradient.
            local_gradient = numerator * world_size / global_denominator
            loss = local_gradient + (global_mean - local_gradient.detach())
        else:
            loss = global_mean
        components[name] = loss
        total = loss if total is None else total + loss
    return components, total


def one_step(config: Config, runtime, example, optimizer=None):
    codes = torch.tensor(example.input_codes, dtype=torch.long, device=config.device).unsqueeze(0)
    output = model_forward_train(runtime.model, codes)
    total, components = loss_components(output, codes, example, runtime.tokenizer.padding_id, torch)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        trainable = [p for p in runtime.model.parameters() if p.requires_grad]
        if not any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in trainable):
            raise RuntimeError("LoRA gradients are all zero")
        if any(p.grad is not None for p in runtime.model.parameters() if not p.requires_grad):
            raise RuntimeError("frozen base parameter received a gradient")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    else:
        grad_norm = torch.tensor(0.0)
    return total, components, float(grad_norm)


def save_training_state(
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    optimizer_step: int,
    gradient_accumulation_steps: int,
    num_processes: int,
    per_device_batch_size: int = 1,
) -> Path:
    """Persist enough state to continue the optimizer trajectory exactly."""
    path = checkpoint_dir / "training_state.pt"
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    if isinstance(scheduler_state, dict) and callable(scheduler_state.get("anneal_func")):
        # OneCycleLR stores a Python function in its state. Persist its known
        # function name so the checkpoint remains loadable with weights_only=True.
        scheduler_state["anneal_func"] = scheduler_state["anneal_func"].__name__
    torch.save(
        {
            "format_version": 1,
            "optimizer_step": optimizer_step,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_processes": num_processes,
            "per_device_batch_size": per_device_batch_size,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler_state,
        },
        path,
    )
    return path


def load_training_state(
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    gradient_accumulation_steps: int,
    num_processes: int,
    per_device_batch_size: int = 1,
    total_steps: int | None = None,
) -> int | None:
    """Restore an optimizer checkpoint, rejecting incompatible DDP topology."""
    path = checkpoint_dir / "training_state.pt"
    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("gradient_accumulation_steps") != gradient_accumulation_steps:
        raise RuntimeError("resume checkpoint gradient_accumulation_steps differs from this run")
    if state.get("num_processes") != num_processes:
        raise RuntimeError("resume checkpoint num_processes differs from this run")
    if state.get("per_device_batch_size", 1) != per_device_batch_size:
        raise RuntimeError("resume checkpoint per_device_batch_size differs from this run")
    saved_scheduler = state.get("scheduler")
    saved_total_steps = saved_scheduler.get("total_steps") if isinstance(saved_scheduler, dict) else None
    if total_steps is not None and saved_total_steps != total_steps:
        raise RuntimeError(
            f"resume checkpoint max_steps ({saved_total_steps}) differs from this run ({total_steps}); "
            "start a fresh run to use a different OneCycleLR schedule"
        )
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is None and saved_scheduler is not None:
        raise RuntimeError("resume checkpoint has scheduler state but this run has no scheduler")
    if scheduler is not None and saved_scheduler is None:
        raise RuntimeError("resume checkpoint has no scheduler state but this run requires one")
    if scheduler is not None and saved_scheduler is not None:
        anneal_name = saved_scheduler.get("anneal_func")
        if isinstance(anneal_name, str):
            supported_annealers = {
                "_annealing_cos": scheduler._annealing_cos,
                "_annealing_linear": scheduler._annealing_linear,
            }
            if anneal_name not in supported_annealers:
                raise RuntimeError(f"unsupported scheduler anneal function in checkpoint: {anneal_name}")
            saved_scheduler["anneal_func"] = supported_annealers[anneal_name]
        scheduler.load_state_dict(saved_scheduler)
    return int(state["optimizer_step"])


def save_adapter(run_dir: Path, model, config: Config, step: int, optimizer=None, scheduler=None, gradient_accumulation_steps: int = 1, num_processes: int = 1) -> Path:
    return save_adapter_state(
        run_dir, adapter_state_dict(model), config, step, optimizer, scheduler,
        gradient_accumulation_steps, num_processes,
    )


def save_adapter_state(run_dir: Path, state_dict, config: Config, step: int, optimizer=None, scheduler=None, gradient_accumulation_steps: int = 1, num_processes: int = 1) -> Path:
    path = run_dir / "checkpoints" / f"checkpoint_{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    adapter = path / "lora.safetensors"
    save_file(state_dict, str(adapter))
    (path / "adapter.json").write_text(
        json.dumps(
            {"step": step, "model_root": str(config.model_root), "rank": config.lora_rank, "alpha": config.lora_alpha,
             "scaling": config.lora_scaling,
             "gradient_accumulation_steps": gradient_accumulation_steps, "num_processes": num_processes,
             "per_device_batch_size": config.per_device_batch_size},
            indent=2,
        )
    )
    if optimizer is not None:
        save_training_state(path, optimizer, scheduler, step, gradient_accumulation_steps, num_processes, config.per_device_batch_size)
    return adapter


def save_best_adapter(
    run_dir: Path, model, config: Config, step: int, val_loss: float,
    optimizer=None, scheduler=None, gradient_accumulation_steps: int = 1, num_processes: int = 1,
) -> Path:
    return save_best_adapter_state(
        run_dir, adapter_state_dict(model), config, step, val_loss, optimizer, scheduler,
        gradient_accumulation_steps, num_processes,
    )


def save_best_adapter_state(
    run_dir: Path, state_dict, config: Config, step: int, val_loss: float,
    optimizer=None, scheduler=None, gradient_accumulation_steps: int = 1, num_processes: int = 1,
    checkpoint_name: str = "best",
) -> Path:
    path = run_dir / "checkpoints" / checkpoint_name
    path.mkdir(parents=True, exist_ok=True)
    adapter = path / "lora.safetensors"
    save_file(state_dict, str(adapter))
    (path / "adapter.json").write_text(
        json.dumps(
            {"step": step, "val_loss": val_loss, "model_root": str(config.model_root), "rank": config.lora_rank, "alpha": config.lora_alpha,
             "scaling": config.lora_scaling,
             "gradient_accumulation_steps": gradient_accumulation_steps, "num_processes": num_processes,
             "per_device_batch_size": config.per_device_batch_size},
            indent=2,
        )
    )
    if optimizer is not None:
        save_training_state(path, optimizer, scheduler, step, gradient_accumulation_steps, num_processes, config.per_device_batch_size)
    return adapter


def evaluate_validation(config: Config, runtime, val_samples: list, rank: int, world_size: int, device) -> dict[str, float]:
    if not val_samples:
        return {}
    unwrapped = runtime.model
    unwrapped.eval()
    totals = {"total": 0.0, "text": 0.0, "semantic": 0.0, "nonsemantic": 0.0}
    nonpadding_text_loss_sum = 0.0
    nonpadding_text_token_count = 0
    count = 0
    with torch.no_grad():
        selected_samples = evenly_spaced_validation_samples(val_samples, config.validation_max_samples)
        local_samples = selected_samples[rank::world_size]
        for sample in local_samples:
            example = build_example(config, sample, runtime, random_crop=False)
            codes = torch.tensor(example.input_codes, dtype=torch.long, device=device).unsqueeze(0)
            output = model_forward_train(unwrapped, codes)
            total, comps = loss_components(
                output, codes, example, runtime.tokenizer.padding_id, torch,
                config.first_codebook_weight_multiplier, config.text_padding_weight,
            )
            totals["total"] += float(total.detach())
            totals["text"] += float(comps["text"].detach())
            totals["semantic"] += float(comps["audio_semantic"].detach())
            totals["nonsemantic"] += float(comps["audio_nonsemantic"].detach())
            labels = torch.tensor(example.labels, dtype=torch.long, device=device).unsqueeze(0)
            loss_mask = torch.tensor(example.loss_mask, dtype=torch.bool, device=device).unsqueeze(0)
            text_loss_sum, text_token_count = text_target_token_loss_stats(
                {"labels": labels, "loss_mask": loss_mask}, output, runtime.tokenizer.padding_id,
            )
            nonpadding_text_loss_sum += float(text_loss_sum)
            nonpadding_text_token_count += int(text_token_count)
            count += 1
    unwrapped.train()
    reduced = torch.tensor([
        totals["total"], totals["text"], totals["semantic"], totals["nonsemantic"], count,
        nonpadding_text_loss_sum, nonpadding_text_token_count,
    ], device=device)
    if world_size > 1:
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
    total_count = float(reduced[4].item())
    if total_count == 0:
        raise RuntimeError("validation has no samples after rank partitioning")
    text_token_count = float(reduced[6].item())
    text_nonpadding_loss = float(reduced[5].item()) / text_token_count if text_token_count else float("inf")
    audio_semantic_loss = float(reduced[2].item()) / total_count
    audio_nonsemantic_loss = float(reduced[3].item()) / total_count
    return {
        "val/loss_total": float(reduced[0].item()) / total_count,
        "val/loss_text": float(reduced[1].item()) / total_count,
        "val/loss_text_nonpadding": text_nonpadding_loss,
        "val/text_nonpadding_tokens": text_token_count,
        "val/loss_audio_semantic": audio_semantic_loss,
        "val/loss_audio_nonsemantic": audio_nonsemantic_loss,
        "val/loss_selection": text_nonpadding_loss + audio_semantic_loss + audio_nonsemantic_loss,
    }


def validation_selection_loss(metrics: dict[str, float]) -> float:
    """Choose checkpoints using valid text targets plus the two agent-audio losses."""
    keys = (
        "val/loss_text_nonpadding",
        "val/loss_audio_semantic",
        "val/loss_audio_nonsemantic",
    )
    if any(key not in metrics for key in keys):
        return float("inf")
    return sum(metrics[key] for key in keys)


def generation_checkpoint_score(metrics: dict, baseline_cer: float | None = None) -> float:
    """Return CER only for complete, non-empty generation that beats the starting adapter."""
    if metrics.get("val/generation_empty_samples", 0):
        return float("inf")
    cer = float(metrics.get("val/generation_cer", float("inf")))
    if baseline_cer is not None and cer >= baseline_cer:
        return float("inf")
    return cer


def inference_config_snapshot(config: Config, adapter_path: Path, output_dir: Path) -> dict:
    """Create a standalone inference config matching the effective training inputs."""
    data = {
        "prepared_dir": str(config.prepared_dir),
        "window_seconds": config.window_seconds,
        "val_ratio": config.val_ratio,
    }
    if config.val_manifest is not None:
        data["val_manifest"] = str(config.val_manifest)
    if config.test_manifest is not None:
        data["test_manifest"] = str(config.test_manifest)
    return {
        "model": {
            "root": str(config.model_root),
            "source": str(config.personaplex_source),
            "device": config.device,
        },
        "data": data,
        "lora": {"qlora": config.qlora, "quant_type": config.quant_type},
        "adapter": {"path": str(adapter_path)},
        "seed": config.seed,
        "generation": config.generation_settings.as_dict(),
        "inference": {"output_dir": str(output_dir)},
    }


def evenly_spaced_validation_samples(samples: list, limit: int) -> list:
    """Choose a deterministic spread over a conversation-ordered validation set."""
    if limit < 1:
        raise ValueError("validation sample limit must be positive")
    if len(samples) <= limit:
        return list(samples)
    if limit == 1:
        return [samples[len(samples) // 2]]
    indices = [round(index * (len(samples) - 1) / (limit - 1)) for index in range(limit)]
    return [samples[index] for index in indices]


def evaluate_free_running(runtime, samples: list, config: Config) -> dict:
    """Score autoregressive text on a small, fixed held-out subset."""
    evaluated = []
    unique_conversations = []
    seen_groups = set()
    for sample in samples:
        keys = conversation_group_keys(sample)
        if any(key in seen_groups for key in keys):
            continue
        seen_groups.update(keys)
        unique_conversations.append(sample)
    candidates = evenly_spaced_validation_samples(
        unique_conversations, max(config.free_running_eval_samples * 4, config.free_running_eval_samples),
    )
    for sample in candidates:
        agent_words = [
            word for word in sample.words
            if word.speaker == "agent" and sample.window_start_sec <= word.start < sample.window_end_sec
        ]
        if not agent_words:
            continue
        clipped_end = min(sample.window_end_sec, sample.audio.duration_sec)
        latest_start = max(sample.window_start_sec, clipped_end - config.free_running_eval_window_seconds)
        window_start = max(sample.window_start_sec, min(min(word.start for word in agent_words), latest_start))
        end_sec = min(
            sample.window_end_sec,
            window_start + config.free_running_eval_window_seconds,
            sample.audio.duration_sec,
        )
        if end_sec <= window_start:
            continue
        window = sample.with_window(window_start, end_sec)
        reference = " ".join(
            word.word for word in window.words
            if word.speaker == "agent" and window.window_start_sec <= word.start < window.window_end_sec
        )
        if not reference.strip():
            continue
        hypothesis = generate_text_with_runtime(
            runtime, window,
            generation=getattr(config, "generation_settings", GenerationSettings()),
            seed=config.seed + len(evaluated),
        )
        metrics = text_error_metrics(reference, hypothesis)
        if metrics is not None:
            evaluated.append((metrics, sample.sample_id, reference, hypothesis, window.window_start_sec, window.window_end_sec))
        if len(evaluated) >= config.free_running_eval_samples:
            break
    if not evaluated:
        raise RuntimeError("free-running validation found no samples with agent text references")
    empty_hypotheses = sum(not hypothesis.strip() for _, _, _, hypothesis, _, _ in evaluated)
    return {
        "val/generation_cer": sum(float(item[0]["cer"]) for item in evaluated) / len(evaluated),
        "val/generation_wer": sum(float(item[0]["wer"]) for item in evaluated) / len(evaluated),
        "val/generation_samples": len(evaluated),
        "val/generation_empty_samples": empty_hypotheses,
        "samples": [
            {"sample_id": sample_id, "reference": reference, "hypothesis": hypothesis,
             "cer": metrics["cer"], "wer": metrics["wer"],
             "window_start_sec": start_sec, "window_end_sec": end_sec,
             "source_duration_sec": next(
                 sample.audio.duration_sec for sample in unique_conversations
                 if sample.sample_id == sample_id
             )}
            for metrics, sample_id, reference, hypothesis, start_sec, end_sec in evaluated
        ],
    }


def verify_reloaded_adapter(config: Config, sample, adapter: Path, runtime) -> float:
    """Round-trip saved LoRA weights in-place without allocating a second 7B base."""
    parallel_model = runtime.model
    model = unwrap_parallel_model(parallel_model)
    was_training = parallel_model.training
    runtime.model = model
    try:
        model.eval()
        example = build_example(config, sample, runtime)
        with torch.no_grad():
            reference_loss, _, _ = one_step(config, runtime, example)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".lora_a." in name or ".lora_b." in name:
                    parameter.add_(1.0)
        load_adapter(model, adapter)
        with torch.no_grad():
            reloaded_loss, _, _ = one_step(config, runtime, example)
        reference_loss = float(reference_loss)
        reloaded_loss = float(reloaded_loss)
        if abs(reloaded_loss - reference_loss) > 1e-5:
            raise RuntimeError(
                f"reloaded adapter loss drifted: {reloaded_loss} vs {reference_loss}"
            )
        return reloaded_loss
    finally:
        runtime.model = parallel_model
        parallel_model.train(was_training)


def run(
    config: Config,
    smoke: bool = False,
    resume_from: str | None = None,
) -> Path | None:
    # Moshi's lazy compile wrappers can trigger graph/compile shape issues on
    # the fixed, padded sequences used by distributed training.
    os.environ.setdefault("NO_TORCH_COMPILE", "1")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required; install the project requirements before training") from exc

    if smoke and config.shuffle:
        raise ValueError("overfit/smoke configuration must set data.shuffle: false")
    accum_steps = max(1, config.gradient_accumulation_steps)
    if "LOCAL_RANK" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun distributed mode requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank, world_size, local_rank = 0, 1, 0
    distributed = world_size > 1
    if distributed and not torch.cuda.is_available():
        raise RuntimeError("multi-process DDP training requires CUDA")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    global_batch_size = effective_global_batch_size(config.per_device_batch_size, world_size, accum_steps)
    cpu_threads = limit_cpu_threads(torch)
    seed_everything(config.seed, torch)
    main_process = rank == 0
    def barrier():
        if distributed:
            torch.distributed.barrier(device_ids=[local_rank])
    def all_reduce_sum(value):
        if distributed:
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        return value
    # Dataset loading
    test_samples = []
    if config.eval_on_train_samples:
        if config.no_eval or config.sample_number is None:
            raise ValueError("eval_on_train_samples requires evaluation enabled and a finite sample_number")
        if config.val_manifest_path is not None:
            raise ValueError("eval_on_train_samples cannot be combined with a separate validation manifest")
        train_samples = PreparedDataset(config.manifest, config.window_seconds).load()
        val_samples = []
    elif config.val_manifest:
        train_samples = PreparedDataset(config.manifest, config.window_seconds).load()
        val_samples = PreparedDataset(config.val_manifest, config.window_seconds).load()
    elif (config.eval_every_steps > 0 or config.free_running_eval_every_steps > 0) and not config.no_eval:
        train_samples, val_samples = PreparedDataset(config.manifest, config.window_seconds).split(
            val_ratio=config.val_ratio, seed=config.seed
        )
    else:
        train_samples = PreparedDataset(config.manifest, config.window_seconds).load()
        val_samples = []

    if config.test_manifest:
        test_samples = PreparedDataset(config.test_manifest, config.window_seconds).load()
    train_groups = {key for sample in train_samples for key in conversation_group_keys(sample)}
    val_groups = {key for sample in val_samples for key in conversation_group_keys(sample)}
    test_groups = {key for sample in test_samples for key in conversation_group_keys(sample)}
    if (
        (not config.eval_on_train_samples and train_groups & val_groups)
        or train_groups & test_groups
        or val_groups & test_groups
    ):
        raise ValueError("train/validation/test manifests contain overlapping conversation groups")

    train_conversations = limit_conversations(train_samples, config.sample_number)
    train_samples = duration_chunks(train_conversations, config.duration_sec)
    val_samples = duration_chunks(val_samples, config.duration_sec) if val_samples else []
    if config.eval_on_train_samples:
        # Explicitly scoped overfit gate: monitor autoregressive behavior on the
        # exact fixed training examples. Full runs continue using disjoint splits.
        val_samples = list(train_samples)
    test_samples = duration_chunks(test_samples, config.duration_sec) if test_samples else []

    # Run directory creation (coordinated across ranks)
    run_dir = None
    if main_process:
        run_dir = create_run_dir(config.output_dir, smoke)
        config_record = {
            "event": "configuration",
            "seed": config.seed,
            "model_root": str(config.model_root),
            "personaplex_source": str(config.personaplex_source),
            "manifest": str(config.manifest),
            "codec_cache_dir": str(config.codec_cache_dir) if config.codec_cache_dir else None,
            "output_dir": str(run_dir),
            "duration_sec": config.duration_sec,
            "sample_number": config.sample_number,
            "profile_steps": config.profile_steps,
            "max_steps": 1 if smoke else config.max_steps,
            "learning_rate": config.learning_rate,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_scaling": config.lora_scaling,
            "lora_enabled": config.lora_enabled,
            "ft_embed": config.ft_embed,
            "weight_decay": config.weight_decay,
            "pct_start": config.pct_start,
            "first_codebook_weight_multiplier": config.first_codebook_weight_multiplier,
            "text_padding_weight": config.text_padding_weight,
            "log_freq": config.log_freq,
            "no_eval": config.no_eval,
            "eval_on_train_samples": config.eval_on_train_samples,
            "ckpt_freq": config.ckpt_freq,
            "torch_compile_disabled": os.environ.get("NO_TORCH_COMPILE", ""),
            "qlora": config.qlora,
            "quant_type": config.quant_type if config.qlora else None,
            "device": str(device),
            "num_processes": world_size,
            "cpu_threads": cpu_threads,
            "gradient_accumulation_steps": accum_steps,
            "per_device_batch_size": config.per_device_batch_size,
            "num_workers": 0 if smoke else config.num_workers,
            "prefetch_factor": config.prefetch_factor,
            "pin_memory": config.pin_memory,
            "persistent_workers": config.persistent_workers and not smoke and config.num_workers > 0,
            "global_batch_size": global_batch_size,
            "warmup_steps": config.warmup_steps,
            "eval_every_steps": config.eval_every_steps,
            "free_running_eval_every_steps": config.free_running_eval_every_steps,
            "free_running_eval_samples": config.free_running_eval_samples,
            "free_running_eval_window_seconds": config.free_running_eval_window_seconds,
            "validation_max_samples": config.validation_max_samples,
            "validation_generation_settings": config.generation_settings.as_dict(),
            "save_every_steps": config.save_every_steps,
            "random_crop": config.random_crop,
            "randomize_train": config.randomize_train,
            "prompt_aug_prob": config.prompt_aug_prob,
            "static_chunking": config.static_chunking,
            "swap_roles_after_pass": config.swap_roles_after_pass,
            "gradient_checkpointing": config.gradient_checkpointing,
            "mixed_precision": config.mixed_precision,
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "num_test_samples": len(test_samples),
            "num_train_conversations": len(train_conversations),
            "num_train_role_views": len(train_samples),
        }
        (run_dir / "config.json").write_text(json.dumps(config_record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(config_record))
        print(f"[Main Process] Active across {world_size} process(es) on device {device}")

    if world_size > 1:
        run_dir_list = [str(run_dir) if main_process else ""]
        torch.distributed.broadcast_object_list(run_dir_list, src=0)
        run_dir = Path(run_dir_list[0])

    if run_dir is not None:
        write_rank_info(run_dir, rank, world_size, device, len(train_samples))
    if distributed:
        barrier()

    # DDP keeps a complete frozen base model on every device. Each rank loads
    # the same explicit local PersonaPlex checkpoint before DDP synchronizes it.
    runtime = load_runtime(
        RuntimePaths(config.model_root, config.personaplex_source),
        str(device), config.qlora, config.quant_type,
        model_device=str(device),
        load_model_weights=True,
        codec_cache_dir=config.codec_cache_dir,
    )
    inspection = None
    for sample in train_samples:
        try:
            inspection = inspect_training_sample(sample, runtime.tokenizer)
            break
        except ValueError as exc:
            if "no agent text target" not in str(exc):
                raise
    if inspection is None:
        raise ValueError("training chunks contain no agent text targets")
    if main_process:
        print("[Training sample inspection] " + json.dumps(inspection, ensure_ascii=False))

    # Inject LoRA with Stage-Wise Freezing support
    stage = getattr(config, "train_stage", "joint").lower()
    lora_prefixes = lora_prefixes_for_stage(config)
    if stage in {"temporal_only", "freeze_depformer"}:
        if hasattr(runtime.model, "depformer"):
            runtime.model.depformer.requires_grad_(False)
        if main_process:
            print("[Stage-Wise Training] Active Stage: TEMPORAL ONLY (Depth Transformer / Depformer is 100% frozen).")
    elif stage in {"depth_only", "freeze_tempformer"}:
        if hasattr(runtime.model, "transformer"):
            runtime.model.transformer.requires_grad_(False)
        if main_process:
            print("[Stage-Wise Training] Active Stage: DEPTH ONLY (Temporal 7B Transformer is 100% frozen).")
    else:
        if main_process:
            print("[Stage-Wise Training] Active Stage: JOINT (Both Temporal & Depth active).")

    if not config.lora_enabled:
        raise ValueError("this training path requires lora.enable=true")
    lora_alpha = config.lora_alpha
    targets = inject_lora(runtime.model, config.lora_rank, lora_alpha, prefixes=lora_prefixes)

    # Optional gradient checkpointing
    if config.gradient_checkpointing:
        enable_gradient_checkpointing(runtime.model)
        if main_process:
            print("Gradient checkpointing enabled on transformer layers.")

    # Load adapter weights before DDP wrapping.
    resume_checkpoint_dir = None
    adapter_resume_step = 0
    if resume_from:
        resume_path = Path(resume_from)
        adapter_file = resume_path if resume_path.is_file() else resume_path / "lora.safetensors"
        if not adapter_file.is_file():
            raise FileNotFoundError(f"resume adapter does not exist: {adapter_file}")
        resume_checkpoint_dir = adapter_file.parent
        if main_process:
            print(f"Resuming LoRA weights from {adapter_file}")
        load_adapter(runtime.model, adapter_file)
        meta_file = adapter_file.parent / "adapter.json"
        if meta_file.is_file():
            try:
                adapter_resume_step = int(json.loads(meta_file.read_text(encoding="utf-8")).get("step", 0))
                if main_process:
                    print(f"Adapter checkpoint is at optimizer step {adapter_resume_step}")
            except Exception:
                pass

    # Alias LMModel.forward to forward_train for training execution
    from moshi.models.lm import LMModel
    LMModel.forward = LMModel.forward_train

    trainable = [p for p in runtime.model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable LoRA parameters were injected")
    unexpected_trainable = [
        name for name, parameter in runtime.model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected_trainable:
        raise RuntimeError(f"unexpected non-LoRA trainable parameters: {unexpected_trainable[:5]}")
    if main_process:
        print(f"LoRA targets: {len(targets)}; trainable parameters: {sum(p.numel() for p in trainable):,}")

    temp_lr = config.learning_rate
    dep_lr = config.depformer_learning_rate
    if dep_lr is not None and dep_lr != temp_lr:
        temp_params = [p for n, p in runtime.model.named_parameters() if p.requires_grad and "depformer" not in n]
        dep_params = [p for n, p in runtime.model.named_parameters() if p.requires_grad and "depformer" in n]
        param_groups = []
        if temp_params:
            param_groups.append({"params": temp_params, "lr": temp_lr})
        if dep_params:
            param_groups.append({"params": dep_params, "lr": dep_lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay, fused=device.type == "cuda")
        if main_process:
            print(f"Using Dual Learning Rates -> Temporal Transformer: {temp_lr:.2e}, Depth Transformer: {dep_lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(trainable, lr=temp_lr, weight_decay=config.weight_decay, fused=device.type == "cuda")
    max_steps = 1 if smoke else config.max_steps
    scheduler = None
    if max_steps > 1:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            total_steps=max_steps,
            pct_start=config.pct_start,
        )
    if distributed:
        runtime.model = torch.nn.parallel.DistributedDataParallel(
            runtime.model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        barrier()

    start_step = adapter_resume_step
    if resume_checkpoint_dir is not None:
        restored_step = load_training_state(
            resume_checkpoint_dir,
            optimizer,
            scheduler,
            accum_steps,
            world_size,
            config.per_device_batch_size,
            total_steps=max_steps if max_steps > 1 else None,
        )
        if restored_step is not None:
            start_step = restored_step
            if main_process:
                print(f"Restored optimizer and scheduler state at optimizer step {start_step}")
        elif main_process:
            print("Resume checkpoint has no training_state.pt; resuming adapter weights with a fresh optimizer.")
    start_micro_step = start_step * accum_steps
    if start_step > max_steps:
        raise RuntimeError(f"resume checkpoint step {start_step} exceeds train.max_steps={max_steps}")

    writer = None
    log_file = None
    if main_process and run_dir is not None:
        log_path = run_dir / "metrics.jsonl"
        log_file = log_path.open("w", encoding="utf-8")
        tensorboard_dir = run_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        writer.add_text("configuration", json.dumps(config_record, indent=2), 0)

    saved = None
    best_saved = None
    best_val_loss = float("inf")
    best_inference_saved = None
    best_generation_cer = float("inf")
    baseline_generation_cer = None
    best_generation_sample = None
    last_record = None
    reload_checks: list[dict[str, float | int]] = []
    started = time.monotonic()

    if val_samples and not config.no_eval and config.free_running_eval_every_steps > 0:
        # Establish the pre-training generation baseline with the same samples,
        # seed, and LMGen settings used for later checkpoint selection.
        barrier()
        baseline_result = [None]
        if main_process:
            try:
                baseline_result[0] = evaluate_free_running(runtime, val_samples, config)
            except Exception as exc:
                baseline_result[0] = {"error": f"{type(exc).__name__}: {exc}"}
        if distributed:
            torch.distributed.broadcast_object_list(baseline_result, src=0, device=device)
        if isinstance(baseline_result[0], dict) and "error" in baseline_result[0]:
            raise RuntimeError(f"base-model free-running validation failed: {baseline_result[0]['error']}")
        baseline_generation_cer = float(baseline_result[0]["val/generation_cer"])
        if main_process and run_dir is not None:
            baseline_row = {
                "step": 0, "val/generation_baseline": True, **baseline_result[0],
            }
            with (run_dir / "free_running_metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(baseline_row, ensure_ascii=False) + "\n")
            if writer:
                writer.add_scalar("val/generation_cer_base", baseline_generation_cer, 0)
                writer.add_scalar(
                    "val/generation_empty_samples_base",
                    baseline_result[0]["val/generation_empty_samples"], 0,
                )
            print(json.dumps({
                "event": "starting_generation_baseline",
                "cer": baseline_generation_cer,
                "wer": baseline_result[0]["val/generation_wer"],
                "empty_samples": baseline_result[0]["val/generation_empty_samples"],
            }, ensure_ascii=False))
        barrier()

    try:
        max_micro_steps = max_steps * accum_steps
        progress = tqdm(total=max_steps, initial=start_step, desc="training (DDP)" if distributed else "training", unit="update", dynamic_ncols=True) if main_process else None
        last_update_time = time.monotonic()
        batch_iterator = iter(iter_training_batches(
            config, train_conversations, runtime, device, rank,
            world_size, smoke, skip_batches=start_micro_step,
        ))
        samples_seen = 0
        audio_seconds_seen = 0.0
        audio_frames_seen = 0
        text_target_tokens_seen = 0
        text_padding_positions_seen = 0
        pending_text_target_tokens = torch.zeros((), dtype=torch.int64, device=device)
        pending_text_padding_positions = torch.zeros((), dtype=torch.int64, device=device)
        pending_text_target_ce_sum = torch.zeros((), dtype=torch.float32, device=device)
        pending_text_target_ce_count = torch.zeros((), dtype=torch.int64, device=device)
        pending_profile_times = torch.zeros(3, dtype=torch.float64, device=device)

        def profile_sync() -> None:
            if config.profile_steps and device.type == "cuda":
                torch.cuda.synchronize(device)

        optimizer_step = start_step
        for micro_step in range(start_micro_step, max_micro_steps):
            profile_sync()
            data_started = time.monotonic() if config.profile_steps else 0.0
            try:
                batch, epoch, local_samples, local_audio_seconds, local_audio_frames, batch_samples = next(batch_iterator)
            except StopIteration as exc:
                raise RuntimeError("training batch iterator stopped before max_steps") from exc
            profile_sync()
            if config.profile_steps:
                pending_profile_times[0] += time.monotonic() - data_started
            codes = batch["codes"]
            samples_seen += local_samples * world_size
            global_audio_seconds = all_reduce_sum(
                torch.tensor([local_audio_seconds, local_audio_frames], dtype=torch.float32, device=device)
            )
            audio_seconds_seen += float(global_audio_seconds[0])
            audio_frames_seen += int(global_audio_seconds[1])

            sync_gradients = (micro_step + 1 - start_micro_step) % accum_steps == 0 or micro_step + 1 == max_micro_steps
            sync_context = runtime.model.no_sync() if distributed and not sync_gradients else nullcontext()
            with sync_context:
                profile_sync()
                forward_started = time.monotonic() if config.profile_steps else 0.0
                output = model_forward_train(runtime.model, codes)
                text_targets, text_padding = text_supervision_counts(
                    batch, output, runtime.tokenizer.padding_id
                )
                pending_text_target_tokens += text_targets.detach()
                pending_text_padding_positions += text_padding.detach()
                target_ce_sum, target_ce_count = text_target_token_loss_stats(
                    batch, output, runtime.tokenizer.padding_id
                )
                pending_text_target_ce_sum += target_ce_sum
                pending_text_target_ce_count += target_ce_count
                loss_result = loss_components(
                    output, codes, batch, runtime.tokenizer.padding_id, torch,
                    config.first_codebook_weight_multiplier, config.text_padding_weight,
                    distributed=distributed,
                )
                if distributed:
                    components, total = reduce_distributed_loss(
                        loss_result[1], all_reduce_sum, preserve_grad=True, world_size=world_size
                    )
                else:
                    total = loss_result[0]
                    components = loss_result[1]
                profile_sync()
                if config.profile_steps:
                    pending_profile_times[1] += time.monotonic() - forward_started
                profile_sync()
                backward_started = time.monotonic() if config.profile_steps else 0.0
                (total / accum_steps).backward()
                profile_sync()
                if config.profile_steps:
                    pending_profile_times[2] += time.monotonic() - backward_started
            profile_sync()
            optimizer_started = time.monotonic() if config.profile_steps else 0.0
            sync_state = type("SyncState", (), {"sync_gradients": sync_gradients})()
            grad_norm = step_optimizer_if_ready(sync_state, optimizer, scheduler, trainable, runtime.model)
            profile_sync()
            optimizer_seconds = time.monotonic() - optimizer_started if config.profile_steps else 0.0

            if not sync_gradients:
                continue

            global_text_stats = all_reduce_sum(torch.stack([
                pending_text_target_tokens.float(), pending_text_padding_positions.float(),
                pending_text_target_ce_sum, pending_text_target_ce_count.float(),
            ]))
            text_target_tokens_seen += int(global_text_stats[0])
            text_padding_positions_seen += int(global_text_stats[1])
            text_nonpadding_loss = float(
                global_text_stats[2] / global_text_stats[3].clamp_min(1.0)
            )
            pending_text_target_tokens.zero_()
            pending_text_padding_positions.zero_()
            pending_text_target_ce_sum.zero_()
            pending_text_target_ce_count.zero_()

            profile_times = None
            if config.profile_steps:
                profile_times = torch.cat((
                    pending_profile_times,
                    torch.tensor([optimizer_seconds], dtype=torch.float64, device=device),
                ))
                if distributed:
                    torch.distributed.all_reduce(profile_times, op=torch.distributed.ReduceOp.MAX)
                profile_times = profile_times.cpu().tolist()
                pending_profile_times.zero_()

            optimizer_step += 1

            reduced_total = total.detach()
            reduced_text = components["text"].detach()
            reduced_sem = components["audio_semantic"].detach()
            reduced_nonsem = components["audio_nonsemantic"].detach()

            if main_process and (optimizer_step % config.log_freq == 0 or optimizer_step == max_steps):
                record = {
                    "step": optimizer_step,
                    "micro_step": micro_step + 1,
                    "epoch": epoch,
                    "samples_seen": samples_seen,
                    "audio_seconds_seen": audio_seconds_seen,
                    "audio_frames_seen": audio_frames_seen,
                    "text_target_tokens_seen": text_target_tokens_seen,
                    "text_padding_positions_seen": text_padding_positions_seen,
                    "text_target_fraction": text_target_tokens_seen / max(
                        1, text_target_tokens_seen + text_padding_positions_seen
                    ),
                    "global_batch_size": global_batch_size,
                    "loss/total": float(reduced_total.detach()),
                    "loss/text": float(reduced_text.detach()),
                    "loss/text_nonpadding": text_nonpadding_loss,
                    "loss/audio_semantic": float(reduced_sem.detach()),
                    "loss/audio_nonsemantic": float(reduced_nonsem.detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                    "samples_per_second": global_batch_size / max(time.monotonic() - last_update_time, 1e-9),
                    "gpu_peak_bytes": torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0,
                }
                if profile_times is not None:
                    record.update({
                        "timing/data_sec": profile_times[0],
                        "timing/forward_loss_sec": profile_times[1],
                        "timing/backward_sec": profile_times[2],
                        "timing/optimizer_sec": profile_times[3],
                        "timing/profiled_phase_sum_sec": sum(profile_times),
                    })
                last_update_time = time.monotonic()
                last_record = record
                if log_file:
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                if writer:
                    write_tensorboard_scalars(writer, record, sum(p.numel() for p in trainable), cpu_threads)
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(loss=f"{record['loss/total']:.4f}", grad=f"{grad_norm:.3f}")
                if max_steps == 1:
                    tqdm.write(json.dumps({"event": "training_step", **record}))

            # Validation evaluation
            teacher_eval_due = (
                config.eval_every_steps > 0
                and (optimizer_step % config.eval_every_steps == 0 or optimizer_step == max_steps)
            )
            generation_eval_due = (
                config.free_running_eval_every_steps > 0
                and (optimizer_step % config.free_running_eval_every_steps == 0 or optimizer_step == max_steps)
            )
            if val_samples and not config.no_eval and (teacher_eval_due or generation_eval_due):
                barrier()
                val_metrics = evaluate_validation(config, runtime, val_samples, rank, world_size, device)
                barrier()
                best_state = adapter_state_dict(unwrap_parallel_model(runtime.model))
                generation_metrics = None
                if generation_eval_due:
                    barrier()
                    generation_result = [None]
                    if main_process:
                        try:
                            generation_result[0] = evaluate_free_running(
                                runtime, val_samples, config,
                            )
                        except Exception as exc:
                            generation_result[0] = {"error": f"{type(exc).__name__}: {exc}"}
                    if distributed:
                        torch.distributed.broadcast_object_list(generation_result, src=0, device=device)
                    if isinstance(generation_result[0], dict) and "error" in generation_result[0]:
                        raise RuntimeError(f"free-running validation failed: {generation_result[0]['error']}")
                    generation_metrics = generation_result[0]
                    if main_process:
                        generation_row = {"step": optimizer_step, **generation_metrics}
                        with (run_dir / "free_running_metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
                            metrics_file.write(json.dumps(generation_row, ensure_ascii=False) + "\n")
                        if writer:
                            writer.add_scalar("val/generation_cer", generation_metrics["val/generation_cer"], optimizer_step)
                            writer.add_scalar("val/generation_wer", generation_metrics["val/generation_wer"], optimizer_step)
                        # A blank transcript is a failed generation even when its
                        # normalized CER happens to beat earlier checkpoints.
                        generation_score = generation_checkpoint_score(
                            generation_metrics, baseline_cer=baseline_generation_cer,
                        )
                        if generation_score < best_generation_cer:
                            best_generation_cer = generation_score
                            best_generation_sample = generation_metrics["samples"][0]
                            best_inference_saved = save_best_adapter_state(
                                run_dir, best_state, config, optimizer_step,
                                val_metrics.get("val/loss_selection", float("inf")), optimizer, scheduler,
                                gradient_accumulation_steps=accum_steps, num_processes=world_size,
                                checkpoint_name="best_inference",
                            )
                            tqdm.write(json.dumps({
                                "event": "new_best_free_running_checkpoint", "step": optimizer_step,
                                "cer": best_generation_cer,
                                "wer": generation_metrics["val/generation_wer"],
                                "adapter": str(best_inference_saved),
                            }, ensure_ascii=False))
                    barrier()
                if main_process:
                    for k, v in val_metrics.items():
                        if writer:
                            writer.add_scalar(k, v, optimizer_step)
                    val_loss = validation_selection_loss(val_metrics)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_saved = save_best_adapter_state(
                            run_dir, best_state, config, optimizer_step, val_loss, optimizer, scheduler,
                            gradient_accumulation_steps=accum_steps, num_processes=world_size,
                            checkpoint_name="best_loss",
                        )
                        tqdm.write(json.dumps({
                            "event": "new_best_val_checkpoint", "step": optimizer_step,
                            "selection_loss": val_loss,
                            "text_nonpadding_loss": val_metrics.get("val/loss_text_nonpadding"),
                            "audio_semantic_loss": val_metrics.get("val/loss_audio_semantic"),
                            "audio_nonsemantic_loss": val_metrics.get("val/loss_audio_nonsemantic"),
                        }))

            # Periodic saving & smoke reload verification
            save_interval = 1 if smoke else config.ckpt_freq
            if optimizer_step % save_interval == 0 or optimizer_step == max_steps:
                barrier()
                checkpoint_state = adapter_state_dict(unwrap_parallel_model(runtime.model))
                if main_process:
                    saved = save_adapter_state(
                        run_dir, checkpoint_state, config, optimizer_step, optimizer, scheduler,
                        accum_steps, world_size,
                    )
                    if smoke:
                        reload_loss = verify_reloaded_adapter(
                            config, batch_samples[0], saved, runtime,
                        )
                        reload_checks.append({"step": optimizer_step, "loss": reload_loss})
                barrier()

        if log_file:
            log_file.close()
        if progress is not None:
            progress.close()
    finally:
        if writer:
            writer.close()

    if run_dir is not None:
        rank_peak = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        write_rank_info(run_dir, rank, world_size, device, len(train_samples), rank_peak)
    barrier()

    if main_process and run_dir is not None:
        peak = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        run_info = {
            "seconds": time.monotonic() - started,
            "peak_gpu_bytes": peak,
            "checkpoint": str(saved) if saved else None,
            "best_checkpoint": str(best_saved) if best_saved else None,
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "best_inference_checkpoint": str(best_inference_saved) if best_inference_saved else None,
            "best_generation_cer": best_generation_cer if best_generation_cer < float("inf") else None,
            "starting_generation_cer": baseline_generation_cer,
            "inference_checkpoint_status": (
                "validated_generation_improves_base"
                if best_inference_saved is not None else "no_checkpoint_improved_base_generation"
            ),
            "best_inference_sample": best_generation_sample,
            "reload_checks": reload_checks,
        }
        report = run_dir / "training_report.md"
        inference_adapter = best_inference_saved or saved
        inference_config_path = run_dir / "inference_config.json"
        inference_output_dir = run_dir / "inference_smoke"
        inference_config_path.write_text(
            json.dumps(
                inference_config_snapshot(config, inference_adapter, inference_output_dir),
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        run_info["inference_config"] = str(inference_config_path)
        (run_dir / "run.json").write_text(json.dumps(run_info, indent=2))
        inference_args = [
            sys.executable, "-m", "tools.inference_smoke", "--config", str(inference_config_path),
            "--adapter", str(inference_adapter),
            "--output-dir", str(inference_output_dir),
        ]
        if best_generation_sample is not None:
            inference_args.extend([
                "--split", "train" if config.eval_on_train_samples else "validation",
                "--sample-id", str(best_generation_sample["sample_id"]),
                "--start", str(best_generation_sample["window_start_sec"]),
                "--window-seconds", str(best_generation_sample["window_end_sec"] - best_generation_sample["window_start_sec"]),
            ])
        training_duration_inference_args = None
        if (
            best_generation_sample is not None
            and best_generation_sample["source_duration_sec"]
            >= best_generation_sample["window_start_sec"] + config.duration_sec
        ):
            training_duration_inference_args = list(inference_args)
            duration_index = training_duration_inference_args.index("--window-seconds") + 1
            training_duration_inference_args[duration_index] = str(config.duration_sec)
            training_duration_inference_args.extend([
                "--output-dir", str(run_dir / "inference_training_duration"),
            ])
        report.write_text(
            "# PersonaPlex Training Report\n\n"
            f"Status: {'GENERATION-VALIDATED' if best_inference_saved is not None else 'NO GENERATION-VALID ADAPTER'} — "
            "training loss alone does not establish usable inference.\n\n"
            f"- Command: `{shlex.join([sys.executable, *sys.argv])}`\n"
            f"- Dataset: {config.manifest}\n- Model: {config.model_root}\n- LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}\n"
            f"- Processes (GPUs): {world_size}\n"
            f"- Steps: {max_steps}\n- Final loss: {last_record['loss/total'] if last_record else 'n/a'}\n"
            f"- Text targets/padding positions: "
            f"{last_record['text_target_tokens_seen'] if last_record else 'n/a'}/"
            f"{last_record['text_padding_positions_seen'] if last_record else 'n/a'}\n"
            f"- Non-padding text loss: {last_record['loss/text_nonpadding'] if last_record else 'n/a'}\n"
            f"- Best val loss: {best_val_loss if best_val_loss < float('inf') else 'n/a'}\n"
            f"- Peak GPU bytes: {peak}\n- Checkpoint: {saved}\n"
            f"- Best checkpoint: {best_saved}\n"
            f"- Best inference checkpoint: {best_inference_saved}\n"
            f"- Starting free-running CER: {baseline_generation_cer if baseline_generation_cer is not None else 'n/a'}\n"
            f"- Best free-running CER: {best_generation_cer if best_generation_cer < float('inf') else 'n/a'}\n"
            f"- Best inference sample: {best_generation_sample['sample_id'] if best_generation_sample else 'n/a'}\n"
            f"- Inference command: `{shlex.join(inference_args)}`\n",
            encoding="utf-8",
        )
        if training_duration_inference_args is not None:
            with report.open("a", encoding="utf-8") as report_file:
                report_file.write(
                    f"- Same adapter/sample with training-duration input ({config.duration_sec:g}s): "
                    f"`{shlex.join(training_duration_inference_args)}`\n"
                )

    barrier()
    return best_inference_saved or best_saved or saved


def main() -> int:
    parser = argparse.ArgumentParser(description="PersonaPlex Fine-Tuning with Moshi-style training")
    parser.add_argument("config_file", nargs="?", help="Path to Hydra YAML configuration file")
    parser.add_argument("--config", type=str, default=None, help="Path to Hydra YAML configuration file")
    parser.add_argument("--config-name", type=str, default=None, help="Name of config in configs/ directory (Hydra style)")
    parser.add_argument("--smoke", action="store_true", help="Run 1-step smoke test with verification")
    parser.add_argument("--qlora", action="store_true", default=None, help="Enable 4-bit QLoRA")
    parser.add_argument("--no-qlora", dest="qlora", action="store_false", help="Disable QLoRA")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint directory to resume from")

    args, unknown = parser.parse_known_args()
    overrides = [arg for arg in unknown if "=" in arg]

    if args.config and args.config_file:
        parser.error("pass the config either positionally or with --config, not both")
    config_path = args.config or args.config_file
    if config_path is None:
        if args.config_name:
            config_path = Path("configs") / f"{args.config_name}.yaml"
        else:
            config_path = Path("configs/config.yaml")

    config = load_config(config_path, overrides=overrides)
    if args.qlora is not None:
        config = config.replace(qlora=args.qlora)

    run(
        config,
        smoke=args.smoke,
        resume_from=args.resume_from,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
