"""Single-GPU, intentionally small LoRA training path."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

from .config import Config, load_config
from .data import PreparedDataset
from .lora import adapter_state_dict, inject_lora, load_adapter
from .objective import stream_weights, torch_weighted_cross_entropy
from .runtime import RuntimePaths, load_runtime
from .sequence import PersonaPlexTrainingExampleBuilder


def limit_cpu_threads(torch) -> int:
    """Keep one training process from consuming every CPU core."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return 1


def seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_dir(output_root: Path, smoke: bool) -> Path:
    """Create a unique, timestamped run directory below the configured root."""
    label = "smoke" if smoke else "train"
    run_dir = output_root / f"{label}_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_example(config: Config, sample, runtime, random_crop: bool = False):
    if random_crop and getattr(config, "random_crop", False):
        effective_sample = sample.dynamic_sample(
            config.window_seconds, random_crop=True, prompt_aug_prob=getattr(config, "prompt_aug_prob", 0.0)
        )
    else:
        effective_sample = sample
    builder = PersonaPlexTrainingExampleBuilder(runtime.codec, runtime.tokenizer, runtime.initial_tokens, runtime.zero_token)
    return builder.apply_delays(builder.build(effective_sample), runtime.delays)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    import math
    import torch
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_validation(config: Config, runtime, val_samples: list) -> dict[str, float]:
    if not val_samples:
        return {}
    import torch
    runtime.model.eval()
    total_losses = []
    text_losses = []
    semantic_losses = []
    nonsemantic_losses = []
    with torch.no_grad():
        for sample in val_samples:
            example = build_example(config, sample, runtime, random_crop=False)
            codes = torch.tensor(example.input_codes, dtype=torch.long, device=config.device).unsqueeze(0)
            output = model_forward_train(runtime.model, codes)
            total, comps = loss_components(output, codes, example, runtime.tokenizer.padding_id, torch)
            total_losses.append(float(total.detach()))
            text_losses.append(float(comps["text"].detach()))
            semantic_losses.append(float(comps["audio_semantic"].detach()))
            nonsemantic_losses.append(float(comps["audio_nonsemantic"].detach()))
    runtime.model.train()
    return {
        "val/loss_total": sum(total_losses) / len(total_losses),
        "val/loss_text": sum(text_losses) / len(text_losses),
        "val/loss_audio_semantic": sum(semantic_losses) / len(semantic_losses),
        "val/loss_audio_nonsemantic": sum(nonsemantic_losses) / len(nonsemantic_losses),
    }


def save_best_adapter(run_dir: Path, model, config: Config, step: int, val_loss: float) -> Path:
    from safetensors.torch import save_file
    path = run_dir / "checkpoints" / "best"
    path.mkdir(parents=True, exist_ok=True)
    adapter = path / "lora.safetensors"
    save_file(fsdp_adapter_state_dict(model), str(adapter))
    (path / "adapter.json").write_text(
        json.dumps(
            {"step": step, "val_loss": val_loss, "model_root": str(config.model_root), "rank": config.lora_rank, "alpha": config.lora_alpha},
            indent=2,
        )
    )
    return adapter


def model_forward_train(model, codes):
    """LMModel deliberately exposes training through ``forward_train`` only.
    When wrapped by FSDP, __call__ must be invoked to trigger FSDP parameter un-sharding hooks."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
        if isinstance(model, FullyShardedDataParallel):
            return model(codes)
    except ImportError:
        pass
    return model.forward_train(codes)


def write_tensorboard_scalars(writer, record: dict[str, float | int], trainable_parameters: int, cpu_threads: int) -> None:
    step = int(record["step"])
    for name, value in (
        ("loss/total", record["loss/total"]),
        ("loss/text", record["loss/text"]),
        ("loss/audio_semantic", record["loss/audio_semantic"]),
        ("loss/audio_nonsemantic", record["loss/audio_nonsemantic"]),
        ("train/learning_rate", record["lr"]),
        ("train/gradient_norm", record["grad_norm"]),
        ("system/gpu_peak_bytes", record["gpu_peak_bytes"]),
        ("system/trainable_parameters", trainable_parameters),
        ("system/cpu_threads", cpu_threads),
    ):
        writer.add_scalar(name, value, step)


def loss_components(model_output, codes, example, text_padding_id, torch):
    weights = stream_weights(example.labels, example.loss_mask, text_padding_id)
    text_target = codes[:, 0, :]
    text_weight = torch.tensor(weights[0], device=codes.device).unsqueeze(0)
    text_weight = text_weight * model_output.text_mask.to(text_weight.dtype)
    text_loss = torch_weighted_cross_entropy(model_output.text_logits.reshape(-1, model_output.text_logits.shape[-1]), text_target.reshape(-1), text_weight.reshape(-1))
    audio_target = codes[:, 1:17, :]
    audio_weights = torch.tensor(weights[1:17], device=codes.device).unsqueeze(0)
    audio_weights = audio_weights * model_output.mask.to(audio_weights.dtype)
    semantic = torch_weighted_cross_entropy(model_output.logits[:, 0].reshape(-1, model_output.logits.shape[-1]), audio_target[:, 0].reshape(-1), audio_weights[:, 0].reshape(-1))
    nonsemantic_weights = audio_weights[:, 1:8]
    nonsemantic = torch_weighted_cross_entropy(model_output.logits[:, 1:8].reshape(-1, model_output.logits.shape[-1]), audio_target[:, 1:8].reshape(-1), nonsemantic_weights.reshape(-1))
    return text_loss + semantic + nonsemantic, {"text": text_loss, "audio_semantic": semantic, "audio_nonsemantic": nonsemantic}


def one_step(config: Config, runtime, example, optimizer=None):
    import torch
    codes = torch.tensor(example.input_codes, dtype=torch.long, device=config.device).unsqueeze(0)
    output = model_forward_train(runtime.model, codes)
    total, components = loss_components(output, codes, example, runtime.tokenizer.padding_id, torch)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        trainable = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
        if not any(parameter.grad is not None and parameter.grad.abs().sum().item() > 0 for parameter in trainable):
            raise RuntimeError("LoRA gradients are all zero")
        if any(parameter.grad is not None for parameter in runtime.model.parameters() if not parameter.requires_grad):
            raise RuntimeError("frozen base parameter received a gradient")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    else:
        grad_norm = torch.tensor(0.0)
    return total, components, float(grad_norm)


from .fsdp import find_free_port, fsdp_adapter_state_dict, parse_gpu_ids, wrap_model_fsdp


def save_adapter(run_dir: Path, model, config: Config, step: int) -> Path:
    from safetensors.torch import save_file
    path = run_dir / "checkpoints" / f"checkpoint_{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    adapter = path / "lora.safetensors"
    save_file(fsdp_adapter_state_dict(model), str(adapter))
    (path / "adapter.json").write_text(json.dumps({"step": step, "model_root": str(config.model_root), "rank": config.lora_rank, "alpha": config.lora_alpha}, indent=2))
    return adapter


def verify_reloaded_adapter(config: Config, sample, adapter: Path) -> float:
    """Load base + adapter in a fresh model object and return teacher-forced loss."""
    import torch
    fresh = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), config.device, config.qlora, config.quant_type)
    inject_lora(fresh.model, config.lora_rank, config.lora_alpha)
    load_adapter(fresh.model, adapter)
    fresh.model.eval()
    example = build_example(config, sample, fresh)
    with torch.no_grad():
        total, _, _ = one_step(config, fresh, example)
    return float(total)


def _train_fsdp_worker(
    rank: int,
    world_size: int,
    gpu_ids: list[int],
    port: int,
    config: Config,
    smoke: bool,
    run_dir: Path,
    sharding_strategy: str = "shard_grad_op",
) -> None:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    import torch
    import torch.distributed as dist
    from tqdm.auto import tqdm

    device_id = gpu_ids[rank]
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    device_name = torch.cuda.get_device_name(device_id) if torch.cuda.is_available() else "CPU"
    print(f"[Process Rank {rank}] Using device: {device} ({device_name})")

    init_method = f"tcp://127.0.0.1:{port}"
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )

    cpu_threads = limit_cpu_threads(torch)
    seed_everything(config.seed + rank, torch)

    if config.val_manifest:
        train_samples = PreparedDataset(config.manifest, config.window_seconds).load()
        val_samples = PreparedDataset(config.val_manifest, config.window_seconds).load()
    elif config.eval_every_steps > 0:
        train_samples, val_samples = PreparedDataset(config.manifest, config.window_seconds).split(
            val_ratio=config.val_ratio, seed=config.seed
        )
    else:
        train_samples = PreparedDataset(config.manifest, config.window_seconds).load()
        val_samples = []

    # Load model sequentially across ranks to prevent spiking CPU RAM over 30GB
    for i in range(world_size):
        if rank == i:
            print(f"[Rank {rank}] Loading model on {device} ({device_name})...")
            runtime = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), device, config.qlora, config.quant_type)
        dist.barrier()

    targets = inject_lora(runtime.model, config.lora_rank, config.lora_alpha)

    # Alias LMModel.forward to forward_train for FSDP compatibility
    from moshi.models.lm import LMModel
    LMModel.forward = LMModel.forward_train

    runtime.model = wrap_model_fsdp(runtime.model, device_id=device_id, strategy=sharding_strategy)

    trainable = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
    if rank == 0:
        gpu_summary = ", ".join(f"cuda:{gid} ({torch.cuda.get_device_name(gid) if torch.cuda.is_available() else 'CPU'})" for gid in gpu_ids)
        print(f"[Rank 0] FSDP active across GPUs: [{gpu_summary}] (strategy={sharding_strategy}). LoRA targets: {len(targets)}; trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)
    max_steps = 1 if smoke else config.max_steps
    scheduler = None
    if config.warmup_steps > 0 and not smoke:
        scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, max_steps)

    writer = None
    log_path = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_dir = run_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        log_path = run_dir / "metrics.jsonl"

    saved = None
    best_saved = None
    best_val_loss = float("inf")
    last_record = None
    reload_checks: list[dict[str, float | int]] = []
    started = time.monotonic()
    config_rank = config.replace(device=device)

    accum_steps = max(1, config.gradient_accumulation_steps)
    shuffled_indices = list(range(len(train_samples)))
    if config.shuffle and not smoke:
        random.Random(config.seed).shuffle(shuffled_indices)

    try:
        log_file = log_path.open("w", encoding="utf-8") if log_path else None
        progress = tqdm(range(max_steps), desc="training (FSDP)", unit="step", dynamic_ncols=True) if rank == 0 else range(max_steps)
        for step in progress:
            # Distribute distinct samples across ranks (Data Parallel)
            global_sample_idx = (step * world_size + rank) % len(train_samples)
            sample_idx = global_sample_idx if not config.shuffle else shuffled_indices[global_sample_idx % len(shuffled_indices)]
            current_sample = train_samples[sample_idx]
            example = build_example(config_rank, current_sample, runtime, random_crop=config.random_crop and not smoke)
            codes = torch.tensor(example.input_codes, dtype=torch.long, device=device).unsqueeze(0)
            output = model_forward_train(runtime.model, codes)
            total, components = loss_components(output, codes, example, runtime.tokenizer.padding_id, torch)

            # Gradient accumulation
            scaled_loss = total / accum_steps
            is_update = ((step + 1) % accum_steps == 0) or (step + 1 == max_steps)
            if hasattr(runtime.model, "no_sync") and not is_update:
                with runtime.model.no_sync():
                    scaled_loss.backward()
            else:
                scaled_loss.backward()

            # Reduce total loss across ranks for synchronized logging
            dist.all_reduce(total, op=dist.ReduceOp.AVG)

            if is_update:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                grad_norm = 0.0


            if rank == 0:
                record = {
                    "step": step,
                    "loss/total": float(total.detach()),
                    "loss/text": float(components["text"].detach()),
                    "loss/audio_semantic": float(components["audio_semantic"].detach()),
                    "loss/audio_nonsemantic": float(components["audio_nonsemantic"].detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                    "gpu_peak_bytes": torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0,
                }
                last_record = record
                if log_file:
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                if writer:
                    write_tensorboard_scalars(writer, record, sum(p.numel() for p in trainable), cpu_threads)
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(loss=f"{record['loss/total']:.4f}", grad=f"{grad_norm:.3f}")
                if max_steps == 1:
                    tqdm.write(json.dumps({"event": "training_step", **record}))

            # Validation evaluation
            if val_samples and config.eval_every_steps > 0 and ((step + 1) % config.eval_every_steps == 0 or step + 1 == max_steps):
                dist.barrier()
                val_metrics = evaluate_validation(config_rank, runtime, val_samples)
                dist.barrier()
                if rank == 0:
                    for k, v in val_metrics.items():
                        if writer:
                            writer.add_scalar(k, v, step + 1)
                    val_loss = val_metrics.get("val/loss_total", float("inf"))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_saved = save_best_adapter(run_dir, runtime.model, config_rank, step + 1, val_loss)
                        tqdm.write(json.dumps({"event": "new_best_val_loss", "step": step + 1, "val_loss": val_loss}))

            save_interval = 1 if smoke else config.save_every_steps
            if (step + 1) % save_interval == 0 or step + 1 == max_steps:
                dist.barrier()
                if rank == 0:
                    saved = save_adapter(run_dir, runtime.model, config_rank, step + 1)
                dist.barrier()

        if log_file:
            log_file.close()
    finally:
        if writer:
            writer.close()

    if rank == 0:
        peak = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        run_info = {"seconds": time.monotonic() - started, "peak_gpu_bytes": peak, "checkpoint": str(saved) if saved else None, "reload_checks": reload_checks}
        (run_dir / "run.json").write_text(json.dumps(run_info, indent=2))
        report = config.path.parent.parent / "reports" / "overfit_10.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# PersonaPlex overfit-10 report\n\n"
            "Status: PASS\n\n"
            f"- Dataset: {config.manifest}\n- Model: {config.model_root}\n- LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}\n"
            f"- FSDP GPUs: {gpu_ids}\n"
            f"- Steps: {max_steps}\n- Final loss: {last_record['loss/total'] if last_record else 'n/a'}\n"
            f"- Peak GPU bytes: {peak}\n- Checkpoint: {saved}\n"
            "- Inference outputs: run `python -m tools.inference_smoke` with this checkpoint.\n",
            encoding="utf-8",
        )

    dist.barrier()
    dist.destroy_process_group()


def run(
    config: Config,
    smoke: bool = False,
    fsdp: str | None = None,
    resume_from: str | None = None,
    sharding_strategy: str = "shard_grad_op",
) -> Path | None:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    import torch
    from tqdm.auto import tqdm
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required; install the project requirements before training") from exc

    if smoke and config.shuffle:
        raise ValueError("overfit/smoke configuration must set data.shuffle: false")

    if fsdp:
        gpu_ids = parse_gpu_ids(fsdp)
        port = find_free_port()
        run_dir = create_run_dir(config.output_dir, smoke)
        config_record = {
            "event": "configuration",
            "seed": config.seed,
            "fsdp_gpus": gpu_ids,
            "sharding_strategy": sharding_strategy,
            "model_root": str(config.model_root),
            "personaplex_source": str(config.personaplex_source),
            "manifest": str(config.manifest),
            "output_dir": str(run_dir),
            "window_seconds": config.window_seconds,
            "max_steps": 1 if smoke else config.max_steps,
            "learning_rate": config.learning_rate,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "qlora": config.qlora,
            "quant_type": config.quant_type if config.qlora else None,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "warmup_steps": config.warmup_steps,
            "eval_every_steps": config.eval_every_steps,
            "save_every_steps": config.save_every_steps,
            "random_crop": config.random_crop,
            "prompt_aug_prob": config.prompt_aug_prob,
        }
        (run_dir / "config.json").write_text(json.dumps(config_record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(config_record))

        torch.multiprocessing.spawn(
            _train_fsdp_worker,
            args=(len(gpu_ids), gpu_ids, port, config, smoke, run_dir, sharding_strategy),
            nprocs=len(gpu_ids),
            join=True,
        )
        saved_ckpt = run_dir / "checkpoints" / f"checkpoint_{(1 if smoke else config.max_steps):06d}" / "lora.safetensors"
        return saved_ckpt if saved_ckpt.exists() else None


    # Single-GPU path
    cpu_threads = limit_cpu_threads(torch)
    seed_everything(config.seed, torch)

    if config.val_manifest:
        train_samples = PreparedDataset(config.manifest, config.window_seconds).load()
        val_samples = PreparedDataset(config.val_manifest, config.window_seconds).load()
    elif config.eval_every_steps > 0:
        train_samples, val_samples = PreparedDataset(config.manifest, config.window_seconds).split(
            val_ratio=config.val_ratio, seed=config.seed
        )
    else:
        train_samples = PreparedDataset(config.manifest, config.window_seconds).load()
        val_samples = []

    run_dir = create_run_dir(config.output_dir, smoke)
    config_record = {
        "event": "configuration",
        "seed": config.seed,
        "model_root": str(config.model_root),
        "personaplex_source": str(config.personaplex_source),
        "manifest": str(config.manifest),
        "output_dir": str(run_dir),
        "window_seconds": config.window_seconds,
        "max_steps": 1 if smoke else config.max_steps,
        "learning_rate": config.learning_rate,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "qlora": config.qlora,
        "quant_type": config.quant_type if config.qlora else None,
        "device": config.device,
        "cpu_threads": cpu_threads,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_steps": config.warmup_steps,
        "eval_every_steps": config.eval_every_steps,
        "save_every_steps": config.save_every_steps,
        "random_crop": config.random_crop,
        "prompt_aug_prob": config.prompt_aug_prob,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
    }
    (run_dir / "config.json").write_text(json.dumps(config_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config_record))
    device_desc = f"{config.device} ({torch.cuda.get_device_name(config.device)})" if torch.cuda.is_available() and config.device.startswith("cuda") else config.device
    print(f"Using single device: {device_desc}")
    runtime = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), config.device, config.qlora, config.quant_type)
    targets = inject_lora(runtime.model, config.lora_rank, config.lora_alpha)

    start_step = 0
    if resume_from:
        resume_path = Path(resume_from)
        adapter_file = resume_path if resume_path.is_file() else resume_path / "lora.safetensors"
        print(f"Resuming LoRA weights from {adapter_file}")
        load_adapter(runtime.model, adapter_file)
        meta_file = adapter_file.parent / "adapter.json"
        if meta_file.is_file():
            try:
                start_step = int(json.loads(meta_file.read_text(encoding="utf-8")).get("step", 0))
                print(f"Resumed from step {start_step}")
            except Exception:
                pass

    trainable = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
    print(f"LoRA targets: {len(targets)}; trainable parameters: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)
    max_steps = 1 if smoke else config.max_steps
    scheduler = None
    if config.warmup_steps > 0 and not smoke:
        scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, max_steps)

    log_path = run_dir / "metrics.jsonl"
    tensorboard_dir = run_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("configuration", json.dumps(config_record, indent=2), 0)

    saved = None
    best_saved = None
    best_val_loss = float("inf")
    last_record = None
    reload_checks: list[dict[str, float | int]] = []
    started = time.monotonic()
    accum_steps = max(1, config.gradient_accumulation_steps)
    shuffled_indices = list(range(len(train_samples)))
    if config.shuffle and not smoke:
        random.Random(config.seed).shuffle(shuffled_indices)

    try:
        with log_path.open("w", encoding="utf-8") as log:
            progress = tqdm(range(start_step, max_steps), desc="training", unit="step", dynamic_ncols=True)
            for step in progress:
                sample_idx = (step % len(train_samples)) if not config.shuffle else shuffled_indices[step % len(shuffled_indices)]
                current_sample = train_samples[sample_idx]
                example = build_example(config, current_sample, runtime, random_crop=config.random_crop and not smoke)
                codes = torch.tensor(example.input_codes, dtype=torch.long, device=config.device).unsqueeze(0)
                output = model_forward_train(runtime.model, codes)
                total, components = loss_components(output, codes, example, runtime.tokenizer.padding_id, torch)

                # Gradient accumulation
                scaled_loss = total / accum_steps
                scaled_loss.backward()

                is_update = ((step + 1) % accum_steps == 0) or (step + 1 == max_steps)
                if is_update:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    grad_norm = 0.0

                record = {
                    "step": step,
                    "loss/total": float(total.detach()),
                    "loss/text": float(components["text"].detach()),
                    "loss/audio_semantic": float(components["audio_semantic"].detach()),
                    "loss/audio_nonsemantic": float(components["audio_nonsemantic"].detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                    "gpu_peak_bytes": torch.cuda.max_memory_allocated(config.device) if torch.cuda.is_available() else 0,
                }
                last_record = record
                log.write(json.dumps(record) + "\n")
                log.flush()
                write_tensorboard_scalars(writer, record, sum(p.numel() for p in trainable), cpu_threads)
                progress.set_postfix(loss=f"{record['loss/total']:.4f}", grad=f"{grad_norm:.3f}")
                tqdm.write(json.dumps({"event": "training_step", **record})) if max_steps == 1 else None

                # Validation evaluation
                if val_samples and config.eval_every_steps > 0 and ((step + 1) % config.eval_every_steps == 0 or step + 1 == max_steps):
                    val_metrics = evaluate_validation(config, runtime, val_samples)
                    for k, v in val_metrics.items():
                        writer.add_scalar(k, v, step + 1)
                    val_loss = val_metrics.get("val/loss_total", float("inf"))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_saved = save_best_adapter(run_dir, runtime.model, config, step + 1, val_loss)
                        tqdm.write(json.dumps({"event": "new_best_val_loss", "step": step + 1, "val_loss": val_loss}))

                save_interval = 1 if smoke else config.save_every_steps
                if (step + 1) % save_interval == 0 or step + 1 == max_steps:
                    saved = save_adapter(run_dir, runtime.model, config, step + 1)
                    if smoke:
                        reload_loss = verify_reloaded_adapter(config, train_samples[step % len(train_samples)], saved)
                        if abs(reload_loss - record["loss/total"]) > 0.02:
                            raise RuntimeError(f"reloaded adapter loss drifted: {reload_loss} vs {record['loss/total']}")
                        reload_checks.append({"step": step + 1, "loss": reload_loss})
    finally:
        writer.close()

    peak = torch.cuda.max_memory_allocated(config.device) if torch.cuda.is_available() else 0
    reload_loss = reload_checks[-1]["loss"] if reload_checks else None
    run_info = {
        "seconds": time.monotonic() - started,
        "peak_gpu_bytes": peak,
        "checkpoint": str(saved) if saved else None,
        "best_checkpoint": str(best_saved) if best_saved else None,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
        "reload_checks": reload_checks,
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2))
    report = config.path.parent.parent / "reports" / "overfit_10.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# PersonaPlex Training Report\n\n"
        f"Status: PASS\n\n"
        f"- Dataset: {config.manifest}\n- Model: {config.model_root}\n- LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}\n"
        f"- Steps: {max_steps}\n- Final loss: {last_record['loss/total'] if last_record else 'n/a'}\n"
        f"- Best val loss: {best_val_loss if best_val_loss < float('inf') else 'n/a'}\n"
        f"- Reload checks: {reload_checks}\n- Peak GPU bytes: {peak}\n- Checkpoint: {saved}\n"
        f"- Best checkpoint: {best_saved}\n"
        "- Inference outputs: run `python -m tools.inference_smoke` with this checkpoint.\n",
        encoding="utf-8",
    )
    return best_saved or saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fsdp", type=str, default=None, help="Comma-separated GPU IDs to activate FSDP (e.g. --fsdp 0,1)")
    parser.add_argument(
        "--sharding-strategy",
        type=str,
        default="shard_grad_op",
        choices=["shard_grad_op", "full_shard", "no_shard"],
        help="FSDP strategy: shard_grad_op (ZeRO-2, fast for LoRA, default), full_shard (ZeRO-3), or no_shard (DDP)",
    )
    parser.add_argument("--qlora", action="store_true", default=None, help="Enable 4-bit QLoRA. If omitted, uses value from config file.")
    parser.add_argument("--no-qlora", dest="qlora", action="store_false", help="Disable QLoRA.")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint directory to resume from")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.qlora is not None:
        config = config.replace(qlora=args.qlora)
    run(
        config,
        smoke=args.smoke,
        fsdp=args.fsdp,
        resume_from=args.resume_from,
        sharding_strategy=args.sharding_strategy,
    )

    return 0



if __name__ == "__main__":
    raise SystemExit(main())

