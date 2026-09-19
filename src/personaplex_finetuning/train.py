"""PersonaPlex LoRA training path supporting Accelerate, DDP, BF16, and single-GPU execution."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file
from tqdm.auto import tqdm

from .config import Config, load_config
from .data import PreparedDataset
from .lora import adapter_state_dict, inject_lora, load_adapter
from .objective import stream_weights_torch, torch_weighted_cross_entropy
from .runtime import RuntimePaths, load_runtime
from .sequence import PersonaPlexTrainingExampleBuilder


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
    """LMModel exposes training through forward_train."""
    if hasattr(model, "forward_train"):
        return model.forward_train(codes)
    return model(codes)


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


def loss_components(model_output, codes, example, text_padding_id, torch_module):
    """Compute per-stream losses using GPU-native vectorized weights."""
    labels_tensor = torch_module.tensor(example.labels, dtype=torch_module.long, device=codes.device)
    mask_tensor = torch_module.tensor(example.loss_mask, dtype=torch_module.bool, device=codes.device)
    weights = stream_weights_torch(labels_tensor, mask_tensor, text_padding_id)

    text_target = codes[:, 0, :]
    text_weight = weights[0].unsqueeze(0) * model_output.text_mask.to(weights.dtype)
    text_loss = torch_weighted_cross_entropy(
        model_output.text_logits.reshape(-1, model_output.text_logits.shape[-1]),
        text_target.reshape(-1),
        text_weight.reshape(-1),
    )

    audio_target = codes[:, 1:17, :]
    audio_weights = weights[1:17].unsqueeze(0) * model_output.mask.to(weights.dtype)

    semantic = torch_weighted_cross_entropy(
        model_output.logits[:, 0].reshape(-1, model_output.logits.shape[-1]),
        audio_target[:, 0].reshape(-1),
        audio_weights[:, 0].reshape(-1),
    )
    nonsemantic = torch_weighted_cross_entropy(
        model_output.logits[:, 1:8].reshape(-1, model_output.logits.shape[-1]),
        audio_target[:, 1:8].reshape(-1),
        audio_weights[:, 1:8].reshape(-1),
    )
    return text_loss + semantic + nonsemantic, {"text": text_loss, "audio_semantic": semantic, "audio_nonsemantic": nonsemantic}


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


def save_adapter(run_dir: Path, model, config: Config, step: int) -> Path:
    path = run_dir / "checkpoints" / f"checkpoint_{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    adapter = path / "lora.safetensors"
    save_file(adapter_state_dict(model), str(adapter))
    (path / "adapter.json").write_text(
        json.dumps(
            {"step": step, "model_root": str(config.model_root), "rank": config.lora_rank, "alpha": config.lora_alpha},
            indent=2,
        )
    )
    return adapter


def save_best_adapter(run_dir: Path, model, config: Config, step: int, val_loss: float) -> Path:
    path = run_dir / "checkpoints" / "best"
    path.mkdir(parents=True, exist_ok=True)
    adapter = path / "lora.safetensors"
    save_file(adapter_state_dict(model), str(adapter))
    (path / "adapter.json").write_text(
        json.dumps(
            {"step": step, "val_loss": val_loss, "model_root": str(config.model_root), "rank": config.lora_rank, "alpha": config.lora_alpha},
            indent=2,
        )
    )
    return adapter


def evaluate_validation(config: Config, runtime, val_samples: list, accelerator: Accelerator) -> dict[str, float]:
    if not val_samples:
        return {}
    unwrapped = accelerator.unwrap_model(runtime.model)
    unwrapped.eval()
    total_losses = []
    text_losses = []
    semantic_losses = []
    nonsemantic_losses = []
    device = accelerator.device
    with torch.no_grad():
        for sample in val_samples:
            example = build_example(config, sample, runtime, random_crop=False)
            codes = torch.tensor(example.input_codes, dtype=torch.long, device=device).unsqueeze(0)
            output = unwrapped(codes)
            total, comps = loss_components(output, codes, example, runtime.tokenizer.padding_id, torch)
            reduced_total = accelerator.reduce(total, reduction="mean")
            reduced_text = accelerator.reduce(comps["text"], reduction="mean")
            reduced_sem = accelerator.reduce(comps["audio_semantic"], reduction="mean")
            reduced_nonsem = accelerator.reduce(comps["audio_nonsemantic"], reduction="mean")
            total_losses.append(float(reduced_total.detach()))
            text_losses.append(float(reduced_text.detach()))
            semantic_losses.append(float(reduced_sem.detach()))
            nonsemantic_losses.append(float(reduced_nonsem.detach()))
    unwrapped.train()
    return {
        "val/loss_total": sum(total_losses) / len(total_losses),
        "val/loss_text": sum(text_losses) / len(text_losses),
        "val/loss_audio_semantic": sum(semantic_losses) / len(semantic_losses),
        "val/loss_audio_nonsemantic": sum(nonsemantic_losses) / len(nonsemantic_losses),
    }


def verify_reloaded_adapter(config: Config, sample, adapter: Path) -> float:
    """Load base + adapter in a fresh model object and return teacher-forced loss."""
    fresh = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), config.device, config.qlora, config.quant_type)
    inject_lora(fresh.model, config.lora_rank, config.lora_alpha)
    load_adapter(fresh.model, adapter)
    fresh.model.eval()
    example = build_example(config, sample, fresh)
    with torch.no_grad():
        total, _, _ = one_step(config, fresh, example)
    return float(total)


def run(
    config: Config,
    smoke: bool = False,
    resume_from: str | None = None,
) -> Path | None:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required; install the project requirements before training") from exc

    if smoke and config.shuffle:
        raise ValueError("overfit/smoke configuration must set data.shuffle: false")

    accum_steps = max(1, config.gradient_accumulation_steps)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision if torch.cuda.is_available() else "no",
        gradient_accumulation_steps=accum_steps,
    )
    device = accelerator.device
    cpu_threads = limit_cpu_threads(torch)
    set_seed(config.seed + accelerator.process_index)

    # Dataset loading
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

    # Run directory creation (coordinated across ranks)
    run_dir = None
    if accelerator.is_main_process:
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
            "device": str(device),
            "num_processes": accelerator.num_processes,
            "cpu_threads": cpu_threads,
            "gradient_accumulation_steps": accum_steps,
            "warmup_steps": config.warmup_steps,
            "eval_every_steps": config.eval_every_steps,
            "save_every_steps": config.save_every_steps,
            "random_crop": config.random_crop,
            "prompt_aug_prob": config.prompt_aug_prob,
            "gradient_checkpointing": config.gradient_checkpointing,
            "mixed_precision": config.mixed_precision,
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
        }
        (run_dir / "config.json").write_text(json.dumps(config_record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(config_record))
        print(f"[Main Process] Active across {accelerator.num_processes} process(es) on device {device}")

    if accelerator.num_processes > 1:
        run_dir_list = [str(run_dir) if accelerator.is_main_process else ""]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(run_dir_list, src=0)
        run_dir = Path(run_dir_list[0])

    # Sequential model loading across ranks to avoid host CPU RAM spikes
    for i in range(accelerator.num_processes):
        if accelerator.process_index == i:
            runtime = load_runtime(
                RuntimePaths(config.model_root, config.personaplex_source),
                str(device),
                config.qlora,
                config.quant_type,
            )
        accelerator.wait_for_everyone()

    # Inject LoRA
    targets = inject_lora(runtime.model, config.lora_rank, config.lora_alpha)

    # Optional gradient checkpointing
    if config.gradient_checkpointing:
        enable_gradient_checkpointing(runtime.model)
        if accelerator.is_main_process:
            print("Gradient checkpointing enabled on transformer layers.")

    # Resume checkpoint if provided
    start_step = 0
    if resume_from:
        resume_path = Path(resume_from)
        adapter_file = resume_path if resume_path.is_file() else resume_path / "lora.safetensors"
        if accelerator.is_main_process:
            print(f"Resuming LoRA weights from {adapter_file}")
        load_adapter(runtime.model, adapter_file)
        meta_file = adapter_file.parent / "adapter.json"
        if meta_file.is_file():
            try:
                start_step = int(json.loads(meta_file.read_text(encoding="utf-8")).get("step", 0))
                if accelerator.is_main_process:
                    print(f"Resumed from step {start_step}")
            except Exception:
                pass

    # Alias LMModel.forward to forward_train for training execution
    from moshi.models.lm import LMModel
    LMModel.forward = LMModel.forward_train

    trainable = [p for p in runtime.model.parameters() if p.requires_grad]
    if accelerator.is_main_process:
        print(f"LoRA targets: {len(targets)}; trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)
    max_steps = 1 if smoke else config.max_steps
    scheduler = None
    if config.warmup_steps > 0 and not smoke:
        scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, max_steps)

    # Prepare model, optimizer, scheduler with Accelerator (DDP wrapping)
    runtime.model, optimizer = accelerator.prepare(runtime.model, optimizer)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    writer = None
    log_file = None
    if accelerator.is_main_process and run_dir is not None:
        log_path = run_dir / "metrics.jsonl"
        log_file = log_path.open("w", encoding="utf-8")
        tensorboard_dir = run_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        writer.add_text("configuration", json.dumps(config_record, indent=2), 0)

    saved = None
    best_saved = None
    best_val_loss = float("inf")
    last_record = None
    reload_checks: list[dict[str, float | int]] = []
    started = time.monotonic()

    shuffled_indices = list(range(len(train_samples)))
    if config.shuffle and not smoke:
        random.Random(config.seed).shuffle(shuffled_indices)

    try:
        progress = (
            tqdm(range(start_step, max_steps), desc="training (DDP)", unit="step", dynamic_ncols=True)
            if accelerator.is_main_process
            else range(start_step, max_steps)
        )

        for step in progress:
            global_sample_idx = (step * accelerator.num_processes + accelerator.process_index) % len(train_samples)
            sample_idx = global_sample_idx if not config.shuffle else shuffled_indices[global_sample_idx % len(shuffled_indices)]
            current_sample = train_samples[sample_idx]
            example = build_example(config, current_sample, runtime, random_crop=config.random_crop and not smoke)
            codes = torch.tensor(example.input_codes, dtype=torch.long, device=device).unsqueeze(0)

            with accelerator.accumulate(runtime.model):
                output = runtime.model(codes)
                total, components = loss_components(output, codes, example, runtime.tokenizer.padding_id, torch)
                accelerator.backward(total)

                if accelerator.sync_gradients:
                    grad_norm = float(accelerator.clip_grad_norm_(trainable, 1.0))
                else:
                    grad_norm = 0.0

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            reduced_total = accelerator.reduce(total, reduction="mean")
            reduced_text = accelerator.reduce(components["text"], reduction="mean")
            reduced_sem = accelerator.reduce(components["audio_semantic"], reduction="mean")
            reduced_nonsem = accelerator.reduce(components["audio_nonsemantic"], reduction="mean")

            if accelerator.is_main_process:
                record = {
                    "step": step,
                    "loss/total": float(reduced_total.detach()),
                    "loss/text": float(reduced_text.detach()),
                    "loss/audio_semantic": float(reduced_sem.detach()),
                    "loss/audio_nonsemantic": float(reduced_nonsem.detach()),
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
                accelerator.wait_for_everyone()
                val_metrics = evaluate_validation(config, runtime, val_samples, accelerator)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    for k, v in val_metrics.items():
                        if writer:
                            writer.add_scalar(k, v, step + 1)
                    val_loss = val_metrics.get("val/loss_total", float("inf"))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        unwrapped = accelerator.unwrap_model(runtime.model)
                        best_saved = save_best_adapter(run_dir, unwrapped, config, step + 1, val_loss)
                        tqdm.write(json.dumps({"event": "new_best_val_loss", "step": step + 1, "val_loss": val_loss}))

            # Periodic saving & smoke reload verification
            save_interval = 1 if smoke else config.save_every_steps
            if (step + 1) % save_interval == 0 or step + 1 == max_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(runtime.model)
                    saved = save_adapter(run_dir, unwrapped, config, step + 1)
                    if smoke:
                        reload_loss = verify_reloaded_adapter(config, train_samples[step % len(train_samples)], saved)
                        if abs(reload_loss - record["loss/total"]) > 0.05:
                            raise RuntimeError(f"reloaded adapter loss drifted: {reload_loss} vs {record['loss/total']}")
                        reload_checks.append({"step": step + 1, "loss": reload_loss})
                accelerator.wait_for_everyone()

        if log_file:
            log_file.close()
    finally:
        if writer:
            writer.close()

    if accelerator.is_main_process and run_dir is not None:
        peak = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
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
            "Status: PASS\n\n"
            f"- Dataset: {config.manifest}\n- Model: {config.model_root}\n- LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}\n"
            f"- Processes (GPUs): {accelerator.num_processes}\n"
            f"- Steps: {max_steps}\n- Final loss: {last_record['loss/total'] if last_record else 'n/a'}\n"
            f"- Best val loss: {best_val_loss if best_val_loss < float('inf') else 'n/a'}\n"
            f"- Peak GPU bytes: {peak}\n- Checkpoint: {saved}\n"
            f"- Best checkpoint: {best_saved}\n"
            "- Inference outputs: run `python -m tools.inference_smoke` with this checkpoint.\n",
            encoding="utf-8",
        )

    accelerator.wait_for_everyone()
    return best_saved or saved


def main() -> int:
    parser = argparse.ArgumentParser(description="PersonaPlex Fine-Tuning with Accelerate and DDP")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON configuration file")
    parser.add_argument("--config-name", type=str, default=None, help="Name of config in configs/ directory (Hydra style)")
    parser.add_argument("--smoke", action="store_true", help="Run 1-step smoke test with verification")
    parser.add_argument("--qlora", action="store_true", default=None, help="Enable 4-bit QLoRA")
    parser.add_argument("--no-qlora", dest="qlora", action="store_false", help="Disable QLoRA")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint directory to resume from")

    args, unknown = parser.parse_known_args()
    overrides = [arg for arg in unknown if "=" in arg]

    config_path = args.config
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
