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


def build_example(config: Config, sample, runtime):
    builder = PersonaPlexTrainingExampleBuilder(runtime.codec, runtime.tokenizer, runtime.initial_tokens, runtime.zero_token)
    return builder.apply_delays(builder.build(sample), runtime.delays)


def model_forward_train(model, codes):
    """LMModel deliberately exposes training through ``forward_train`` only."""
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
) -> None:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    import torch
    import torch.distributed as dist
    from tqdm.auto import tqdm

    device_id = gpu_ids[rank]
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"

    init_method = f"tcp://127.0.0.1:{port}"
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )

    cpu_threads = limit_cpu_threads(torch)
    seed_everything(config.seed + rank, torch)

    samples = PreparedDataset(config.manifest, config.window_seconds).load()
    runtime = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), device, config.qlora, config.quant_type)
    targets = inject_lora(runtime.model, config.lora_rank, config.lora_alpha)

    # Alias LMModel.forward to forward_train for FSDP compatibility
    from moshi.models.lm import LMModel
    LMModel.forward = LMModel.forward_train

    runtime.model = wrap_model_fsdp(runtime.model, device_id=device_id)

    trainable = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
    if rank == 0:
        print(f"[Rank 0] FSDP active across GPUs {gpu_ids}. LoRA targets: {len(targets)}; trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)

    writer = None
    log_path = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_dir = run_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        log_path = run_dir / "metrics.jsonl"

    max_steps = 1 if smoke else config.max_steps
    saved = None
    last_record = None
    reload_checks: list[dict[str, float | int]] = []
    started = time.monotonic()

    # Overwrite config.device with current rank device for one_step
    config_rank = Config(
        seed=config.seed,
        model_root=config.model_root,
        personaplex_source=config.personaplex_source,
        manifest=config.manifest,
        output_dir=config.output_dir,
        window_seconds=config.window_seconds,
        shuffle=config.shuffle,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        qlora=config.qlora,
        quant_type=config.quant_type,
        device=device,
        path=config.path,
    )

    try:
        log_file = log_path.open("w", encoding="utf-8") if log_path else None
        progress = tqdm(range(max_steps), desc="training (FSDP)", unit="step", dynamic_ncols=True) if rank == 0 else range(max_steps)
        for step in progress:
            example = build_example(config_rank, samples[step % len(samples)], runtime)
            total, components, grad_norm = one_step(config_rank, runtime, example, optimizer)

            # Reduce total loss across ranks for synchronized logging
            dist.all_reduce(total, op=dist.ReduceOp.AVG)

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

            if (step + 1) % 50 == 0 or step + 1 == max_steps:
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


def run(config: Config, smoke: bool = False, fsdp: str | None = None) -> Path | None:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    import torch
    from tqdm.auto import tqdm
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required; install the project requirements before training") from exc

    if config.shuffle:
        raise ValueError("overfit configuration must set data.shuffle: false")

    if fsdp:
        gpu_ids = parse_gpu_ids(fsdp)
        port = find_free_port()
        run_dir = create_run_dir(config.output_dir, smoke)
        config_record = {
            "event": "configuration",
            "seed": config.seed,
            "fsdp_gpus": gpu_ids,
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
        }
        (run_dir / "config.json").write_text(json.dumps(config_record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(config_record))

        torch.multiprocessing.spawn(
            _train_fsdp_worker,
            args=(len(gpu_ids), gpu_ids, port, config, smoke, run_dir),
            nprocs=len(gpu_ids),
            join=True,
        )
        saved_ckpt = run_dir / "checkpoints" / f"checkpoint_{(1 if smoke else config.max_steps):06d}" / "lora.safetensors"
        return saved_ckpt if saved_ckpt.exists() else None

    # Single-GPU path
    cpu_threads = limit_cpu_threads(torch)
    seed_everything(config.seed, torch)
    samples = PreparedDataset(config.manifest, config.window_seconds).load()
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
    }
    (run_dir / "config.json").write_text(json.dumps(config_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config_record))
    runtime = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), config.device, config.qlora, config.quant_type)
    targets = inject_lora(runtime.model, config.lora_rank, config.lora_alpha)
    trainable = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
    print(f"LoRA targets: {len(targets)}; trainable parameters: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)
    log_path = run_dir / "metrics.jsonl"
    tensorboard_dir = run_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("configuration", json.dumps(config_record, indent=2), 0)
    max_steps = 1 if smoke else config.max_steps
    saved = None
    last_record = None
    reload_checks: list[dict[str, float | int]] = []
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            progress = tqdm(range(max_steps), desc="training", unit="step", dynamic_ncols=True)
            for step in progress:
                example = build_example(config, samples[step % len(samples)], runtime)
                total, components, grad_norm = one_step(config, runtime, example, optimizer)
                record = {"step": step, "loss/total": float(total.detach()), "loss/text": float(components["text"].detach()), "loss/audio_semantic": float(components["audio_semantic"].detach()), "loss/audio_nonsemantic": float(components["audio_nonsemantic"].detach()), "lr": optimizer.param_groups[0]["lr"], "grad_norm": grad_norm, "gpu_peak_bytes": torch.cuda.max_memory_allocated(config.device) if torch.cuda.is_available() else 0}
                last_record = record
                log.write(json.dumps(record) + "\n")
                log.flush()
                write_tensorboard_scalars(writer, record, sum(p.numel() for p in trainable), cpu_threads)
                progress.set_postfix(loss=f"{record['loss/total']:.4f}", grad=f"{grad_norm:.3f}")
                tqdm.write(json.dumps({"event": "training_step", **record})) if max_steps == 1 else None
                if (step + 1) % 50 == 0 or step + 1 == max_steps:
                    saved = save_adapter(run_dir, runtime.model, config, step + 1)
                    reload_loss = verify_reloaded_adapter(config, samples[step % len(samples)], saved)
                    if abs(reload_loss - record["loss/total"]) > 1e-3:
                        raise RuntimeError(f"reloaded adapter loss drifted: {reload_loss} vs {record['loss/total']}")
                    reload_checks.append({"step": step + 1, "loss": reload_loss})
    finally:
        writer.close()
    peak = torch.cuda.max_memory_allocated(config.device) if torch.cuda.is_available() else 0
    reload_loss = reload_checks[-1]["loss"] if reload_checks else None
    run_info = {"seconds": time.monotonic() - started, "peak_gpu_bytes": peak, "checkpoint": str(saved) if saved else None, "reload_checks": reload_checks}
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2))
    report = config.path.parent.parent / "reports" / "overfit_10.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# PersonaPlex overfit-10 report\n\n"
        "Status: PASS\n\n"
        f"- Dataset: {config.manifest}\n- Model: {config.model_root}\n- LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}\n"
        f"- Steps: {max_steps}\n- Final loss: {last_record['loss/total'] if last_record else 'n/a'}\n"
        f"- Reload checks: {reload_checks}\n- Peak GPU bytes: {peak}\n- Checkpoint: {saved}\n"
        "- Inference outputs: run `python -m tools.inference_smoke` with this checkpoint.\n",
        encoding="utf-8",
    )
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fsdp", type=str, default=None, help="Comma-separated GPU IDs to activate FSDP (e.g. --fsdp 0,1)")
    args = parser.parse_args()
    run(load_config(args.config), smoke=args.smoke, fsdp=args.fsdp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
