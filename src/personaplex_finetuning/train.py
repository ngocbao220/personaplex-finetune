"""Single-GPU, intentionally small LoRA training path."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .config import Config, load_config
from .data import PreparedDataset
from .lora import adapter_state_dict, inject_lora, load_adapter
from .objective import stream_weights, torch_weighted_cross_entropy
from .runtime import RuntimePaths, load_runtime
from .sequence import PersonaPlexTrainingExampleBuilder


def seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_example(config: Config, sample, runtime):
    builder = PersonaPlexTrainingExampleBuilder(runtime.codec, runtime.tokenizer, runtime.initial_tokens, runtime.zero_token)
    return builder.apply_delays(builder.build(sample), runtime.delays)


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
    output = runtime.model(codes)
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


def save_adapter(run_dir: Path, model, config: Config, step: int) -> Path:
    from safetensors.torch import save_file
    path = run_dir / "checkpoints" / f"checkpoint_{step:06d}"
    path.mkdir(parents=True, exist_ok=False)
    adapter = path / "lora.safetensors"
    save_file(adapter_state_dict(model), str(adapter))
    (path / "adapter.json").write_text(json.dumps({"step": step, "model_root": str(config.model_root), "rank": config.lora_rank, "alpha": config.lora_alpha}, indent=2))
    return adapter


def verify_reloaded_adapter(config: Config, sample, adapter: Path) -> float:
    """Load base + adapter in a fresh model object and return teacher-forced loss."""
    import torch
    fresh = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), config.device)
    inject_lora(fresh.model, config.lora_rank, config.lora_alpha)
    load_adapter(fresh.model, adapter)
    fresh.model.eval()
    example = build_example(config, sample, fresh)
    with torch.no_grad():
        total, _, _ = one_step(config, fresh, example)
    return float(total)


def run(config: Config, smoke: bool = False) -> Path | None:
    import torch
    seed_everything(config.seed, torch)
    if config.shuffle:
        raise ValueError("overfit configuration must set data.shuffle: false")
    samples = PreparedDataset(config.manifest, config.window_seconds).load()
    runtime = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), config.device)
    targets = inject_lora(runtime.model, config.lora_rank, config.lora_alpha)
    trainable = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
    print(f"LoRA targets: {len(targets)}; trainable parameters: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    log_path = config.output_dir / "metrics.jsonl"
    max_steps = 1 if smoke else config.max_steps
    saved = None
    last_record = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(max_steps):
            example = build_example(config, samples[step % len(samples)], runtime)
            total, components, grad_norm = one_step(config, runtime, example, optimizer)
            record = {"step": step, "loss/total": float(total.detach()), "loss/text": float(components["text"].detach()), "loss/audio_semantic": float(components["audio_semantic"].detach()), "loss/audio_nonsemantic": float(components["audio_nonsemantic"].detach()), "lr": optimizer.param_groups[0]["lr"], "grad_norm": grad_norm}
            last_record = record
            log.write(json.dumps(record) + "\n")
            print(json.dumps(record))
            if (step + 1) % 50 == 0 or step + 1 == max_steps:
                saved = save_adapter(config.output_dir, runtime.model, config, step + 1)
    peak = torch.cuda.max_memory_allocated(config.device) if torch.cuda.is_available() else 0
    reload_loss = verify_reloaded_adapter(config, samples[(max_steps - 1) % len(samples)], saved) if saved else None
    if reload_loss is not None and last_record is not None and abs(reload_loss - last_record["loss/total"]) > 1e-3:
        raise RuntimeError(f"reloaded adapter loss drifted: {reload_loss} vs {last_record['loss/total']}")
    run_info = {"seconds": time.monotonic() - started, "peak_gpu_bytes": peak, "checkpoint": str(saved) if saved else None, "reload_loss": reload_loss}
    (config.output_dir / "run.json").write_text(json.dumps(run_info, indent=2))
    report = config.path.parent.parent / "reports" / "overfit_10.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# PersonaPlex overfit-10 report\n\n"
        "Status: PASS\n\n"
        f"- Dataset: {config.manifest}\n- Model: {config.model_root}\n- LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}\n"
        f"- Steps: {max_steps}\n- Final loss: {last_record['loss/total'] if last_record else 'n/a'}\n"
        f"- Reload loss: {reload_loss}\n- Peak GPU bytes: {peak}\n- Checkpoint: {saved}\n"
        "- Inference outputs: run `python -m tools.inference_smoke` with this checkpoint.\n",
        encoding="utf-8",
    )
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config), smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
