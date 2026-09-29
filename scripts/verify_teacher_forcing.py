#!/usr/bin/env python3
"""Teacher-Forcing Accuracy & Loss Diagnostic Tool for PersonaPlex.

Runs Teacher-Forcing forward on one exact training sample and evaluates:
1. Valid text tokens vs padding tokens %
2. Text Token Accuracy (Argmax Logits vs Ground Truth)
3. Audio Codebook 0-7 Accuracy (Argmax Logits vs Ground Truth)
4. Per-codebook Cross-Entropy Loss
5. Overall Audio & Text Loss

If model is truly overfitted, Text Acc and all 8 Audio Codebook Accuracies should be ~99-100%.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project src to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
from rich.console import Console
from rich.table import Table

from personaplex_finetuning.config import load_config
from personaplex_finetuning.data import PreparedDataset, duration_chunks, limit_conversations
from personaplex_finetuning.lora import inject_lora, load_adapter
from personaplex_finetuning.runtime import RuntimePaths, load_runtime
from personaplex_finetuning.sequence import PersonaPlexTrainingExampleBuilder, pad_training_example
from personaplex_finetuning.batching import post_encode_collate
from personaplex_finetuning.train import (
    codebook_diagnostic_stats,
    loss_components,
    model_forward_train,
    tokenizer_text_padding_ids,
)


def evaluate_teacher_forcing_sample(
    config_path: Path,
    adapter_path: Path | None = None,
    sample_index: int = 0,
    device: str = "cuda",
) -> dict:
    console = Console()
    console.print(f"\n[bold cyan]=== PersonaPlex Teacher-Forcing Diagnostic Test ===[/bold cyan]")
    console.print(f"Config: [yellow]{config_path}[/yellow]")
    console.print(f"Adapter: [yellow]{adapter_path or 'Base Model (None)'}[/yellow]")
    console.print(f"Sample Index: [green]{sample_index}[/green]\n")

    config = load_config(config_path)
    device_obj = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    # 1. Load dataset & extract exact sample
    console.print("[dim]Loading dataset...[/dim]")
    dataset = PreparedDataset(config.manifest, config.window_seconds).load()
    conversations = limit_conversations(dataset, sample_index=sample_index)
    chunks = duration_chunks(conversations, config.duration_sec)
    sample = chunks[0]  # First 100s window

    console.print(f"Selected Sample ID: [bold]{sample.sample_id}[/bold]")
    console.print(f"Window: [dim]{sample.window_start_sec:.2f}s - {sample.window_end_sec:.2f}s[/dim]")
    console.print(f"Text Prompt: [italic]{sample.text_prompt[:80]}...[/italic]\n")

    # 2. Load runtime & weights
    console.print("[dim]Loading PersonaPlex model and Mimi Codec...[/dim]")
    runtime = load_runtime(
        RuntimePaths(config.model_root, config.personaplex_source),
        str(device_obj),
        config.qlora,
        config.quant_type,
    )

    if adapter_path is not None and Path(adapter_path).exists():
        console.print(f"[dim]Injecting LoRA and loading adapter from {adapter_path}...[/dim]")
        adapter_file = Path(adapter_path)
        if adapter_file.is_dir():
            adapter_file = adapter_file / "lora.safetensors"
        inject_lora(runtime.model, config.lora_rank, config.lora_alpha)
        load_adapter(runtime.model, adapter_file)

    runtime.model.eval()

    # 3. Build sequence and collate
    console.print("[dim]Building training sequence & tokenizing...[/dim]")
    builder = PersonaPlexTrainingExampleBuilder(
        runtime.codec, runtime.tokenizer, runtime.initial_tokens, runtime.zero_token,
        pause_frames=getattr(getattr(config, "generation_settings", None), "audio_silence_frame_cnt", 6),
    )
    example = builder.build(sample)
    fixed_frames = round(config.duration_sec * runtime.codec.frame_rate) + example.prompt_frames
    example = pad_training_example(example, fixed_frames, runtime.tokenizer.padding_id, runtime.zero_token)
    batch = post_encode_collate([example], runtime.tokenizer.padding_id, runtime.zero_token, device_obj)

    # 4. Teacher-Forcing Forward Pass
    console.print("[dim]Running Model Forward Pass (Teacher Forcing)...[/dim]")
    codes = batch["codes"]
    padding_ids = tokenizer_text_padding_ids(runtime.tokenizer)

    with torch.no_grad():
        output = model_forward_train(runtime.model, codes)
        t_corr, t_cnt, t_loss, cb_corr, cb_cnt, cb_loss = codebook_diagnostic_stats(
            batch, output, padding_ids
        )
        _, loss_dict = loss_components(
            output, codes, batch, padding_ids, torch,
            config.first_codebook_weight_multiplier, config.text_padding_weight,
            distributed=False,
        )

    # 5. Format results
    text_acc = float(t_corr / t_cnt.clamp_min(1)) if t_cnt > 0 else 0.0
    text_loss_val = float(t_loss / t_cnt.clamp_min(1)) if t_cnt > 0 else 0.0

    cb_accs = [float(cb_corr[i] / cb_cnt[i].clamp_min(1)) for i in range(8)]
    cb_losses = [float(cb_loss[i] / cb_cnt[i].clamp_min(1)) for i in range(8)]
    mean_audio_acc = float(cb_corr.sum() / cb_cnt.sum().clamp_min(1))
    mean_audio_loss = sum(cb_losses) / 8.0

    # Display Rich Table
    table = Table(title=f"Teacher-Forcing Verification Results (Sample {sample.sample_id})")
    table.add_column("Stream / Codebook", style="cyan", no_wrap=True)
    table.add_column("Valid Tokens", justify="right")
    table.add_column("Correct Tokens", justify="right")
    table.add_column("Accuracy (Argmax vs GT)", justify="right", style="bold")
    table.add_column("Cross-Entropy Loss", justify="right")
    table.add_column("Evaluation", justify="center")

    def eval_status(acc: float, is_text: bool = False) -> str:
        thresh = 0.95 if is_text else 0.90
        if acc >= 0.98:
            return "[bold green]EXCELLENT (Overfitted)[/bold green]"
        elif acc >= thresh:
            return "[green]GOOD[/green]"
        elif acc >= 0.50:
            return "[yellow]MODERATE[/yellow]"
        else:
            return "[red]POOR / RANDOM[/red]"

    table.add_row(
        "Agent Text (Stream 0)",
        f"{int(t_cnt):,}",
        f"{int(t_corr):,}",
        f"{text_acc * 100:.2f}%",
        f"{text_loss_val:.4f}",
        eval_status(text_acc, is_text=True),
    )
    table.add_section()

    table.add_row(
        "Audio CB0 (Semantic, Stream 1)",
        f"{int(cb_cnt[0]):,}",
        f"{int(cb_corr[0]):,}",
        f"{cb_accs[0] * 100:.2f}%",
        f"{cb_losses[0]:.4f}",
        eval_status(cb_accs[0]),
    )
    for i in range(1, 8):
        table.add_row(
            f"Audio CB{i} (Acoustic, Stream {i+1})",
            f"{int(cb_cnt[i]):,}",
            f"{int(cb_corr[i]):,}",
            f"{cb_accs[i] * 100:.2f}%",
            f"{cb_losses[i]:.4f}",
            eval_status(cb_accs[i]),
        )
    table.add_section()
    table.add_row(
        "[bold]Overall Audio (CB 0-7)[/bold]",
        f"{int(cb_cnt.sum()):,}",
        f"{int(cb_corr.sum()):,}",
        f"[bold]{mean_audio_acc * 100:.2f}%[/bold]",
        f"[bold]{mean_audio_loss:.4f}[/bold]",
        eval_status(mean_audio_acc),
    )

    console.print(table)

    summary = {
        "sample_id": sample.sample_id,
        "sample_index": sample_index,
        "text_valid_tokens": int(t_cnt),
        "text_accuracy": text_acc,
        "text_loss": text_loss_val,
        "audio_semantic_accuracy": cb_accs[0],
        "audio_semantic_loss": cb_losses[0],
        "mean_audio_accuracy": mean_audio_acc,
        "mean_audio_loss": mean_audio_loss,
        "codebook_accuracies": cb_accs,
        "codebook_losses": cb_losses,
        "loss_components": {k: float(v) for k, v in loss_dict.items()},
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Teacher-Forcing Diagnostic on One Training Sample")
    parser.add_argument("--config", type=Path, default=Path("configs/overfit.yaml"), help="Path to config yaml")
    parser.add_argument("--adapter", type=Path, default=None, help="Path to LoRA adapter file or checkpoint dir")
    parser.add_argument("--index", type=int, default=0, help="Sample index in dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run forward pass on")
    parser.add_argument("--json", action="store_true", help="Output raw JSON summary")

    args = parser.parse_args()
    result = evaluate_teacher_forcing_sample(
        config_path=args.config,
        adapter_path=args.adapter,
        sample_index=args.index,
        device=args.device,
    )
    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
