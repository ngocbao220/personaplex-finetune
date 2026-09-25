"""Mimi Neural Audio Codec evaluation and benchmark tool.

Tests reconstruction quality (SNR, SI-SDR) on arbitrary audio files (especially
Vietnamese speech samples with tone marks) by passing audio through Mimi Encode -> Decode.
Supports recursive directory traversal with nested subfolders and random sampling (e.g. 600 files).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich import box

from moshi.models import loaders


def compute_metrics(ref: np.ndarray, deg: np.ndarray) -> dict[str, float]:
    """Calculate reconstruction metrics: SNR and SI-SDR in dB."""
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len].astype(np.float64)
    deg = deg[:min_len].astype(np.float64)

    # Standard SNR
    noise = ref - deg
    ref_power = np.sum(ref**2)
    noise_power = np.sum(noise**2)
    snr = 10.0 * np.log10(max(ref_power, 1e-12) / max(noise_power, 1e-12))

    # Scale-Invariant SDR (SI-SDR)
    alpha = np.dot(deg, ref) / max(ref_power, 1e-12)
    e_target = alpha * ref
    e_res = deg - e_target
    target_power = np.sum(e_target**2)
    res_power = np.sum(e_res**2)
    si_sdr = 10.0 * np.log10(max(target_power, 1e-12) / max(res_power, 1e-12))

    return {"snr": float(snr), "si_sdr": float(si_sdr)}


def load_audio(path: Path, target_sr: int = 24000) -> tuple[torch.Tensor, float]:
    """Load audio file and convert to mono 24kHz tensor of shape [1, 1, T]."""
    try:
        import sphn
        data, sr = sphn.read(str(path))
        wav = torch.as_tensor(data, dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        # Convert stereo to mono by averaging channels if needed
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # Resample with torchaudio if needed
        if sr != target_sr:
            import torchaudio.functional as F
            wav = F.resample(wav, orig_freq=sr, new_freq=target_sr)
    except Exception:
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            import torchaudio.functional as F
            wav = F.resample(wav, orig_freq=sr, new_freq=target_sr)

    duration_sec = wav.shape[-1] / target_sr
    return wav.unsqueeze(0), duration_sec  # [1, 1, T]


def find_mimi_weight(model_root: Path) -> Path:
    """Find the Mimi safetensors checkpoint inside the model root."""
    target = "tokenizer-e351c8d8-checkpoint125.safetensors"
    direct = model_root / target
    if direct.is_file():
        return direct
    matches = list(model_root.rglob(target))
    if matches:
        return matches[0]
    # Fallback to any safetensors with tokenizer
    tokenizers = list(model_root.glob("*tokenizer*.safetensors"))
    if tokenizers:
        return tokenizers[0]
    raise FileNotFoundError(f"Could not find '{target}' in {model_root}")


def discover_audio_files(input_path: Path, exts: set[str] | None = None) -> list[Path]:
    """Recursively discover all audio files in input_path and its nested subdirectories."""
    if exts is None:
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    if input_path.is_file():
        if input_path.suffix.lower() in exts:
            return [input_path]
        return []

    # Recursive scan through all subdirectories
    found_files = sorted([p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return found_files


def run_mimi_test(
    audio_paths: list[Path],
    mimi_weight: Path,
    output_dir: Path,
    num_codebooks: int = 8,
    save_samples: int = 10,
    device: str = "cpu",
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    console = Console()
    console.print(f"[bold green]Loading Mimi Codec from:[/bold green] {mimi_weight}")
    console.print(f"[cyan]Device:[/cyan] {device} | [cyan]Num Codebooks:[/cyan] {num_codebooks}")
    console.print(f"[cyan]Evaluating:[/cyan] {len(audio_paths)} audio files (Saving first {min(save_samples, len(audio_paths))} sample WAVs)")

    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(num_codebooks)
    mimi.eval()

    results = []
    saved_count = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task_id = progress.add_task("[cyan]Benchmarking Mimi...", total=len(audio_paths))

        for p in audio_paths:
            try:
                wav, duration = load_audio(p, target_sr=mimi.sample_rate)
                wav = wav.to(device)

                with torch.no_grad():
                    # Encode to RVQ codes
                    codes = mimi.encode(wav)
                    # Decode back to waveform
                    rec = mimi.decode(codes)

                ref_np = wav.squeeze().cpu().numpy()
                rec_np = rec.squeeze().cpu().numpy()

                metrics = compute_metrics(ref_np, rec_np)

                # Save reconstructed wav for inspecting sound quality (up to save_samples)
                out_path_str = None
                if saved_count < save_samples:
                    out_name = f"{p.stem}_mimi_{num_codebooks}cb.wav"
                    out_path = output_dir / out_name
                    try:
                        import sphn
                        sphn.write_wav(str(out_path), rec_np, mimi.sample_rate)
                    except Exception:
                        import torchaudio
                        torchaudio.save(str(out_path), torch.from_numpy(rec_np).unsqueeze(0), mimi.sample_rate)
                    out_path_str = str(out_path)
                    saved_count += 1

                results.append({
                    "filename": p.name,
                    "rel_path": str(p),
                    "duration": duration,
                    "snr": metrics["snr"],
                    "si_sdr": metrics["si_sdr"],
                    "output_path": out_path_str,
                })
            except Exception as exc:
                console.print(f"[yellow]Warning: Failed to process {p.name}: {exc}[/yellow]")

            progress.update(task_id, advance=1)

    return results


def print_report(results: list[dict], num_codebooks: int, output_dir: Path):
    console = Console()
    if not results:
        console.print("[red]No audio files were successfully evaluated.[/red]")
        return

    # 1. Detail Table (Show preview if > 25 samples)
    table = Table(
        title=f"Mimi Neural Audio Codec Reconstruction Samples ({num_codebooks} Codebooks)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Sample Audio", style="cyan", justify="left")
    table.add_column("Duration (s)", justify="right")
    table.add_column("SNR (dB)", justify="right")
    table.add_column("SI-SDR (dB)", justify="right")
    table.add_column("Assessment", justify="center")
    table.add_column("Saved Output Path", style="dim", justify="left")

    display_results = results if len(results) <= 25 else (results[:12] + results[-8:])

    for idx, r in enumerate(display_results):
        if len(results) > 25 and idx == 12:
            table.add_row(
                f"[dim]... and {len(results) - 20} more samples ...[/dim]",
                "...",
                "...",
                "...",
                "...",
                "...",
            )

        si_sdr = r["si_sdr"]
        if si_sdr >= 12.0:
            assess = "[green]Xuất sắc (Ready)[/green]"
        elif si_sdr >= 8.0:
            assess = "[yellow]Khá tốt (Acceptable)[/yellow]"
        else:
            assess = "[bold red]Cần Fine-tune[/bold red]"

        table.add_row(
            r["filename"],
            f"{r['duration']:.2f}",
            f"{r['snr']:.2f}",
            f"{r['si_sdr']:.2f}",
            assess,
            r["output_path"] if r["output_path"] else "[dim]Not saved[/dim]",
        )

    console.print()
    console.print(table)

    # 2. Comprehensive Statistical Summary Table
    si_sdrs = [r["si_sdr"] for r in results]
    snrs = [r["snr"] for r in results]
    durations = [r["duration"] for r in results]

    total_samples = len(results)
    total_dur_sec = sum(durations)
    total_dur_min = total_dur_sec / 60.0

    mean_si_sdr = float(np.mean(si_sdrs))
    median_si_sdr = float(np.median(si_sdrs))
    std_si_sdr = float(np.std(si_sdrs))
    min_si_sdr = float(np.min(si_sdrs))
    max_si_sdr = float(np.max(si_sdrs))

    mean_snr = float(np.mean(snrs))
    median_snr = float(np.median(snrs))
    std_snr = float(np.std(snrs))
    min_snr = float(np.min(snrs))
    max_snr = float(np.max(snrs))

    # Distribution counts
    count_excellent = sum(1 for v in si_sdrs if v >= 12.0)
    count_good = sum(1 for v in si_sdrs if 8.0 <= v < 12.0)
    count_poor = sum(1 for v in si_sdrs if v < 8.0)

    summary_table = Table(
        title=f"Aggregate Benchmark Statistics ({total_samples} Evaluated Samples, {total_dur_min:.2f} mins)",
        box=box.DOUBLE_EDGE,
        header_style="bold green",
    )
    summary_table.add_column("Metric", style="bold white")
    summary_table.add_column("Mean", justify="right")
    summary_table.add_column("Median", justify="right")
    summary_table.add_column("Min", justify="right")
    summary_table.add_column("Max", justify="right")
    summary_table.add_column("Std Dev", justify="right")

    summary_table.add_row(
        "SI-SDR (dB)",
        f"{mean_si_sdr:.2f}",
        f"{median_si_sdr:.2f}",
        f"{min_si_sdr:.2f}",
        f"{max_si_sdr:.2f}",
        f"{std_si_sdr:.2f}",
    )
    summary_table.add_row(
        "SNR (dB)",
        f"{mean_snr:.2f}",
        f"{median_snr:.2f}",
        f"{min_snr:.2f}",
        f"{max_snr:.2f}",
        f"{std_snr:.2f}",
    )
    console.print()
    console.print(summary_table)

    # 3. Distribution Breakdown
    dist_table = Table(title="Quality Distribution Breakdown", box=box.SIMPLE_HEAVY)
    dist_table.add_column("Category", style="bold")
    dist_table.add_column("SI-SDR Threshold", justify="center")
    dist_table.add_column("Count", justify="right")
    dist_table.add_column("Percentage", justify="right")
    dist_table.add_column("Recommendation", style="italic")

    dist_table.add_row(
        "[green]Xuất sắc (Excellent)[/green]",
        ">= 12.0 dB",
        str(count_excellent),
        f"{(count_excellent / total_samples) * 100:.1f}%",
        "Tái tạo hoàn hảo ngữ âm & thanh điệu",
    )
    dist_table.add_row(
        "[yellow]Khá tốt (Acceptable)[/yellow]",
        "8.0 - 12.0 dB",
        str(count_good),
        f"{(count_good / total_samples) * 100:.1f}%",
        "Âm thanh rõ ràng, nghe tốt",
    )
    dist_table.add_row(
        "[red]Cần Fine-tune (Poor)[/red]",
        "< 8.0 dB",
        str(count_poor),
        f"{(count_poor / total_samples) * 100:.1f}%",
        "Nguy cơ méo tiếng hoặc lệch cao độ F0",
    )
    console.print(dist_table)

    # 4. Final Verdict
    console.print(f"\n[bold]Điểm SI-SDR Trung bình:[/bold] [bold cyan]{mean_si_sdr:.2f} dB[/bold cyan] (Median: [bold cyan]{median_si_sdr:.2f} dB[/bold cyan])")
    if mean_si_sdr < 8.0 or (count_poor / total_samples) > 0.25:
        console.print(
            "[bold red]Khuyến nghị:[/bold red] Mimi tái tạo âm thanh chưa đủ độ trung thực đối với tập dữ liệu này. "
            "Bạn nên thực hiện [bold yellow]Stage 0: Fine-tune Mimi[/bold yellow] (chạy `scripts/train_mimi.sh`) "
            "trước khi huấn luyện PersonaPlex 7B."
        )
    else:
        console.print(
            "[bold green]Đánh giá:[/bold green] Mimi bảo toàn tốt đặc trưng âm thanh và thanh điệu. "
            "Bạn có thể tự tin sử dụng Mimi để trích xuất tokens cho PersonaPlex."
        )

    # 5. Export JSON
    report_data = {
        "num_codebooks": num_codebooks,
        "total_samples": total_samples,
        "total_duration_sec": total_dur_sec,
        "mean_si_sdr": mean_si_sdr,
        "median_si_sdr": median_si_sdr,
        "std_si_sdr": std_si_sdr,
        "min_si_sdr": min_si_sdr,
        "max_si_sdr": max_si_sdr,
        "mean_snr": mean_snr,
        "median_snr": median_snr,
        "std_snr": std_snr,
        "min_snr": min_snr,
        "max_snr": max_snr,
        "distribution": {
            "excellent_ge_12db": count_excellent,
            "good_8_to_12db": count_good,
            "poor_lt_8db": count_poor,
        },
        "samples": results,
    }
    json_path = output_dir / "mimi_benchmark_results.json"
    json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    console.print(f"\n[dim]Detailed benchmark results saved to: {json_path}[/dim]")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test Mimi Neural Audio Codec reconstruction fidelity on audio samples with recursive search & sampling."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to single audio file or directory (will recursively search all nested subdirectories for audio files)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=600,
        help="Number of audio files to randomly sample for benchmark (default: 600, set 0 to evaluate all found files)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling of audio files (default: 42)",
    )
    parser.add_argument(
        "--model-root",
        default="../models",
        help="Path to folder containing Mimi weights",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mimi_test",
        help="Directory to save reconstructed WAVs and JSON benchmark report",
    )
    parser.add_argument(
        "--num-codebooks",
        type=int,
        default=8,
        help="Number of RVQ codebooks (default: 8, options: 8, 16, 32)",
    )
    parser.add_argument(
        "--save-samples",
        type=int,
        default=20,
        help="Maximum number of reconstructed WAV files to save to disk to save space (default: 20)",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save reconstructed WAV files for ALL evaluated samples",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device ('cuda', 'cpu', 'mps', or 'auto')",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.", file=sys.stderr)
        return 1

    console = Console()

    # Discover files recursively
    all_audio_files = discover_audio_files(input_path)
    if not all_audio_files:
        print(f"Error: No audio files (.wav, .mp3, .flac, etc.) found in '{input_path}' or any of its subfolders.", file=sys.stderr)
        return 1

    console.print(f"[bold cyan]Discovered {len(all_audio_files)} audio files[/bold cyan] in [dim]{input_path}[/dim] (including nested folders).")

    # Sample random subset (e.g. 600 files)
    if args.num_samples > 0 and len(all_audio_files) > args.num_samples:
        rng = random.Random(args.seed)
        selected_files = sorted(rng.sample(all_audio_files, args.num_samples))
        console.print(f"[yellow]Randomly sampled {len(selected_files)} files[/yellow] for benchmark using seed={args.seed}.")
    else:
        selected_files = all_audio_files
        console.print(f"[green]Using all {len(selected_files)} files[/green] for benchmark.")

    # Model root resolution
    model_root = Path(args.model_root)
    if not model_root.is_dir():
        cand = Path("models/personaplex-7b-v1")
        if cand.is_dir():
            model_root = cand

    try:
        mimi_weight = find_mimi_weight(model_root)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    save_samples = len(selected_files) if args.save_all else args.save_samples

    output_dir = Path(args.output_dir)
    results = run_mimi_test(
        audio_paths=selected_files,
        mimi_weight=mimi_weight,
        output_dir=output_dir,
        num_codebooks=args.num_codebooks,
        save_samples=save_samples,
        device=device,
    )

    print_report(results, args.num_codebooks, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
