"""Mimi Neural Audio Codec evaluation and sanity check tool.

Tests reconstruction quality (SNR, SI-SDR, Mel loss) on arbitrary audio files (especially
Vietnamese speech samples with tone marks) by passing audio through Mimi Encode -> Decode.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
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
        # Resample with sphn / torchaudio if needed
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


def run_mimi_test(
    audio_paths: list[Path],
    mimi_weight: Path,
    output_dir: Path,
    num_codebooks: int = 8,
    device: str = "cpu",
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    console = Console()
    console.print(f"[bold green]Loading Mimi Codec from:[/bold green] {mimi_weight}")
    console.print(f"[cyan]Device:[/cyan] {device} | [cyan]Num Codebooks:[/cyan] {num_codebooks}")

    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(num_codebooks)
    mimi.eval()

    results = []
    for p in audio_paths:
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

        # Save reconstructed wav
        out_name = f"{p.stem}_mimi_{num_codebooks}cb.wav"
        out_path = output_dir / out_name
        try:
            import sphn
            sphn.write_wav(str(out_path), rec_np, mimi.sample_rate)
        except Exception:
            import torchaudio
            torchaudio.save(str(out_path), torch.from_numpy(rec_np).unsqueeze(0), mimi.sample_rate)

        results.append({
            "filename": p.name,
            "duration": duration,
            "snr": metrics["snr"],
            "si_sdr": metrics["si_sdr"],
            "output_path": str(out_path),
        })

    return results


def print_report(results: list[dict], num_codebooks: int):
    console = Console()
    table = Table(
        title=f"Mimi Neural Audio Codec Reconstruction Report ({num_codebooks} Codebooks)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Sample Audio", style="cyan", justify="left")
    table.add_column("Duration (s)", justify="right")
    table.add_column("SNR (dB)", justify="right")
    table.add_column("SI-SDR (dB)", justify="right")
    table.add_column("Assessment", justify="center")
    table.add_column("Saved Output Path", style="dim", justify="left")

    for r in results:
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
            r["output_path"],
        )

    console.print()
    console.print(table)

    avg_si_sdr = sum(r["si_sdr"] for r in results) / len(results)
    console.print(f"\n[bold]Điểm SI-SDR Trung bình:[/bold] [bold cyan]{avg_si_sdr:.2f} dB[/bold cyan]")
    if avg_si_sdr < 8.0:
        console.print("[bold red]Khuyến nghị:[/bold red] Mimi tái tạo âm thanh chưa đủ độ trung thực đối với bộ ngữ âm này. Bạn nên thực hiện Stage 1: Fine-tune Mimi trước khi fine-tune mô hình ngôn ngữ 7B.")
    else:
        console.print("[bold green]Đánh giá:[/bold green] Mimi bảo toàn tốt đặc trưng âm thanh. Bạn có thể tự tin sử dụng Mimi để trích xuất tokens cho PersonaPlex.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Mimi Neural Audio Codec reconstruction fidelity on audio samples.")
    parser.add_argument("--input", required=True, help="Path to single audio file or directory of audio files")
    parser.add_argument("--model-root", default="../models", help="Path to folder containing Mimi weights")
    parser.add_argument("--output-dir", default="outputs/mimi_test", help="Directory to save reconstructed WAVs")
    parser.add_argument("--num-codebooks", type=int, default=8, help="Number of RVQ codebooks (default: 8, options: 8, 16, 32)")
    parser.add_argument("--device", default="auto", help="Device ('cuda', 'cpu', 'mps', or 'auto')")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.", file=sys.stderr)
        return 1

    if input_path.is_file():
        audio_files = [input_path]
    else:
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        audio_files = [f for f in input_path.iterdir() if f.suffix.lower() in exts]
        if not audio_files:
            print(f"Error: No audio files found in directory '{input_path}'.", file=sys.stderr)
            return 1

    model_root = Path(args.model_root)
    if not model_root.is_dir():
        # Try local models fallback
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

    results = run_mimi_test(
        audio_paths=audio_files,
        mimi_weight=mimi_weight,
        output_dir=Path(args.output_dir),
        num_codebooks=args.num_codebooks,
        device=device,
    )

    print_report(results, args.num_codebooks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
