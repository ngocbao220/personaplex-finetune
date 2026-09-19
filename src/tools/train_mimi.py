"""Stage 0: Mimi Neural Audio Codec Fine-Tuning Pipeline.

Fine-tunes the Mimi Audio Codec (RVQ AutoEncoder) on target language audio (e.g. Vietnamese)
to adapt phonetic representations, pitch/F0 contours, and tone marks prior to LLM training.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from moshi.models import loaders


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi-resolution STFT loss for high-fidelity audio reconstruction (spectral convergence + log-magnitude)."""

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: tuple[int, ...] = (120, 240, 480),
        win_lengths: tuple[int, ...] = (480, 960, 1920),
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def _mag_stft(self, x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        wav = x.squeeze(1) if x.dim() == 3 else x
        window = torch.hann_window(win, device=wav.device)
        stft = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
        return torch.abs(stft)

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            mag_ref = self._mag_stft(x, n_fft, hop, win)
            mag_deg = self._mag_stft(x_hat, n_fft, hop, win)
            # Spectral convergence
            sc = torch.norm(mag_ref - mag_deg, p="fro") / (torch.norm(mag_ref, p="fro") + 1e-7)
            # Log magnitude
            log_mag = F.l1_loss(torch.log(mag_deg + 1e-5), torch.log(mag_ref + 1e-5))
            loss = loss + sc + log_mag
        return loss / len(self.fft_sizes)


class RawAudioDataset(Dataset):
    """Loads arbitrary mono audio files, slices deterministic/random crops at 24kHz."""

    def __init__(self, data_dir: Path, sample_rate: int = 24000, chunk_seconds: float = 2.0):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_seconds)
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        self.files = sorted([p for p in data_dir.rglob("*") if p.suffix.lower() in exts])
        if not self.files:
            raise FileNotFoundError(f"No audio files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        try:
            import sphn
            data, sr = sphn.read(str(path))
            wav = torch.as_tensor(data, dtype=torch.float32)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                import torchaudio.functional as AF
                wav = AF.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        except Exception:
            import torchaudio
            wav, sr = torchaudio.load(str(path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                import torchaudio.functional as AF
                wav = AF.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        # Random or center crop to chunk_samples
        t = wav.shape[-1]
        if t > self.chunk_samples:
            max_start = t - self.chunk_samples
            start = random.randint(0, max_start)
            wav = wav[:, start : start + self.chunk_samples]
        elif t < self.chunk_samples:
            wav = F.pad(wav, (0, self.chunk_samples - t))

        return wav  # [1, T]


def find_mimi_weight(model_root: Path) -> Path:
    target = "tokenizer-e351c8d8-checkpoint125.safetensors"
    direct = model_root / target
    if direct.is_file():
        return direct
    matches = list(model_root.rglob(target))
    if matches:
        return matches[0]
    tokenizers = list(model_root.glob("*tokenizer*.safetensors"))
    if tokenizers:
        return tokenizers[0]
    raise FileNotFoundError(f"Could not find '{target}' in {model_root}")


def train_mimi(
    data_dir: Path,
    model_root: Path,
    output_dir: Path,
    max_steps: int = 5000,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    chunk_seconds: float = 2.0,
    num_codebooks: int = 8,
    device: str = "cuda",
    save_every_steps: int = 500,
    freeze_encoder: bool = False,
    smoke: bool = False,
):
    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)

    mimi_weight = find_mimi_weight(model_root)
    console.print(f"[bold cyan]=== STAGE 0: MIMI CODEC FINE-TUNING ===[/bold cyan]")
    console.print(f"Base Mimi weights : {mimi_weight}")
    console.print(f"Dataset root      : {data_dir}")
    console.print(f"Output directory  : {output_dir}")
    console.print(f"Device            : {device} | Codebooks: {num_codebooks}")

    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(num_codebooks)
    mimi.train()

    if freeze_encoder:
        console.print("[yellow]Freeze Encoder active:[/yellow] Training only Quantizer and Decoder.")
        mimi.encoder.requires_grad_(False)
        if mimi.encoder_transformer is not None:
            mimi.encoder_transformer.requires_grad_(False)

    trainable_params = [p for p in mimi.parameters() if p.requires_grad]
    console.print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, betas=(0.8, 0.99), weight_decay=0.0)
    stft_criterion = MultiResolutionSTFTLoss().to(device)

    dataset = RawAudioDataset(data_dir, sample_rate=mimi.sample_rate, chunk_seconds=chunk_seconds)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=(device == "cuda"))

    data_iter = iter(dataloader)
    target_steps = 1 if smoke else max_steps

    console.print(f"Starting optimization for {target_steps} steps...\n")

    start_time = time.monotonic()
    running_loss = 0.0

    for step in range(1, target_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(device)  # [B, 1, T]

        optimizer.zero_grad()
        q_res = mimi(batch)
        reconstructed = q_res.x
        commitment = q_res.penalty if q_res.penalty is not None else 0.0

        # Losses
        loss_time = F.l1_loss(reconstructed, batch)
        loss_stft = stft_criterion(reconstructed, batch)
        total_loss = loss_time + loss_stft + 0.25 * commitment

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        running_loss += total_loss.item()

        if step % 20 == 0 or step == 1 or step == target_steps:
            elapsed = time.monotonic() - start_time
            steps_per_sec = step / max(1.0, elapsed)
            console.print(
                f"Step {step:05d}/{target_steps:05d} | "
                f"Loss: [bold green]{total_loss.item():.4f}[/bold green] "
                f"(Time L1: {loss_time.item():.4f}, STFT: {loss_stft.item():.4f}, VQ: {float(commitment):.4f}) | "
                f"Speed: {steps_per_sec:.2f} it/s"
            )

        if (step % save_every_steps == 0 or step == target_steps) and not smoke:
            ckpt_path = output_dir / f"mimi_step_{step:06d}.safetensors"
            canonical_path = output_dir / "tokenizer-e351c8d8-checkpoint125.safetensors"
            state = mimi.state_dict()
            save_file(state, str(ckpt_path))
            save_file(state, str(canonical_path))
            console.print(f"[bold green]Saved Mimi checkpoint:[/bold green] {ckpt_path}")

    if smoke:
        console.print("[bold green]Smoke test complete: Forward + Backward + Gradients PASS![/bold green]")
    else:
        final_file = output_dir / "tokenizer-e351c8d8-checkpoint125.safetensors"
        console.print(f"\n[bold green]Training finished![/bold green] Final Mimi weights saved at:\n{final_file}")
        console.print("[cyan]You can now use this adapted Mimi checkpoint in Stage 1/2/3 training of PersonaPlex.[/cyan]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Mimi Neural Audio Codec on speech data.")
    parser.add_argument("--data-dir", required=True, help="Directory containing mono audio WAV files")
    parser.add_argument("--model-root", default="../models", help="Directory containing base Mimi safetensors")
    parser.add_argument("--output-dir", default="../runs/mimi_finetuned", help="Directory to save checkpoints")
    parser.add_argument("--max-steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for AdamW")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--chunk-seconds", type=float, default=2.0, help="Audio crop duration per sample in seconds")
    parser.add_argument("--num-codebooks", type=int, default=8, help="Number of RVQ codebooks")
    parser.add_argument("--device", default="auto", help="Device ('cuda', 'cpu', 'mps', or 'auto')")
    parser.add_argument("--save-every-steps", type=int, default=500, help="Save periodic checkpoint every N steps")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder and train only quantizer + decoder")
    parser.add_argument("--smoke", action="store_true", help="Run 1-step verification smoke test")

    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: Data directory '{data_dir}' does not exist.", file=sys.stderr)
        return 1

    train_mimi(
        data_dir=data_dir,
        model_root=Path(args.model_root),
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        chunk_seconds=args.chunk_seconds,
        num_codebooks=args.num_codebooks,
        device=device,
        save_every_steps=args.save_every_steps,
        freeze_encoder=args.freeze_encoder,
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
