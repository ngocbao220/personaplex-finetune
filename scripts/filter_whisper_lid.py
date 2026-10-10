"""Score each fixed training chunk with Whisper language identification."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np
import torch
import whisper


def score_chunk(model, wav: wave.Wave_read, start: float, end: float, device: str) -> tuple[float, list[float]]:
    rate = wav.getframerate()
    channels = wav.getnchannels()
    width = wav.getsampwidth()
    if width != 2 or channels != 2:
        raise ValueError("Whisper LID expects PCM16 stereo conversation.wav")
    duration = wav.getnframes() / rate
    valid_end = min(end, duration)
    if valid_end <= start:
        raise ValueError("empty chunk")
    # Three evenly spaced 30 s probes cover the start, middle and end of a
    # 100 s chunk. Both speakers contribute equally to the mono LID signal.
    probe_seconds = min(30.0, valid_end - start)
    offsets = [start, start + (valid_end - start - probe_seconds) / 2, valid_end - probe_seconds]
    scores = []
    for offset in offsets:
        wav.setpos(min(wav.getnframes(), int(offset * rate)))
        count = min(int(probe_seconds * rate), wav.getnframes() - wav.tell())
        raw = wav.readframes(count)
        audio = np.frombuffer(raw, dtype="<i2").reshape(-1, 2).astype(np.float32).mean(axis=1) / 32768.0
        if rate != whisper.audio.SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=rate, target_sr=whisper.audio.SAMPLE_RATE)
        padded = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(padded, n_mels=model.dims.n_mels).to(device)
        with torch.no_grad():
            _, probabilities = model.detect_language(mel.unsqueeze(0))
        scores.append(float(probabilities[0].get("vi", 0.0)))
    return sum(scores) / len(scores), scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="small")
    parser.add_argument("--duration-sec", type=float, default=100.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.duration_sec <= 0:
        parser.error("threshold must be in [0,1] and duration-sec must be positive")
    root = args.prepared_dir.resolve()
    manifest = root / "train.jsonl"
    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(args.model, device=device)
    results = []
    for index, entry in enumerate(entries, 1):
        path = root / entry["sample_dir"] / "conversation.wav"
        with wave.open(str(path), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
            for chunk_index in range(math.ceil(duration / args.duration_sec)):
                start = chunk_index * args.duration_sec
                end = start + args.duration_sec
                probability, probes = score_chunk(model, wav, start, end, device)
                row = {
                    "sample_id": entry["sample_id"], "window_start_sec": float(start),
                    "window_end_sec": float(end), "vi_probability": probability,
                    "probe_probabilities": probes, "keep": probability >= args.threshold,
                }
                results.append(row)
        print(f"[Whisper LID] conversation={index}/{len(entries)} id={entry['sample_id']} "
              f"kept={sum(x['keep'] for x in results)} scored={len(results)}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": args.model, "threshold": args.threshold, "duration_sec": args.duration_sec,
                   "source_manifest": str(manifest), "source_manifest_sha256": __import__('hashlib').sha256(manifest.read_bytes()).hexdigest(),
                   "chunks": results}
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    print(f"[Whisper LID] complete kept={sum(x['keep'] for x in results)}/{len(results)} report={args.output}")


if __name__ == "__main__":
    main()
