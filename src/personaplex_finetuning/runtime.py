"""Explicit local PersonaPlex runtime; deliberately contains no Hub fallback."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolvedRuntimePaths:
    source: Path
    model_root: Path
    moshi_weight: Path
    mimi_weight: Path
    tokenizer: Path


@dataclass(frozen=True)
class RuntimePaths:
    model_root: Path
    source: Path

    def validate(self) -> ResolvedRuntimePaths:
        root = Path(self.model_root).resolve()
        source = Path(self.source).resolve()
        required = {
            "model.safetensors": root / "model.safetensors",
            "tokenizer-e351c8d8-checkpoint125.safetensors": root / "tokenizer-e351c8d8-checkpoint125.safetensors",
            "tokenizer_spm_32k_3.model": root / "tokenizer_spm_32k_3.model",
        }
        required_source = source / "moshi" / "models" / "loaders.py"
        if not required_source.is_file():
            raise FileNotFoundError(
                f"PersonaPlex source must contain moshi/models/loaders.py: {source}"
            )
        for name, path in required.items():
            if not path.is_file():
                raise FileNotFoundError(f"required local PersonaPlex asset missing: {path} ({name})")
        return ResolvedRuntimePaths(
            source=source, model_root=root, moshi_weight=required["model.safetensors"],
            mimi_weight=required["tokenizer-e351c8d8-checkpoint125.safetensors"],
            tokenizer=required["tokenizer_spm_32k_3.model"],
        )


class SentencePieceTokenizer:
    padding_id = 3
    end_padding_id = 0

    def __init__(self, path: Path) -> None:
        sentencepiece = importlib.import_module("sentencepiece")
        self._processor = sentencepiece.SentencePieceProcessor(str(path))

    def encode(self, text: str) -> list[int]:
        return list(self._processor.encode(text))


class MimiCodec:
    """Mimi adapter using the same source helpers as PersonaPlex inference."""

    codebooks = 8

    def __init__(self, mimi, sample_rate: int, frame_rate: float, device: str, lm_helpers) -> None:
        self.mimi = mimi
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.device = device
        self._helpers = lm_helpers

    def encode_conversation(self, path: Path, channel: int, start_sec: float, end_sec: float):
        import sphn
        import torch
        audio, source_rate = sphn.read(str(path))
        audio = sphn.resample(audio, src_sample_rate=source_rate, dst_sample_rate=self.sample_rate)
        start, end = int(start_sec * self.sample_rate), int(end_sec * self.sample_rate)
        if channel not in (0, 1) or end <= start or end > audio.shape[-1]:
            raise ValueError(f"invalid conversation window {start_sec}:{end_sec} for {path}")
        return self._encode(audio[channel : channel + 1, start:end], torch)

    def encode_voice_prompt(self, path: Path):
        import torch
        audio = self._helpers.load_audio(str(path), self.sample_rate)
        audio = self._helpers.normalize_audio(audio, self.sample_rate, -24.0)
        if audio.ndim == 1:
            audio = audio[None, :]
        return self._encode(audio[:1], torch)

    def sine(self, frames: int):
        import numpy as np
        import torch
        duration = frames / self.frame_rate
        return self._encode(self._helpers.create_sinewave(duration, self.sample_rate)[None, :], torch)

    def silence(self, frames: int):
        import numpy as np
        import torch
        return self._encode(np.zeros((1, int(frames * self.sample_rate / self.frame_rate)), dtype=np.float32), torch)

    def _encode(self, audio, torch):
        with torch.no_grad():
            codes = self.mimi.encode(torch.as_tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
        return tuple(tuple(int(token) for token in stream.tolist()) for stream in codes)


@dataclass
class PersonaPlexRuntime:
    model: object
    codec: MimiCodec
    tokenizer: SentencePieceTokenizer
    initial_tokens: tuple[int, ...]
    zero_token: int
    delays: tuple[int, ...]


def load_runtime(paths: RuntimePaths, device: str = "cuda", qlora: bool = False, quant_type: str = "nf4") -> PersonaPlexRuntime:
    """Load from explicit local assets only. No Hugging Face function is imported."""
    resolved = paths.validate()
    source = str(resolved.source)
    if source not in sys.path:
        sys.path.insert(0, source)
    torch = importlib.import_module("torch")
    loaders = importlib.import_module("moshi.models.loaders")
    lm_helpers = importlib.import_module("moshi.models.lm")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    mimi = loaders.get_mimi(resolved.mimi_weight, device=device)
    if qlora:
        from .lora import quantize_model_4bit
        model = loaders.get_moshi_lm(resolved.moshi_weight, device="cpu")
        model = quantize_model_4bit(model, device=device, quant_type=quant_type)
    else:
        model = loaders.get_moshi_lm(resolved.moshi_weight, device=device)

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()
    initial = tuple(int(value) for value in model._get_initial_token()[0, :, 0].tolist())
    if len(initial) != 17 or model.dep_q != 16 or model.n_q != 16:
        raise RuntimeError("loaded checkpoint is not the expected 17-stream PersonaPlex model")
    return PersonaPlexRuntime(
        model=model,
        codec=MimiCodec(mimi, mimi.sample_rate, mimi.frame_rate, device, lm_helpers),
        tokenizer=SentencePieceTokenizer(resolved.tokenizer),
        initial_tokens=initial,
        zero_token=int(model.zero_token_id),
        delays=tuple(int(delay) for delay in model.delays),
    )
