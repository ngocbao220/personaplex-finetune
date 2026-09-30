"""Explicit local PersonaPlex runtime; deliberately contains no Hub fallback."""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from .data import read_stereo_window


def _pad_audio_window(audio, sample_rate: int, duration_sec: float):
    """Match the fixed final-chunk padding performed by the reference dataset loader."""
    import numpy as np

    if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[-1] == 0:
        raise ValueError(f"cannot pad missing or non-stereo decoded audio: {audio.shape}")

    expected_samples = round(sample_rate * duration_sec)
    if audio.shape[-1] >= expected_samples:
        return audio[..., :expected_samples]
    return np.pad(audio, [(0, 0)] * (audio.ndim - 1) + [(0, expected_samples - audio.shape[-1])])


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

    def validate(self, require_model: bool = True) -> ResolvedRuntimePaths:
        root = Path(self.model_root).resolve()
        source = Path(self.source).resolve()
        required = {
            "tokenizer-e351c8d8-checkpoint125.safetensors": root / "tokenizer-e351c8d8-checkpoint125.safetensors",
            "tokenizer_spm_32k_3.model": root / "tokenizer_spm_32k_3.model",
        }
        if require_model:
            required["model.safetensors"] = root / "model.safetensors"
        required_source = source / "moshi" / "models" / "loaders.py"
        if not required_source.is_file():
            raise FileNotFoundError(
                f"PersonaPlex source must contain moshi/models/loaders.py: {source}"
            )
        for name, path in required.items():
            if not path.is_file():
                raise FileNotFoundError(f"required local PersonaPlex asset missing: {path} ({name})")
        return ResolvedRuntimePaths(
            source=source, model_root=root, moshi_weight=root / "model.safetensors",
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

    def decode(self, tokens: list[int]) -> str:
        return str(self._processor.decode(tokens))


class MimiCodec:
    """Mimi adapter using the same source helpers as PersonaPlex inference."""

    codebooks = 8

    def __init__(
        self, mimi, sample_rate: int, frame_rate: float, device: str, lm_helpers,
        cache_dir: Path | None = None, cache_namespace: str = "default",
    ) -> None:
        self.mimi = mimi
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.device = device
        self._helpers = lm_helpers
        self._voice_cache: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._cache_dir = cache_dir  # Optional persistent disk cache for conversation encoding
        self._cache_namespace = cache_namespace

    def encode_conversation_stereo(self, path: Path, agent_channel: int, user_channel: int, start_sec: float, end_sec: float):
        return self.encode_conversation_stereo_batch([
            (path, agent_channel, user_channel, start_sec, end_sec),
        ])[0]

    def encode_conversation_stereo_batch(self, windows, raw_audio=None):
        """Encode each batch's stereo windows in one Mimi call, preserving per-window padding.

        ``windows`` contains ``(path, agent_channel, user_channel, start_sec, end_sec)``
        tuples. Each conversation contributes two independent Mimi batch items, in agent/user
        order. This replaces repeated tiny codec launches when the LM microbatch is greater
        than one without changing the audio window or channel semantics.
        """
        import numpy as np
        import torch

        if not windows:
            raise ValueError("cannot encode an empty conversation batch")
        results = [None] * len(windows)
        pending = []
        cache_info = [None] * len(windows)
        for index, (path, agent_channel, user_channel, start_sec, end_sec) in enumerate(windows):
            if agent_channel not in (0, 1) or user_channel not in (0, 1) or agent_channel == user_channel:
                raise ValueError(f"invalid agent/user channels {agent_channel}, {user_channel} for {path}")
            duration_sec = end_sec - start_sec
            if duration_sec <= 0:
                raise ValueError(f"invalid conversation window {start_sec}:{end_sec} for {path}")
            info = self._conversation_cache_info(path, agent_channel, user_channel, start_sec, end_sec)
            cache_info[index] = info
            if info is not None and info[1].is_file():
                identity, cache_file = info
                data = torch.load(str(cache_file), map_location="cpu", weights_only=True)
                if data.get("identity") != identity:
                    raise ValueError(f"Mimi cache identity mismatch: {cache_file}")
                agent, user = data.get("agent"), data.get("user")
                self._validate_cached_codes(agent, user, cache_file)
                results[index] = (agent, user)
            else:
                pending.append((index, Path(path), agent_channel, user_channel, start_sec, end_sec))

        if pending:
            audio_items = []
            for result_index, path, agent_channel, user_channel, start_sec, end_sec in pending:
                duration_sec = end_sec - start_sec
                if raw_audio is None:
                    # Windowed seek + resample avoids decoding an entire long conversation.
                    audio = read_stereo_window(
                        path, start_sec, end_sec, self.sample_rate, str(path),
                    )
                else:
                    valid_samples = int(raw_audio["valid_samples"][result_index])
                    audio = raw_audio["waveforms"][result_index, :, :valid_samples].numpy()
                if audio.ndim == 2 and audio.shape[0] != 2 and audio.shape[1] == 2:
                    audio = audio.T
                audio = _pad_audio_window(audio, self.sample_rate, duration_sec)
                if audio.ndim != 2 or audio.shape[0] != 2:
                    raise ValueError(f"conversation audio must decode as [2,T], got {audio.shape}: {path}")
                audio_items.append(audio[[agent_channel, user_channel]])
            lengths = {audio.shape[-1] for audio in audio_items}
            if len(lengths) != 1:
                raise ValueError("batched Mimi windows must have one fixed duration")
            # [B, speaker, T] -> [B*speaker, mono, T], matching the former [2,1,T] call.
            host_batch = torch.from_numpy(
                np.stack(audio_items, axis=0).reshape(-1, 1, audio_items[0].shape[-1])
            )
            use_pinned_transfer = (
                str(self.device).startswith("cuda")
                and raw_audio is not None
                and raw_audio["waveforms"].is_pinned()
            )
            if use_pinned_transfer and not host_batch.is_pinned():
                host_batch = host_batch.pin_memory()
            batch = host_batch.to(
                self.device, dtype=torch.float32, non_blocking=use_pinned_transfer,
            )
            with torch.no_grad():
                encoded = self.mimi.encode(batch).detach().cpu().tolist()
            for pending_index, (result_index, *_rest) in enumerate(pending):
                agent_codes = tuple(tuple(int(token) for token in stream) for stream in encoded[pending_index * 2])
                user_codes = tuple(tuple(int(token) for token in stream) for stream in encoded[pending_index * 2 + 1])
                self._validate_cached_codes(agent_codes, user_codes, Path("<Mimi batch>"))
                results[result_index] = (agent_codes, user_codes)
                info = cache_info[result_index]
                if info is not None:
                    self._write_conversation_cache(info[0], info[1], agent_codes, user_codes)

        if any(result is None for result in results):
            raise RuntimeError("Mimi batch encoding did not produce codes for every conversation")
        return results

    def encode_stereo_waveform(self, waveform, agent_channel: int, user_channel: int):
        """Encode one unpadded [2, T] waveform, preserving Mimi's true end context."""
        import torch

        if waveform.ndim != 2 or waveform.shape[0] != 2 or waveform.shape[-1] == 0:
            raise ValueError("Mimi input must be an unpadded stereo waveform [2, T]")
        if {agent_channel, user_channel} != {0, 1}:
            raise ValueError("agent/user channels must be a permutation of LEFT/RIGHT")
        batch = waveform[[agent_channel, user_channel]].to(self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            codes = self.mimi.encode(batch)
        agent_codes = tuple(tuple(int(token) for token in stream.tolist()) for stream in codes[0, 0])
        user_codes = tuple(tuple(int(token) for token in stream.tolist()) for stream in codes[0, 1])
        return agent_codes, user_codes

    def encode_conversation_stereo_cached(self, path: Path, agent_channel: int, user_channel: int, start_sec: float, end_sec: float):
        """Stereo encode with persistent disk cache.

        On first call, encodes via Mimi GPU forward and saves a ``.pt`` sidecar.
        On subsequent calls, loads from the sidecar – no GPU work needed.
        Falls back to live encoding if cache_dir is not set.
        """
        return self.encode_conversation_stereo_batch([
            (path, agent_channel, user_channel, start_sec, end_sec),
        ])[0]

    def _conversation_cache_info(self, path, agent_channel, user_channel, start_sec, end_sec):
        if self._cache_dir is None:
            return None
        path = Path(path).expanduser().resolve()
        stat = path.stat()
        identity = json.dumps({
            "format_version": 1,
            "source_path": str(path),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "agent_channel": agent_channel,
            "user_channel": user_channel,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "sample_rate": self.sample_rate,
            "mimi_namespace": self._cache_namespace,
        }, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return identity, self._cache_dir / (key + ".pt")

    def _write_conversation_cache(self, identity, cache_file, agent_codes, user_codes):
        import torch

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            torch.save({"identity": identity, "agent": agent_codes, "user": user_codes}, str(temporary))
            os.replace(temporary, cache_file)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_cached_codes(agent, user, path: Path) -> None:
        for label, codes in (("agent", agent), ("user", user)):
            if not isinstance(codes, (tuple, list)) or len(codes) != 8:
                raise ValueError(f"invalid {label} codebooks in Mimi cache: {path}")
            if not codes or len({len(stream) for stream in codes}) != 1:
                raise ValueError(f"inconsistent {label} code lengths in Mimi cache: {path}")
        if len(agent[0]) == 0 or len(agent[0]) != len(user[0]):
            raise ValueError(f"agent/user Mimi frame counts differ in cache: {path}")

    def encode_conversation(self, path: Path, channel: int, start_sec: float, end_sec: float):
        import torch
        duration_sec = end_sec - start_sec
        audio = read_stereo_window(path, start_sec, end_sec, self.sample_rate, str(path))
        audio = _pad_audio_window(audio, self.sample_rate, duration_sec)
        if channel not in (0, 1):
            raise ValueError(f"invalid conversation window {start_sec}:{end_sec} for {path}")
        return self._encode(audio[channel : channel + 1], torch)

    def encode_voice_prompt(self, path: Path):
        key = str(path)
        if key in self._voice_cache:
            return self._voice_cache[key]
        import torch
        audio = self._helpers.load_audio(str(path), self.sample_rate)
        audio = self._helpers.normalize_audio(audio, self.sample_rate, -24.0)
        if audio.ndim == 1:
            audio = audio[None, :]
        codes = self._encode(audio[:1], torch)
        self._voice_cache[key] = codes
        return codes

    def sine(self, frames: int):
        """Repeat PersonaPlex's native sine-conditioning frame without Mimi re-encoding."""
        return self._repeat_native_prompt_frame("SINE_TOKENS", frames)

    def silence(self, frames: int):
        """Repeat PersonaPlex's native silence-conditioning frame without Mimi re-encoding."""
        return self._repeat_native_prompt_frame("SILENCE_TOKENS", frames)

    def _repeat_native_prompt_frame(self, name: str, frames: int):
        if frames < 0:
            raise ValueError("prompt frame count must be non-negative")
        tokens = getattr(self._helpers, name, None)
        if tokens is None or len(tokens) != self.codebooks:
            raise RuntimeError(f"PersonaPlex LM does not expose a valid {name} frame")
        return tuple((int(token),) * frames for token in tokens)

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


def load_runtime(
    paths: RuntimePaths,
    device: str = "cuda",
    qlora: bool = False,
    quant_type: str = "nf4",
    *,
    model_device: str | None = None,
    load_model_weights: bool = True,
    codec_cache_dir: Path | None = None,
) -> PersonaPlexRuntime:
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
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    mimi = loaders.get_mimi(resolved.mimi_weight, device=device)
    mimi.eval()
    mimi.requires_grad_(False)
    mimi_stat = resolved.mimi_weight.stat()
    mimi_cache_identity = json.dumps(
        {
            "path": str(resolved.mimi_weight),
            "size": mimi_stat.st_size,
            "mtime_ns": mimi_stat.st_mtime_ns,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    lm_dtype = torch.bfloat16
    dev_type = getattr(device, "type", str(device))
    if dev_type == "mps":
        try:
            _ = torch.zeros((1,), dtype=torch.bfloat16, device="mps")
        except Exception:
            lm_dtype = torch.float16

    target_model_device = model_device or device
    if qlora:
        if not load_model_weights or target_model_device == "meta":
            raise ValueError("QLoRA cannot be combined with meta-device distributed initialization")
        from .lora import quantize_model_4bit
        model = loaders.get_moshi_lm(resolved.moshi_weight, device="cpu", dtype=lm_dtype)
        model = quantize_model_4bit(model, device=device, quant_type=quant_type)
    else:
        model_path = resolved.moshi_weight if load_model_weights else None
        model = loaders.get_moshi_lm(model_path, device=target_model_device, dtype=lm_dtype)

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
        codec=MimiCodec(
            mimi, mimi.sample_rate, mimi.frame_rate, device, lm_helpers,
            cache_dir=codec_cache_dir,
            cache_namespace=mimi_cache_identity,
        ),
        tokenizer=SentencePieceTokenizer(resolved.tokenizer),
        initial_tokens=initial,
        zero_token=int(model.zero_token_id),
        delays=tuple(int(delay) for delay in model.delays),
    )
