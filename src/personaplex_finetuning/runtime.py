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

from .data import read_audio_window, read_stereo_window

PERSONAPLEX_MIMI_FRAME_RATE = 12.5


def _pad_audio_window(audio, sample_rate: int, duration_sec: float):
    """Match the fixed final-chunk padding performed by the reference dataset loader."""
    import numpy as np

    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or audio.shape[-1] == 0:
        raise ValueError(f"cannot pad missing or non-stereo/mono decoded audio: {audio.shape}")

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
        self._path = Path(path)
        sentencepiece = importlib.import_module("sentencepiece")
        self._processor = sentencepiece.SentencePieceProcessor(str(self._path))

    def __getstate__(self) -> dict[str, str]:
        # Spawned CPU workers should reopen the small local model instead of
        # trying to pickle SentencePiece's native processor object.
        return {"path": str(self._path)}

    def __setstate__(self, state: dict[str, str]) -> None:
        self.__init__(Path(state["path"]))

    def encode(self, text: str) -> list[int]:
        return list(self._processor.encode(text))

    def decode(self, tokens: list[int]) -> str:
        return str(self._processor.decode(tokens))


def torch_device_type(device) -> str:
    """``cuda:1`` -> ``cuda``; Mimi codes do not depend on which GPU encoded them."""
    return str(device).split(":", 1)[0]


class MimiCodec:
    """Mimi adapter using the same source helpers as PersonaPlex inference."""

    codebooks = 8
    conversation_encoding_contract = "mono-batch1-fp32-no-autocast"

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
        # The training batch prefetch thread and evaluation may both encode; Mimi's
        # streaming state is not thread-safe, so live encodes are serialized.
        self._mimi_lock = threading.RLock()
        # Cumulative per-process counters: disk-cache hits vs live Mimi encodes.
        self.cache_stats = {"dialogue_cache_hit": 0, "dialogue_encoded": 0,
                            "voice_cache_hit": 0, "voice_encoded": 0}
        self._cache_dir = cache_dir  # Optional persistent disk cache for conversation encoding
        self._cache_namespace = cache_namespace

    def encode_conversation_stereo(self, path: Path, agent_channel: int, user_channel: int, start_sec: float, end_sec: float):
        return self.encode_conversation_stereo_batch([
            (path, agent_channel, user_channel, start_sec, end_sec),
        ])[0]

    def encode_conversation_stereo_batch(self, windows, raw_audio=None):
        """Encode stereo windows with the same mono batch-1 contract as inference.

        ``windows`` contains ``(path, agent_channel, user_channel, start_sec, end_sec)``
        tuples. Agent/user channels are encoded separately, independent of LM batch size
        or cache hits. Grouped transfer preserves per-window padding and channel order.
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
            info = self._conversation_cache_info(path, start_sec, end_sec)
            cache_info[index] = info
            if info is not None and info[1].is_file():
                identity, cache_file = info
                data = torch.load(str(cache_file), map_location="cpu", weights_only=True)
                if data.get("identity") != identity:
                    raise ValueError(f"Mimi cache identity mismatch: {cache_file}")
                # Stored by physical LEFT/RIGHT channel, so role-swapped passes hit too.
                channels = (data.get("left"), data.get("right"))
                agent, user = channels[agent_channel], channels[user_channel]
                self._validate_cached_codes(agent, user, cache_file)
                results[index] = (agent, user)
                self.cache_stats["dialogue_cache_hit"] += 1
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
            # Group only the transfer; Mimi always receives [1,1,T] per channel.
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
            encoded = [self._encode(channel, torch) for channel in batch]
            self.cache_stats["dialogue_encoded"] += len(pending)
            for pending_index, (result_index, *_rest) in enumerate(pending):
                agent_codes = encoded[pending_index * 2]
                user_codes = encoded[pending_index * 2 + 1]
                self._validate_cached_codes(agent_codes, user_codes, Path("<Mimi batch>"))
                results[result_index] = (agent_codes, user_codes)
                info = cache_info[result_index]
                if info is not None:
                    agent_channel = pending[pending_index][2]
                    left, right = (agent_codes, user_codes) if agent_channel == 0 else (user_codes, agent_codes)
                    self._write_conversation_cache(info[0], info[1], left, right)

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
        agent_codes = self._encode(waveform[agent_channel : agent_channel + 1], torch)
        user_codes = self._encode(waveform[user_channel : user_channel + 1], torch)
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

    def _conversation_cache_info(self, path, start_sec, end_sec):
        if self._cache_dir is None:
            return None
        path = Path(path).expanduser().resolve()
        stat = path.stat()
        identity = json.dumps({
            "format_version": 3,
            "encoding_contract": self.conversation_encoding_contract,
            # Device type only: DDP ranks (cuda:0/cuda:1) share one cache across reshuffles.
            "encoding_device": torch_device_type(self.device),
            "source_path": str(path),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "sample_rate": self.sample_rate,
            "mimi_namespace": self._cache_namespace,
        }, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return identity, self._cache_dir / (key + ".pt")

    def _write_conversation_cache(self, identity, cache_file, left_codes, right_codes):
        import torch

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            torch.save({"identity": identity, "left": left_codes, "right": right_codes}, str(temporary))
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
        if channel not in (0, 1):
            raise ValueError(f"invalid conversation window {start_sec}:{end_sec} for {path}")
        audio = read_audio_window(path, start_sec, end_sec, self.sample_rate, str(path), channels=(1, 2))
        if channel >= audio.shape[0]:
            raise ValueError(f"channel {channel} is unavailable in {audio.shape[0]}-channel audio: {path}")
        audio = _pad_audio_window(audio, self.sample_rate, duration_sec)
        return self._encode(audio[channel : channel + 1], torch)

    def _voice_prompt_cache_info(self, path):
        if self._cache_dir is None:
            return None
        path = Path(path).expanduser().resolve()
        stat = path.stat()
        identity = json.dumps({
            "kind": "voice_prompt", "format_version": 1,
            "encoding_contract": "native-streaming-batch1-normalized-24lufs",
            "encoding_device": torch_device_type(self.device),
            "source_path": str(path), "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
            "sample_rate": self.sample_rate, "mimi_namespace": self._cache_namespace,
        }, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return identity, self._cache_dir / "voice" / (key + ".pt")

    def encode_voice_prompt(self, path: Path):
        key = str(path)
        if key in self._voice_cache:
            return self._voice_cache[key]
        import torch
        # Streaming Mimi encodes the prompt frame by frame on the training thread;
        # a disk hit keeps that latency off every first-seen conversation.
        info = self._voice_prompt_cache_info(path)
        if info is not None and info[1].is_file():
            data = torch.load(str(info[1]), map_location="cpu", weights_only=True)
            if data.get("identity") != info[0]:
                raise ValueError(f"Mimi voice-prompt cache identity mismatch: {info[1]}")
            codes = data["codes"]
            if not isinstance(codes, (tuple, list)) or len(codes) != 8 or not codes[0]:
                raise ValueError(f"invalid voice-prompt codes in Mimi cache: {info[1]}")
            codes = tuple(tuple(stream) for stream in codes)
            self._voice_cache[key] = codes
            self.cache_stats["voice_cache_hit"] += 1
            return codes
        audio = self._helpers.load_audio(str(path), self.sample_rate)
        audio = self._helpers.normalize_audio(audio, self.sample_rate, -24.0)
        if audio.ndim == 1:
            audio = audio[None, :]
        frame_size = int(self.sample_rate / self.frame_rate)
        with self._mimi_lock, torch.no_grad(), self.mimi.streaming(1):
            native_frames = list(self._helpers.encode_from_sphn(
                self.mimi,
                self._helpers._iterate_audio(audio[:1], frame_size, pad=True),
                max_batch=1,
            ))
        if not native_frames:
            raise ValueError(f"voice prompt has no Mimi frames: {path}")
        encoded = torch.cat(native_frames, dim=2)[0]
        codes = tuple(tuple(int(token) for token in stream.tolist()) for stream in encoded)
        self._voice_cache[key] = codes
        self.cache_stats["voice_encoded"] += 1
        if info is not None:
            cache_file = info[1]
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                torch.save({"identity": info[0], "codes": codes}, str(temporary))
                os.replace(temporary, cache_file)
            finally:
                temporary.unlink(missing_ok=True)
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
        audio = torch.as_tensor(audio, dtype=torch.float32, device=self.device)
        if audio.ndim != 2 or audio.shape[0] != 1 or audio.shape[-1] == 0:
            raise ValueError("Mimi encoding requires one nonempty mono channel [1,T]")
        # A shared contract avoids batch-dependent quantization and inherited LM autocast.
        with self._mimi_lock, torch.no_grad(), \
                torch.autocast(device_type=torch.device(self.device).type, enabled=False):
            codes = self.mimi.encode(audio.unsqueeze(0))[0]
        return tuple(tuple(int(token) for token in stream.tolist()) for stream in codes)



@dataclass
class PersonaPlexRuntime:
    model: object
    codec: MimiCodec
    tokenizer: SentencePieceTokenizer
    initial_tokens: tuple[int, ...]
    zero_token: int
    delays: tuple[int, ...]


def _load_mimi_codec(resolved, device: str, codec_cache_dir: Path | None) -> MimiCodec:
    """Frozen Mimi codec; the cache namespace is shared by training and precompute."""
    loaders = importlib.import_module("moshi.models.loaders")
    lm_helpers = importlib.import_module("moshi.models.lm")
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
    return MimiCodec(
        mimi, mimi.sample_rate, mimi.frame_rate, device, lm_helpers,
        cache_dir=codec_cache_dir,
        cache_namespace=mimi_cache_identity,
    )


def load_mimi_codec(paths: RuntimePaths, device: str, codec_cache_dir: Path | None) -> MimiCodec:
    """Load only Mimi from explicit local assets, without the 7B language model."""
    resolved = paths.validate(require_model=False)
    source = str(resolved.source)
    if source not in sys.path:
        sys.path.insert(0, source)
    return _load_mimi_codec(resolved, device, codec_cache_dir)


def load_runtime(
    paths: RuntimePaths,
    device: str = "cuda",
    qlora: bool = False,
    quant_type: str = "nf4",
    *,
    model_device: str | None = None,
    load_model_weights: bool = True,
    codec_cache_dir: Path | None = None,
    full_precision_model: bool = False,
) -> PersonaPlexRuntime:
    """Load from explicit local assets only. No Hugging Face function is imported."""
    resolved = paths.validate()
    source = str(resolved.source)
    if source not in sys.path:
        sys.path.insert(0, source)
    torch = importlib.import_module("torch")
    loaders = importlib.import_module("moshi.models.loaders")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if torch.cuda.is_available():
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    codec = _load_mimi_codec(resolved, device, codec_cache_dir)
    lm_dtype = torch.float32 if full_precision_model else torch.bfloat16
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
        codec=codec,
        tokenizer=SentencePieceTokenizer(resolved.tokenizer),
        initial_tokens=initial,
        zero_token=int(model.zero_token_id),
        delays=tuple(int(delay) for delay in model.delays),
    )
