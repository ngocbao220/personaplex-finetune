"""Fresh-process adapter reload and PersonaPlex-native conditioning smoke."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .data import PreparedSample
from .lora import inject_lora, load_adapter
from .runtime import RuntimePaths, load_runtime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationSettings:
    """Sampling settings forwarded to moshi's ``LMGen``.

    The defaults mirror Moshi/PersonaPlex (``LMGen.__init__``) and are exactly what the
    inference smoke test hard-coded before the ``generation:`` config block existed, so an
    absent or partial block keeps the previous behaviour unchanged.
    """

    use_sampling: bool = True
    temp: float = 0.8
    temp_text: float = 0.7
    top_k: int = 250
    top_k_text: int = 25
    audio_silence_frame_cnt: int = 6

    def lmgen_kwargs(self) -> dict[str, Any]:
        """Keyword arguments handed to ``LMGen`` and the structured run-report payload."""
        return {
            "use_sampling": self.use_sampling,
            "temp": self.temp,
            "temp_text": self.temp_text,
            "top_k": self.top_k,
            "top_k_text": self.top_k_text,
            "audio_silence_frame_cnt": self.audio_silence_frame_cnt,
        }

    def as_dict(self) -> dict[str, Any]:
        return self.lmgen_kwargs()

    def label(self) -> str:
        """Describe the mechanism ``sample_token`` really uses, per stream.

        ``moshi.utils.sampling.sample_token`` falls back to argmax when sampling is disabled
        or the temperature is not positive, so the audio and text streams are labelled
        separately instead of claiming one global mode.
        """
        audio = "greedy" if (not self.use_sampling or self.temp <= 0.0) else f"sampling temp={self.temp:g}"
        text = "greedy" if (not self.use_sampling or self.temp_text <= 0.0) else f"sampling temp={self.temp_text:g}"
        return (
            f"native LMGen audio={audio}, text={text}, top_k={self.top_k}, "
            f"top_k_text={self.top_k_text}, audio_silence_frame_cnt={self.audio_silence_frame_cnt}"
        )


def _as_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return value.strip().lower() in {"true", "1"}
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _as_float(name: str, value: Any, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not np.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum:g}, got {value!r}")
    return number


def _as_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if float(value) != int(value):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return number


def generation_from_config(raw: Mapping[str, Any] | None) -> GenerationSettings:
    """Read the optional ``generation:`` section of a raw (Hydra/OmegaConf) config.

    Missing keys fall back to the Moshi defaults. Unknown keys and out-of-range values
    raise immediately, so a typo in the YAML can never be mistaken for a working setting.
    """
    defaults = GenerationSettings()
    section = {} if raw is None else raw.get("generation")
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError("generation config section must be a mapping")
    supported = sorted(defaults.as_dict())
    unknown = sorted(set(section) - set(supported))
    if unknown:
        raise ValueError(f"unknown generation config keys: {unknown}; supported keys are {supported}")
    return GenerationSettings(
        use_sampling=_as_bool("generation.use_sampling", section.get("use_sampling", defaults.use_sampling)),
        temp=_as_float("generation.temp", section.get("temp", defaults.temp), minimum=0.0),
        temp_text=_as_float("generation.temp_text", section.get("temp_text", defaults.temp_text), minimum=0.0),
        top_k=_as_int("generation.top_k", section.get("top_k", defaults.top_k), minimum=0),
        top_k_text=_as_int("generation.top_k_text", section.get("top_k_text", defaults.top_k_text), minimum=0),
        audio_silence_frame_cnt=_as_int(
            "generation.audio_silence_frame_cnt",
            section.get("audio_silence_frame_cnt", defaults.audio_silence_frame_cnt),
            minimum=0,
        ),
    )


def _prepare_input_audio(input_file: Path, output_dir: Path) -> Path:
    """Normalize an external input file to the codec sample rate."""
    import sphn

    if not input_file.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_file}")

    audio, sample_rate = sphn.read(str(input_file))
    if sample_rate != 24000:
        audio = sphn.resample(
            audio,
            src_sample_rate=sample_rate,
            dst_sample_rate=24000,
        )

    normalized_path = output_dir / "input_24k.wav"
    sphn.write_wav(str(normalized_path), audio, 24000)
    return normalized_path


def resolve_adapter_checkpoint(adapter: Path) -> tuple[Path, int, int]:
    """Resolve adapter weights and LoRA dimensions from checkpoint contents."""
    adapter = Path(adapter).expanduser()
    if adapter.is_dir():
        checkpoint_dir = adapter
        adapter_file = checkpoint_dir / "lora.safetensors"
    else:
        adapter_file = adapter
        checkpoint_dir = adapter.parent
    if not adapter_file.is_file():
        raise FileNotFoundError(f"LoRA adapter weights not found: {adapter_file}")

    metadata_file = checkpoint_dir / "adapter.json"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"LoRA checkpoint metadata not found: {metadata_file}")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read LoRA checkpoint metadata: {metadata_file}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"LoRA checkpoint metadata must be a JSON object: {metadata_file}")

    # Rank is encoded directly in adapter tensor shapes and is more reliable
    # than metadata from older runs, which could record the config default.
    from safetensors import safe_open

    ranks = set()
    with safe_open(str(adapter_file), framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            shape = checkpoint.get_slice(name).get_shape()
            if name.endswith(".lora_a.weight") and len(shape) == 2:
                ranks.add(shape[0])
            elif name.endswith(".lora_b.weight") and len(shape) == 2:
                ranks.add(shape[1])
    if not ranks:
        raise ValueError(f"adapter file contains no LoRA A/B weight tensors: {adapter_file}")
    if len(ranks) != 1 or next(iter(ranks)) <= 0:
        raise ValueError(f"adapter tensors contain inconsistent LoRA ranks {sorted(ranks)}: {adapter_file}")
    rank = next(iter(ranks))
    metadata_rank = metadata.get("rank")
    if metadata_rank != rank:
        logger.warning(
            "LoRA rank metadata (%r) disagrees with adapter tensor shapes (rank=%d); using tensor rank",
            metadata_rank, rank,
        )

    alpha = metadata.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, int) or alpha <= 0:
        raise ValueError(f"LoRA checkpoint metadata must contain a positive integer 'alpha': {metadata_file}")
    scaling = metadata.get("scaling")
    if scaling is not None:
        if isinstance(scaling, bool) or not isinstance(scaling, (int, float)) or scaling <= 0:
            raise ValueError(f"LoRA checkpoint metadata must contain a positive 'scaling': {metadata_file}")
        alpha = round(rank * scaling)
    else:
        # Older checkpoints wrote config.lora_alpha into adapter.json even
        # though training injected alpha=rank*lora_scaling. Their run-level
        # config.json records the effective scaling needed to recover alpha.
        run_config_file = checkpoint_dir.parent.parent / "config.json"
        if run_config_file.is_file():
            try:
                run_config = json.loads(run_config_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                run_config = None
            if isinstance(run_config, dict):
                run_rank = run_config.get("lora_rank")
                run_scaling = run_config.get("lora_scaling")
                if run_rank == rank and isinstance(run_scaling, (int, float)) and run_scaling > 0:
                    alpha = round(rank * run_scaling)
    return adapter_file, rank, alpha


def generate(
    config: Config,
    sample: PreparedSample,
    output_wav: Path,
    output_text: Path,
    adapter: Path | None,
    generation: GenerationSettings | None = None,
    seed: int | None = None,
) -> None:
    import importlib
    import numpy as np
    import sphn
    import torch

    # Mimi's streaming decoder uses ConvTranspose1d shapes that have failed in
    # Moshi's lazy torch.compile path on deployed CUDA/cuDNN stacks. Set this
    # before load_runtime imports any Moshi modules so the native eager path is
    # selected for both model and codec.
    os.environ.setdefault("NO_TORCH_COMPILE", "1")

    settings = GenerationSettings() if generation is None else generation
    adapter_file = None
    adapter_rank = None
    adapter_alpha = None
    if adapter is not None:
        adapter_file, adapter_rank, adapter_alpha = resolve_adapter_checkpoint(adapter)
    if settings.use_sampling and seed is not None:
        # Sampling is only reproducible with a pinned RNG (AGENTS.md: seed all randomness).
        torch.manual_seed(int(seed))
        logger.info("seeded torch RNG with %s for reproducible sampling", seed)
    runtime = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), config.device, config.qlora, config.quant_type)
    if adapter_file is not None:
        assert adapter_rank is not None and adapter_alpha is not None
        inject_lora(runtime.model, adapter_rank, adapter_alpha)
        load_adapter(runtime.model, adapter_file)
    runtime.model.eval()
    lm_module = importlib.import_module("moshi.models.lm")
    generator = lm_module.LMGen(
        runtime.model, sample_rate=runtime.codec.sample_rate,
        frame_rate=runtime.codec.frame_rate, device=config.device,
        **settings.lmgen_kwargs(),
    )
    generator.load_voice_prompt(str(sample.voice_prompt_wav))
    generator.text_prompt_tokens = runtime.tokenizer.encode(f"<system> {sample.text_prompt.strip()} <system>")
    user_codes = runtime.codec.encode_conversation(sample.conversation_wav, sample.user_channel, sample.window_start_sec, sample.window_end_sec)
    user = torch.tensor(user_codes, device=config.device).unsqueeze(0)
    pcm_frames: list[np.ndarray] = []
    text_token_ids: list[int] = []
    # Mimi's decoder is causal/streaming: resetting it for every 80 ms frame
    # inserts boundary transients that sound like clicks and clipped syllables.
    with torch.no_grad(), runtime.codec.mimi.streaming(1), generator.streaming(1):
        generator.step_system_prompts(runtime.codec.mimi)
        for frame in range(user.shape[-1]):
            tokens = generator.step(input_tokens=user[:, :, frame : frame + 1])
            if tokens is None:
                continue
            decoded = runtime.codec.mimi.decode(tokens[:, 1:9]).squeeze().detach().float().cpu().numpy()
            pcm_frames.append(decoded)
            token = int(tokens[0, 0, 0])
            if token not in (0, runtime.tokenizer.padding_id):
                text_token_ids.append(token)
    if not pcm_frames:
        raise RuntimeError("native PersonaPlex generation produced no frames")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(output_wav), np.concatenate(pcm_frames), runtime.codec.sample_rate)
    # SentencePiece decode_ids merges multi-byte tokens into clean Vietnamese text
    if hasattr(runtime.tokenizer._processor, "decode_ids"):
        cleaned_text = runtime.tokenizer._processor.decode_ids(text_token_ids)
    else:
        pieces = [runtime.tokenizer._processor.id_to_piece(t) for t in text_token_ids]
        cleaned_text = "".join(pieces).replace(" ", " ").strip()
    output_text.write_text(cleaned_text, encoding="utf-8")


def _export_context(sample: PreparedSample, output_dir: Path) -> None:
    import shutil
    import sphn

    # Export user audio window
    audio, source_rate = sphn.read(str(sample.conversation_wav))
    if source_rate != 24000:
        audio = sphn.resample(audio, src_sample_rate=source_rate, dst_sample_rate=24000)
    start = int(sample.window_start_sec * 24000)
    end = int(sample.window_end_sec * 24000)
    original_window = np.ascontiguousarray(audio[..., start:end])
    if original_window.size == 0:
        raise ValueError(f"{sample.sample_id}: original dialogue window contains no audio")
    sphn.write_wav(str(output_dir / "dialogue_original.wav"), original_window, 24000)
    user_audio = audio[sample.user_channel, start:end]
    sphn.write_wav(str(output_dir / "user.wav"), user_audio, 24000)

    # Export user text in window
    user_words = [
        w.word for w in sample.words
        if w.speaker == "user" and w.end >= sample.window_start_sec and w.start <= sample.window_end_sec
    ]
    (output_dir / "user.txt").write_text(" ".join(user_words), encoding="utf-8")

    # Export prompts for reference
    (output_dir / "prompt_text.txt").write_text(sample.text_prompt.strip(), encoding="utf-8")
    if sample.voice_prompt_wav.is_file():
        shutil.copyfile(sample.voice_prompt_wav, output_dir / "prompt_voice.wav")


def smoke(
    config: Config,
    sample: PreparedSample,
    adapter: Path,
    output_dir: Path,
    input_file: Path | None = None,
    generation: GenerationSettings | None = None,
) -> None:
    settings = GenerationSettings() if generation is None else generation
    seed = int(getattr(config, "seed", 42))
    logger.info("generation settings: %s (seed=%s)", settings.label(), seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_file is not None:
        normalized_input = _prepare_input_audio(input_file, output_dir)

        import sphn

        audio, sample_rate = sphn.read(str(normalized_input))
        if audio.ndim == 1:
            audio = audio[None, :]

        sample = replace(
            sample,
            conversation_wav=normalized_input,
            user_channel=0,
            window_start_sec=0.0,
            window_end_sec=audio.shape[-1] / sample_rate,
        )

    _export_context(sample, output_dir)
    # Base and fine-tuned runs share the same sampling settings and the same seed, so any
    # audible difference comes from the adapter and not from a different random draw.
    generate(config, sample, output_dir / "base.wav", output_dir / "base.txt", None, generation=settings, seed=seed)
    generate(
        config, sample, output_dir / "finetuned.wav", output_dir / "finetuned.txt", adapter,
        generation=settings, seed=seed,
    )
    output_warnings = []
    for name in ("dialogue_original.wav", "user.wav", "base.wav", "finetuned.wav", "base.txt", "finetuned.txt"):
        path = output_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            message = f"inference output is missing or empty: {path}"
            logger.warning(message)
            output_warnings.append(message)

    import numpy as np
    import sphn

    def _read_audio(name: str) -> np.ndarray | None:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            return None
        try:
            audio, _ = sphn.read(str(path))
        except Exception as exc:
            message = f"could not read inference output {path}: {exc}"
            logger.warning(message)
            output_warnings.append(message)
            return None
        return audio

    base_audio = _read_audio("base.wav")
    finetuned_audio = _read_audio("finetuned.wav")
    user_audio = _read_audio("user.wav")
    for name, audio in (("base.wav", base_audio), ("finetuned.wav", finetuned_audio)):
        if audio is not None and not np.isfinite(audio).all():
            message = f"inference output contains non-finite audio: {output_dir / name}"
            logger.warning(message)
            output_warnings.append(message)
            if name == "base.wav":
                base_audio = None
            else:
                finetuned_audio = None
    if base_audio is not None and finetuned_audio is not None and np.array_equal(base_audio, finetuned_audio):
        logger.warning("adapter output is identical to base output")

    # Export stereo dialogue: LEFT = Agent, RIGHT = User
    def _make_stereo(agent_pcm: np.ndarray, user_pcm: np.ndarray) -> np.ndarray:
        a = agent_pcm.squeeze()
        u = user_pcm.squeeze()
        min_len = min(len(a), len(u))
        return np.stack([a[:min_len], u[:min_len]], axis=0)

    if base_audio is not None and user_audio is not None:
        sphn.write_wav(str(output_dir / "dialogue_base.wav"), _make_stereo(base_audio, user_audio), 24000)
    if finetuned_audio is not None and user_audio is not None:
        sphn.write_wav(str(output_dir / "dialogue_finetune.wav"), _make_stereo(finetuned_audio, user_audio), 24000)

    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "sample_id": sample.sample_id,
                "input_file": str(input_file) if input_file is not None else None,
                "window_start_sec": sample.window_start_sec,
                "window_end_sec": sample.window_end_sec,
                "adapter": str(adapter),
                "base_model": str(config.model_root),
                "generation": settings.label(),
                "generation_settings": settings.as_dict(),
                "seed": seed,
                "stereo_mapping": "Channel 0 (LEFT) = Agent, Channel 1 (RIGHT) = User",
                "warnings": output_warnings,
            },
            indent=2,
        )
    )
