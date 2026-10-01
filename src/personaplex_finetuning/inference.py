"""Fresh-process adapter reload and PersonaPlex-native conditioning smoke."""

from __future__ import annotations

import json
import logging
import os
import copy
import unicodedata
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import Config
from .data import AudioInfo, PreparedSample
from .generation import GenerationSettings, generation_from_config
from .lora import inject_lora, load_adapter
from .runtime import RuntimePaths, load_runtime
from .text_normalization import strip_vietnamese_diacritics

logger = logging.getLogger(__name__)


def inference_autocast_context(device):
    """Match BF16 PersonaPlex weights with autocast during CUDA generation."""
    import torch

    if torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _normalize_text_for_metrics(text: str) -> tuple[list[str], str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    words = normalized.split()
    return words, "".join(words)


def _edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def text_error_metrics(
    reference: str,
    hypothesis: str,
    normalize_vietnamese_diacritics: bool = False,
) -> dict[str, float | int] | None:
    """Compute normalized Vietnamese-friendly WER and whitespace-free CER."""
    if normalize_vietnamese_diacritics:
        reference = strip_vietnamese_diacritics(reference)
        hypothesis = strip_vietnamese_diacritics(hypothesis)
    reference_words, reference_characters = _normalize_text_for_metrics(reference)
    hypothesis_words, hypothesis_characters = _normalize_text_for_metrics(hypothesis)
    if not reference_words:
        return None
    word_errors = _edit_distance(reference_words, hypothesis_words)
    character_errors = _edit_distance(reference_characters, hypothesis_characters)
    return {
        "wer": word_errors / len(reference_words),
        "cer": character_errors / max(1, len(reference_characters)),
        "reference_words": len(reference_words),
        "reference_characters": len(reference_characters),
        "word_errors": word_errors,
        "character_errors": character_errors,
    }


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


def _input_audio_window(
    duration_sec: float,
    start_sec: float | None,
    window_seconds: float | None,
) -> tuple[float, float]:
    start = 0.0 if start_sec is None else start_sec
    if start < 0 or start >= duration_sec:
        raise ValueError(f"input start {start:g}s is outside audio duration {duration_sec:g}s")
    if window_seconds is not None and window_seconds <= 0:
        raise ValueError("input window_seconds must be positive")
    end = duration_sec if window_seconds is None else min(start + window_seconds, duration_sec)
    return start, end


def resolve_adapter_checkpoint(adapter: Path) -> tuple[Path, int, int, Path | None, tuple[str, ...]]:
    """Resolve adapter weights, LoRA dimensions/targets, and the training base-model path."""
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
    raw_model_root = metadata.get("model_root")
    if raw_model_root is None:
        model_root = None
    elif isinstance(raw_model_root, str) and raw_model_root.strip():
        model_root = Path(raw_model_root).expanduser().resolve()
    else:
        raise ValueError(f"LoRA checkpoint metadata has an invalid model_root: {metadata_file}")

    # Rank is encoded directly in adapter tensor shapes and is more reliable
    # than metadata from older runs, which could record the config default.
    from safetensors import safe_open

    ranks = set()
    target_prefixes = set()
    with safe_open(str(adapter_file), framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            shape = checkpoint.get_slice(name).get_shape()
            if name.endswith(".lora_a.weight") and len(shape) == 2:
                ranks.add(shape[0])
                target_prefixes.add(name.split(".", 1)[0])
            elif name.endswith(".lora_b.weight") and len(shape) == 2:
                ranks.add(shape[1])
                target_prefixes.add(name.split(".", 1)[0])
    if not ranks:
        raise ValueError(f"adapter file contains no LoRA A/B weight tensors: {adapter_file}")
    if len(ranks) != 1 or next(iter(ranks)) <= 0:
        raise ValueError(f"adapter tensors contain inconsistent LoRA ranks {sorted(ranks)}: {adapter_file}")
    supported_prefixes = {"transformer", "depformer"}
    if not target_prefixes or target_prefixes - supported_prefixes:
        raise ValueError(
            f"adapter tensors contain unsupported LoRA module prefixes {sorted(target_prefixes)}: {adapter_file}"
        )
    prefixes = tuple(name for name in ("transformer", "depformer") if name in target_prefixes)
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
    return adapter_file, rank, alpha, model_root, prefixes


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
    adapter_model_root = None
    adapter_prefixes = None
    if adapter is not None:
        adapter_file, adapter_rank, adapter_alpha, adapter_model_root, adapter_prefixes = resolve_adapter_checkpoint(adapter)
    if settings.use_sampling and seed is not None:
        # Sampling is only reproducible with a pinned RNG (AGENTS.md: seed all randomness).
        torch.manual_seed(int(seed))
        logger.info("seeded torch RNG with %s for reproducible sampling", seed)
    model_root = adapter_model_root or Path(config.model_root)
    if adapter_model_root is not None and adapter_model_root != Path(config.model_root).expanduser().resolve():
        logger.info("using training base model from adapter metadata: %s", model_root)
    runtime = load_runtime(RuntimePaths(model_root, config.personaplex_source), config.device, config.qlora, config.quant_type)
    if adapter_file is not None:
        assert adapter_rank is not None and adapter_alpha is not None
        inject_lora(runtime.model, adapter_rank, adapter_alpha, prefixes=adapter_prefixes)
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
    with (
        torch.no_grad(),
        inference_autocast_context(config.device),
        runtime.codec.mimi.streaming(1),
        generator.streaming(1),
    ):
        generator.step_system_prompts(runtime.codec.mimi)
        for frame in range(user.shape[-1]):
            tokens = generator.step(input_tokens=user[:, :, frame : frame + 1])
            if tokens is None:
                continue
            decoded = runtime.codec.mimi.decode(tokens[:, 1:9]).squeeze().detach().float().cpu().numpy()
            pcm_frames.append(decoded)
            token = int(tokens[0, 0, 0])
            ignored_tokens = (0, runtime.tokenizer.padding_id, getattr(runtime.tokenizer, "end_padding_id", 0))
            if token not in ignored_tokens:
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


def generate_text_with_runtime(
    runtime, sample: PreparedSample, generation: GenerationSettings | None = None, seed: int = 42,
) -> str:
    """Run PersonaPlex free-running text generation with an already-loaded model.

    This is used for validation between training checkpoints so validation measures the
    autoregressive path without loading a second 7B model or decoding generated audio.
    """
    import importlib
    import torch

    settings = GenerationSettings() if generation is None else generation
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    model = runtime.model.module if hasattr(runtime.model, "module") else runtime.model
    was_training = model.training
    model.eval()
    try:
        lm_module = importlib.import_module("moshi.models.lm")
        device = runtime.codec.device
        generator = lm_module.LMGen(
            model, sample_rate=runtime.codec.sample_rate,
            frame_rate=runtime.codec.frame_rate, device=device,
            **settings.lmgen_kwargs(),
        )
        generator.load_voice_prompt(str(sample.voice_prompt_wav))
        generator.text_prompt_tokens = runtime.tokenizer.encode(
            f"<system> {sample.text_prompt.strip()} <system>"
        )
        user_codes = runtime.codec.encode_conversation(
            sample.conversation_wav, sample.user_channel,
            sample.window_start_sec, min(sample.window_end_sec, sample.audio.duration_sec),
        )
        user = torch.tensor(user_codes, device=device).unsqueeze(0)
        token_ids: list[int] = []
        with (
            torch.no_grad(),
            inference_autocast_context(device),
            runtime.codec.mimi.streaming(1),
            generator.streaming(1),
        ):
            generator.step_system_prompts(runtime.codec.mimi)
            for frame in range(user.shape[-1]):
                tokens = generator.step(input_tokens=user[:, :, frame : frame + 1])
                if tokens is None:
                    continue
                token = int(tokens[0, 0, 0])
                ignored_tokens = (0, runtime.tokenizer.padding_id, getattr(runtime.tokenizer, "end_padding_id", 0))
                if token not in ignored_tokens:
                    token_ids.append(token)
        processor = runtime.tokenizer._processor
        if hasattr(processor, "decode_ids"):
            return str(processor.decode_ids(token_ids))
        return "".join(processor.id_to_piece(token) for token in token_ids).strip()
    finally:
        model.train(was_training)


def _export_context(sample: PreparedSample, output_dir: Path) -> None:
    import shutil
    import sphn

    # Export user audio window
    audio, source_rate = sphn.read(str(sample.conversation_wav))
    if source_rate != 24000:
        audio = sphn.resample(audio, src_sample_rate=source_rate, dst_sample_rate=24000)
    if audio.ndim == 1:
        audio = audio[None, :]
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
    agent_words = [
        w.word for w in sample.words
        if w.speaker == "agent" and sample.window_start_sec <= w.start < sample.window_end_sec
    ]
    (output_dir / "agent_reference.txt").write_text(" ".join(agent_words), encoding="utf-8")

    # Export prompts for reference
    (output_dir / "prompt_text.txt").write_text(sample.text_prompt.strip(), encoding="utf-8")
    if sample.voice_prompt_wav.is_file():
        shutil.copyfile(sample.voice_prompt_wav, output_dir / "prompt_voice.wav")


def smoke(
    config: Config,
    sample: PreparedSample | None,
    adapter: Path,
    output_dir: Path,
    input_file: Path | None = None,
    input_start_sec: float | None = None,
    input_window_seconds: float | None = None,
    generation: GenerationSettings | None = None,
    voice_prompt_wav: Path | None = None,
    text_prompt: str | None = None,
) -> str:
    settings = GenerationSettings() if generation is None else generation
    seed = int(getattr(config, "seed", 42))
    logger.info("generation settings: %s (seed=%s)", settings.label(), seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = Path(adapter).expanduser()
    adapter_model_root = None
    if adapter_path.is_file() or adapter_path.is_dir():
        adapter_model_root = resolve_adapter_checkpoint(adapter_path)[3]
    if adapter_model_root is not None:
        # Compare the adapter against the exact base checkpoint it was trained on.
        if hasattr(config, "replace"):
            config = config.replace(model_root=adapter_model_root)
        else:
            config = copy.copy(config)
            config.model_root = adapter_model_root
    if sample is None:
        if input_file is None:
            raise ValueError("inference without external audio requires a prepared sample")
        if voice_prompt_wav is None or text_prompt is None:
            raise ValueError("external inference without a manifest sample requires voice and text prompts")
        if not voice_prompt_wav.is_file():
            raise FileNotFoundError(f"voice prompt does not exist: {voice_prompt_wav}")
    if input_file is not None:
        normalized_input = _prepare_input_audio(input_file, output_dir)

        import sphn

        audio, sample_rate = sphn.read(str(normalized_input))
        if audio.ndim == 1:
            audio = audio[None, :]

        audio_duration_sec = audio.shape[-1] / sample_rate
        window_start_sec, window_end_sec = _input_audio_window(
            audio_duration_sec, input_start_sec, input_window_seconds,
        )

        if sample is None:
            sample = PreparedSample(
                sample_id=input_file.stem,
                conversation_wav=normalized_input,
                voice_prompt_wav=voice_prompt_wav,
                words=(),
                text_prompt=text_prompt,
                metadata={},
                audio=AudioInfo(sample_rate, audio.shape[0], audio_duration_sec),
                window_start_sec=window_start_sec,
                window_end_sec=window_end_sec,
                user_channel=0,
            )
        else:
            sample = replace(
                sample,
                conversation_wav=normalized_input,
                user_channel=0,
                window_start_sec=window_start_sec,
                window_end_sec=window_end_sec,
                words=(),
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

    reference_path = output_dir / "agent_reference.txt"
    reference_text = reference_path.read_text(encoding="utf-8") if reference_path.is_file() else ""
    base_text_path = output_dir / "base.txt"
    finetuned_text_path = output_dir / "finetuned.txt"
    base_text = base_text_path.read_text(encoding="utf-8") if base_text_path.is_file() else ""
    finetuned_text = finetuned_text_path.read_text(encoding="utf-8") if finetuned_text_path.is_file() else ""
    for name, text in (("base", base_text), ("finetuned", finetuned_text)):
        if not text.strip():
            message = f"{name} inference produced an empty transcript"
            logger.warning(message)
            output_warnings.append(message)
    normalize_diacritics = getattr(config, "normalize_vietnamese_diacritics", False)
    base_text_metrics = text_error_metrics(reference_text, base_text, normalize_diacritics)
    finetuned_text_metrics = text_error_metrics(reference_text, finetuned_text, normalize_diacritics)
    if not finetuned_text.strip():
        text_quality_status = "empty_transcript"
    elif finetuned_text_metrics is None:
        text_quality_status = "reference_unavailable"
    elif base_text_metrics is None:
        text_quality_status = "base_reference_unavailable"
    elif finetuned_text_metrics["cer"] < base_text_metrics["cer"]:
        text_quality_status = "improves_over_base"
    else:
        text_quality_status = "does_not_improve_over_base"
        message = (
            "finetuned transcript CER does not improve over base: "
            f"{finetuned_text_metrics['cer']:.4f} >= {base_text_metrics['cer']:.4f}"
        )
        logger.warning(message)
        output_warnings.append(message)
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "sample_id": sample.sample_id,
                "input_file": str(input_file) if input_file is not None else None,
                "window_start_sec": sample.window_start_sec,
                "window_end_sec": sample.window_end_sec,
                "adapter": str(adapter),
                "base_model": str(Path(config.model_root).expanduser().resolve()),
                "text_prompt": sample.text_prompt,
                "voice_prompt": str(sample.voice_prompt_wav),
                "agent_reference_text": reference_text,
                "text_metrics_available": bool(reference_text.strip()),
                "base_text_metrics": base_text_metrics,
                "finetuned_text_metrics": finetuned_text_metrics,
                "text_quality_status": text_quality_status,
                "finetuned_improves_base_cer": (
                    finetuned_text_metrics["cer"] < base_text_metrics["cer"]
                    if finetuned_text_metrics is not None and base_text_metrics is not None else None
                ),
                "generation": settings.label(),
                "generation_settings": settings.as_dict(),
                "seed": seed,
                "stereo_mapping": (
                    "Input channel 0 = User context" if input_file is not None
                    else "Channel 0 (LEFT) = Agent, Channel 1 (RIGHT) = User"
                ),
                "warnings": output_warnings,
            },
            indent=2,
        )
    )
    return text_quality_status
