"""Fresh-process adapter reload and PersonaPlex-native conditioning smoke."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .data import PreparedSample
from .lora import inject_lora, load_adapter
from .runtime import RuntimePaths, load_runtime


def generate(config: Config, sample: PreparedSample, output_wav: Path, output_text: Path, adapter: Path | None) -> None:
    import importlib
    import numpy as np
    import sphn
    import torch

    runtime = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), config.device, config.qlora, config.quant_type)
    if adapter is not None:
        inject_lora(runtime.model, config.lora_rank, config.lora_alpha)
        load_adapter(runtime.model, adapter)
    runtime.model.eval()
    lm_module = importlib.import_module("moshi.models.lm")
    generator = lm_module.LMGen(
        runtime.model, audio_silence_frame_cnt=6, sample_rate=runtime.codec.sample_rate,
        frame_rate=runtime.codec.frame_rate, device=config.device, use_sampling=False,
    )
    generator.load_voice_prompt(str(sample.voice_prompt_wav))
    generator.text_prompt_tokens = runtime.tokenizer.encode(f"<system> {sample.text_prompt.strip()} <system>")
    user_codes = runtime.codec.encode_conversation(sample.conversation_wav, sample.user_channel, sample.window_start_sec, sample.window_end_sec)
    user = torch.tensor(user_codes, device=config.device).unsqueeze(0)
    pcm_frames: list[np.ndarray] = []
    text_tokens: list[str] = []
    with torch.no_grad(), generator.streaming(1):
        for frame in range(user.shape[-1]):
            tokens = generator.step(input_tokens=user[:, :, frame : frame + 1])
            if tokens is None:
                continue
            decoded = runtime.codec.mimi.decode(tokens[:, 1:9]).squeeze().detach().float().cpu().numpy()
            pcm_frames.append(decoded)
            token = int(tokens[0, 0, 0])
            if token not in (0, runtime.tokenizer.padding_id):
                text_tokens.append(runtime.tokenizer._processor.id_to_piece(token).replace("▁", " "))
    if not pcm_frames:
        raise RuntimeError("native PersonaPlex generation produced no frames")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(output_wav), np.concatenate(pcm_frames), runtime.codec.sample_rate)
    output_text.write_text("".join(text_tokens), encoding="utf-8")


def _export_context(sample: PreparedSample, output_dir: Path) -> None:
    import shutil
    import sphn

    # Export user audio window
    audio, source_rate = sphn.read(str(sample.conversation_wav))
    if source_rate != 24000:
        audio = sphn.resample(audio, src_sample_rate=source_rate, dst_sample_rate=24000)
    start = int(sample.window_start_sec * 24000)
    end = int(sample.window_end_sec * 24000)
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


def smoke(config: Config, sample: PreparedSample, adapter: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    _export_context(sample, output_dir)
    generate(config, sample, output_dir / "base.wav", output_dir / "base.txt", None)
    generate(config, sample, output_dir / "finetuned.wav", output_dir / "finetuned.txt", adapter)
    for name in ("user.wav", "base.wav", "finetuned.wav", "base.txt", "finetuned.txt"):
        if not (output_dir / name).is_file() or (output_dir / name).stat().st_size == 0:
            raise RuntimeError(f"inference output is missing or empty: {name}")
    import numpy as np
    import sphn
    base_audio, _ = sphn.read(str(output_dir / "base.wav"))
    finetuned_audio, _ = sphn.read(str(output_dir / "finetuned.wav"))
    user_audio, _ = sphn.read(str(output_dir / "user.wav"))
    if not np.isfinite(base_audio).all() or not np.isfinite(finetuned_audio).all():
        raise RuntimeError("inference generated non-finite audio")
    if np.array_equal(base_audio, finetuned_audio):
        raise RuntimeError("adapter output is identical to base output")

    # Export stereo dialogue: LEFT = Agent, RIGHT = User
    def _make_stereo(agent_pcm: np.ndarray, user_pcm: np.ndarray) -> np.ndarray:
        a = agent_pcm.squeeze()
        u = user_pcm.squeeze()
        min_len = min(len(a), len(u))
        return np.stack([a[:min_len], u[:min_len]], axis=0)

    sphn.write_wav(str(output_dir / "dialogue_base.wav"), _make_stereo(base_audio, user_audio), 24000)
    sphn.write_wav(str(output_dir / "dialogue_finetune.wav"), _make_stereo(finetuned_audio, user_audio), 24000)

    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "sample_id": sample.sample_id,
                "window_start_sec": sample.window_start_sec,
                "window_end_sec": sample.window_end_sec,
                "adapter": str(adapter),
                "base_model": str(config.model_root),
                "generation": "greedy native LMGen",
                "stereo_mapping": "Channel 0 (LEFT) = Agent, Channel 1 (RIGHT) = User",
            },
            indent=2,
        )
    )
