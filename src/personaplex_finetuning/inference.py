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

    runtime = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), config.device)
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


def smoke(config: Config, sample: PreparedSample, adapter: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    generate(config, sample, output_dir / "base.wav", output_dir / "base.txt", None)
    generate(config, sample, output_dir / "finetuned.wav", output_dir / "finetuned.txt", adapter)
    for name in ("base.wav", "finetuned.wav", "base.txt", "finetuned.txt"):
        if not (output_dir / name).is_file() or (output_dir / name).stat().st_size == 0:
            raise RuntimeError(f"inference output is missing or empty: {name}")
    import numpy as np
    import sphn
    base_audio, _ = sphn.read(str(output_dir / "base.wav"))
    finetuned_audio, _ = sphn.read(str(output_dir / "finetuned.wav"))
    if not np.isfinite(base_audio).all() or not np.isfinite(finetuned_audio).all():
        raise RuntimeError("inference generated non-finite audio")
    if np.array_equal(base_audio, finetuned_audio):
        raise RuntimeError("adapter output is identical to base output")
    (output_dir / "run.json").write_text(json.dumps({"sample_id": sample.sample_id, "adapter": str(adapter), "base_model": str(config.model_root), "generation": "greedy native LMGen"}, indent=2))
