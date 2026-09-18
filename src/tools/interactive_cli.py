"""Terminal CLI for live interactive conversation with PersonaPlex using sounddevice."""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("interactive_cli")


def list_audio_devices():
    """List available audio devices via sounddevice."""
    try:
        import sounddevice as sd
        print("\n=== Available Audio Devices ===")
        print(sd.query_devices())
        print("===============================\n")
    except Exception as exc:
        print(f"Error querying audio devices: {exc}", file=sys.stderr)


def make_ready_chime(sample_rate: int, frame_size: int) -> list[np.ndarray]:
    """Generate a soft two-tone chime (F5 698Hz -> A5 880Hz) to signal the system is ready."""
    try:
        dur1, dur2 = 0.12, 0.14
        t1 = np.linspace(0, dur1, int(dur1 * sample_rate), endpoint=False)
        t2 = np.linspace(0, dur2, int(dur2 * sample_rate), endpoint=False)
        tone1 = 0.08 * np.sin(2 * np.pi * 698 * t1) * np.exp(-t1 * 18)
        tone2 = 0.08 * np.sin(2 * np.pi * 880 * t2) * np.exp(-t2 * 12)
        full_chime = np.concatenate([tone1, tone2]).astype(np.float32)

        # Pad to multiple of frame_size
        pad_len = (frame_size - (len(full_chime) % frame_size)) % frame_size
        if pad_len > 0:
            full_chime = np.pad(full_chime, (0, pad_len))

        return [full_chime[i : i + frame_size] for i in range(0, len(full_chime), frame_size)]
    except Exception:
        return []


def run_live_interaction(
    session,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    save_session_dir: Path | None = None,
):
    """Run real-time duplex speech interaction using sounddevice."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is required for terminal CLI interaction. Install via: pip install sounddevice") from exc

    sample_rate = session.sample_rate  # 24000
    frame_size = session.frame_size    # 1920 (80ms)

    input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
    output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)

    recorded_user_pcm: list[np.ndarray] = []
    recorded_agent_pcm: list[np.ndarray] = []
    full_transcript: list[str] = []

    stop_event = threading.Event()
    last_user_activity = 0.0

    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal last_user_activity
        # Capture mono microphone input
        mono_input = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(mono_input ** 2))) if np is not None else 0.0
        if rms > 0.012:
            last_user_activity = time.time()

        try:
            input_queue.put_nowait(mono_input)
        except queue.Full:
            pass

        # Output agent audio to speaker
        try:
            agent_chunk = output_queue.get_nowait()
            outdata[:, 0] = agent_chunk
        except queue.Empty:
            outdata.fill(0)

    # Enqueue a soft chime to alert the user that audio is active
    for chime_chunk in make_ready_chime(sample_rate, frame_size):
        output_queue.put_nowait(chime_chunk)

    def worker_loop():
        """Process incoming user audio chunks and queue outgoing agent audio chunks."""
        first_token = True
        agent_speaking = False
        indicator_printed = False

        while not stop_event.is_set():
            try:
                user_chunk = input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if save_session_dir is not None:
                recorded_user_pcm.append(user_chunk)

            agent_chunk, text_piece = session.step_audio_chunk(user_chunk)

            if agent_chunk is not None:
                if save_session_dir is not None:
                    recorded_agent_pcm.append(agent_chunk)
                try:
                    output_queue.put_nowait(agent_chunk)
                except queue.Full:
                    pass

            if text_piece:
                if not agent_speaking:
                    sys.stdout.write("\n\033[1;32m[PersonaPlex 🗣️]\033[0m: ")
                    agent_speaking = True
                    indicator_printed = False
                sys.stdout.write(text_piece)
                sys.stdout.flush()
                full_transcript.append(text_piece)
            else:
                if agent_speaking and agent_chunk is None:
                    agent_speaking = False
                    sys.stdout.write("\n\033[2m[🟢 Đang lắng nghe...]\033[0m\n")
                    sys.stdout.flush()

    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()

    print("\n" + "═" * 68)
    print("  \033[1;32m🟢 HỆ THỐNG ĐÃ KÍCH HOẠT (ACTIVE) — MICROPHONE ĐANG LẮNG NGHE!\033[0m")
    print("═" * 68)
    print("  • \033[1mHãy nói vào microphone của bạn ngay bây giờ.\033[0m")
    print("  • PersonaPlex sẽ tự động phát âm thanh phản hồi qua loa và hiển thị chữ.")
    print("  • Bạn có thể nói ngắt lời (interruption) tự nhiên bất kỳ lúc nào.")
    print("  • Nhấn \033[1;31mCtrl+C\033[0m để kết thúc cuộc trò chuyện.")
    print("═" * 68 + "\n")
    sys.stdout.write("\033[2m[🟢 Đang lắng nghe...]\033[0m\n")
    sys.stdout.flush()

    try:
        with sd.Stream(
            samplerate=sample_rate,
            blocksize=frame_size,
            channels=(1, 1),
            dtype="float32",
            device=(input_device, output_device),
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n\033[1;33mĐang dừng cuộc trò chuyện và đóng stream âm thanh...\033[0m")
    finally:
        stop_event.set()
        worker_thread.join(timeout=1.0)
        session.close()

    # Save session outputs if directory specified
    if save_session_dir is not None and recorded_user_pcm and recorded_agent_pcm:
        save_dir = Path(save_session_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        import sphn

        user_full = np.concatenate(recorded_user_pcm)
        agent_full = np.concatenate(recorded_agent_pcm)
        min_len = min(len(user_full), len(agent_full))

        user_full = user_full[:min_len]
        agent_full = agent_full[:min_len]

        sphn.write_wav(str(save_dir / "user.wav"), user_full, sample_rate)
        sphn.write_wav(str(save_dir / "agent.wav"), agent_full, sample_rate)

        # Stereo: Channel 0 (LEFT) = Agent, Channel 1 (RIGHT) = User
        stereo_pcm = np.stack([agent_full, user_full], axis=0)
        sphn.write_wav(str(save_dir / "dialogue_stereo.wav"), stereo_pcm, sample_rate)

        (save_dir / "transcript.txt").write_text("".join(full_transcript), encoding="utf-8")
        logger.info("Session audio and transcript saved to: %s", save_dir)


def run_file_interaction(
    session,
    input_wav_path: Path | str,
    output_wav_path: Path | str,
    output_text_path: Path | str | None = None,
):
    """Run interaction by feeding an input WAV file and recording generated agent speech."""
    import sphn

    input_path = Path(input_wav_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input WAV file does not exist: {input_path}")

    audio, source_rate = sphn.read(str(input_path))
    if source_rate != session.sample_rate:
        audio = sphn.resample(audio, src_sample_rate=source_rate, dst_sample_rate=session.sample_rate)

    # Convert to mono
    mono_audio = audio[0] if audio.ndim > 1 else audio

    frame_size = session.frame_size
    num_frames = len(mono_audio) // frame_size

    logger.info("Processing input WAV: %s (%d frames, %.2f sec)", input_path, num_frames, len(mono_audio) / session.sample_rate)

    agent_pcm_frames: list[np.ndarray] = []
    text_pieces: list[str] = []

    print("\n\033[92m[PersonaPlex Response]\033[0m: ", end="", flush=True)
    for frame_idx in range(num_frames):
        chunk = mono_audio[frame_idx * frame_size : (frame_idx + 1) * frame_size]
        agent_chunk, text_piece = session.step_audio_chunk(chunk)
        if agent_chunk is not None:
            agent_pcm_frames.append(agent_chunk)
        if text_piece:
            sys.stdout.write(text_piece)
            sys.stdout.flush()
            text_pieces.append(text_piece)

    print("\n")
    session.close()

    out_wav = Path(output_wav_path).resolve()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if agent_pcm_frames:
        sphn.write_wav(str(out_wav), np.concatenate(agent_pcm_frames), session.sample_rate)
        logger.info("Generated agent speech saved to: %s", out_wav)

    if output_text_path:
        out_txt = Path(output_text_path).resolve()
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("".join(text_pieces), encoding="utf-8")
        logger.info("Generated text saved to: %s", out_txt)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PersonaPlex Live Interactive Terminal Demo with Checkpoint & Adapter Support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration file
    parser.add_argument("--config", type=str, default=None, help="[Optional] Path to YAML/JSON configuration file.")

    # Checkpoint & model paths
    parser.add_argument("--model-root", type=str, default=None, help="Path to local base PersonaPlex checkpoint directory.")
    parser.add_argument("--adapter", type=str, default=None, help="[Optional] Path to fine-tuned LoRA checkpoint file or directory.")
    parser.add_argument("--personaplex-source", type=str, default=None, help="[Optional] Path to PersonaPlex source root (defaults to bundled src).")

    # Prompt conditioning
    parser.add_argument("--voice-prompt", type=str, default=None, help="Path to voice prompt WAV or .pt file.")
    parser.add_argument(
        "--text-prompt",
        type=str,
        default=None,
        help="[Optional] Role/system text prompt string or path to .txt file.",
    )

    # Hardware & optimization
    parser.add_argument("--device", type=str, default=None, help="[Optional] Device to run model on: 'cuda' or 'cpu'.")
    parser.add_argument("--qlora", action="store_true", default=None, help="[Optional] Enable 4-bit quantization for base model.")
    parser.add_argument("--lora-rank", type=int, default=None, help="[Optional] Override LoRA rank (if not in adapter.json).")
    parser.add_argument("--lora-alpha", type=int, default=None, help="[Optional] Override LoRA alpha (if not in adapter.json).")

    # Generation sampling knobs
    parser.add_argument("--greedy", action="store_true", default=None, help="[Optional] Use greedy decoding instead of sampling.")
    parser.add_argument("--temp", type=float, default=None, help="[Optional] Audio sampling temperature.")
    parser.add_argument("--temp-text", type=float, default=None, help="[Optional] Text sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="[Optional] Audio sampling top-k.")
    parser.add_argument("--top-k-text", type=int, default=None, help="[Optional] Text sampling top-k.")

    # Audio device selection
    parser.add_argument("--list-devices", action="store_true", help="[Optional] List available sounddevice audio devices and exit.")
    parser.add_argument("--input-device", type=str, default=None, help="[Optional] Audio input device ID or name substring.")
    parser.add_argument("--output-device", type=str, default=None, help="[Optional] Audio output device ID or name substring.")

    # Offline file mode / session recording
    parser.add_argument("--input-wav", type=str, default=None, help="[Optional] Path to input WAV file to chat with instead of microphone.")
    parser.add_argument("--output-wav", type=str, default="outputs/interactive/agent.wav", help="[Optional] Path to save agent output WAV in file mode.")
    parser.add_argument("--save-session-dir", type=str, default=None, help="[Optional] Directory to record live conversation (user.wav, agent.wav, stereo).")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return 0

    # Load from config file if provided
    cfg = {}
    cfg_dir = Path.cwd()
    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.is_file():
            parser.error(f"Config file not found: {config_path}")
        cfg_dir = config_path.parent
        from personaplex_finetuning.config import _read_yaml_or_json
        cfg = _read_yaml_or_json(config_path)

    def resolve_path(val: str | None) -> Path | None:
        if not val:
            return None
        p = Path(val)
        return p.resolve() if p.is_absolute() else (cfg_dir / p).resolve()

    model_sec = cfg.get("model", {})
    adapter_sec = cfg.get("adapter", {})
    prompt_sec = cfg.get("prompt", {})
    gen_sec = cfg.get("generation", {})
    audio_sec = cfg.get("audio", {})

    model_root = args.model_root or (str(resolve_path(model_sec.get("root"))) if model_sec.get("root") else None)
    if not model_root:
        parser.error("--model-root is required (or specify model.root in --config)")

    personaplex_source = args.personaplex_source or (str(resolve_path(model_sec.get("source"))) if model_sec.get("source") else None)
    device = args.device or model_sec.get("device", "cuda")

    adapter = args.adapter or (str(resolve_path(adapter_sec.get("path"))) if adapter_sec.get("path") else None)
    lora_rank = args.lora_rank if args.lora_rank is not None else adapter_sec.get("rank", None)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else adapter_sec.get("alpha", None)
    qlora = args.qlora if args.qlora is not None else adapter_sec.get("qlora", False)

    voice_prompt = args.voice_prompt or (str(resolve_path(prompt_sec.get("voice_prompt"))) if prompt_sec.get("voice_prompt") else None)
    if not voice_prompt:
        parser.error("--voice-prompt is required (or specify prompt.voice_prompt in --config)")

    raw_text_prompt = args.text_prompt or prompt_sec.get("text_prompt", "You enjoy having a good conversation. You are a helpful and friendly assistant.")
    text_prompt_str = raw_text_prompt
    if Path(raw_text_prompt).is_file():
        text_prompt_str = Path(raw_text_prompt).read_text(encoding="utf-8").strip()

    greedy = args.greedy if args.greedy is not None else not gen_sec.get("use_sampling", True)
    temp = args.temp if args.temp is not None else float(gen_sec.get("temp", 0.8))
    temp_text = args.temp_text if args.temp_text is not None else float(gen_sec.get("temp_text", 0.7))
    top_k = args.top_k if args.top_k is not None else int(gen_sec.get("top_k", 250))
    top_k_text = args.top_k_text if args.top_k_text is not None else int(gen_sec.get("top_k_text", 25))

    in_dev = args.input_device or audio_sec.get("input_device")
    out_dev = args.output_device or audio_sec.get("output_device")
    save_session_dir = args.save_session_dir or (str(resolve_path(audio_sec.get("save_session_dir"))) if audio_sec.get("save_session_dir") else None)

    # Import and initialize interactive session
    from personaplex_finetuning.session import InteractiveSession

    session = InteractiveSession(
        model_root=model_root,
        personaplex_source=personaplex_source,
        adapter_path=adapter,
        device=device,
        qlora=bool(qlora),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        use_sampling=not greedy,
        temp=temp,
        temp_text=temp_text,
        top_k=top_k,
        top_k_text=top_k_text,
    )

    # Warmup model
    session.warmup()

    # Setup conditioning (Hybrid System Prompt)
    session.setup_prompts(
        voice_prompt_path=voice_prompt,
        text_prompt=text_prompt_str,
    )

    # Parse audio device identifiers (handle numeric index if provided)
    in_dev_parsed = int(in_dev) if in_dev is not None and str(in_dev).isdigit() else in_dev
    out_dev_parsed = int(out_dev) if out_dev is not None and str(out_dev).isdigit() else out_dev

    if args.input_wav:
        run_file_interaction(
            session=session,
            input_wav_path=args.input_wav,
            output_wav_path=args.output_wav,
        )
    else:
        run_live_interaction(
            session=session,
            input_device=in_dev_parsed,
            output_device=out_dev_parsed,
            save_session_dir=Path(save_session_dir) if save_session_dir else None,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
