"""PersonaPlex Web Demo using Gradio with public temporary share URL support."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Generator

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("interactive_web")


def find_preset_voice_prompts(search_dir: Path) -> list[tuple[str, str]]:
    """Scan search_dir for voice_prompt.wav files and return (label, path) pairs."""
    prompts: list[tuple[str, str]] = []
    if not search_dir.is_dir():
        return prompts
    for p in sorted(search_dir.rglob("*voice_prompt*.wav")):
        label = f"{p.parent.name} ({p.name})"
        prompts.append((label, str(p.resolve())))
    return prompts


class WebDemoApp:
    def __init__(
        self,
        model_root: str,
        adapter_path: str | None = None,
        personaplex_source: str | None = None,
        device: str = "cuda",
        qlora: bool = False,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        default_voice_prompt: str | None = None,
        default_text_prompt: str = "You enjoy having a good conversation. You are a helpful and friendly assistant.",
    ):
        from personaplex_finetuning.session import InteractiveSession

        self.model_root = model_root
        self.adapter_path = adapter_path
        self.device = device
        self.qlora = qlora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.personaplex_source = personaplex_source
        self.default_voice_prompt = default_voice_prompt
        self.default_text_prompt = default_text_prompt

        logger.info("Initializing PersonaPlex Session for Gradio Web Demo...")
        self.session = InteractiveSession(
            model_root=self.model_root,
            personaplex_source=self.personaplex_source,
            adapter_path=self.adapter_path,
            device=self.device,
            qlora=self.qlora,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_sampling=True,
        )

        logger.info("Warming up model...")
        self.session.warmup(num_frames=2)

        self.current_voice_prompt: str | None = None
        self.current_text_prompt: str | None = None

        if self.default_voice_prompt and Path(self.default_voice_prompt).is_file():
            self._setup_prompts(self.default_voice_prompt, self.default_text_prompt)

    def _setup_prompts(self, voice_prompt: str, text_prompt: str) -> None:
        if voice_prompt != self.current_voice_prompt or text_prompt != self.current_text_prompt:
            logger.info("Updating conditioning: voice=%s", voice_prompt)
            self.session.setup_prompts(voice_prompt, text_prompt)
            self.current_voice_prompt = voice_prompt
            self.current_text_prompt = text_prompt

    def process_speech(
        self,
        audio_file: str | None,
        voice_prompt: str | None,
        custom_voice_file: str | None,
        text_prompt: str,
        temperature: float,
        temp_text: float,
        top_k: int,
        top_k_text: int,
    ) -> Generator[tuple[str | None, str, str | None, str], None, None]:
        if not audio_file:
            yield None, "", None, "⚠️ Please record or upload your audio first."
            return

        effective_voice = custom_voice_file or voice_prompt or self.default_voice_prompt
        if not effective_voice or not Path(effective_voice).is_file():
            yield None, "", None, "⚠️ Please select or upload a valid voice prompt."
            return

        import sphn

        # Update generation parameters
        self.session.generator.temp = float(temperature)
        self.session.generator.temp_text = float(temp_text)
        self.session.generator.top_k = int(top_k)
        self.session.generator.top_k_text = int(top_k_text)

        # Setup conditioning if changed
        try:
            self._setup_prompts(effective_voice, text_prompt)
        except Exception as exc:
            yield None, "", None, f"❌ Error setting prompts: {exc}"
            return

        # Load user audio and resample to 24000 Hz mono
        raw_audio, sr = sphn.read(str(audio_file))
        if sr != self.session.sample_rate:
            raw_audio = sphn.resample(raw_audio, src_sample_rate=sr, dst_sample_rate=self.session.sample_rate)
        user_pcm = raw_audio[0] if raw_audio.ndim > 1 else raw_audio

        frame_size = self.session.frame_size
        num_frames = len(user_pcm) // frame_size

        if num_frames == 0:
            yield None, "", None, "⚠️ Audio is too short (minimum 80ms required)."
            return

        yield None, "", None, f"⏳ Processing {num_frames} frames ({len(user_pcm)/self.session.sample_rate:.1f}s)..."

        agent_pcm_frames: list[np.ndarray] = []
        transcript_pieces: list[str] = []
        start_time = time.time()

        for idx in range(num_frames):
            chunk = user_pcm[idx * frame_size : (idx + 1) * frame_size]
            agent_chunk, piece = self.session.step_audio_chunk(chunk)
            if agent_chunk is not None:
                agent_pcm_frames.append(agent_chunk)
            if piece:
                transcript_pieces.append(piece)
                # Stream transcript updates periodically
                if idx % 5 == 0 or idx == num_frames - 1:
                    yield None, "".join(transcript_pieces), None, f"🗣️ PersonaPlex speaking... ({idx+1}/{num_frames} frames)"

        elapsed = time.time() - start_time
        rtf = elapsed / (len(user_pcm) / self.session.sample_rate)

        # Build output WAV files
        tmp_dir = Path(tempfile.mkdtemp(prefix="personaplex_demo_"))
        agent_wav_path = str(tmp_dir / "agent_response.wav")
        stereo_wav_path = str(tmp_dir / "dialogue_stereo.wav")

        if agent_pcm_frames:
            agent_pcm = np.concatenate(agent_pcm_frames)
        else:
            agent_pcm = np.zeros(frame_size, dtype=np.float32)

        sphn.write_wav(agent_wav_path, agent_pcm, self.session.sample_rate)

        # Create stereo dialogue: Channel 0 (LEFT) = Agent, Channel 1 (RIGHT) = User
        min_len = min(len(agent_pcm), len(user_pcm))
        stereo = np.stack([agent_pcm[:min_len], user_pcm[:min_len]], axis=0)
        sphn.write_wav(stereo_wav_path, stereo, self.session.sample_rate)

        final_transcript = "".join(transcript_pieces) or "(No speech detected from agent in this window)"
        status_msg = f"✅ Completed! Processed {num_frames} frames ({len(user_pcm)/self.session.sample_rate:.1f}s) in {elapsed:.2f}s (RTF: {rtf:.2f}x)"

        yield agent_wav_path, final_transcript, stereo_wav_path, status_msg


def create_gradio_ui(app: WebDemoApp, preset_prompts: list[tuple[str, str]]):
    import gradio as gr

    title = "🎙️ PersonaPlex: Voice & Role-Controlled Conversational AI"
    adapter_status = f"✅ LoRA Active ({Path(app.adapter_path).name})" if app.adapter_path else "⚪ Base Model"
    device_status = f"{app.device.upper()}"

    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="indigo")

    with gr.Blocks(title="PersonaPlex Web Demo") as demo:
        gr.Markdown(
            f"""
            # {title}
            **NVIDIA PersonaPlex 7B Duplex Speech-to-Speech** | Checkpoint: `{adapter_status}` | Device: `{device_status}`
            """
        )

        with gr.Row():
            # Left Column: Configuration & Prompting
            with gr.Column(scale=1):
                gr.Markdown("### 🎭 Persona & Conditioning Settings")

                prompt_choices = [p[1] for p in preset_prompts]
                prompt_labels = {p[1]: p[0] for p in preset_prompts}
                default_choice = prompt_choices[0] if prompt_choices else None

                voice_dropdown = gr.Dropdown(
                    choices=prompt_choices,
                    value=app.default_voice_prompt or default_choice,
                    label="Agent Voice Prompt (Select Preset)",
                    info="Choose a reference speaker voice from prepared samples",
                )

                custom_voice_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Or Upload Custom Voice Prompt WAV",
                )

                text_prompt_input = gr.Textbox(
                    value=app.default_text_prompt,
                    lines=3,
                    label="System / Role Text Prompt",
                    placeholder="Enter PersonaPlex persona prompt...",
                )

                with gr.Row():
                    btn_assistant = gr.Button("🎓 Teacher / Assistant", size="sm")
                    btn_customer = gr.Button("💼 Customer Service", size="sm")
                    btn_friend = gr.Button("☕ Casual Friend", size="sm")

                btn_assistant.click(
                    fn=lambda: "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
                    outputs=text_prompt_input,
                )
                btn_customer.click(
                    fn=lambda: "You work for CitySan Services which is a waste management company. You are polite, helpful, and professional.",
                    outputs=text_prompt_input,
                )
                btn_friend.click(
                    fn=lambda: "You enjoy having a good conversation. Have a casual and friendly discussion about life, technology, and travel.",
                    outputs=text_prompt_input,
                )

                with gr.Accordion("⚙️ Generation Hyperparameters", open=False):
                    temp_slider = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Audio Temperature")
                    temp_text_slider = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Text Temperature")
                    top_k_slider = gr.Slider(10, 500, value=250, step=10, label="Audio Top-K")
                    top_k_text_slider = gr.Slider(5, 100, value=25, step=5, label="Text Top-K")

            # Right Column: Live Conversation & Audio Output
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Duplex Conversation")

                user_audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="🎙️ Speak to PersonaPlex (Record or Upload Audio)",
                )

                send_btn = gr.Button("🚀 Chat with PersonaPlex", variant="primary", size="lg")
                status_box = gr.Markdown("🟢 System ready. Click record above and talk!")

                gr.Markdown("#### 🗣️ Model Speech Response")
                agent_audio_output = gr.Audio(label="PersonaPlex Audio", autoplay=True)

                gr.Markdown("#### 📝 Live Streaming Transcript")
                transcript_output = gr.Textbox(label="Agent Words", lines=4, interactive=False)

                with gr.Accordion("🎧 Full Stereo Dialogue (LEFT = Agent, RIGHT = User)", open=False):
                    stereo_audio_output = gr.Audio(label="Stereo Recording")

        send_btn.click(
            fn=app.process_speech,
            inputs=[
                user_audio_input,
                voice_dropdown,
                custom_voice_input,
                text_prompt_input,
                temp_slider,
                temp_text_slider,
                top_k_slider,
                top_k_text_slider,
            ],
            outputs=[
                agent_audio_output,
                transcript_output,
                stereo_audio_output,
                status_box,
            ],
        )

    return demo


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PersonaPlex Web Demo using Gradio with Temporary Public Share Link",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON configuration file (e.g. configs/demo.yaml).")
    parser.add_argument("--model-root", type=str, default=None, help="Path to local base PersonaPlex checkpoint directory.")
    parser.add_argument("--adapter", type=str, default=None, help="Path to fine-tuned LoRA checkpoint file or directory.")
    parser.add_argument("--voice-prompt", type=str, default=None, help="Path to default voice prompt WAV or .pt file.")
    parser.add_argument("--text-prompt", type=str, default=None, help="Role/system text prompt string or path to .txt file.")
    parser.add_argument("--device", type=str, default=None, help="Device to run model on: 'cuda' or 'cpu'.")
    parser.add_argument("--qlora", action="store_true", default=None, help="Enable 4-bit quantization.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind server.")
    parser.add_argument("--port", type=int, default=8998, help="Port to listen on.")
    parser.add_argument("--share", action="store_true", default=True, help="Create a temporary public URL link (gradio.live).")
    parser.add_argument("--no-share", dest="share", action="store_false", help="Disable public URL link; only listen locally.")

    args = parser.parse_args()

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
        if p.is_absolute():
            return p
        candidates = [
            (Path.cwd() / p).resolve(),
            (cfg_dir / p).resolve(),
            (cfg_dir.parent / p).resolve(),
            (cfg_dir.parent.parent / p).resolve(),
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        return (Path.cwd() / p).resolve() if (Path.cwd() / p).exists() else (cfg_dir / p).resolve()

    model_sec = cfg.get("model", {})
    adapter_sec = cfg.get("adapter", {})
    prompt_sec = cfg.get("prompt", {})
    server_sec = cfg.get("server", {})

    model_root = args.model_root or (str(resolve_path(model_sec.get("root"))) if model_sec.get("root") else None)
    if not model_root:
        parser.error("--model-root is required (or specify model.root in --config)")

    personaplex_source = str(resolve_path(model_sec.get("source"))) if model_sec.get("source") else None
    device = args.device or model_sec.get("device", "cuda")
    adapter = args.adapter or (str(resolve_path(adapter_sec.get("path"))) if adapter_sec.get("path") else None)
    qlora = args.qlora if args.qlora is not None else adapter_sec.get("qlora", False)
    voice_prompt = args.voice_prompt or (str(resolve_path(prompt_sec.get("voice_prompt"))) if prompt_sec.get("voice_prompt") else None)
    text_prompt = args.text_prompt or prompt_sec.get("text_prompt", "You enjoy having a good conversation. You are a helpful and friendly assistant.")

    host = server_sec.get("host", args.host)
    port = server_sec.get("port", args.port)
    share = args.share if args.share is not None else server_sec.get("share", True)

    # Scan preset voice prompts
    prepared_dir = cfg_dir / "prepared" if (cfg_dir / "prepared").is_dir() else Path("../prepared").resolve()
    preset_prompts = find_preset_voice_prompts(prepared_dir)
    if voice_prompt and (voice_prompt not in [p[1] for p in preset_prompts]):
        preset_prompts.insert(0, (f"Config Voice ({Path(voice_prompt).name})", voice_prompt))

    # Initialize app
    app = WebDemoApp(
        model_root=model_root,
        adapter_path=adapter,
        personaplex_source=personaplex_source,
        device=device,
        qlora=bool(qlora),
        default_voice_prompt=voice_prompt,
        default_text_prompt=text_prompt,
    )

    demo = create_gradio_ui(app, preset_prompts)

    print("\n" + "═" * 68)
    print("  \033[1;32m🚀 KHỞI ĐỘNG PERSONAPLEX GRADIO WEB DEMO\033[0m")
    print(f"  • Local URL: \033[1;36mhttp://localhost:{port}\033[0m")
    if share:
        print("  • Public Temporary URL: \033[1;35mĐang tạo link công khai tạm thời (gradio.live)...\033[0m")
    print("═" * 68 + "\n")

    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="indigo"),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
