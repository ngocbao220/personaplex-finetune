"""Interactive session engine for PersonaPlex with base checkpoint and LoRA adapter support."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    import torch
except ImportError:
    np = None  # type: ignore
    torch = None  # type: ignore

from .lora import inject_lora, load_adapter
from .runtime import RuntimePaths, load_runtime

logger = logging.getLogger(__name__)


def wrap_with_system_tags(text: str) -> str:
    """Ensure text prompt is enclosed in <system> tags as required by PersonaPlex."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def resolve_adapter_info(adapter_path: Path | str) -> tuple[Path, int, int]:
    """Resolve adapter safetensors file and its rank/alpha parameters.

    Returns:
        tuple of (safetensors_file_path, rank, alpha)
    """
    path = Path(adapter_path).resolve()
    if path.is_dir():
        safetensors_file = path / "lora.safetensors"
        meta_file = path / "adapter.json"
    else:
        safetensors_file = path
        meta_file = path.parent / "adapter.json"

    if not safetensors_file.is_file():
        raise FileNotFoundError(f"LoRA adapter file not found: {safetensors_file}")

    rank = 16
    alpha = 32
    if meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            rank = int(meta.get("rank", rank))
            alpha = int(meta.get("alpha", alpha))
            logger.info("Loaded LoRA metadata from %s: rank=%d, alpha=%d", meta_file, rank, alpha)
        except Exception as exc:
            logger.warning("Could not parse %s: %s (using defaults rank=%d, alpha=%d)", meta_file, exc, rank, alpha)

    return safetensors_file, rank, alpha


class InteractiveSession:
    """Encapsulates PersonaPlex runtime, LoRA adapter injection, and streaming dialogue generation."""

    def __init__(
        self,
        model_root: Path | str,
        personaplex_source: Path | str | None = None,
        adapter_path: Path | str | None = None,
        device: str = "cuda",
        qlora: bool = False,
        quant_type: str = "nf4",
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
    ):
        self.device = device
        self.model_root = Path(model_root).resolve()
        if personaplex_source is None:
            # Default to bundled src in current package
            personaplex_source = Path(__file__).resolve().parent.parent
        self.personaplex_source = Path(personaplex_source).resolve()

        logger.info("Initializing PersonaPlex runtime from: %s", self.model_root)
        self.runtime = load_runtime(
            RuntimePaths(self.model_root, self.personaplex_source),
            device=self.device,
            qlora=qlora,
            quant_type=quant_type,
        )

        # Inject and load LoRA adapter if provided
        self.adapter_file: Optional[Path] = None
        if adapter_path is not None:
            adapter_file, resolved_rank, resolved_alpha = resolve_adapter_info(adapter_path)
            self.adapter_file = adapter_file
            effective_rank = lora_rank if lora_rank is not None else resolved_rank
            effective_alpha = lora_alpha if lora_alpha is not None else resolved_alpha

            logger.info("Injecting LoRA adapter (rank=%d, alpha=%d) from %s", effective_rank, effective_alpha, adapter_file)
            inject_lora(self.runtime.model, rank=effective_rank, alpha=effective_alpha)
            load_adapter(self.runtime.model, adapter_file)
            logger.info("LoRA adapter loaded successfully.")
        else:
            logger.info("No adapter provided: running base PersonaPlex model.")

        self.runtime.model.eval()

        # Import LMGen from PersonaPlex
        import importlib
        lm_module = importlib.import_module("moshi.models.lm")
        self.lm_module = lm_module

        self.frame_size = int(self.runtime.codec.sample_rate / self.runtime.codec.frame_rate)  # 1920 samples for 24kHz/12.5Hz
        self.sample_rate = self.runtime.codec.sample_rate
        self.frame_rate = self.runtime.codec.frame_rate

        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text

        # Instantiate LMGen
        self.generator = lm_module.LMGen(
            self.runtime.model,
            audio_silence_frame_cnt=int(0.5 * self.frame_rate),
            sample_rate=self.sample_rate,
            frame_rate=self.frame_rate,
            device=self.device,
            use_sampling=self.use_sampling,
            temp=self.temp,
            temp_text=self.temp_text,
            top_k=self.top_k,
            top_k_text=self.top_k_text,
        )

        # Keep modules in permanent streaming mode (matching server.py / offline.py)
        self.runtime.codec.mimi.streaming_forever(1)
        self.generator.streaming_forever(1)

    def warmup(self, num_frames: int = 4) -> None:
        """Warm up model and CUDA kernels with dummy frames."""
        logger.info("Warming up model with %d frames...", num_frames)
        chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            codes = self.runtime.codec.mimi.encode(chunk)
            for _ in range(num_frames):
                tokens = self.generator.step(codes[:, :, :1])
                if tokens is not None:
                    _ = self.runtime.codec.mimi.decode(tokens[:, 1:9])
        if torch is not None and torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.runtime.codec.mimi.reset_streaming()
        self.generator.reset_streaming()
        logger.info("Warmup complete.")

    def setup_prompts(
        self,
        voice_prompt_path: Path | str,
        text_prompt: str | None = None,
    ) -> None:
        """Set up voice conditioning and role/system text prompt."""
        voice_path = Path(voice_prompt_path).resolve()
        if not voice_path.is_file():
            raise FileNotFoundError(f"Voice prompt file does not exist: {voice_path}")

        logger.info("Loading voice prompt from: %s", voice_path)
        if voice_path.suffix == ".pt":
            self.generator.load_voice_prompt_embeddings(str(voice_path))
        else:
            self.generator.load_voice_prompt(str(voice_path))

        if text_prompt and text_prompt.strip():
            formatted_text = wrap_with_system_tags(text_prompt)
            logger.info("Setting system text prompt: %s", formatted_text)
            self.generator.text_prompt_tokens = self.runtime.tokenizer.encode(formatted_text)
        else:
            self.generator.text_prompt_tokens = None

        # Reset streaming states and step through Hybrid System Prompts
        logger.info("Stepping through Hybrid System Prompts (Voice + Silence + Text + Silence)...")
        self.runtime.codec.mimi.reset_streaming()
        self.generator.reset_streaming()

        with torch.no_grad():
            self.generator.step_system_prompts(self.runtime.codec.mimi)

        # Reset mimi streaming state for dialogue phase
        self.runtime.codec.mimi.reset_streaming()
        logger.info("Hybrid System Prompts conditioning initialized. Ready for conversation.")

    def step_audio_chunk(self, user_pcm_chunk: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        """Process one chunk of user PCM audio (length frame_size) and return agent audio and text.

        Args:
            user_pcm_chunk: numpy float32 1D array of length frame_size (1920 samples).

        Returns:
            tuple of (agent_pcm_chunk, text_piece):
                agent_pcm_chunk: numpy float32 1D array of 1920 samples (or None).
                text_piece: text token string (or None if pad/special/empty).
        """

        # Ensure correct shape and device: (1, 1, frame_size)
        if user_pcm_chunk.ndim == 1:
            user_tensor = torch.from_numpy(user_pcm_chunk).to(dtype=torch.float32, device=self.device)[None, None]
        else:
            user_tensor = torch.as_tensor(user_pcm_chunk, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Encode user audio frame through Mimi
            user_codes = self.runtime.codec.mimi.encode(user_tensor)

            agent_pcm = None
            text_piece = None

            # Feed codes into LMGen step
            tokens = self.generator.step(input_tokens=user_codes[:, :, :1])
            if tokens is not None:
                # Decode agent audio codebooks 1:9
                decoded = self.runtime.codec.mimi.decode(tokens[:, 1:9])
                agent_pcm = decoded.squeeze().detach().float().cpu().numpy()

                # Decode agent text token
                token_id = int(tokens[0, 0, 0].item())
                if token_id not in (0, self.runtime.tokenizer.padding_id, self.runtime.tokenizer.end_padding_id):
                    piece = self.runtime.tokenizer._processor.id_to_piece(token_id)
                    text_piece = piece.replace("▁", " ")

            return agent_pcm, text_piece

    def close(self) -> None:
        """Clean up streaming session."""
        try:
            self.generator._stop_streaming()
        except Exception:
            pass
        try:
            self.runtime.codec.mimi._stop_streaming()
        except Exception:
            pass
        if torch is not None and torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.empty_cache()
