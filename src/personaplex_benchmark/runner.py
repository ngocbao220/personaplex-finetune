"""Streaming benchmark runner extracting native frame-level events from PersonaPlex."""

from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .contract import BenchmarkSample
from .metrics import (
    SpeechEvent,
    compute_tor,
    compute_speaker_switch_latency,
    compute_interruption_latency,
    compute_backchannel_metrics,
    compute_response_quality,
)


@dataclass
class SampleEvaluationResult:
    sample_id: str
    task: str
    tor: float
    latency: float | None = None
    freq: float | None = None
    jsd: float | None = None
    response_quality: float | None = None
    agent_text: str = ""
    events: list[SpeechEvent] = field(default_factory=list)


def cluster_tokens_into_speech_events(
    frame_tokens: list[tuple[float, str]],
    max_gap_sec: float = 0.4,
) -> list[SpeechEvent]:
    """Clusters frame-by-frame text tokens into contiguous SpeechEvents."""
    if not frame_tokens:
        return []

    events: list[SpeechEvent] = []
    current_words: list[str] = []
    start_sec = frame_tokens[0][0]
    last_sec = frame_tokens[0][0]

    for t, piece in frame_tokens:
        clean_p = piece.strip()
        if not clean_p:
            continue

        if (t - last_sec) > max_gap_sec and current_words:
            # Finalize previous event
            text = " ".join(current_words)
            events.append(
                SpeechEvent(
                    start_sec=start_sec,
                    end_sec=last_sec + 0.08,
                    text=text,
                )
            )
            current_words = []
            start_sec = t

        current_words.append(clean_p)
        last_sec = t

    if current_words:
        text = " ".join(current_words)
        events.append(
            SpeechEvent(
                start_sec=start_sec,
                end_sec=last_sec + 0.08,
                text=text,
            )
        )

    return events


class MockBenchmarkRunner:
    """Mock runner for dry-runs, unit tests, and CI without loading 7B model."""

    def __init__(self, behavior_profile: str = "ideal"):
        self.profile = behavior_profile

    def evaluate_sample(self, sample: BenchmarkSample) -> SampleEvaluationResult:
        events: list[SpeechEvent] = []
        lat: float | None = None
        freq: float | None = None
        jsd: float | None = None
        quality: float | None = None

        if sample.task in ("pause_synthetic", "pause_natural"):
            # In ideal profile, agent remains silent during pause (TOR = 0)
            if self.profile == "talkative":
                events.append(
                    SpeechEvent(
                        start_sec=sample.pause_start_sec or sample.window_start_sec,
                        end_sec=(sample.pause_start_sec or sample.window_start_sec) + 2.0,
                        text="I interrupt your thought right now",
                    )
                )
            tor = compute_tor(
                events,
                window_start_sec=sample.pause_start_sec,
                window_end_sec=sample.pause_end_sec,
            )

        elif sample.task == "turn_taking":
            u_end = sample.user_turn_end_sec or (sample.window_start_sec + 2.0)
            events.append(
                SpeechEvent(
                    start_sec=u_end + 0.12,
                    end_sec=u_end + 2.5,
                    text="Yes I completely agree with your point",
                )
            )
            tor = compute_tor(events, window_start_sec=u_end)
            lat = compute_speaker_switch_latency(events, user_turn_end_sec=u_end)

        elif sample.task == "backchannel":
            dur = max(1.0, sample.window_end_sec - sample.window_start_sec)
            events.append(SpeechEvent(start_sec=sample.window_start_sec + 2.0, end_sec=sample.window_start_sec + 2.5, text="uh-huh"))
            events.append(SpeechEvent(start_sec=sample.window_start_sec + 5.0, end_sec=sample.window_start_sec + 5.4, text="yeah"))
            bc = compute_backchannel_metrics(events, user_duration_sec=dur, gt_distribution=sample.gt_distribution)
            tor = bc["tor"]
            freq = bc["freq"]
            jsd = bc["jsd"]

        elif sample.task == "interruption":
            i_end = sample.interruption_end_sec or (sample.window_start_sec + 3.0)
            events.append(
                SpeechEvent(
                    start_sec=i_end + 0.35,
                    end_sec=i_end + 3.0,
                    text="Sure let me address your new question",
                )
            )
            tor = compute_tor(events, window_start_sec=i_end)
            lat = compute_interruption_latency(events, interruption_end_sec=i_end)
            quality = compute_response_quality(sample.interruption_query, "Sure let me address your new question", sample.agent_pre_speech)

        else:
            tor = 0.0

        return SampleEvaluationResult(
            sample_id=sample.sample_id,
            task=sample.task,
            tor=tor,
            latency=lat,
            freq=freq,
            jsd=jsd,
            response_quality=quality,
            agent_text=" ".join(e.text for e in events),
            events=events,
        )


class PersonaPlexStreamingRunner:
    """Streaming evaluator running directly on PersonaPlex (Base or LoRA)."""

    def __init__(
        self,
        model_root: Path,
        adapter_path: Path | None = None,
        device: str = "auto",
        personaplex_source: Path | None = None,
    ):
        import importlib
        import torch
        from personaplex_finetuning.runtime import RuntimePaths, load_runtime
        from personaplex_finetuning.lora import inject_lora, load_adapter

        # Resolve device
        if device == "auto" or device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Resolve source
        source = personaplex_source
        if source is None or not (source / "moshi" / "models" / "loaders.py").is_file():
            candidates = [
                Path("refs/personaplex-original/moshi"),
                Path("../refs/personaplex-original/moshi"),
                Path("../../refs/personaplex-original/moshi"),
                Path("personaplex-finetuning/src"),
            ]
            for c in candidates:
                if (c / "moshi" / "models" / "loaders.py").is_file():
                    source = c
                    break
        if source is None or not (source / "moshi" / "models" / "loaders.py").is_file():
            raise FileNotFoundError(f"Could not resolve PersonaPlex source containing moshi/models/loaders.py")

        self.runtime = load_runtime(
            RuntimePaths(model_root, source),
            device=device,
            qlora=False,
            quant_type=None,
        )

        if adapter_path is not None and adapter_path.is_file():
            inject_lora(self.runtime.model, rank=32, alpha=64.0)
            load_adapter(self.runtime.model, adapter_path)

        self.runtime.model.eval()
        self.lm_module = importlib.import_module("moshi.models.lm")

    def evaluate_sample(self, sample: BenchmarkSample) -> SampleEvaluationResult:
        import torch

        generator = self.lm_module.LMGen(
            self.runtime.model,
            audio_silence_frame_cnt=6,
            sample_rate=self.runtime.codec.sample_rate,
            frame_rate=self.runtime.codec.frame_rate,
            device=self.device,
            use_sampling=False,
        )

        generator.load_voice_prompt(str(sample.voice_prompt_wav))
        generator.text_prompt_tokens = self.runtime.tokenizer.encode(
            f"<system> {sample.text_prompt.strip()} <system>"
        )

        user_codes = self.runtime.codec.encode_conversation(
            sample.conversation_wav,
            sample.user_channel,
            sample.window_start_sec,
            sample.window_end_sec,
        )
        user = torch.tensor(user_codes, device=self.device).unsqueeze(0)

        frame_tokens: list[tuple[float, str]] = []
        all_text_tokens: list[str] = []

        frame_duration_sec = 1.0 / self.runtime.codec.frame_rate  # 0.08s for 12.5 Hz

        with torch.no_grad(), generator.streaming(1):
            generator.step_system_prompts(self.runtime.codec.mimi)
            for f in range(user.shape[-1]):
                curr_time = sample.window_start_sec + (f * frame_duration_sec)
                tokens = generator.step(input_tokens=user[:, :, f : f + 1])
                if tokens is None:
                    continue

                token_id = int(tokens[0, 0, 0])
                if token_id not in (0, self.runtime.tokenizer.padding_id):
                    piece = self.runtime.tokenizer._processor.id_to_piece(token_id).replace("▁", " ")
                    frame_tokens.append((curr_time, piece))
                    all_text_tokens.append(piece)

        events = cluster_tokens_into_speech_events(frame_tokens, max_gap_sec=0.4)
        agent_full_text = "".join(all_text_tokens).strip()

        # Compute task metrics
        lat: float | None = None
        freq: float | None = None
        jsd: float | None = None
        quality: float | None = None

        if sample.task in ("pause_synthetic", "pause_natural"):
            tor = compute_tor(
                events,
                window_start_sec=sample.pause_start_sec,
                window_end_sec=sample.pause_end_sec,
            )

        elif sample.task == "turn_taking":
            u_end = sample.user_turn_end_sec or sample.window_start_sec
            tor = compute_tor(events, window_start_sec=u_end)
            lat = compute_speaker_switch_latency(events, user_turn_end_sec=u_end)

        elif sample.task == "backchannel":
            dur = max(1.0, sample.window_end_sec - sample.window_start_sec)
            bc = compute_backchannel_metrics(events, user_duration_sec=dur, gt_distribution=sample.gt_distribution)
            tor = bc["tor"]
            freq = bc["freq"]
            jsd = bc["jsd"]

        elif sample.task == "interruption":
            i_end = sample.interruption_end_sec or sample.window_start_sec
            tor = compute_tor(events, window_start_sec=i_end)
            lat = compute_interruption_latency(events, interruption_end_sec=i_end)
            quality = compute_response_quality(sample.interruption_query, agent_full_text, sample.agent_pre_speech)

        else:
            tor = 0.0

        return SampleEvaluationResult(
            sample_id=sample.sample_id,
            task=sample.task,
            tor=tor,
            latency=lat,
            freq=freq,
            jsd=jsd,
            response_quality=quality,
            agent_text=agent_full_text,
            events=events,
        )
