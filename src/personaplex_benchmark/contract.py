"""Benchmark dataset contracts and sample extractors from prepared conversations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkSample:
    """A single evaluation sample for full-duplex benchmark."""
    sample_id: str
    task: str  # "pause_synthetic", "pause_natural", "backchannel", "turn_taking", "interruption"
    conversation_wav: Path
    user_channel: int
    voice_prompt_wav: Path
    text_prompt: str
    window_start_sec: float
    window_end_sec: float

    # Task specific ground-truth metadata
    user_turn_end_sec: float | None = None
    pause_start_sec: float | None = None
    pause_end_sec: float | None = None
    interruption_start_sec: float | None = None
    interruption_end_sec: float | None = None
    interruption_query: str = ""
    agent_pre_speech: str = ""
    gt_distribution: list[float] = field(default_factory=list)


def extract_samples_from_prepared_dir(
    sample_dir: Path,
    min_pause_sec: float = 1.0,
    max_window_sec: float = 30.0,
) -> list[BenchmarkSample]:
    """Extracts benchmark scenarios from a prepared sample directory.

    Leverages words.json, metadata.json, and conversation.wav.
    """
    words_file = sample_dir / "words.json"
    meta_file = sample_dir / "metadata.json"
    conv_wav = sample_dir / "conversation.wav"
    voice_wav = sample_dir / "voice_prompt_left.wav"

    if not (words_file.is_file() and conv_wav.is_file()):
        return []

    meta = {}
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

    sample_id = meta.get("sample_id", sample_dir.name)
    user_channel = 1 if meta.get("user_channel", "right").lower() == "right" else 0
    text_prompt = meta.get("text_prompt_left", "You are a helpful and natural conversational AI.")

    raw_words = json.loads(words_file.read_text(encoding="utf-8"))
    user_words = [w for w in raw_words if w.get("speaker") == "user"]
    agent_words = [w for w in raw_words if w.get("speaker") == "agent"]

    samples: list[BenchmarkSample] = []

    # 1. Extract Natural Pause Samples
    # Find intra-turn pause in user speech where gap >= min_pause_sec
    for i in range(len(user_words) - 1):
        w_curr = user_words[i]
        w_next = user_words[i + 1]
        gap = w_next["start"] - w_curr["end"]
        if min_pause_sec <= gap <= 4.0:
            # Check there is no agent speech in between (true user pause)
            agent_in_between = any(
                w_curr["end"] < aw["start"] < w_next["start"] for aw in agent_words
            )
            if not agent_in_between:
                win_start = max(0.0, w_curr["start"] - 2.0)
                win_end = min(w_next["end"] + 2.0, win_start + max_window_sec)
                samples.append(
                    BenchmarkSample(
                        sample_id=f"{sample_id}_pause_{len(samples):02d}",
                        task="pause_natural",
                        conversation_wav=conv_wav,
                        user_channel=user_channel,
                        voice_prompt_wav=voice_wav,
                        text_prompt=text_prompt,
                        window_start_sec=win_start,
                        window_end_sec=win_end,
                        pause_start_sec=w_curr["end"],
                        pause_end_sec=w_next["start"],
                    )
                )
                if len([s for s in samples if s.task == "pause_natural"]) >= 2:
                    break

    # 2. Extract Smooth Turn Taking Samples
    # Find transition where user finishes turn and agent replies
    for i in range(len(user_words) - 1):
        u_end = user_words[i]["end"]
        # Find next agent word
        subsequent_agent = [aw for aw in agent_words if aw["start"] >= u_end]
        if subsequent_agent:
            first_aw = subsequent_agent[0]
            lat = first_aw["start"] - u_end
            if 0.0 <= lat <= 3.0:
                win_start = max(0.0, user_words[i]["start"] - 4.0)
                win_end = min(first_aw["end"] + 4.0, win_start + max_window_sec)
                samples.append(
                    BenchmarkSample(
                        sample_id=f"{sample_id}_turn_{len(samples):02d}",
                        task="turn_taking",
                        conversation_wav=conv_wav,
                        user_channel=user_channel,
                        voice_prompt_wav=voice_wav,
                        text_prompt=text_prompt,
                        window_start_sec=win_start,
                        window_end_sec=win_end,
                        user_turn_end_sec=u_end,
                    )
                )
                if len([s for s in samples if s.task == "turn_taking"]) >= 2:
                    break

    # 3. Extract Backchannel Candidate Windows
    # Continuous user speech for > 8s
    chunk_start: float | None = None
    chunk_end: float | None = None
    for i in range(len(user_words) - 1):
        if chunk_start is None:
            chunk_start = user_words[i]["start"]
        chunk_end = user_words[i]["end"]
        gap = user_words[i + 1]["start"] - chunk_end
        if gap > 1.5 or (chunk_end - chunk_start >= 10.0):
            if chunk_end - chunk_start >= 8.0:
                samples.append(
                    BenchmarkSample(
                        sample_id=f"{sample_id}_bc_{len(samples):02d}",
                        task="backchannel",
                        conversation_wav=conv_wav,
                        user_channel=user_channel,
                        voice_prompt_wav=voice_wav,
                        text_prompt=text_prompt,
                        window_start_sec=chunk_start,
                        window_end_sec=chunk_end + 1.0,
                        gt_distribution=[0.1] * 10,
                    )
                )
                break
            chunk_start = None

    # 4. Extract Interruption Sample
    # Find user speaking while agent is speaking (overlap) or user breaking agent turn
    for aw in agent_words:
        overlapping_user = [
            uw for uw in user_words
            if uw["start"] > aw["start"] and uw["start"] < aw["end"]
        ]
        if overlapping_user:
            u_intr = overlapping_user[0]
            win_start = max(0.0, aw["start"] - 1.0)
            win_end = min(u_intr["end"] + 5.0, win_start + max_window_sec)
            samples.append(
                BenchmarkSample(
                    sample_id=f"{sample_id}_intr_{len(samples):02d}",
                    task="interruption",
                    conversation_wav=conv_wav,
                    user_channel=user_channel,
                    voice_prompt_wav=voice_wav,
                    text_prompt=text_prompt,
                    window_start_sec=win_start,
                    window_end_sec=win_end,
                    interruption_start_sec=u_intr["start"],
                    interruption_end_sec=u_intr["end"],
                    interruption_query=u_intr.get("word", ""),
                    agent_pre_speech=aw.get("word", ""),
                )
            )
            break

    return samples


def load_fdb_v1_dataset(
    extracted_root: Path,
    default_voice_prompt: Path,
    default_text_prompt: str = "You are a helpful, attentive, and natural conversational AI.",
    gt_distribution_path: Path | None = None,
) -> list[BenchmarkSample]:
    """Loads samples from the official Full-Duplex-Bench v1.0 extracted directory."""
    import sphn

    samples: list[BenchmarkSample] = []

    gt_dist: list[float] = []
    if gt_distribution_path and gt_distribution_path.is_file():
        raw_gt = json.loads(gt_distribution_path.read_text(encoding="utf-8"))
        if isinstance(raw_gt, dict) and "0" in raw_gt:
            gt_dist = raw_gt["0"]
        elif isinstance(raw_gt, list):
            gt_dist = raw_gt

    # 1. Synthetic Pause
    p_synth_dir = extracted_root / "synthetic_pause_handling"
    if p_synth_dir.is_dir():
        for s_dir in sorted(p_synth_dir.iterdir()):
            if not s_dir.is_dir() or s_dir.name.startswith("."):
                continue
            wav_path = s_dir / "input.wav"
            pause_json = s_dir / "pause.json"
            if wav_path.is_file() and pause_json.is_file():
                try:
                    p_info = json.loads(pause_json.read_text(encoding="utf-8"))
                    t_start, t_end = p_info[0]["timestamp"]
                    pcm, sr = sphn.read(str(wav_path))
                    dur = len(pcm[0]) / sr
                    samples.append(
                        BenchmarkSample(
                            sample_id=f"fdb_synth_pause_{s_dir.name}",
                            task="pause_synthetic",
                            conversation_wav=wav_path,
                            user_channel=0,
                            voice_prompt_wav=default_voice_prompt,
                            text_prompt=default_text_prompt,
                            window_start_sec=0.0,
                            window_end_sec=dur,
                            pause_start_sec=float(t_start),
                            pause_end_sec=float(t_end),
                        )
                    )
                except Exception:
                    continue

    # 2. Candor Pause
    p_candor_dir = extracted_root / "candor_pause_handling"
    if p_candor_dir.is_dir():
        for s_dir in sorted(p_candor_dir.iterdir()):
            if not s_dir.is_dir() or s_dir.name.startswith("."):
                continue
            wav_path = s_dir / "input.wav"
            pause_json = s_dir / "pause.json"
            if wav_path.is_file() and pause_json.is_file():
                try:
                    p_info = json.loads(pause_json.read_text(encoding="utf-8"))
                    t_start, t_end = p_info[0]["timestamp"]
                    pcm, sr = sphn.read(str(wav_path))
                    dur = len(pcm[0]) / sr
                    samples.append(
                        BenchmarkSample(
                            sample_id=f"fdb_candor_pause_{s_dir.name}",
                            task="pause_natural",
                            conversation_wav=wav_path,
                            user_channel=0,
                            voice_prompt_wav=default_voice_prompt,
                            text_prompt=default_text_prompt,
                            window_start_sec=0.0,
                            window_end_sec=dur,
                            pause_start_sec=float(t_start),
                            pause_end_sec=float(t_end),
                        )
                    )
                except Exception:
                    continue

    # 3. Candor Turn-Taking
    turn_dir = extracted_root / "candor_turn_taking"
    if turn_dir.is_dir():
        for s_dir in sorted(turn_dir.iterdir()):
            if not s_dir.is_dir() or s_dir.name.startswith("."):
                continue
            wav_path = s_dir / "input.wav"
            turn_json = s_dir / "turn_taking.json"
            if wav_path.is_file() and turn_json.is_file():
                try:
                    t_info = json.loads(turn_json.read_text(encoding="utf-8"))
                    u_end = t_info[0]["timestamp"][0]
                    pcm, sr = sphn.read(str(wav_path))
                    dur = len(pcm[0]) / sr
                    samples.append(
                        BenchmarkSample(
                            sample_id=f"fdb_turn_{s_dir.name}",
                            task="turn_taking",
                            conversation_wav=wav_path,
                            user_channel=0,
                            voice_prompt_wav=default_voice_prompt,
                            text_prompt=default_text_prompt,
                            window_start_sec=0.0,
                            window_end_sec=dur,
                            user_turn_end_sec=float(u_end),
                        )
                    )
                except Exception:
                    continue

    # 4. ICC Backchannel
    bc_dir = extracted_root / "icc_backchannel"
    if bc_dir.is_dir():
        for s_dir in sorted(bc_dir.iterdir()):
            if not s_dir.is_dir() or s_dir.name.startswith("."):
                continue
            wav_path = s_dir / "input.wav"
            if wav_path.is_file():
                try:
                    pcm, sr = sphn.read(str(wav_path))
                    dur = len(pcm[0]) / sr
                    samples.append(
                        BenchmarkSample(
                            sample_id=f"fdb_bc_{s_dir.name}",
                            task="backchannel",
                            conversation_wav=wav_path,
                            user_channel=0,
                            voice_prompt_wav=default_voice_prompt,
                            text_prompt=default_text_prompt,
                            window_start_sec=0.0,
                            window_end_sec=dur,
                            gt_distribution=gt_dist,
                        )
                    )
                except Exception:
                    continue

    # 5. Synthetic User Interruption
    intr_dir = extracted_root / "synthetic_user_interruption"
    if intr_dir.is_dir():
        for s_dir in sorted(intr_dir.iterdir()):
            if not s_dir.is_dir() or s_dir.name.startswith("."):
                continue
            wav_path = s_dir / "input.wav"
            intr_json = s_dir / "interrupt.json"
            if wav_path.is_file() and intr_json.is_file():
                try:
                    i_info = json.loads(intr_json.read_text(encoding="utf-8"))
                    t_start, t_end = i_info[0]["timestamp"]
                    pcm, sr = sphn.read(str(wav_path))
                    dur = len(pcm[0]) / sr
                    samples.append(
                        BenchmarkSample(
                            sample_id=f"fdb_intr_{s_dir.name}",
                            task="interruption",
                            conversation_wav=wav_path,
                            user_channel=0,
                            voice_prompt_wav=default_voice_prompt,
                            text_prompt=default_text_prompt,
                            window_start_sec=0.0,
                            window_end_sec=dur,
                            interruption_start_sec=float(t_start),
                            interruption_end_sec=float(t_end),
                            interruption_query=i_info[0].get("interrupt", ""),
                            agent_pre_speech=i_info[0].get("context", ""),
                        )
                    )
                except Exception:
                    continue

    return samples
