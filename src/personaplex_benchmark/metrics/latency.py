"""Latency metrics computation following Full-Duplex-Bench."""

from __future__ import annotations

from typing import Sequence
from .tor import SpeechEvent


def compute_speaker_switch_latency(
    events: Sequence[SpeechEvent],
    user_turn_end_sec: float,
    min_words: int = 1,
) -> float | None:
    """Computes the latency from user turn completion to agent speech onset.

    Returns the latency in seconds, or None if no valid response occurred.
    """
    for ev in events:
        if ev.end_sec <= user_turn_end_sec:
            continue
        if ev.word_count >= min_words:
            return round(max(0.0, ev.start_sec - user_turn_end_sec), 4)

    return None


def compute_interruption_latency(
    events: Sequence[SpeechEvent],
    interruption_end_sec: float,
    min_words: int = 1,
) -> float | None:
    """Computes the latency from user interruption end to agent resumption onset.

    Returns latency in seconds, or None if agent never responded post-interruption.
    """
    for ev in events:
        if ev.start_sec >= interruption_end_sec:
            if ev.word_count >= min_words:
                return round(max(0.0, ev.start_sec - interruption_end_sec), 4)

    return None
