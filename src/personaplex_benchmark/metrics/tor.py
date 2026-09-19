"""Takeover Rate (TOR) computation following Full-Duplex-Bench standards."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SpeechEvent:
    """A contiguous speech utterance detected from the model."""
    start_sec: float
    end_sec: float
    text: str = ""
    words: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.words and self.text:
            cleaned = re.sub(r"[^\w\s]", "", self.text).strip()
            self.words = [w for w in cleaned.split() if w]

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def word_count(self) -> int:
        return len(self.words)


def compute_tor(
    events: Sequence[SpeechEvent],
    window_start_sec: float | None = None,
    window_end_sec: float | None = None,
    duration_threshold: float = 1.0,
    words_threshold: int = 3,
) -> float:
    """Computes whether a takeover occurred (1.0) or not (0.0).

    Following FDB: A takeover occurs when the model produces speech
    with duration >= duration_threshold AND word_count >= words_threshold.
    """
    for ev in events:
        # Check if the event overlaps with the window if window is specified
        if window_start_sec is not None and ev.end_sec <= window_start_sec:
            continue
        if window_end_sec is not None and ev.start_sec >= window_end_sec:
            continue

        # Check thresholds
        if ev.duration >= duration_threshold and ev.word_count >= words_threshold:
            return 1.0

    return 0.0
