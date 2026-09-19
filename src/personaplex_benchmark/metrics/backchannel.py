"""Backchannel metrics: TOR, Frequency, and Jensen-Shannon Divergence (JSD)."""

from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy.spatial.distance import jensenshannon

from .tor import SpeechEvent, compute_tor


def compute_backchannel_metrics(
    events: Sequence[SpeechEvent],
    user_duration_sec: float,
    gt_distribution: Sequence[float] | None = None,
    num_bins: int = 10,
    bc_duration_threshold: float = 3.0,
    bc_words_threshold: int = 2,
) -> dict[str, float]:
    """Computes Backchannel metrics:

    - tor: 1.0 if agent spoke > bc_duration_threshold or > bc_words_threshold, else 0.0
    - freq: number of backchannels per second (or total count if duration is 0)
    - jsd: Jensen-Shannon Divergence vs ground-truth distribution
    """
    # 1. Check TOR (whether agent hijacked the turn)
    tor = compute_tor(
        events,
        duration_threshold=bc_duration_threshold,
        words_threshold=bc_words_threshold + 1,
    )

    # 2. Identify backchannel candidates
    bc_events: list[SpeechEvent] = []
    for ev in events:
        if ev.duration <= bc_duration_threshold and ev.word_count <= bc_words_threshold:
            bc_events.append(ev)

    # 3. Frequency (count per second of user speech)
    freq = (len(bc_events) / max(1.0, user_duration_sec)) if user_duration_sec > 0 else 0.0

    # 4. JSD
    if gt_distribution is not None and len(gt_distribution) > 0 and len(bc_events) > 0 and user_duration_sec > 0:
        # Construct normalized timing points (0.0 to 1.0)
        norm_times = [min(1.0, max(0.0, ev.start_sec / user_duration_sec)) for ev in bc_events]
        n_bins = len(gt_distribution)
        hist, _ = np.histogram(norm_times, bins=n_bins, range=(0.0, 1.0))
        # Add epsilon smoothing
        prob_model = (hist + 1e-6) / (np.sum(hist) + 1e-6 * n_bins)
        prob_gt = np.array(gt_distribution, dtype=float)
        prob_gt = (prob_gt + 1e-6) / (np.sum(prob_gt) + 1e-6 * len(prob_gt))
        jsd = float(jensenshannon(prob_model, prob_gt))
    else:
        # Default JSD if no backchannels or no reference: 1.0 (maximum divergence) or 0.0 if both empty
        jsd = 1.0 if len(bc_events) == 0 else 0.5

    return {
        "tor": tor,
        "freq": round(freq, 4),
        "jsd": round(jsd, 4),
        "bc_count": len(bc_events),
    }
