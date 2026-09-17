"""PersonaPlex loss masks and weights, separated from model execution."""

from __future__ import annotations

from typing import Sequence


def stream_weights(
    codes: Sequence[Sequence[int]],
    loss_mask: Sequence[Sequence[bool]],
    text_padding_id: int,
    nonsemantic_audio_weight: float = 0.02,
    text_padding_weight: float = 0.3,
) -> tuple[tuple[float, ...], ...]:
    """Return explicit weights for [text, agent audio x8, user audio x8]."""
    if len(codes) != 17 or len(loss_mask) != 17:
        raise ValueError("PersonaPlex objective requires exactly 17 streams")
    if not 0 <= nonsemantic_audio_weight <= 1 or not 0 <= text_padding_weight <= 1:
        raise ValueError("loss weights must be within [0, 1]")
    weighted: list[tuple[float, ...]] = []
    for stream_index, (stream, mask) in enumerate(zip(codes, loss_mask, strict=True)):
        if len(stream) != len(mask):
            raise ValueError(f"stream {stream_index} code/mask length mismatch")
        values: list[float] = []
        for token, enabled in zip(stream, mask, strict=True):
            if not enabled or stream_index >= 9:
                values.append(0.0)
            elif stream_index == 0:
                values.append(text_padding_weight if token == text_padding_id else 1.0)
            elif stream_index == 1:
                values.append(1.0)
            else:
                values.append(nonsemantic_audio_weight)
        weighted.append(tuple(values))
    return tuple(weighted)


def torch_weighted_cross_entropy(logits, targets, weights):
    """Compute a weighted mean cross entropy without importing Torch at module import."""
    import torch
    import torch.nn.functional as functional

    if logits.ndim != 2 or targets.ndim != 1 or weights.ndim != 1:
        raise ValueError("expected flattened logits [N,V], targets [N], weights [N]")
    if logits.shape[0] != targets.numel() or targets.numel() != weights.numel():
        raise ValueError("logits, targets, and weights must have matching token counts")
    denominator = weights.sum()
    if denominator.item() == 0:
        return logits.sum() * 0.0
    # CUDA cross_entropy validates every target before weights are applied.
    # Delay/padding positions deliberately contain the PersonaPlex zero token
    # (-1), so map only zero-weight positions to a harmless valid class.
    safe_targets = targets.masked_fill(weights == 0, 0)
    per_token = functional.cross_entropy(logits.float(), safe_targets, reduction="none")
    return (per_token * weights).sum() / denominator
