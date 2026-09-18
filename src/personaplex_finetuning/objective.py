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
    """Return explicit per-token weights for [text, agent audio x8, user audio x8].

    Vectorized via numpy to avoid a ~6000-iteration Python loop per step.
    The result is identical to the original element-wise loop.
    """
    if len(codes) != 17 or len(loss_mask) != 17:
        raise ValueError("PersonaPlex objective requires exactly 17 streams")
    if not 0 <= nonsemantic_audio_weight <= 1 or not 0 <= text_padding_weight <= 1:
        raise ValueError("loss weights must be within [0, 1]")

    import numpy as np

    weighted: list[tuple[float, ...]] = []
    for stream_index, (stream, mask) in enumerate(zip(codes, loss_mask, strict=True)):
        if len(stream) != len(mask):
            raise ValueError(f"stream {stream_index} code/mask length mismatch")

        mask_arr = np.asarray(mask, dtype=np.bool_)

        # User audio streams (indices 9-16) are always zero-weighted
        if stream_index >= 9:
            weighted.append(tuple([0.0] * len(stream)))
            continue

        if stream_index == 0:
            # Text stream: 1.0 for real tokens, text_padding_weight for padding
            tokens_arr = np.asarray(stream, dtype=np.int64)
            values = np.where(
                tokens_arr == text_padding_id,
                np.float64(text_padding_weight),
                np.float64(1.0),
            ).astype(np.float64)
        elif stream_index == 1:
            # First (semantic) audio codebook: full weight
            values = np.ones(len(stream), dtype=np.float64)
        else:
            # Non-semantic audio codebooks (2-8): downweighted
            values = np.full(len(stream), nonsemantic_audio_weight, dtype=np.float64)

        # Zero out prompt/padding positions where loss_mask is False
        values = np.where(mask_arr, values, np.float64(0.0)).astype(np.float64)
        weighted.append(tuple(values.flat))

    return tuple(weighted)


def stream_weights_torch(
    codes: "torch.Tensor",
    loss_mask: "torch.Tensor",
    text_padding_id: int,
    nonsemantic_audio_weight: float = 0.02,
    text_padding_weight: float = 0.3,
) -> "torch.Tensor":
    """GPU-native weight computation for use inside the training loop.

    Args:
        codes:    (S, T) long tensor — all 17 streams, S=17, T=sequence length.
        loss_mask:(S, T) bool tensor — True where loss should be computed.

    Returns:
        (S, T) float32 weight tensor on the same device as ``codes``.
    """
    import torch

    S, T = codes.shape
    assert S == 17, f"Expected 17 streams, got {S}"
    weights = torch.zeros(S, T, dtype=torch.float32, device=codes.device)

    # Agent text stream (index 0): 1.0 for real tokens, text_padding_weight for padding
    text_mask = loss_mask[0]
    is_pad = codes[0] == text_padding_id
    weights[0] = torch.where(
        text_mask,
        torch.where(is_pad, torch.tensor(text_padding_weight, device=codes.device), torch.tensor(1.0, device=codes.device)),
        torch.tensor(0.0, device=codes.device),
    )

    # Agent semantic audio (index 1): full weight where unmasked
    weights[1] = loss_mask[1].float()

    # Agent non-semantic audio (indices 2-8): downweighted
    weights[2:9] = loss_mask[2:9].float() * nonsemantic_audio_weight

    # User audio streams (indices 9-16): zero (no loss on user stream)
    # weights[9:17] already 0

    return weights


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
    # (-1) and NaN logits, so map only zero-weight positions to harmless
    # values before CUDA cross_entropy evaluates them.
    ignored = weights == 0
    safe_targets = targets.masked_fill(ignored, 0)
    safe_logits = logits.float().masked_fill(ignored.unsqueeze(1), 0.0)
    per_token = functional.cross_entropy(safe_logits, safe_targets, reduction="none")
    return (per_token * weights).sum() / denominator
