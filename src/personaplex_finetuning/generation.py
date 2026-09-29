"""PersonaPlex generation settings shared by training validation and inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationSettings:
    """Sampling settings forwarded to Moshi's native ``LMGen``."""

    use_sampling: bool = True
    temp: float = 0.8
    temp_text: float = 0.7
    top_k: int = 250
    top_k_text: int = 25
    audio_silence_frame_cnt: int = 6

    def lmgen_kwargs(self) -> dict[str, Any]:
        return {
            "use_sampling": self.use_sampling,
            "temp": self.temp,
            "temp_text": self.temp_text,
            "top_k": self.top_k,
            "top_k_text": self.top_k_text,
            "audio_silence_frame_cnt": self.audio_silence_frame_cnt,
        }

    def as_dict(self) -> dict[str, Any]:
        return self.lmgen_kwargs()

    def label(self) -> str:
        audio = "greedy" if (not self.use_sampling or self.temp <= 0.0) else f"sampling temp={self.temp:g}"
        text = "greedy" if (not self.use_sampling or self.temp_text <= 0.0) else f"sampling temp={self.temp_text:g}"
        return (
            f"native LMGen audio={audio}, text={text}, top_k={self.top_k}, "
            f"top_k_text={self.top_k_text}, audio_silence_frame_cnt={self.audio_silence_frame_cnt}"
        )


def _as_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return value.strip().lower() in {"true", "1"}
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _as_float(name: str, value: Any, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum:g}, got {value!r}")
    return number


def _as_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if float(value) != int(value):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return number


def generation_from_config(raw: Mapping[str, Any] | None) -> GenerationSettings:
    """Parse a generation block, rejecting unknown and invalid values."""
    defaults = GenerationSettings()
    section = {} if raw is None else raw.get("generation")
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError("generation config section must be a mapping")
    supported = sorted(defaults.as_dict())
    unknown = sorted(set(section) - set(supported))
    if unknown:
        raise ValueError(f"unknown generation config keys: {unknown}; supported keys are {supported}")
    return GenerationSettings(
        use_sampling=_as_bool("generation.use_sampling", section.get("use_sampling", defaults.use_sampling)),
        temp=_as_float("generation.temp", section.get("temp", defaults.temp), minimum=0.0),
        temp_text=_as_float("generation.temp_text", section.get("temp_text", defaults.temp_text), minimum=0.0),
        top_k=_as_int("generation.top_k", section.get("top_k", defaults.top_k), minimum=0),
        top_k_text=_as_int("generation.top_k_text", section.get("top_k_text", defaults.top_k_text), minimum=0),
        audio_silence_frame_cnt=_as_int(
            "generation.audio_silence_frame_cnt",
            section.get("audio_silence_frame_cnt", defaults.audio_silence_frame_cnt),
            minimum=0,
        ),
    )
