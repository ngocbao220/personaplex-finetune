"""PersonaPlex's 17-stream hybrid-prompt training sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .data import PreparedSample


class Codec(Protocol):
    frame_rate: float
    codebooks: int

    def encode_conversation(self, path, channel: int, start_sec: float, end_sec: float) -> tuple[tuple[int, ...], ...]: ...
    def encode_voice_prompt(self, path) -> tuple[tuple[int, ...], ...]: ...
    def sine(self, frames: int) -> tuple[tuple[int, ...], ...]: ...
    def silence(self, frames: int) -> tuple[tuple[int, ...], ...]: ...


class Tokenizer(Protocol):
    padding_id: int
    end_padding_id: int

    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class WordTokenAlignment:
    speaker: str
    word: str
    start_sec: float
    end_sec: float
    start_frame: int
    token_frames: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrainingExample:
    input_codes: tuple[tuple[int, ...], ...]
    labels: tuple[tuple[int, ...], ...]
    loss_mask: tuple[tuple[bool, ...], ...]
    stream_names: tuple[str, ...]
    prompt_frames: int
    dialogue_frames: int
    voice_prompt_frames: int = 0
    text_prompt_frames: int = 0
    word_alignments: tuple[WordTokenAlignment, ...] = ()

    @property
    def total_frames(self) -> int:
        return len(self.input_codes[0])


def pad_training_example(example: TrainingExample, total_frames: int, text_padding_id: int, zero_token: int) -> TrainingExample:
    """Pad complete stream sequences and mask synthetic tail positions."""
    if total_frames < example.total_frames:
        raise ValueError("target sequence length cannot truncate a training example")
    padding = total_frames - example.total_frames
    if padding == 0:
        return example
    codes = tuple(
        stream + ((text_padding_id if index == 0 else zero_token),) * padding
        for index, stream in enumerate(example.input_codes)
    )
    labels = tuple(
        stream + ((text_padding_id if index == 0 else zero_token),) * padding
        for index, stream in enumerate(example.labels)
    )
    masks = tuple(mask + (False,) * padding for mask in example.loss_mask)
    return TrainingExample(
        input_codes=codes, labels=labels, loss_mask=masks,
        stream_names=example.stream_names, prompt_frames=example.prompt_frames,
        dialogue_frames=example.dialogue_frames, voice_prompt_frames=example.voice_prompt_frames,
        text_prompt_frames=example.text_prompt_frames, word_alignments=example.word_alignments,
    )


class PersonaPlexTrainingExampleBuilder:
    """Builds only the agent target contract; user audio remains conditioning."""

    pause_frames = 6

    def __init__(self, codec: Codec, tokenizer: Tokenizer, initial_tokens: Sequence[int], zero_token: int) -> None:
        self.codec = codec
        self.tokenizer = tokenizer
        self.initial_tokens = tuple(initial_tokens)
        self.zero_token = zero_token
        if codec.codebooks != 8 or len(self.initial_tokens) != 17:
            raise ValueError("PersonaPlex requires 8 codebooks per speaker and 17 initial tokens")

    def build(self, sample: PreparedSample, dialogue_codes=None) -> TrainingExample:
        if dialogue_codes is not None:
            agent, user = dialogue_codes
        elif hasattr(self.codec, "encode_conversation_stereo"):
            agent, user = self.codec.encode_conversation_stereo(
                sample.conversation_wav, sample.agent_channel, sample.user_channel, sample.window_start_sec, sample.window_end_sec
            )
        else:
            agent = self.codec.encode_conversation(sample.conversation_wav, sample.agent_channel, sample.window_start_sec, sample.window_end_sec)
            user = self.codec.encode_conversation(sample.conversation_wav, sample.user_channel, sample.window_start_sec, sample.window_end_sec)
        voice = self.codec.encode_voice_prompt(sample.voice_prompt_wav)

        self._assert_codebooks(agent, "agent dialogue")
        self._assert_codebooks(user, "user dialogue")
        self._assert_codebooks(voice, "voice prompt")
        dialogue_frames = len(agent[0])
        if dialogue_frames == 0 or len(user[0]) != dialogue_frames:
            raise ValueError(f"{sample.sample_id}: agent/user Mimi frame counts differ or are empty")
        voice_frames = len(voice[0])
        text_prompt = tuple(self.tokenizer.encode(f"<system> {sample.text_prompt.strip()} <system>"))
        if not text_prompt:
            raise ValueError(f"{sample.sample_id}: text prompt tokenized to no tokens")

        prompt_audio_frames = voice_frames + self.pause_frames + len(text_prompt) + self.pause_frames
        user_prompt = self.codec.sine(prompt_audio_frames)
        silence_prompt = self.codec.silence(self.pause_frames + len(text_prompt) + self.pause_frames)
        self._assert_codebooks(user_prompt, "sine prompt")
        self._assert_codebooks(silence_prompt, "silent prompt")

        agent_audio = tuple(
            voice[index] + silence_prompt[index] + agent[index]
            for index in range(8)
        )
        user_audio = tuple(user_prompt[index] + user[index] for index in range(8))
        dialogue_text, word_alignments = self._dialogue_text(sample, dialogue_frames)
        agent_text = (
            (self.tokenizer.padding_id,) * (voice_frames + self.pause_frames)
            + text_prompt
            + (self.tokenizer.padding_id,) * self.pause_frames
            + dialogue_text
        )
        streams = (agent_text,) + agent_audio + user_audio
        if any(len(stream) != len(agent_text) for stream in streams):
            raise AssertionError("hybrid streams have unequal lengths")
        prompt_mask = (False,) * prompt_audio_frames
        valid_duration = max(
            0.0,
            min(sample.window_end_sec, sample.audio.duration_sec) - sample.window_start_sec,
        )
        valid_dialogue_frames = min(
            dialogue_frames, round(valid_duration * self.codec.frame_rate)
        )
        dialogue_validity = (True,) * valid_dialogue_frames + (False,) * (dialogue_frames - valid_dialogue_frames)
        dialogue_text_mask = dialogue_validity
        loss_mask = (
            (prompt_mask + dialogue_text_mask),
            *((prompt_mask + dialogue_validity) for _ in range(8)),
            *((False,) * len(agent_text) for _ in range(8)),
        )
        return TrainingExample(
            input_codes=streams,
            labels=streams,
            loss_mask=loss_mask,
            stream_names=("agent_text",) + tuple(f"agent_audio_{i}" for i in range(8)) + tuple(f"user_audio_{i}" for i in range(8)),
            prompt_frames=prompt_audio_frames,
            dialogue_frames=dialogue_frames,
            voice_prompt_frames=voice_frames,
            text_prompt_frames=len(text_prompt),
            word_alignments=word_alignments,
        )

    def apply_delays(self, example: TrainingExample, delays: Sequence[int]) -> TrainingExample:
        if len(delays) != 17 or any(delay < 0 for delay in delays):
            raise ValueError("delays must contain 17 non-negative values")
        max_delay = max(delays)
        streams: list[tuple[int, ...]] = []
        masks: list[tuple[bool, ...]] = []
        for stream, mask, initial, delay in zip(example.input_codes, example.loss_mask, self.initial_tokens, delays, strict=True):
            streams.append((initial,) + (initial,) * delay + stream + (self.zero_token,) * (max_delay - delay))
            masks.append((False,) + (False,) * delay + mask + (False,) * (max_delay - delay))
        return TrainingExample(
            input_codes=tuple(streams), labels=tuple(streams), loss_mask=tuple(masks),
            stream_names=example.stream_names, prompt_frames=example.prompt_frames,
            dialogue_frames=example.dialogue_frames,
            voice_prompt_frames=example.voice_prompt_frames,
            text_prompt_frames=example.text_prompt_frames,
            word_alignments=example.word_alignments,
        )

    def _dialogue_text(self, sample: PreparedSample, frames: int) -> tuple[tuple[int, ...], tuple[WordTokenAlignment, ...]]:
        text = [self.tokenizer.padding_id] * frames
        alignments: list[WordTokenAlignment] = []
        for word in sample.words:
            if not sample.window_start_sec <= word.start < sample.window_end_sec:
                continue
            frame = min(frames - 1, int((word.start - sample.window_start_sec) * self.codec.frame_rate))
            start_frame = frame
            if word.speaker != "agent":
                alignments.append(WordTokenAlignment(
                    word.speaker, word.word, word.start, word.end, start_frame
                ))
                continue
            token_frames: list[int] = []
            for token in self.tokenizer.encode(" " + word.word):
                while frame < frames and text[frame] != self.tokenizer.padding_id:
                    frame += 1
                if frame >= frames:
                    break
                if frame > 0 and text[frame - 1] == self.tokenizer.padding_id:
                    text[frame - 1] = self.tokenizer.end_padding_id
                text[frame] = token
                token_frames.append(frame)
                frame += 1
            alignments.append(WordTokenAlignment(
                word.speaker, word.word, word.start, word.end, start_frame, tuple(token_frames)
            ))
        return tuple(text), tuple(alignments)

    def _assert_codebooks(self, streams: tuple[tuple[int, ...], ...], label: str) -> None:
        if len(streams) != 8 or len({len(stream) for stream in streams}) != 1:
            raise ValueError(f"{label} must contain eight equally sized codebooks")
