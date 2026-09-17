import unittest
from pathlib import Path

from personaplex_finetuning.data import AudioInfo, PreparedSample, Word
from personaplex_finetuning.sequence import PersonaPlexTrainingExampleBuilder


class FakeCodec:
    frame_rate = 12.5
    codebooks = 8

    def encode_conversation(self, _path, _channel, _start, _end):
        return tuple(tuple(100 * codebook + frame for frame in range(4)) for codebook in range(8))

    def encode_voice_prompt(self, _path):
        return tuple(tuple(10 * codebook + frame for frame in range(2)) for codebook in range(8))

    def sine(self, frames):
        return tuple(tuple(700 + codebook for _ in range(frames)) for codebook in range(8))

    def silence(self, frames):
        return tuple(tuple(800 + codebook for _ in range(frames)) for codebook in range(8))


class FakeTokenizer:
    padding_id = 3
    end_padding_id = 0

    def encode(self, text):
        return [len(text), len(text) + 1]


def sample() -> PreparedSample:
    return PreparedSample(
        sample_id="conv_0001",
        conversation_wav=Path("conversation.wav"),
        voice_prompt_wav=Path("voice_prompt.wav"),
        words=(
            Word("agent", "Hello", 10.0, 10.4),
            Word("user", "Hi", 10.5, 10.7),
            Word("agent", "there", 10.8, 11.0),
        ),
        text_prompt="Be helpful.",
        metadata={},
        audio=AudioInfo(24000, 2, 60),
        window_start_sec=10,
        window_end_sec=14,
    )


class SequenceBuilderTest(unittest.TestCase):
    def test_builds_17_streams_with_masked_hybrid_prompt_and_agent_targets(self) -> None:
        builder = PersonaPlexTrainingExampleBuilder(
            codec=FakeCodec(), tokenizer=FakeTokenizer(), initial_tokens=[1] * 17, zero_token=-1
        )

        example = builder.build(sample())

        self.assertEqual(example.stream_names[0], "agent_text")
        self.assertEqual(len(example.input_codes), 17)
        self.assertTrue(all(len(stream) == example.total_frames for stream in example.input_codes))
        self.assertEqual(example.prompt_frames, 16)  # voice=2, two 6-frame pauses, prompt text=2
        self.assertTrue(all(not enabled for stream in example.loss_mask for enabled in stream[:16]))
        self.assertTrue(all(example.loss_mask[stream][16] for stream in range(1, 9)))
        self.assertTrue(all(not example.loss_mask[stream][16] for stream in range(9, 17)))
        self.assertEqual(example.loss_mask[0][16], True)

    def test_delay_keeps_masks_and_stream_lengths_aligned(self) -> None:
        builder = PersonaPlexTrainingExampleBuilder(
            codec=FakeCodec(), tokenizer=FakeTokenizer(), initial_tokens=[1] * 17, zero_token=-1
        )
        example = builder.build(sample())

        delayed = builder.apply_delays(example, [0, 0] + [1] * 7 + [0] + [1] * 7)

        # One initial frame plus the largest stream delay are prepended/appended.
        self.assertEqual(delayed.total_frames, example.total_frames + 2)
        self.assertFalse(delayed.loss_mask[2][0])
        self.assertFalse(delayed.loss_mask[1][-1])
