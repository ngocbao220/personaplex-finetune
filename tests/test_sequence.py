import unittest
from dataclasses import replace
from pathlib import Path

from personaplex_finetuning.data import AudioInfo, PreparedSample, Word
from personaplex_finetuning.sequence import (
    PersonaPlexTrainingExampleBuilder,
    align_dialogue_text_targets,
    pad_training_example,
)


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


class RecordingTokenizer(FakeTokenizer):
    def __init__(self):
        self.inputs = []

    def encode(self, text):
        self.inputs.append(text)
        return super().encode(text)


def sample() -> PreparedSample:
    return PreparedSample(
        sample_id="conv_0001",
        conversation_wav=Path("conversation.wav"),
        voice_prompt_wav=Path("voice_prompt.wav"),
        words=(
            Word("agent", "Hello", 10.0, 10.08),
            Word("user", "Hi", 10.08, 10.16),
            Word("agent", "there", 10.16, 10.32),
        ),
        text_prompt="Be helpful.",
        metadata={},
        audio=AudioInfo(24000, 2, 60),
        window_start_sec=10,
        window_end_sec=14,
    )


class SequenceBuilderTest(unittest.TestCase):
    def test_alignment_diagnostic_reports_local_overflow_below_total_frame_capacity(self):
        late_word = replace(
            sample(), words=(Word("agent", "x", 0.9, 0.95),),
            window_start_sec=0.0, window_end_sec=1.0,
        )

        result = align_dialogue_text_targets(late_word, 10, 10.0, FakeTokenizer())

        self.assertEqual(result.required_tokens, 2)
        self.assertEqual(result.placed_tokens, 1)
        self.assertEqual(result.overflow_word.word, "x")

    def test_word_tokens_use_sentencepiece_prefix_without_an_extra_leading_space(self):
        tokenizer = RecordingTokenizer()
        builder = PersonaPlexTrainingExampleBuilder(
            codec=FakeCodec(), tokenizer=tokenizer, initial_tokens=[1] * 17, zero_token=-1
        )

        builder.build(sample())

        self.assertIn("Hello", tokenizer.inputs)
        self.assertIn("there", tokenizer.inputs)
        self.assertNotIn(" Hello", tokenizer.inputs)
        self.assertNotIn(" there", tokenizer.inputs)

    def test_vietnamese_target_normalization_does_not_modify_source_word_alignment(self):
        tokenizer = RecordingTokenizer()
        vietnamese_sample = replace(
            sample(), words=(Word("agent", "người", 10.0, 10.3),),
        )
        builder = PersonaPlexTrainingExampleBuilder(
            codec=FakeCodec(), tokenizer=tokenizer, initial_tokens=[1] * 17,
            zero_token=-1, normalize_vietnamese_diacritics=True,
        )

        example = builder.build(vietnamese_sample)

        self.assertIn("nguoi", tokenizer.inputs)
        self.assertNotIn("người", tokenizer.inputs)
        self.assertEqual(example.word_alignments[0].word, "người")

    def test_prefers_cached_stereo_mimi_codes_when_available(self) -> None:
        class CachedCodec(FakeCodec):
            def __init__(self):
                self.cached_calls = 0

            def encode_conversation_stereo_cached(self, *_args):
                self.cached_calls += 1
                codes = tuple(tuple(100 * cb + frame for frame in range(4)) for cb in range(8))
                return codes, codes

            def encode_conversation_stereo(self, *_args):
                raise AssertionError("cache-backed training should not encode Mimi again")

        codec = CachedCodec()
        builder = PersonaPlexTrainingExampleBuilder(
            codec=codec, tokenizer=FakeTokenizer(), initial_tokens=[1] * 17, zero_token=-1
        )

        example = builder.build(sample())

        self.assertEqual(codec.cached_calls, 1)
        self.assertEqual(example.dialogue_frames, 4)

    def test_final_chunk_padding_is_masked_while_real_silence_remains_valid(self) -> None:
        codec = FakeCodec()
        codec.frame_rate = 0.5
        short_conversation = replace(
            sample(), words=(), audio=AudioInfo(24000, 2, 2.0),
            window_start_sec=0.0, window_end_sec=100.0,
        )
        builder = PersonaPlexTrainingExampleBuilder(
            codec=codec, tokenizer=FakeTokenizer(), initial_tokens=[1] * 17, zero_token=-1
        )

        example = builder.build(short_conversation)

        self.assertEqual(example.dialogue_frames, 4)
        self.assertTrue(example.loss_mask[0][example.prompt_frames])
        self.assertFalse(example.loss_mask[0][example.prompt_frames + 1])
        self.assertTrue(example.loss_mask[1][example.prompt_frames])
        self.assertFalse(example.loss_mask[1][example.prompt_frames + 1])

    def test_fractional_final_mimi_frame_is_valid_instead_of_rounded_away(self) -> None:
        codec = FakeCodec()
        codec.frame_rate = 0.5
        short_conversation = replace(
            sample(), words=(), audio=AudioInfo(24000, 2, 1.0),
            window_start_sec=0.0, window_end_sec=1.0,
        )
        builder = PersonaPlexTrainingExampleBuilder(
            codec=codec, tokenizer=FakeTokenizer(), initial_tokens=[1] * 17, zero_token=-1
        )

        example = builder.build(short_conversation)

        self.assertTrue(example.loss_mask[0][example.prompt_frames])
        self.assertFalse(example.loss_mask[0][example.prompt_frames + 1])
        self.assertTrue(example.loss_mask[1][example.prompt_frames])
        self.assertFalse(example.loss_mask[1][example.prompt_frames + 1])

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

    def test_hybrid_prompt_uses_the_configured_native_inference_pause_length(self) -> None:
        builder = PersonaPlexTrainingExampleBuilder(
            codec=FakeCodec(), tokenizer=FakeTokenizer(), initial_tokens=[1] * 17,
            zero_token=-1, pause_frames=2,
        )

        example = builder.build(sample())

        self.assertEqual(example.prompt_frames, 8)  # voice=2, two 2-frame pauses, prompt text=2

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

    def test_batch_padding_adds_equal_shape_and_zero_loss_tail(self) -> None:
        builder = PersonaPlexTrainingExampleBuilder(
            codec=FakeCodec(), tokenizer=FakeTokenizer(), initial_tokens=[1] * 17, zero_token=-1
        )
        example = builder.build(sample())
        padded = pad_training_example(example, example.total_frames + 5, 3, -1)

        self.assertEqual(padded.total_frames, example.total_frames + 5)
        self.assertTrue(all(not mask[-5:].count(True) for mask in padded.loss_mask))
        self.assertEqual(padded.input_codes[0][-5:], (3,) * 5)
        self.assertEqual(padded.input_codes[1][-5:], (-1,) * 5)
