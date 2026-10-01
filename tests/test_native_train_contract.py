"""Compare native training preprocessing with fully forced LMGen history."""

import json
import unittest
from pathlib import Path

import torch

from moshi.models import loaders
from moshi.models.lm import LMGen, LMModel, SILENCE_TOKENS, SINE_TOKENS
from personaplex_finetuning.data import AudioInfo, PreparedSample, Word
from personaplex_finetuning.sequence import PersonaPlexTrainingExampleBuilder
from personaplex_finetuning.runtime import MimiCodec, SentencePieceTokenizer
from personaplex_finetuning.data import read_audio_window
from personaplex_finetuning.objective import stream_weights_torch
from personaplex_finetuning.train import codebook_diagnostic_stats


class _Codec:
    frame_rate = 12.5
    codebooks = 8

    def encode_conversation(self, _path, channel, _start, _end):
        return tuple(tuple(100 * channel + 10 * cb + t for t in range(38)) for cb in range(8))

    def encode_voice_prompt(self, _path):
        return tuple(tuple(20 * cb + t for t in range(3)) for cb in range(8))

    def sine(self, frames):
        return tuple((int(token),) * frames for token in SINE_TOKENS)

    def silence(self, frames):
        return tuple((int(token),) * frames for token in SILENCE_TOKENS)


class _Tokenizer:
    padding_id = 3
    end_padding_id = 0

    def encode(self, text):
        return [10 + (ord(char) % 30) for char in text]


class _RecordingLM(LMModel):
    """Use native LMModel preprocessing without allocating 7B weights."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(())))
        self.n_q = self.dep_q = 16
        self.card = 2048
        self.text_card = loaders._lm_kwargs["text_card"]
        self.existing_text_padding_id = 3
        self.delays = list(loaders._lm_kwargs["delays"])
        self.inputs = []
        self.targets = []

    def forward_codes(self, codes):
        self.inputs.append(codes.detach().clone())
        batch, _, frames = codes.shape
        return torch.zeros(batch, frames, 1), torch.zeros(batch, 1, frames, self.text_card)

    def forward_depformer_training(self, codes, _transformer_out):
        self.targets.append(codes.detach().clone())
        batch, _, frames = codes.shape
        return torch.zeros(batch, self.dep_q, frames, self.card)


class NativeTrainContractTest(unittest.TestCase):
    def test_both_semantic_codebooks_and_all_audio_diagnostics_are_present(self):
        from types import SimpleNamespace

        labels = torch.zeros((1, 17, 2), dtype=torch.long)
        mask = torch.ones_like(labels, dtype=torch.bool)
        mask[:, :, 0] = False  # prompt conditioning only
        weights = stream_weights_torch(labels, mask, (3, 0), text_padding_weight=0.5)
        self.assertEqual(float(weights[0, 1, 1]), 1.0)
        self.assertEqual(float(weights[0, 9, 1]), 1.0)
        self.assertAlmostEqual(float(weights[0, 2, 1]), 0.02)
        self.assertAlmostEqual(float(weights[0, 10, 1]), 0.02)
        self.assertTrue(torch.all(weights[:, :, 0] == 0))
        output = SimpleNamespace(
            text_logits=torch.zeros((1, 1, 2, 5)),
            text_mask=torch.ones((1, 1, 2), dtype=torch.bool),
            logits=torch.zeros((1, 16, 2, 5)),
            mask=torch.ones((1, 16, 2), dtype=torch.bool),
        )
        _, _, _, correct, count, losses = codebook_diagnostic_stats(
            {"labels": labels, "loss_mask": mask}, output, (3, 0),
        )
        self.assertEqual(tuple(correct.shape), (16,))
        self.assertEqual(tuple(count.tolist()), (1,) * 16)
        self.assertEqual(tuple(losses.shape), (16,))

    def test_real_prepared_three_second_window_matches_all_native_inputs_and_targets(self):
        from moshi.models import lm

        root = Path(__file__).resolve().parents[2]
        model_root = root / "models"
        sample_root = root / "otospeech-prepared" / "samples" / "conv_0001"
        required = (
            model_root / "tokenizer-e351c8d8-checkpoint125.safetensors",
            model_root / "tokenizer_spm_32k_3.model",
            sample_root / "conversation.wav", sample_root / "voice_prompt.wav",
        )
        if any(not path.is_file() for path in required):
            self.skipTest("local Mimi/tokenizer or prepared sample unavailable")
        torch.set_num_threads(1)
        mimi = loaders.get_mimi(required[0], device="cpu").eval()
        codec = MimiCodec(mimi, mimi.sample_rate, mimi.frame_rate, "cpu", lm)
        tokenizer = SentencePieceTokenizer(required[1])
        metadata = json.loads((sample_root / "metadata.json").read_text())
        sample = PreparedSample(
            "conv_0001_window", sample_root / "conversation.wav", sample_root / "voice_prompt.wav",
            (Word("agent", "And", 1.3, 1.46),), metadata["text_prompt"], metadata,
            AudioInfo(24000, 2, metadata["duration_sec"]), 0.0, 3.04,
        )
        self.assertEqual((sample.agent_channel, sample.user_channel), (0, 1))
        model = _RecordingLM().eval()
        example = PersonaPlexTrainingExampleBuilder(
            codec, tokenizer, tuple(model._get_initial_token()[0, :, 0].tolist()), model.zero_token_id,
        ).build(sample)
        canonical = torch.tensor(example.input_codes)[None]
        self.assertEqual(canonical.shape, (1, 17, example.prompt_frames + example.dialogue_frames))
        self.assertTrue(any(token != tokenizer.padding_id for token in example.labels[0][example.prompt_frames:]))
        model.forward_train(canonical)
        train_input, train_target = model.inputs.pop(), model.targets.pop()
        generator = LMGen(model, device="cpu", use_sampling=False, check=True)
        generator._streaming_state = generator._init_streaming_state(1)
        self.assertIsNone(generator.prepare_step_input())
        native_inputs, native_targets = [], []
        for frame in range(canonical.shape[2]):
            prepared = generator.prepare_step_input(
                input_tokens=canonical[:, 9:17, frame:frame + 1],
                moshi_tokens=canonical[:, 1:9, frame:frame + 1],
                text_token=canonical[:, 0, frame],
            )
            input_codes, provided, target, _, _ = prepared
            self.assertTrue(provided.all(), f"generated token in LM In at frame {frame}")
            native_inputs.append(input_codes.clone())
            native_targets.append(target.clone())
            generator._streaming_state.offset += 1
        self.assertTrue(torch.equal(train_input, torch.cat(native_inputs, dim=2)))
        self.assertTrue(torch.equal(train_target, torch.cat(native_targets, dim=2)))
        print("real prepared 3.04 s window: 17/17 streams verified; input mismatches = 0; target mismatches = 0")

    def test_real_voice_prompt_codes_match_native_lmgen(self):
        from moshi.models import lm

        root = Path(__file__).resolve().parents[2]
        mimi_path = root / "models" / "tokenizer-e351c8d8-checkpoint125.safetensors"
        voice_path = root / "otospeech-prepared" / "samples" / "conv_0001" / "voice_prompt.wav"
        if not mimi_path.is_file() or not voice_path.is_file():
            self.skipTest("local Mimi weights or prepared voice prompt unavailable")
        torch.set_num_threads(1)
        mimi = loaders.get_mimi(mimi_path, device="cpu").eval()
        codec = MimiCodec(mimi, mimi.sample_rate, mimi.frame_rate, "cpu", lm)
        train_codes = torch.tensor(codec.encode_voice_prompt(voice_path), dtype=torch.long)
        prompt_audio = lm.normalize_audio(lm.load_audio(str(voice_path), mimi.sample_rate), mimi.sample_rate, -24.0)
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio[None, :]
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        with mimi.streaming(1):
            native_frames = list(lm.encode_from_sphn(
                mimi, lm._iterate_audio(prompt_audio, frame_size, pad=True), max_batch=1,
            ))
        native_codes = torch.cat(native_frames, dim=2)[0]
        self.assertTrue(torch.equal(train_codes, native_codes),
                        f"voice prompt differs from LMGen at {(train_codes != native_codes).sum().item()} tokens")

    def test_real_user_codes_match_native_streaming_input(self):
        from moshi.models import lm

        root = Path(__file__).resolve().parents[2]
        mimi_path = root / "models" / "tokenizer-e351c8d8-checkpoint125.safetensors"
        conversation = root / "otospeech-prepared" / "samples" / "conv_0001" / "conversation.wav"
        if not mimi_path.is_file() or not conversation.is_file():
            self.skipTest("local Mimi or prepared conversation unavailable")
        torch.set_num_threads(1)
        mimi = loaders.get_mimi(mimi_path, device="cpu").eval()
        codec = MimiCodec(mimi, mimi.sample_rate, mimi.frame_rate, "cpu", lm)
        train_codes = torch.tensor(codec.encode_conversation(conversation, 1, 0.0, 3.04))
        audio = read_audio_window(conversation, 0.0, 3.04, mimi.sample_rate, str(conversation), channels=(1, 2))
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        with mimi.streaming(1):
            native_frames = list(lm.encode_from_sphn(
                mimi, lm._iterate_audio(audio[1:2], frame_size, pad=True), max_batch=1,
            ))
        native_codes = torch.cat(native_frames, dim=2)[0]
        self.assertTrue(torch.equal(train_codes, native_codes),
                        f"user input differs from native LMGen at {(train_codes != native_codes).sum().item()} tokens")

    def test_all_streams_match_fully_forced_lmgen_before_forward(self):
        model = _RecordingLM().eval()
        sample = PreparedSample(
            "short", Path("stereo.wav"), Path("voice.wav"),
            (Word("agent", "hello", 0.1, 0.5),), "You are a helpful assistant.",
            {}, AudioInfo(24000, 2, 3.04), 0.0, 3.04,
        )
        example = PersonaPlexTrainingExampleBuilder(
            _Codec(), _Tokenizer(), tuple(model._get_initial_token()[0, :, 0].tolist()),
            model.zero_token_id,
        ).build(sample)
        canonical = torch.tensor(example.input_codes, dtype=torch.long)[None]
        self.assertEqual(canonical.shape[1], 17)
        self.assertEqual(canonical.shape[2], example.prompt_frames + 38)
        self.assertEqual((model.audio_offset, model.dep_q), (1, 16))
        self.assertEqual(model.text_padding_token_id, _Tokenizer.padding_id)
        self.assertEqual(model.end_of_text_padding_id, _Tokenizer.end_padding_id)
        self.assertEqual(example.labels, example.input_codes)
        self.assertFalse(any(any(row[:example.prompt_frames]) for row in example.loss_mask))
        self.assertEqual(tuple(canonical[0, 9:17, 0].tolist()), tuple(SINE_TOKENS))
        self.assertEqual(tuple(canonical[0, 1:9, 3].tolist()), tuple(SILENCE_TOKENS))

        # This calls native forward_train, including its delay, BOS and target shift.
        model.forward_train(canonical)
        train_input, train_target = model.inputs.pop(), model.targets.pop()

        gen = LMGen(model, device="cpu", use_sampling=False, check=True)
        # prepare_step_input is the native boundary immediately before LM forward.
        # The empty first call initializes BOS; every later call forces all 17 GT streams.
        gen._streaming_state = gen._init_streaming_state(1)
        self.assertIsNone(gen.prepare_step_input())
        native_inputs, native_targets, source_rows = [], [], []
        for t in range(canonical.shape[2]):
            prepared = gen.prepare_step_input(
                input_tokens=canonical[:, 9:17, t:t + 1],
                moshi_tokens=canonical[:, 1:9, t:t + 1],
                text_token=canonical[:, 0, t],
            )
            input_codes, provided, target, _, _ = prepared
            self.assertTrue(provided.all(), f"LM In at frame {t} includes a sampled token")
            native_inputs.append(input_codes.clone())
            native_targets.append(target.clone())
            source_rows.append(("GT text/history",) + ("GT agent Mimi",) * 8 + ("GT user Mimi",) * 8)
            gen._streaming_state.offset += 1
        native_input = torch.cat(native_inputs, dim=2)
        native_target = torch.cat(native_targets, dim=2)
        input_mismatches = int((train_input != native_input).sum())
        target_mismatches = int((train_target != native_target).sum())
        initial_mismatches = int((train_input[:, :, 0] != model._get_initial_token()[:, :, 0]).sum())
        delay_mismatches = sum(a != b for a, b in zip(model.delays, loaders._lm_kwargs["delays"], strict=True))
        self.assertEqual(initial_mismatches, 0)
        self.assertEqual(delay_mismatches, 0)
        self.assertEqual(input_mismatches, 0)
        self.assertEqual(target_mismatches, 0)
        self.assertEqual(len(source_rows[0]), 17)
        for index, (name, source) in enumerate(zip(example.stream_names, source_rows[0], strict=True)):
            print(f"stream {index:2} {name:<16} <- {source}")
        print(f"prompt boundary={example.prompt_frames}; first dialogue frame={example.prompt_frames}")
        print(f"initial token mismatches={initial_mismatches}; delay mismatches={delay_mismatches}")
        print("17/17 streams verified; input mismatches = 0; target mismatches = 0")
        print("PASS: TRAIN–INFERENCE CONTRACT EQUIVALENT")


if __name__ == "__main__":
    unittest.main()
