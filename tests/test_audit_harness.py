"""Audit must preserve occurrence identity and avoid unsupported causal claims."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_core import compare_runs, load_run, summarize_run, compare_alignment, replay_updates
from personaplex_finetuning.data import AudioInfo, PreparedSample, Word


class AuditLogsTest(unittest.TestCase):
    def test_copied_artifact_names_and_role_losses_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"swap_roles_after_pass": True, "num_train_samples": 57}
            (root / "config.json").write_text(json.dumps(config))
            rows = [{"step": 1, "epoch": 0, "loss/total": 1},
                    {"step": 58, "epoch": 1, "loss/total": 4}]
            (root / "metrics (1).jsonl").write_text("\n".join(map(json.dumps, rows)))
            (root / "free_running_metrics (1).jsonl").write_text(json.dumps({
                "step": 58, "val/generation_cer": .8, "samples": []}))
            result = summarize_run(load_run(root))
            self.assertEqual(result["role_loss_means"]["left-agent"]["loss/total"], 1)
            self.assertEqual(result["role_loss_means"]["right-agent"]["loss/total"], 4)
            self.assertEqual(result["checkpoint_reload"], "not_recorded")
            self.assertFalse(result["same_sample_teacher_forced_verified"])

    def test_compare_flags_text_mode_and_does_not_invent_exposures(self):
        a = {"config": {"vietnamese_text_mode": "diacritics"}, "metrics": [], "generation": [], "run": {}}
        b = {**a, "config": {"vietnamese_text_mode": "no_diacritics"}}
        result = compare_runs(a, b)
        self.assertIn("vietnamese_text_mode", result["config_differences"])
        self.assertEqual(summarize_run(a)["exact_per_chunk_updates"], "unavailable_without_step_identities")

    def test_ambiguous_copied_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            for i in (1, 2):
                (root / f"metrics ({i}).jsonl").write_text("{}")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                load_run(root)

    def test_sampler_replay_counts_each_crop_and_role_separately(self):
        from types import SimpleNamespace
        config = SimpleNamespace(per_device_batch_size=1, seed=42, shuffle=False,
                                 max_steps=5, gradient_accumulation_steps=1, swap_roles_after_pass=True)
        chunks = [SimpleNamespace(sample_id="same", window_start_sec=start, window_end_sec=start + 1)
                  for start in (0, 1)]
        report = replay_updates(config, chunks)
        self.assertEqual(report["status"], "conditional_replay_not_observed")
        counts = {(r["start"], r["role"]): r["updates"] for r in report["chunks"]}
        self.assertEqual(counts[(0, "left-agent")], 2)
        self.assertEqual(counts[(0, "right-agent")], 1)
        self.assertEqual(sum(counts.values()), 5)


class AuditAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from personaplex_finetuning.runtime import SentencePieceTokenizer
        root = Path(__file__).resolve().parents[2]
        cls.reference = root / "refs/personaplex-finetune"
        cls.tokenizer = SentencePieceTokenizer(root / "models/tokenizer_spm_32k_3.model")

    def sample(self, words, duration=2):
        return PreparedSample("collision", Path("conversation.wav"), Path("voice.wav"),
                              tuple(words), "test", {}, AudioInfo(24000, 2, duration), 0, duration)

    def test_same_frame_words_drop_occurrences_in_actual_reference(self):
        sample = self.sample([Word("agent", "nghiêng", .08, .3),
                              Word("agent", "nghiêng", .08, .3)])
        result = compare_alignment(sample, self.tokenizer, 25, 12.5, "diacritics", self.reference)
        self.assertEqual(result["local"]["lost_tokens"], 0)
        self.assertGreater(result["reference"]["lost_tokens"], 0)
        self.assertTrue(all(row["reference_frame"] is None for row in result["occurrences"] if row["word_index"] == 0))
        self.assertEqual(result["local"]["requested_tokens"], len(result["occurrences"]))

    def test_crop_overflow_and_zero_duration_are_distinguished(self):
        sample = self.sample([Word("agent", "nghiêng", 1.99, 1.999)])
        result = compare_alignment(sample, self.tokenizer, 25, 12.5, "diacritics", self.reference)
        self.assertGreater(result["local"]["lost_tokens"], 0)
        self.assertIsNotNone(result["local"]["overflow_word"])
        sample = self.sample([Word("agent", "nghiêng", .1, .1)])
        result = compare_alignment(sample, self.tokenizer, 25, 12.5, "diacritics", self.reference)
        self.assertEqual(result["reference"]["placed_tokens"], 0)
        self.assertGreater(result["reference"]["filtered_tokens"], 0)

    def test_separated_words_and_text_normalization_match(self):
        sample = self.sample([Word("agent", "xin", .08, .4), Word("user", "bỏ", .4, .8),
                              Word("agent", "chào", 1, 1.5)])
        result = compare_alignment(sample, self.tokenizer, 25, 12.5, "no_diacritics", self.reference)
        self.assertEqual(result["local"]["lost_tokens"], 0)
        self.assertEqual(result["reference"]["lost_tokens"], 0)
        self.assertEqual(result["frame_mismatches"], 0)
        self.assertEqual(result["occurrences"][-1]["normalized_word"], "chao")


class AuditNativeTest(unittest.TestCase):
    def test_user_encoding_failure_preserves_tokens_and_repeat_evidence(self):
        from types import SimpleNamespace
        from audit_personaplex import audit_user_encoding
        train = tuple((1, 2) for _ in range(8))
        mono = tuple((1, 3) for _ in range(8))
        codec = SimpleNamespace(device="cpu", _cache_dir=None,
            encode_conversation=lambda *a: mono,
            encode_conversation_stereo=lambda *a: (train, train))
        sample = SimpleNamespace(conversation_wav=Path("test.wav"), agent_channel=0, user_channel=1,
                                 window_start_sec=0, window_end_sec=1)
        example = SimpleNamespace(prompt_frames=1, input_codes=((0, 0, 0),) * 9 +
                                  tuple((0,) + row for row in train))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, "8/16.*user_encoding.json"):
                audit_user_encoding(sample, SimpleNamespace(codec=codec), example, Path(directory))
            report = json.loads((Path(directory) / "user_encoding.json").read_text())
            self.assertEqual(report["training_vs_inference"]["first_mismatch"],
                             {"codebook": 0, "frame": 1, "training": 2, "inference": 3})
            self.assertTrue(report["inference_repeat"]["equal"])
            self.assertTrue(report["training_repeat"]["equal"])
            self.assertEqual(report["training_user_tokens"], [list(row) for row in train])

    def test_tensor_comparison_ignores_invalid_nan_and_locates_first_mismatch(self):
        import torch
        from audit_native import tensor_comparison
        a = torch.tensor([[[[float("nan"), float("nan")], [1., 2.]]]])
        b = torch.tensor([[[[3., 5.], [1., 2.]]]])
        mask = torch.tensor([[[False, True]]])
        self.assertTrue(tensor_comparison(a, b, mask)["passed"])
        b[0, 0, 1, 0] = 9
        result = tensor_comparison(a, b, mask)
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_failure_b_stream_frame"], [0, 0, 1])

    def test_matched_user_conditioning_preserves_agent_prompt_masks_and_original(self):
        from audit_personaplex import match_user_conditioning
        from personaplex_finetuning.sequence import TrainingExample
        codes = tuple((3, 1, 2) for _ in range(17))
        masks = tuple((False, True, True) for _ in range(17))
        example = TrainingExample(codes, codes, masks, tuple(str(k) for k in range(17)), 1, 2)
        mono = tuple((4, 5) for _ in range(8))
        matched = match_user_conditioning(example, mono)
        self.assertEqual(matched.input_codes[:9], example.input_codes[:9])
        self.assertEqual(matched.labels[:9], example.labels[:9])
        self.assertEqual(matched.loss_mask, example.loss_mask)
        self.assertEqual(matched.input_codes[9:], ((3, 4, 5),) * 8)
        self.assertEqual(matched.labels[9:], matched.input_codes[9:])
        self.assertEqual(example.input_codes, codes)
        for bad in (mono[:7], tuple((4,) for _ in range(8))):
            with self.assertRaisesRegex(ValueError, "shape"):
                match_user_conditioning(example, bad)

    def test_matched_user_mode_records_original_mismatch_and_requires_stable_mono(self):
        from types import SimpleNamespace
        from audit_personaplex import audit_user_encoding
        train = ((1, 2),) * 8
        mono = ((1, 3),) * 8
        sample = SimpleNamespace(conversation_wav=Path("test.wav"), agent_channel=0, user_channel=1,
                                 window_start_sec=0, window_end_sec=1)
        example = SimpleNamespace(prompt_frames=1, input_codes=((0, 0, 0),) * 9 +
                                  tuple((0,) + row for row in train))
        for stable in (True, False):
            values = iter([mono, mono if stable else train])
            codec = SimpleNamespace(device="cpu", _cache_dir=None,
                encode_conversation=lambda *a: next(values),
                encode_conversation_stereo=lambda *a: (train, train))
            with tempfile.TemporaryDirectory() as directory:
                if stable:
                    self.assertEqual(audit_user_encoding(sample, SimpleNamespace(codec=codec), example,
                        Path(directory), match_inference=True), mono)
                else:
                    with self.assertRaises(AssertionError):
                        audit_user_encoding(sample, SimpleNamespace(codec=codec), example,
                            Path(directory), match_inference=True)
                report = json.loads((Path(directory) / "user_encoding.json").read_text())
                self.assertFalse(report["training_vs_inference"]["equal"])
                self.assertEqual(report["continued_with_matched_user_conditioning"], stable)

    def test_missing_adapter_is_never_base_fallback(self):
        from personaplex_finetuning.inference import resolve_adapter_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                resolve_adapter_checkpoint(Path(directory) / "missing.safetensors")

    def test_mini_native_training_and_fresh_process_reload(self):
        import os
        import subprocess
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(project / "scripts/audit_personaplex.py"),
                "mini", "--device", "cpu", "--output-dir", directory], cwd=project,
                capture_output=True, text=True, env=dict(os.environ, NO_TORCH_COMPILE="1"))
            self.assertEqual(result.returncode, 0, result.stderr + "\n" + "\n".join(
                p.read_text() for p in Path(directory).glob("*.log")))
            report = json.loads((Path(directory) / "report.json").read_text())
            self.assertTrue(report["frozen_unchanged"])
            for name in ("text", "audio"):
                self.assertTrue(report["zero_init"][name]["passed"])
                self.assertTrue(report["reference_lm"][name]["passed"])
                self.assertTrue(report["fresh_process_reload"][name]["passed"])
            self.assertTrue(report["fresh_process_reload"]["greedy_tokens_equal"])
            self.assertGreater(report["fresh_process_reload"]["greedy_frames"], 0)
            self.assertGreater(report["adapter_effect"]["audio"]["max_abs"], 0)
            self.assertTrue(any(value and value > 0 for value in report["gradients"].values()))
            # A diagnostic must expose, not hide, a common native discrepancy.
            self.assertFalse(report["ring_boundary"]["matches"])
            self.assertFalse(report["streaming_base"]["passed"])
            self.assertTrue(report["streaming_base"]["text"]["passed"])

    def test_generation_observer_restores_runtime_after_failure(self):
        from audit_native import capture_generation
        from moshi.models import lm
        original = lm.LMGen
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "expected"):
                with capture_generation(directory):
                    self.assertIsNot(lm.LMGen, original)
                    raise RuntimeError("expected")
            self.assertIs(lm.LMGen, original)
            self.assertTrue((Path(directory) / "tokens.json").is_file())

    def test_gt_agent_audio_uses_native_forcing_and_keeps_text_free(self):
        import torch
        from audit_native import capture_generation
        from moshi.models import loaders, lm
        kwargs = loaders._lm_kwargs.copy()
        kwargs.update(dim=32, text_card=128, card=64, num_heads=2, num_layers=1, hidden_scale=2,
                      depformer_dim=16, depformer_dim_feedforward=32, depformer_num_heads=2,
                      depformer_num_layers=1, dep_q=16)
        model = loaders.LMModel(device="cpu", dtype=torch.float32, **kwargs).eval()
        gt = [[10 + k] * 6 for k in range(8)]
        original = lm.LMGen
        with tempfile.TemporaryDirectory() as directory:
            with capture_generation(directory, forced_agent_audio=gt):
                generator = lm.LMGen(model, device="cpu", use_sampling=False, temp=0, temp_text=0)
                with generator.streaming(1):
                    for _ in range(generator.max_delay + 2):
                        generator.step(input_tokens=torch.ones(1, 8, 1, dtype=torch.long))
                    generator.in_dialogue = True
                    for _ in range(6):
                        generator.step(input_tokens=torch.ones(1, 8, 1, dtype=torch.long))
                    with self.assertRaisesRegex(ValueError, "shorter"):
                        generator.step(input_tokens=torch.ones(1, 8, 1, dtype=torch.long))
            self.assertIs(lm.LMGen, original)
            report = json.loads((Path(directory) / "tokens.json").read_text())
            prompt = [s for s in report["steps"] if not s["dialogue"]]
            self.assertTrue(prompt)
            self.assertFalse(any(any(s["provided"][1:9]) for s in prompt[-1:]))
            dialogue = [s for s in report["steps"] if s["dialogue"]]
            self.assertEqual(len(dialogue), 6)
            self.assertTrue(all(not s["provided"][0] for s in dialogue))
            self.assertTrue(all(all(s["provided"][1:9]) for s in dialogue[generator.max_delay:]))
            self.assertTrue(all(s["target"][1:9] == list(range(10, 18))
                                for s in dialogue[generator.max_delay:]))
            self.assertTrue(report["diagnostic_forced_agent_audio"])

    def test_gt_agent_audio_requires_eight_equal_nonempty_streams(self):
        from audit_native import capture_generation
        for gt in ([], [[] for _ in range(8)], [[1]] * 7, [[1]] * 7 + [[1, 2]]):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "eight equal nonempty"):
                    with capture_generation(directory, forced_agent_audio=gt):
                        pass

    def test_full_gt_observer_measures_prediction_before_text_is_forced(self):
        import torch
        from unittest.mock import patch
        from audit_native import capture_generation
        from moshi.models import loaders, lm
        kwargs = loaders._lm_kwargs.copy()
        kwargs.update(dim=32, text_card=128, card=64, num_heads=2, num_layers=1, hidden_scale=2,
                      depformer_dim=16, depformer_dim_feedforward=32, depformer_num_heads=2,
                      depformer_num_layers=1, dep_q=16)
        model = loaders.LMModel(device="cpu", dtype=torch.float32, **kwargs).eval()
        native = model.forward_codes
        def biased(codes):
            hidden, logits = native(codes)
            logits = torch.full_like(logits, -10.)
            logits[..., 3] = 10.  # Predict PAD despite the forced GT text=7.
            return hidden, logits
        with tempfile.TemporaryDirectory() as directory, patch.object(model, "forward_codes", side_effect=biased):
            with capture_generation(directory, forced_text=[7] * 6, forced_agent_audio=[[10] * 6 for _ in range(8)]):
                generator = lm.LMGen(model, device="cpu", use_sampling=False, temp=0, temp_text=0)
                with generator.streaming(1):
                    for _ in range(generator.max_delay + 2):
                        generator.step(input_tokens=torch.ones(1, 8, 1, dtype=torch.long))
                    generator.in_dialogue = True
                    for _ in range(6):
                        generator.step(input_tokens=torch.ones(1, 8, 1, dtype=torch.long))
            report = json.loads((Path(directory) / "text_logits.json").read_text())
            rows = report["frames"]
            self.assertTrue(all(r["prediction"] == 3 for r in rows))
            self.assertTrue(all(r["gt_target"] == 7 for r in rows))
            self.assertEqual(report["summary"]["accuracy_on_nonpadding_gt"], 0.)
            self.assertEqual(report["summary"]["pad_prediction_on_nonpadding_gt"], 1.)
            tokens = json.loads((Path(directory) / "tokens.json").read_text())
            self.assertTrue(all(r["tokens"][0] == 7 for r in tokens["returned"][generator.max_delay:]))


if __name__ == "__main__":
    unittest.main()
