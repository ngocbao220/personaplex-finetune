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


if __name__ == "__main__":
    unittest.main()
