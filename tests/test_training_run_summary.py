import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_training_runs import summarize_runs, summarize_training_run


class TrainingRunSummaryTest(unittest.TestCase):
    def test_summary_uses_best_free_running_cer_and_peak_from_all_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "train_001"
            (run_dir / "ranks").mkdir(parents=True)
            (run_dir / "config.json").write_text(json.dumps({
                "num_processes": 2,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "global_batch_size": 16,
                "lora_rank": 128,
                "learning_rate": 2e-6,
                "max_steps": 2000,
            }), encoding="utf-8")
            (run_dir / "run.json").write_text(json.dumps({
                "best_inference_checkpoint": "/runs/train_001/checkpoints/best_inference/lora.safetensors",
                "starting_generation_cer": 0.3,
                "inference_checkpoint_status": "validated_generation_improves_base",
            }), encoding="utf-8")
            (run_dir / "metrics.jsonl").write_text(
                '{"step":1,"samples_per_second":4.0}\n'
                '{"step":2,"samples_per_second":6.0}\n', encoding="utf-8",
            )
            (run_dir / "free_running_metrics.jsonl").write_text(
                '{"step":0,"val/generation_baseline":true,"val/generation_cer":0.3}\n'
                '{"step":500,"val/generation_cer":0.4,"val/generation_wer":0.8,"val/generation_empty_samples":0}\n'
                '{"step":1000,"val/generation_cer":0.2,"val/generation_wer":0.5,"val/generation_empty_samples":0}\n',
                encoding="utf-8",
            )
            (run_dir / "ranks" / "rank_000.json").write_text(
                json.dumps({"peak_gpu_bytes": 1024**3}), encoding="utf-8",
            )
            (run_dir / "ranks" / "rank_001.json").write_text(
                json.dumps({"peak_gpu_bytes": 2 * 1024**3}), encoding="utf-8",
            )

            summary = summarize_training_run(run_dir)

        self.assertEqual(summary["batch_per_device"], 2)
        self.assertEqual(summary["global_batch_size"], 16)
        self.assertEqual(summary["best_generation_cer"], 0.2)
        self.assertEqual(summary["best_generation_step"], 1000)
        self.assertEqual(summary["median_samples_per_second_last_20"], 5.0)
        self.assertEqual(summary["peak_gpu_gib"], 2.0)

    def test_runs_without_generation_metrics_sort_after_scored_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, cer in (("no_eval", None), ("scored", 0.25)):
                run_dir = root / name
                run_dir.mkdir()
                (run_dir / "config.json").write_text("{}", encoding="utf-8")
                if cer is not None:
                    (run_dir / "run.json").write_text(json.dumps({
                        "starting_generation_cer": 0.3,
                        "inference_checkpoint_status": "validated_generation_improves_base",
                    }), encoding="utf-8")
                    (run_dir / "free_running_metrics.jsonl").write_text(
                        json.dumps({"step": 3, "val/generation_cer": cer, "val/generation_empty_samples": 0}) + "\n",
                        encoding="utf-8",
                    )

            runs = summarize_runs(root)

        self.assertEqual([row["run"] for row in runs], ["scored", "no_eval"])

    def test_summary_excludes_base_and_non_improving_generation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "train_001"
            run_dir.mkdir()
            (run_dir / "config.json").write_text("{}", encoding="utf-8")
            (run_dir / "run.json").write_text(json.dumps({
                "starting_generation_cer": 0.3,
                "inference_checkpoint_status": "validated_generation_improves_base",
            }), encoding="utf-8")
            (run_dir / "free_running_metrics.jsonl").write_text(
                '{"step":0,"val/generation_baseline":true,"val/generation_cer":0.3}\n'
                '{"step":100,"val/generation_cer":0.4,"val/generation_empty_samples":0}\n'
                '{"step":200,"val/generation_cer":0.2,"val/generation_empty_samples":1}\n'
                '{"step":300,"val/generation_cer":0.25,"val/generation_empty_samples":0}\n',
                encoding="utf-8",
            )

            summary = summarize_training_run(run_dir)

        self.assertEqual(summary["starting_generation_cer"], 0.3)
        self.assertEqual(summary["best_generation_cer"], 0.25)
        self.assertEqual(summary["best_generation_step"], 300)

    def test_summary_does_not_score_runs_without_validated_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "train_without_baseline"
            run_dir.mkdir()
            (run_dir / "config.json").write_text("{}", encoding="utf-8")
            (run_dir / "free_running_metrics.jsonl").write_text(
                '{"step":100,"val/generation_cer":0.01,"val/generation_empty_samples":0}\n',
                encoding="utf-8",
            )
            summary = summarize_training_run(run_dir)

        self.assertIsNone(summary["best_generation_cer"])
        self.assertIsNone(summary["best_generation_step"])


if __name__ == "__main__":
    unittest.main()
