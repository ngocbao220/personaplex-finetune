"""Unit tests for benchmark runner and reporter."""

import unittest
from pathlib import Path
from personaplex_benchmark.contract import BenchmarkSample
from personaplex_benchmark.runner import MockBenchmarkRunner, cluster_tokens_into_speech_events
from personaplex_benchmark.reporter import aggregate_results, BenchmarkReporter, AggregatedTaskMetrics


class TestBenchmarkRunnerAndReporter(unittest.TestCase):
    def test_cluster_tokens(self):
        # Tokens within 0.2s should form 1 event
        tokens = [
            (1.00, "hello"),
            (1.08, "world"),
            (1.16, "how"),
            # Gap of 0.8s
            (2.00, "are"),
            (2.08, "you"),
        ]
        events = cluster_tokens_into_speech_events(tokens, max_gap_sec=0.4)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].text, "hello world how")
        self.assertEqual(events[1].text, "are you")

    def test_mock_runner_and_aggregate(self):
        sample_pause = BenchmarkSample("s1", "pause_natural", Path("x.wav"), 1, Path("v.wav"), "sys", 0.0, 10.0, pause_start_sec=2.0, pause_end_sec=4.0)
        sample_turn = BenchmarkSample("s2", "turn_taking", Path("x.wav"), 1, Path("v.wav"), "sys", 0.0, 10.0, user_turn_end_sec=3.0)

        runner = MockBenchmarkRunner(behavior_profile="ideal")
        res1 = runner.evaluate_sample(sample_pause)
        res2 = runner.evaluate_sample(sample_turn)

        self.assertEqual(res1.tor, 0.0)
        self.assertEqual(res2.tor, 1.0)
        self.assertIsNotNone(res2.latency)

        row = aggregate_results("TestModel", [res1, res2])
        self.assertEqual(row.model_name, "TestModel")
        self.assertEqual(row.pause_nat_tor, 0.0)
        self.assertEqual(row.turn_tor, 1.0)

    def test_reporter_export(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            row = AggregatedTaskMetrics(
                model_name="DemoModel",
                pause_nat_tor=0.35,
                bc_tor=0.0,
                bc_freq=0.05,
                bc_jsd=0.65,
                turn_tor=0.92,
                turn_lat=0.15,
                intr_tor=0.96,
                intr_quality=4.2,
                intr_lat=0.38,
            )
            reporter = BenchmarkReporter()
            md = reporter.generate_markdown([row])
            self.assertIn("DemoModel", md)
            self.assertIn("fdp-benchmark-result (vi/en)", md)

            reporter.export_reports([row], output_dir=tmp_path)
            self.assertTrue((tmp_path / "fdp-benchmark-result.md").is_file())
            self.assertTrue((tmp_path / "fdp-benchmark-result.json").is_file())


if __name__ == "__main__":
    unittest.main()
