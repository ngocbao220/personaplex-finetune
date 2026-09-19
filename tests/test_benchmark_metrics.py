"""Unit tests for Full-Duplex-Bench metrics."""

import unittest
from personaplex_benchmark.metrics import (
    SpeechEvent,
    compute_tor,
    compute_speaker_switch_latency,
    compute_interruption_latency,
    compute_backchannel_metrics,
    compute_response_quality,
)


class TestBenchmarkMetrics(unittest.TestCase):
    def test_tor_silence_or_short(self):
        # Silence (no events)
        self.assertEqual(compute_tor([]), 0.0)

        # Short backchannel / sound: duration = 0.5s, 1 word
        short_ev = [SpeechEvent(start_sec=1.0, end_sec=1.5, text="uh-huh")]
        self.assertEqual(compute_tor(short_ev, duration_threshold=1.0, words_threshold=3), 0.0)

        # Long utterance with >= 3 words -> Takeover (TOR = 1.0)
        long_ev = [SpeechEvent(start_sec=1.0, end_sec=3.5, text="I think we should do that")]
        self.assertEqual(compute_tor(long_ev, duration_threshold=1.0, words_threshold=3), 1.0)

    def test_latency(self):
        user_end = 2.0
        events = [
            SpeechEvent(start_sec=2.25, end_sec=4.0, text="Yes definitely"),
        ]
        lat = compute_speaker_switch_latency(events, user_turn_end_sec=user_end)
        self.assertAlmostEqual(lat, 0.25, places=3)

    def test_interruption_latency(self):
        interrupt_end = 5.0
        events = [
            SpeechEvent(start_sec=1.0, end_sec=4.0, text="Prior speech"),
            SpeechEvent(start_sec=5.4, end_sec=7.0, text="Okay got it"),
        ]
        lat = compute_interruption_latency(events, interruption_end_sec=interrupt_end)
        self.assertAlmostEqual(lat, 0.4, places=3)

    def test_backchannel_metrics(self):
        user_dur = 10.0
        events = [
            SpeechEvent(start_sec=2.0, end_sec=2.6, text="uh-huh"),
            SpeechEvent(start_sec=6.0, end_sec=6.5, text="yeah"),
        ]
        gt_dist = [0.1] * 10
        res = compute_backchannel_metrics(events, user_duration_sec=user_dur, gt_distribution=gt_dist)
        self.assertEqual(res["tor"], 0.0)
        self.assertEqual(res["bc_count"], 2)
        self.assertAlmostEqual(res["freq"], 0.2, places=2)
        self.assertGreaterEqual(res["jsd"], 0.0)
        self.assertLessEqual(res["jsd"], 1.0)

    def test_response_quality_offline(self):
        query = "What time is the flight?"
        resp = "The flight departs at three in the afternoon."
        score = compute_response_quality(query, resp)
        self.assertGreaterEqual(score, 3.0)
        self.assertLessEqual(score, 5.0)


if __name__ == "__main__":
    unittest.main()
