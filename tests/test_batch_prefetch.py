import threading
import time
import unittest

from personaplex_finetuning.train import BackgroundBatchPrefetcher


class BackgroundBatchPrefetcherTest(unittest.TestCase):
    def test_preserves_order_and_stops(self):
        self.assertEqual(list(BackgroundBatchPrefetcher(iter(range(10)), depth=2)), list(range(10)))

    def test_builds_ahead_in_another_thread(self):
        threads = []

        def batches():
            for index in range(3):
                threads.append(threading.current_thread())
                yield index

        prefetcher = BackgroundBatchPrefetcher(batches(), depth=2)
        time.sleep(0.2)  # Producer fills the queue before the consumer asks.
        self.assertGreaterEqual(len(threads), 2)
        self.assertEqual(list(prefetcher), [0, 1, 2])
        self.assertTrue(all(thread is not threading.main_thread() for thread in threads))

    def test_generator_error_reraises_on_training_thread(self):
        def batches():
            yield 1
            raise RuntimeError("cache identity mismatch")

        prefetcher = BackgroundBatchPrefetcher(batches(), depth=1)
        self.assertEqual(next(prefetcher), 1)
        with self.assertRaisesRegex(RuntimeError, "cache identity mismatch"):
            next(prefetcher)

    def test_close_releases_a_blocked_producer(self):
        def endless():
            while True:
                yield 0

        prefetcher = BackgroundBatchPrefetcher(endless(), depth=1)
        next(prefetcher)
        prefetcher.close()
        prefetcher._thread.join(timeout=2)
        self.assertFalse(prefetcher._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
