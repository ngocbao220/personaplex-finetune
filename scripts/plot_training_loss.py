"""Live terminal loss curves from a TensorBoard event file.

Usage:
    python scripts/plot_training_loss.py /path/to/run
    python scripts/plot_training_loss.py /path/to/run --once
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import plotext
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SERIES = (
    ("audio", "loss/audio_total"),
    ("text", "loss/text"),
    ("cb0", "loss/audio_cb0"),
)


def smooth(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    result = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def render(accumulator: EventAccumulator, smoothing: int) -> tuple[int, str]:
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing = [tag for _, tag in SERIES if tag not in available]
    if missing:
        raise ValueError(f"TensorBoard event file lacks tags: {', '.join(missing)}")
    columns, rows = shutil.get_terminal_size((100, 30))
    figure = plotext.figure
    figure.clear()
    figure.plot_size(max(50, columns - 2), max(15, rows - 4))
    figure.title(f"Training loss (moving average: {smoothing} steps)")
    figure.label("optimizer step", axis="x")
    figure.label("loss", axis="y")
    latest = []
    last_step = 0
    for name, tag in SERIES:
        events = [event for event in accumulator.Scalars(tag) if math.isfinite(event.value)]
        if not events:
            continue
        steps = [event.step for event in events]
        values = [event.value for event in events]
        figure.draw(figure.signal(steps, smooth(values, smoothing)).lines().label(name))
        latest.append(f"{name}={values[-1]:.3f}")
        last_step = max(last_step, steps[-1])
    if not latest:
        raise ValueError("TensorBoard event file has no finite loss points yet")
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    figure.show(flush=True)
    status = f"step {last_step}  |  " + "  ".join(latest)
    print(status, flush=True)
    return last_step, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run directory containing tensorboard/events.out.tfevents.*")
    parser.add_argument("--interval", type=float, default=5.0, help="refresh interval in seconds")
    parser.add_argument("--smooth", type=int, default=15, help="moving-average window; 1 shows raw loss")
    parser.add_argument("--once", action="store_true", help="draw once and exit")
    args = parser.parse_args()
    if args.interval <= 0 or args.smooth < 1:
        parser.error("--interval must be positive and --smooth must be at least 1")
    event_dir = args.run_dir / "tensorboard"
    if not list(event_dir.glob("events.out.tfevents.*")):
        parser.error(f"no TensorBoard event file found in {event_dir}")
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    try:
        while True:
            render(accumulator, args.smooth)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
