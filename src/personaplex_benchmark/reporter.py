"""Table generator and reporter for fdp-benchmark-result (vi/en) using rich."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import numpy as np

from rich.console import Console
from rich.table import Table
from rich import box

from .runner import SampleEvaluationResult


@dataclass
class AggregatedTaskMetrics:
    model_name: str
    pause_synth_tor: float | None = None
    pause_nat_tor: float | None = None
    bc_tor: float | None = None
    bc_freq: float | None = None
    bc_jsd: float | None = None
    turn_tor: float | None = None
    turn_lat: float | None = None
    intr_tor: float | None = None
    intr_quality: float | None = None
    intr_lat: float | None = None


def aggregate_results(model_name: str, results: Sequence[SampleEvaluationResult]) -> AggregatedTaskMetrics:
    """Aggregates a list of sample evaluation results into a single model row."""
    pause_synth = [r.tor for r in results if r.task == "pause_synthetic"]
    pause_nat = [r.tor for r in results if r.task == "pause_natural"]
    bc = [r for r in results if r.task == "backchannel"]
    turn = [r for r in results if r.task == "turn_taking"]
    intr = [r for r in results if r.task == "interruption"]

    def _mean(vals: Sequence[float | None]) -> float | None:
        valid = [v for v in vals if v is not None]
        return round(float(np.mean(valid)), 3) if valid else None

    return AggregatedTaskMetrics(
        model_name=model_name,
        pause_synth_tor=_mean(pause_synth),
        pause_nat_tor=_mean(pause_nat),
        bc_tor=_mean([r.tor for r in bc]),
        bc_freq=_mean([r.freq for r in bc]),
        bc_jsd=_mean([r.jsd for r in bc]),
        turn_tor=_mean([r.tor for r in turn]),
        turn_lat=_mean([r.latency for r in turn]),
        intr_tor=_mean([r.tor for r in intr]),
        intr_quality=_mean([r.response_quality for r in intr]),
        intr_lat=_mean([r.latency for r in intr]),
    )


class BenchmarkReporter:
    """Renders fdp-benchmark-result (vi/en) table and saves markdown/json reports."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def print_rich_table(self, rows: Sequence[AggregatedTaskMetrics], title: str = "fdp-benchmark-result (vi/en)") -> None:
        table = Table(
            title=f"[bold cyan]{title}[/bold cyan]",
            box=box.ROUNDED,
            header_style="bold magenta",
            show_lines=True,
        )

        table.add_column("Model", style="bold yellow", justify="left")
        table.add_column("Pause Synth\nTOR (↓)", justify="center")
        table.add_column("Pause Nat\nTOR (↓)", justify="center")
        table.add_column("Backchannel\nTOR (↓)", justify="center")
        table.add_column("Backchannel\nFreq (↑)", justify="center")
        table.add_column("Backchannel\nJSD (↓)", justify="center")
        table.add_column("Turn-Taking\nTOR (↑)", justify="center")
        table.add_column("Turn-Taking\nLat (↓)", justify="center")
        table.add_column("Interruption\nTOR (↑)", justify="center")
        table.add_column("Interruption\nResp Qual(↑)", justify="center")
        table.add_column("Interruption\nLat (↓)", justify="center")

        def _fmt(val: float | None, unit: str = "") -> str:
            if val is None:
                return "-"
            return f"{val:.3f}{unit}"

        for r in rows:
            table.add_row(
                r.model_name,
                _fmt(r.pause_synth_tor),
                _fmt(r.pause_nat_tor),
                _fmt(r.bc_tor),
                _fmt(r.bc_freq),
                _fmt(r.bc_jsd),
                _fmt(r.turn_tor),
                _fmt(r.turn_lat, "s"),
                _fmt(r.intr_tor),
                _fmt(r.intr_quality),
                _fmt(r.intr_lat, "s"),
            )

        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")

    def generate_markdown(self, rows: Sequence[AggregatedTaskMetrics], title: str = "fdp-benchmark-result (vi/en)") -> str:
        def _fmt(val: float | None, unit: str = "") -> str:
            if val is None:
                return "-"
            return f"{val:.3f}{unit}"

        md_lines = [
            f"### {title}",
            "",
            "| Model | Pause Synth TOR (↓) | Pause Nat TOR (↓) | BC TOR (↓) | BC Freq (↑) | BC JSD (↓) | Turn TOR (↑) | Turn Lat (↓) | Intr TOR (↑) | Intr Qual (↑) | Intr Lat (↓) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in rows:
            md_lines.append(
                f"| **{r.model_name}** | {_fmt(r.pause_synth_tor)} | {_fmt(r.pause_nat_tor)} | {_fmt(r.bc_tor)} | {_fmt(r.bc_freq)} | {_fmt(r.bc_jsd)} | {_fmt(r.turn_tor)} | {_fmt(r.turn_lat, 's')} | {_fmt(r.intr_tor)} | {_fmt(r.intr_quality)} | {_fmt(r.intr_lat, 's')} |"
            )
        return "\n".join(md_lines) + "\n"

    def export_reports(self, rows: Sequence[AggregatedTaskMetrics], output_dir: Path, title: str = "fdp-benchmark-result (vi/en)") -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        md_text = self.generate_markdown(rows, title)
        (output_dir / "fdp-benchmark-result.md").write_text(md_text, encoding="utf-8")

        json_data = [r.__dict__ for r in rows]
        (output_dir / "fdp-benchmark-result.json").write_text(
            json.dumps(json_data, indent=2), encoding="utf-8"
        )
