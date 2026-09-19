"""CLI interface for running Full-Duplex-Bench on PersonaPlex."""

from __future__ import annotations

import argparse
from pathlib import Path
from rich.console import Console

from .contract import extract_samples_from_prepared_dir, BenchmarkSample
from .runner import MockBenchmarkRunner, PersonaPlexStreamingRunner, SampleEvaluationResult
from .reporter import BenchmarkReporter, aggregate_results, AggregatedTaskMetrics


def run_benchmark_cli(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Full-Duplex-Bench on PersonaPlex models.")
    parser.add_argument("--model_root", type=Path, default=Path("models"), help="Path to base PersonaPlex checkpoint directory.")
    parser.add_argument("--adapter", type=Path, default=None, help="Optional path to fine-tuned LoRA adapter.")
    parser.add_argument("--model_name", type=str, default="PersonaPlex Base", help="Display name of the model.")
    parser.add_argument("--fdb_dir", type=Path, default=Path("benchmarks/datasets/fdb_v1/v1.0/extracted"), help="Path to extracted official Full-Duplex-Bench v1.0 directory.")
    parser.add_argument("--prepared_dir", type=Path, default=None, help="Optional path to prepared conversation samples.")
    parser.add_argument("--voice_prompt", type=Path, default=Path("prepared/samples/conv_0001/voice_prompt.wav"), help="Default voice prompt audio.")
    parser.add_argument("--gt_dist", type=Path, default=Path("benchmarks/datasets/fdb_v1/icc_gt_distribution.json"), help="Path to ICC ground-truth distribution.")
    parser.add_argument("--output_dir", type=Path, default=Path("benchmarks/results"), help="Directory to save benchmark reports.")
    parser.add_argument("--max_samples_per_task", type=int, default=10, help="Maximum samples to evaluate per task.")
    parser.add_argument("--mock", action="store_true", help="Run with mock runner for dry-run verification.")
    parser.add_argument("--compare_demo", action="store_true", help="Run side-by-side demo comparison (Base vs LoRA).")
    parser.add_argument("--device", type=str, default="auto", help="Inference device ('auto', 'cuda', 'mps', 'cpu').")

    parsed = parser.parse_args(args)
    console = Console()
    reporter = BenchmarkReporter(console=console)

    console.print("[bold cyan]================================================================[/bold cyan]")
    console.print("[bold cyan]        Full-Duplex-Bench Evaluation: fdp-benchmark-result      [/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]")

    # 1. Collect benchmark samples
    all_samples: list[BenchmarkSample] = []
    from .contract import load_fdb_v1_dataset

    # Priority 1: Official Full-Duplex-Bench dataset
    if parsed.fdb_dir and parsed.fdb_dir.is_dir():
        console.print(f"[dim]Loading official Full-Duplex-Bench v1.0 dataset from: {parsed.fdb_dir}[/dim]")
        fdb_samples = load_fdb_v1_dataset(
            extracted_root=parsed.fdb_dir,
            default_voice_prompt=parsed.voice_prompt,
            gt_distribution_path=parsed.gt_dist,
        )
        all_samples.extend(fdb_samples)

    # Priority 2: In-domain prepared directory
    elif parsed.prepared_dir and parsed.prepared_dir.is_dir():
        console.print(f"[dim]Collecting samples from prepared directory: {parsed.prepared_dir}[/dim]")
        for s_dir in sorted(parsed.prepared_dir.iterdir()):
            if s_dir.is_dir() and not s_dir.name.startswith("."):
                extracted = extract_samples_from_prepared_dir(s_dir)
                all_samples.extend(extracted)

    if not all_samples:
        console.print(f"[yellow]Warning: No dataset found. Using synthetic mock samples.[/yellow]")
        all_samples = [
            BenchmarkSample("synth_pause_01", "pause_synthetic", Path("none.wav"), 1, Path("none.wav"), "sys", 0.0, 10.0, pause_start_sec=3.0, pause_end_sec=5.0),
            BenchmarkSample("synth_turn_01", "turn_taking", Path("none.wav"), 1, Path("none.wav"), "sys", 0.0, 10.0, user_turn_end_sec=4.0),
            BenchmarkSample("synth_bc_01", "backchannel", Path("none.wav"), 1, Path("none.wav"), "sys", 0.0, 15.0, gt_distribution=[0.1]*10),
            BenchmarkSample("synth_intr_01", "interruption", Path("none.wav"), 1, Path("none.wav"), "sys", 0.0, 12.0, interruption_start_sec=4.0, interruption_end_sec=6.0, interruption_query="Wait stop"),
        ]

    # Filter by task and cap max samples
    tasks = ["pause_synthetic", "pause_natural", "backchannel", "turn_taking", "interruption"]
    capped_samples: list[BenchmarkSample] = []
    for t in tasks:
        t_samples = [s for s in all_samples if s.task == t][:parsed.max_samples_per_task]
        capped_samples.extend(t_samples)

    console.print(f"[green]Total benchmark evaluation samples gathered: {len(capped_samples)}[/green]")
    for t in tasks:
        c = len([s for s in capped_samples if s.task == t])
        if c > 0:
            console.print(f"  • Task [bold]{t}[/bold]: {c} samples")

    aggregated_rows: list[AggregatedTaskMetrics] = []

    # Demo comparison mode (Base vs LoRA)
    if parsed.compare_demo:
        console.print("\n[bold yellow]Running side-by-side comparison demo (Base vs LoRA)...[/bold yellow]")
        # 1. Base model
        runner_base = MockBenchmarkRunner(behavior_profile="talkative")
        results_base = [runner_base.evaluate_sample(s) for s in capped_samples]
        row_base = aggregate_results("PersonaPlex Base", results_base)
        aggregated_rows.append(row_base)

        # 2. LoRA Fine-tuned model
        runner_lora = MockBenchmarkRunner(behavior_profile="ideal")
        results_lora = [runner_lora.evaluate_sample(s) for s in capped_samples]
        row_lora = aggregate_results("PersonaPlex LoRA", results_lora)
        aggregated_rows.append(row_lora)

    elif parsed.mock:
        console.print("\n[bold yellow]Running in Mock Mode (Fast Dry-Run)...[/bold yellow]")
        runner = MockBenchmarkRunner(behavior_profile="ideal")
        results = [runner.evaluate_sample(s) for s in capped_samples]
        row = aggregate_results(parsed.model_name, results)
        aggregated_rows.append(row)

    else:
        console.print(f"\n[bold green]Initializing PersonaPlex Streaming Runner on {parsed.device}...[/bold green]")
        runner = PersonaPlexStreamingRunner(
            model_root=parsed.model_root,
            adapter_path=parsed.adapter,
            device=parsed.device,
        )
        results = []
        for i, s in enumerate(capped_samples):
            console.print(f"[{i+1}/{len(capped_samples)}] Evaluating {s.task}: {s.sample_id}...")
            res = runner.evaluate_sample(s)
            results.append(res)
        row = aggregate_results(parsed.model_name, results)
        aggregated_rows.append(row)

    # 4. Render Table and Export
    reporter.print_rich_table(aggregated_rows, title="fdp-benchmark-result (vi/en)")
    reporter.export_reports(aggregated_rows, output_dir=parsed.output_dir, title="fdp-benchmark-result (vi/en)")
    console.print(f"[bold green]✓ Benchmark reports saved to: {parsed.output_dir}/[/bold green]")

    return 0
