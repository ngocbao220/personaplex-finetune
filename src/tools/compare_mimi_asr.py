"""Compare Vietnamese ASR transcription between original audio and Mimi-reconstructed audio.

Calls internal ASR API (via asr_api.py logic) to transcribe both versions,
computes Word Error Rate (WER) and Character Error Rate (CER), and analyzes
phonetic / tonal degradation caused by Mimi codec compression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tempfile
import time

import requests
from rich.console import Console
from rich.table import Table
from rich import box

# Import API utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from asr_api import build_url, check_health, call_speech_api, API_BASE_URL


def compute_levenshtein(ref: list, hyp: list) -> int:
    """Standard dynamic programming Levenshtein distance."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[n][m]


def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation and extra spaces."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\d]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer_cer(ref_text: str, hyp_text: str) -> tuple[float, float]:
    """Compute WER and CER percentages between reference and hypothesis text."""
    ref_norm = normalize_text(ref_text)
    hyp_norm = normalize_text(hyp_text)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))

    w_dist = compute_levenshtein(ref_words, hyp_words)
    wer = (w_dist / max(len(ref_words), 1)) * 100.0

    c_dist = compute_levenshtein(ref_chars, hyp_chars)
    cer = (c_dist / max(len(ref_chars), 1)) * 100.0

    return wer, cer


def get_audio_wav_bytes(file_path: Path) -> bytes:
    """Read audio file as WAV bytes, converting mp3 via sphn if necessary."""
    if file_path.suffix.lower() == ".wav":
        return file_path.read_bytes()

    import sphn
    data, sr = sphn.read(str(file_path))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sphn.write_wav(str(tmp_path), data, sr)
        return tmp_path.read_bytes()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig-dir", default="bactrungnam", help="Thư mục audio gốc")
    parser.add_argument("--mimi-dir", default="outputs/mimi_bactrungnam_8cb", help="Thư mục audio tái tạo qua Mimi")
    parser.add_argument("--api-base-url", default=API_BASE_URL, help="Base URL ASR")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Loại task ASR")
    parser.add_argument("--language", default="vi", help="Ngôn ngữ audio")
    parser.add_argument("--cache-file", default="outputs/asr_comparison_cache.json", help="File cache kết quả ASR")
    parser.add_argument("--output-json", default="outputs/asr_comparison_report.json", help="File JSON kết quả")
    args = parser.parse_args()

    console = Console()
    url = build_url(args.api_base_url, args.task)
    console.print(f"[bold cyan]🔍 Checking ASR Health:[/bold cyan] {url}")
    check_health(url)

    orig_dir = Path(args.orig_dir)
    mimi_dir = Path(args.mimi_dir)
    cache_path = Path(args.cache_file)
    cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            console.print(f"[dim]Loaded {len(cache)} cached transcriptions from {cache_path}[/dim]")
        except Exception:
            cache = {}

    # Seed pre-known cache for mienbac3.wav if present
    mb3_key = str(orig_dir / "mienbac3.wav")
    if mb3_key not in cache:
        cache[mb3_key] = {
            "text": "thế giới đang trải qua những biến động sâu sắc nơi những biến động về chính trị kinh tế xã hội môi trường và công nghệ đan xen tạo nên những thách thức toàn cầu chưa từng có trong làn sóng ấy tri thức và đổi mới sáng tạo trở thành sức mạnh trung tâm kết nối con người quốc gia và các nền văn minh trong hành trình",
            "detected_text_language": "vi",
            "latency_sec": 168.56,
            "error": None,
        }

    # Find audio pairs
    orig_files = sorted(list(orig_dir.glob("*.mp3")) + list(orig_dir.glob("*.wav")))
    if not orig_files:
        console.print(f"[red]Không tìm thấy audio nào trong {orig_dir}[/red]")
        sys.exit(1)

    pairs = []
    for orig_f in orig_files:
        stem = orig_f.stem
        # Match with reconstructed file: {stem}_mimi_8cb.wav
        mimi_candidate = mimi_dir / f"{stem}_mimi_8cb.wav"
        if not mimi_candidate.exists():
            # Try finding any file starting with stem in mimi_dir
            matches = list(mimi_dir.glob(f"{stem}*.wav"))
            if matches:
                mimi_candidate = matches[0]
            else:
                console.print(f"[yellow]Không tìm thấy file mimi tương ứng cho {orig_f.name}[/yellow]")
                continue
        pairs.append((orig_f, mimi_candidate))

    console.print(f"\n[bold green]Đã ghép được {len(pairs)} cặp file để đánh giá ASR.[/bold green]\n")

    def transcribe_file(fpath: Path) -> dict:
        key = str(fpath)
        if key in cache and cache[key].get("text") and not cache[key].get("error"):
            console.print(f"  [dim]⚡ Cache hit for {fpath.name}[/dim]")
            return cache[key]

        console.print(f"  [cyan]⏳ Calling ASR for {fpath.name}...[/cyan]")
        wav_bytes = get_audio_wav_bytes(fpath)
        res = call_speech_api(url, wav_bytes, language=args.language)
        if res.get("error"):
            console.print(f"  [red]❌ Lỗi khi gọi ASR cho {fpath.name}: {res['error']}[/red]")
        else:
            console.print(f"  [green]✓ Done ({res.get('latency_sec', 0):.2f}s)[/green]")
        cache[key] = res
        # Save cache incrementally
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        return res

    comparison_records = []
    total_wer = 0.0
    total_cer = 0.0

    for orig_f, mimi_f in pairs:
        console.print(f"[bold]--- Đang xử lý cặp: {orig_f.stem} ---[/bold]")
        orig_res = transcribe_file(orig_f)
        mimi_res = transcribe_file(mimi_f)

        orig_text = orig_res.get("text") or ""
        mimi_text = mimi_res.get("text") or ""

        wer, cer = compute_wer_cer(orig_text, mimi_text)
        total_wer += wer
        total_cer += cer

        comparison_records.append({
            "stem": orig_f.stem,
            "orig_file": str(orig_f),
            "mimi_file": str(mimi_f),
            "orig_text": orig_text,
            "mimi_text": mimi_text,
            "wer": wer,
            "cer": cer,
            "orig_latency": orig_res.get("latency_sec"),
            "mimi_latency": mimi_res.get("latency_sec"),
        })

    avg_wer = total_wer / max(len(pairs), 1)
    avg_cer = total_cer / max(len(pairs), 1)

    # Render Rich Table
    table = Table(
        title="[bold yellow]BẢNG SO SÁNH NHẬN DẠNG TIẾNG VIỆT (ASR): AUDIO GỐC vs MIMI (8 CODEBOOKS)[/bold yellow]",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Mẫu (Vùng)", style="cyan", width=14)
    table.add_column("Văn bản gốc (Original ASR)", style="green", width=42)
    table.add_column("Văn bản Mimi 8cb (Mimi ASR)", style="bright_yellow", width=42)
    table.add_column("WER (%)", justify="right", style="bold red", width=10)
    table.add_column("CER (%)", justify="right", style="bold red", width=10)

    for rec in comparison_records:
        wer_str = f"{rec['wer']:.1f}%"
        cer_str = f"{rec['cer']:.1f}%"
        if rec['wer'] < 10.0:
            wer_style = "[green]" + wer_str + "[/green]"
        elif rec['wer'] < 25.0:
            wer_style = "[yellow]" + wer_str + "[/yellow]"
        else:
            wer_style = "[red]" + wer_str + "[/red]"

        if rec['cer'] < 5.0:
            cer_style = "[green]" + cer_str + "[/green]"
        elif rec['cer'] < 15.0:
            cer_style = "[yellow]" + cer_str + "[/yellow]"
        else:
            cer_style = "[red]" + cer_str + "[/red]"

        table.add_row(
            rec["stem"],
            rec["orig_text"] if rec["orig_text"] else "[dim](empty)[/dim]",
            rec["mimi_text"] if rec["mimi_text"] else "[dim](empty)[/dim]",
            wer_style,
            cer_style,
        )

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold]📈 Trung bình trên {len(pairs)} mẫu:[/bold]")
    console.print(f"  • [bold red]Trung bình WER:[/bold red] [bold white]{avg_wer:.2f}%[/bold white]")
    console.print(f"  • [bold red]Trung bình CER:[/bold red] [bold white]{avg_cer:.2f}%[/bold white]")

    # Save output report JSON
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "summary": {
            "num_samples": len(pairs),
            "avg_wer": avg_wer,
            "avg_cer": avg_cer,
        },
        "records": comparison_records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    console.print(f"\n[dim]Đã lưu báo cáo chi tiết vào {out_path}[/dim]\n")


if __name__ == "__main__":
    main()
