"""Sequential harness generation; no training is performed.

prepare_test3(args, config, sample, runtime) saves a live trained LoRA baseline.
Release ALL runtime/model/trainer references, then run_test3(ticket).
run_test4(args, config, sample, trained_adapter=...) evaluates a supplied adapter.
run_test6(..., enabled=main_harness_gate) runs only when explicitly gated.
args: output_dir, optional seed, step, worker_timeout, python.
A fresh worker does not imply that the training process itself exited.
Standalone: python scripts/validation_generation.py --worker request.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import subprocess
import sys
import weakref
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("NO_TORCH_COMPILE", "1")
from personaplex_finetuning.generation import GenerationSettings


def greedy_settings():
    # Native API calls audio temperature 'temp', not 'temp_audio'.
    return GenerationSettings(use_sampling=False, temp=0, temp_text=0)


@contextmanager
def verify_argmax():
    """Observe native LMGen sampling calls; require actual argmax for every call."""
    import importlib
    import torch
    lm = importlib.import_module("moshi.models.lm")
    original = lm.sample_token
    counts = {"calls": 0}

    def checked(logits, use_sampling=True, temp=1.0, *args, **kwargs):
        assert not use_sampling and temp == 0, "native sampling settings are not greedy"
        result = original(logits, use_sampling, temp, *args, **kwargs)
        assert torch.equal(result, logits.argmax(-1)), "native sampling result is not argmax"
        counts["calls"] += 1
        return result

    lm.sample_token = checked
    try:
        yield counts
        assert counts["calls"] > 0, "generation never reached native sampling"
    finally:
        lm.sample_token = original


def _json_default(value):
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    return path


def _request(config, sample, adapter, output_dir, seed):
    if not 0 <= sample.window_start_sec < sample.window_end_sec <= sample.audio.duration_sec:
        raise ValueError("generation window must be within source audio")
    return dict(config=asdict(config), sample=asdict(sample),
                adapter=str(Path(adapter).resolve()),
                output_dir=str(Path(output_dir).resolve()), seed=int(seed))


def _decode_request(request):
    from personaplex_finetuning.config import Config
    from personaplex_finetuning.data import PreparedSample, AudioInfo, Word
    config = dict(request["config"])
    for name in ("path", "model_root", "personaplex_source", "prepared_dir", "output_dir",
                 "codec_cache_dir", "val_manifest_path", "test_manifest_path"):
        if config.get(name) is not None:
            config[name] = Path(config[name])
    config["generation_settings"] = GenerationSettings(**config["generation_settings"])
    sample = dict(request["sample"])
    for name in ("conversation_wav", "voice_prompt_wav", "voice_prompt_right_wav"):
        if sample.get(name) is not None:
            sample[name] = Path(sample[name])
    sample["audio"] = AudioInfo(**sample["audio"])
    sample["words"] = tuple(Word(**word) for word in sample["words"])
    return Config(**config), PreparedSample(**sample)


def worker(request_path):
    """Load one local base + native adapter/metadata using existing generate."""
    from personaplex_finetuning.inference import generate, resolve_adapter_checkpoint
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    config, sample = _decode_request(request)
    adapter = Path(request["adapter"])
    resolve_adapter_checkpoint(adapter)
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with verify_argmax() as counts:
        generate(config, sample, output / "generated.wav", output / "generated.txt",
                 adapter, generation=greedy_settings(), seed=request["seed"])
    _write(output / "argmax_checks.json", counts)
    _write(output / "worker.json", {"pid": os.getpid(), "generation": greedy_settings().as_dict(),
                                   "adapter": str(adapter), "sample_id": sample.sample_id})


def _spawn(request, args):
    output = Path(request["output_dir"])
    request_path = _write(output / "request.json", request)
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        subprocess.run([getattr(args, "python", None) or sys.executable,
                        str(Path(__file__).resolve()), "--worker", str(request_path)],
                       check=True, timeout=getattr(args, "worker_timeout", 600),
                       stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, NO_TORCH_COMPILE="1"), cwd=str(ROOT))


def compare_outputs(baseline_dir, fresh_dir):
    """Exact decoded PCM and raw text comparison, not WAV container headers."""
    import numpy as np
    import sphn
    baseline_dir, fresh_dir = Path(baseline_dir), Path(fresh_dir)
    baseline, rate = sphn.read(str(baseline_dir / "generated.wav"))
    fresh, fresh_rate = sphn.read(str(fresh_dir / "generated.wav"))
    if not baseline.size or not fresh.size or not (np.isfinite(baseline).all() and np.isfinite(fresh).all()):
        raise ValueError("missing or non-finite generated PCM")
    shape_equal = baseline.shape == fresh.shape
    pcm_equal = rate == fresh_rate and shape_equal and np.array_equal(baseline, fresh)
    text_equal = ((baseline_dir / "generated.txt").read_text(encoding="utf-8") ==
                  (fresh_dir / "generated.txt").read_text(encoding="utf-8"))
    return {"passed": bool(pcm_equal and text_equal), "pcm_equal": bool(pcm_equal),
            "text_equal": text_equal, "baseline_shape": list(baseline.shape),
            "fresh_shape": list(fresh.shape), "baseline_sample_rate": rate,
            "fresh_sample_rate": fresh_rate,
            "max_abs_pcm_error": float(np.max(np.abs(baseline - fresh))) if shape_equal else None}


@dataclass
class ReloadTicket:
    request: dict
    args: object
    baseline_dir: Path
    runtime_ref: object
    model_ref: object



def prepare_test3(args, config, sample, runtime):
    """Save live trained LoRA + baseline. Caller owns training provenance."""
    from personaplex_finetuning.inference import generate_text_with_runtime, write_generated_text
    from personaplex_finetuning.train import save_adapter
    model = runtime.model
    if hasattr(model, "module"):
        raise ValueError("test3 requires an unwrapped local runtime")
    modules = [m for m in model.modules() if hasattr(m, "lora_a") and hasattr(m, "lora_b")]
    if not modules:
        raise ValueError("test3 requires the loaded trained LoRA adapter")
    ranks = {m.lora_a.weight.shape[0] for m in modules}
    scales = {m.scale for m in modules}
    if len(ranks) != 1 or len(scales) != 1:
        raise ValueError("native metadata requires uniform LoRA rank and scale")
    rank, scale = next(iter(ranks)), next(iter(scales))
    alpha = rank * scale
    if alpha <= 0 or not float(alpha).is_integer():
        raise ValueError("native metadata requires positive integer alpha")
    config = replace(config, lora_rank=rank, lora_alpha=int(alpha), lora_scaling=scale)
    output = Path(args.output_dir).resolve() / "test3"
    seed = getattr(args, "seed", None)
    seed = config.seed if seed is None else seed
    request = _request(config, sample, output, output / "fresh", seed)
    adapter = save_adapter(output / "snapshot", model, config, getattr(args, "step", 0))
    request["adapter"] = str(adapter.resolve())
    baseline = output / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    text = generate_text_with_runtime(runtime, sample, generation=greedy_settings(),
                                     seed=seed, output_wav=baseline / "generated.wav")
    write_generated_text(baseline / "generated.txt", text, config.vietnamese_text_mode)
    _write(output / "request.json", request)
    return ReloadTicket(request, args, baseline, weakref.ref(runtime), weakref.ref(model))


def run_test3(ticket):
    """Complete reload comparison after all live backbone references are gone."""
    gc.collect()
    if ticket.runtime_ref() is not None or ticket.model_ref() is not None:
        raise RuntimeError("release runtime/model and trainer references before fresh load")
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _spawn(ticket.request, ticket.args)
    report = compare_outputs(ticket.baseline_dir, ticket.request["output_dir"])
    report.update(test="test3", baseline="live trained adapter before runtime release",
                  fresh_process=True, training_process_ended=False,
                  generation=greedy_settings().as_dict())
    _write(ticket.baseline_dir.parent / "report.json", report)
    return report


def _evaluate(args, config, sample, trained_adapter, test):
    from personaplex_finetuning.inference import text_error_metrics
    if trained_adapter is None:
        raise ValueError("a provided trained adapter is required; no automatic training")
    output = Path(args.output_dir).resolve() / test
    seed = getattr(args, "seed", None)
    request = _request(config, sample, trained_adapter, output,
                       config.seed if seed is None else seed)
    _spawn(request, args)
    hypothesis = (output / "generated.txt").read_text(encoding="utf-8")
    reference = " ".join(w.word for w in sample.words if w.speaker == "agent"
                         and w.start >= sample.window_start_sec and w.start < sample.window_end_sec)
    metrics = text_error_metrics(reference, hypothesis, config.vietnamese_text_mode)
    report = dict(test=test, generated=True, hypothesis=hypothesis, reference=reference,
                  text_metrics=metrics, overfit_proven=False,
                  generation=greedy_settings().as_dict(), adapter=request["adapter"],
                  sample_id=sample.sample_id, seed=request["seed"])
    _write(output / "report.json", report)
    return report


def run_test4(args, config, sample, *, trained_adapter):
    """Greedy training-sample evaluation; main decides overfit thresholds.

    Caller must release any training backbone before this worker call.
    """
    return _evaluate(args, config, sample, trained_adapter, "test4")


def run_test6(args, config, sample, *, trained_adapter, enabled=False):
    """Native free running only; main harness owns the prerequisite gate."""
    if enabled is not True:
        return {"test": "test6", "skipped": True, "reason": "main harness gate not enabled"}
    return _evaluate(args, config, sample, trained_adapter, "test6")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    worker(parser.parse_args().worker)
