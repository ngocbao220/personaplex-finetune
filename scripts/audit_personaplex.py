#!/usr/bin/env python3
"""Audit logs, real alignment, native LoRA and two production inference paths.

All outputs go to --output-dir. Production sources and prepared inputs are untouched.
Run --help or see audit_personaplex.md. Full models run in sequential child processes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
os.environ.setdefault("NO_TORCH_COMPILE", "1")

from audit_core import compare_alignment, compare_runs, compare_voice_encoding, digest, git_identity, load_run, replay_updates, summarize_run, write_json


def child(args, phase):
    log = args.output_dir / f"{phase}.log"
    with log.open("w") as stream:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "_worker", "--request",
                        str(args.output_dir / "request.json"), "--phase", phase],
                       check=True, timeout=args.timeout, stdout=stream, stderr=subprocess.STDOUT,
                       cwd=PROJECT, env=dict(os.environ, NO_TORCH_COMPILE="1"))


def selected_sample(args):
    from personaplex_finetuning.config import load_config
    from personaplex_finetuning.data import PreparedDataset, duration_chunks, limit_conversations
    from personaplex_finetuning.runtime import RuntimePaths, SentencePieceTokenizer
    from personaplex_finetuning.chunk_filter import filter_text_capacity_chunks
    cfg = replace(load_config(args.config, overrides=args.override), device=args.device)
    if not cfg.eval_on_train_samples:
        raise ValueError("this matched-training audit requires eval_on_train_samples=true; preserve the original run split")
    dataset = PreparedDataset(cfg.manifest, cfg.window_seconds, filter_num_workers=1)
    samples = dataset.load()
    selected = limit_conversations(samples, cfg.sample_number, cfg.sample_index)
    chunks = duration_chunks(selected, cfg.duration_sec)
    if cfg.sample_index is not None:
        chunks = chunks[:1]
    assets = RuntimePaths(cfg.model_root, cfg.personaplex_source).validate(require_model=False)
    tokenizer = SentencePieceTokenizer(assets.tokenizer)
    filtered = filter_text_capacity_chunks(chunks, tokenizer, 12.5, swap_roles=cfg.swap_roles_after_pass,
                                          vietnamese_text_mode=cfg.vietnamese_text_mode, num_workers=1)
    write_json(args.output_dir / "training_chunks.json", {
        "dataset_load": asdict(dataset.load_report),
        "config": asdict(cfg), "manifest_sha256": digest(cfg.manifest),
        "kept": [{"sample_id": s.sample_id, "start": s.window_start_sec, "end": s.window_end_sec} for s in filtered.kept],
        "rejected": [asdict(r) for r in filtered.rejected],
        "updates": replay_updates(cfg, filtered.kept)})
    matches = [s for s in filtered.kept if s.sample_id == args.sample_id
               and s.window_start_sec <= args.start_sec < s.window_end_sec]
    if len(matches) != 1:
        raise ValueError(f"expected one retained training chunk for {args.sample_id} at {args.start_sec}; found {len(matches)}")
    training_chunk = matches[0]
    end = min(training_chunk.window_end_sec, training_chunk.audio.duration_sec) if args.end_sec is None else args.end_sec
    if not 0 <= args.start_sec < end <= min(training_chunk.window_end_sec, training_chunk.audio.duration_sec):
        raise ValueError("audit crop must fit the retained training chunk and actual audio")
    sample = training_chunk.with_window(args.start_sec, end)
    if args.role == "right-agent":
        if not cfg.swap_roles_after_pass:
            raise ValueError("right-agent is not a trained role in this configuration")
        sample = sample.swapped_roles()
    return cfg, sample, tokenizer


def legacy_sample(path, start, end):
    """Explicit audit-only bridge for the supplied old OtoSpeech schema."""
    from personaplex_finetuning.data import PreparedSample, Word, read_wav_info
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("agent_channel") != "left" or metadata.get("user_channel") != "right":
        raise ValueError("legacy bridge requires explicit LEFT=agent, RIGHT=user metadata")
    info = read_wav_info(path / "conversation.wav")
    if info.channels != 2 or info.sample_rate != 24000:
        raise ValueError("legacy bridge requires 24 kHz stereo")
    prompt = metadata.get("text_prompt_left") or metadata.get("text_prompt")
    if not (path / "voice_prompt.wav").is_file() or not prompt:
        raise ValueError("missing prepared voice/text prompt")
    words = tuple(Word(**{k: row[k] for k in ("speaker", "word", "start", "end")})
                  for row in json.loads((path / "words.json").read_text()))
    for w in words:
        if w.speaker not in {"agent", "user"} or not w.word.strip() or not all(math.isfinite(t) for t in (w.start, w.end)):
            raise ValueError("invalid prepared word")
        if not 0 <= w.start <= w.end <= info.duration_sec:
            raise ValueError("word outside audio")
    end = min(end, info.duration_sec)
    if not 0 <= start < end:
        raise ValueError("invalid audit window")
    return PreparedSample(metadata["sample_id"], path / "conversation.wav", path / "voice_prompt.wav",
                          words, prompt, metadata, info, start, end)


def source_identity():
    import importlib
    names = ("personaplex_finetuning.sequence", "personaplex_finetuning.train", "personaplex_finetuning.runtime",
             "personaplex_finetuning.lora", "personaplex_finetuning.inference", "moshi.models.lm", "moshi.modules.transformer")
    return {name: {"path": module.__file__, "sha256": digest(module.__file__)} for name in names
            if (module := importlib.import_module(name))}


def encode_data(cfg, sample, tokenizer, reference, out):
    import torch
    from validate_one_sample import codec_runtime
    from audit_native import require_device
    from personaplex_finetuning.train import build_example
    require_device(cfg.device)
    runtime = codec_runtime(cfg, cfg.device)
    with torch.inference_mode():
        example = build_example(cfg, sample, runtime)
        user = runtime.codec.encode_conversation(sample.conversation_wav, sample.user_channel,
                                                 sample.window_start_sec, sample.window_end_sec)
    train_user = tuple(tuple(row[example.prompt_frames:]) for row in example.input_codes[9:])
    if train_user != user:
        raise AssertionError("training and inference user Mimi tokens differ")
    if any(any(row[:example.prompt_frames]) for row in example.loss_mask):
        raise AssertionError("prompt is supervised")
    if (sample.agent_channel, sample.user_channel) not in {(0, 1), (1, 0)}:
        raise AssertionError("invalid logical role mapping")
    write_json(out / "sequence.json", asdict(example))
    write_json(out / "alignment.json", compare_alignment(sample, tokenizer, example.dialogue_frames,
                                                          runtime.codec.frame_rate, cfg.vietnamese_text_mode, reference))
    voice = compare_voice_encoding(sample, runtime, reference)
    write_json(out / "voice.json", voice)
    from personaplex_finetuning.inference import export_original_audio_window
    export_original_audio_window(sample, out / "source.wav")
    print(f"{sample.sample_id}: channels agent/user={sample.agent_channel}/{sample.user_channel}; "
          f"17 streams; prompt={example.prompt_frames}; dialogue={example.dialogue_frames}", flush=True)
    return {"prompt_masked": True, "user_tokens_equal": True, "agent_channel": sample.agent_channel,
            "user_channel": sample.user_channel, "prompt_frames": example.prompt_frames,
            "dialogue_frames": example.dialogue_frames, "voice_local_native_equal": voice["local_native_equal"],
            "voice_reference_overlap_mismatches": voice["reference_overlap_mismatches"],
            "assets": {str(p): digest(p) for p in (sample.conversation_wav, sample.voice_prompt_wav,
                       cfg.model_root / "tokenizer_spm_32k_3.model",
                       cfg.model_root / "tokenizer-e351c8d8-checkpoint125.safetensors")}}


def full_worker(request, phase):
    import torch
    from audit_native import (require_device, output_metrics, save_output, load_output, output_comparison,
                              generate_observed)
    from validation_generation import _decode_request
    from validate_one_sample import full_runtime, parity
    from personaplex_finetuning.train import build_example, model_forward_train, seed_everything
    from personaplex_finetuning.inference import resolve_adapter_checkpoint
    from personaplex_finetuning.lora import adapter_state_dict
    from safetensors.torch import save_file
    from types import SimpleNamespace
    config, sample = _decode_request(request)
    require_device(config.device)
    seed_everything(config.seed, torch)
    out = Path(request["output_dir"])
    adapter = Path(request["adapter"])
    if phase in {"standalone", "base"}:
        result = generate_observed(config, sample, out / phase, adapter=adapter if phase == "standalone" else None)
        write_json(out / f"{phase}.json", result)
        return
    runtime = full_runtime(config, adapter)
    write_json(out / f"identity_{phase}.json", {"source": source_identity(), "git": git_identity(PROJECT),
               "config": asdict(config), "sample": asdict(sample), "torch": torch.__version__,
               "assets": {str(p): digest(p) for p in (adapter, adapter.with_name("adapter.json"),
                            config.model_root / "model.safetensors", config.model_root / "tokenizer_spm_32k_3.model",
                            config.model_root / "tokenizer-e351c8d8-checkpoint125.safetensors",
                            sample.conversation_wav, sample.voice_prompt_wav)}})
    example = build_example(config, sample, runtime)
    if any(any(row[:example.prompt_frames]) for row in example.loss_mask):
        raise AssertionError("conditioning prompt is supervised")
    codes = torch.tensor(example.input_codes, device=config.device).unsqueeze(0)
    with torch.no_grad():
        output = model_forward_train(runtime.model, codes)
        metrics = output_metrics(config, runtime, example, output)
    save_output(out / f"{phase}_logits.safetensors", output)
    metrics["scope"] = "exact requested crop; not historical training step loss"
    metrics["frames"] = example.total_frames
    write_json(out / f"{phase}_metrics.json", metrics)
    del output
    # Match the production padding helper for a single-example batch. Historical
    # multi-example batches can have a larger maximum prompt length.
    from personaplex_finetuning.sequence import pad_training_example
    from personaplex_finetuning.chunk_filter import expected_mimi_frames
    padded = pad_training_example(example,
        expected_mimi_frames(config.duration_sec, runtime.codec.frame_rate) + example.prompt_frames,
        runtime.tokenizer.padding_id, runtime.zero_token)
    with torch.no_grad():
        padded_output = model_forward_train(runtime.model,
            torch.tensor(padded.input_codes, device=config.device).unsqueeze(0))
        padded_metrics = output_metrics(config, runtime, padded, padded_output)
    padded_metrics.update(scope="production padding for single-example batch; not historical step loss",
                          frames=padded.total_frames)
    write_json(out / f"{phase}_padded_metrics.json", padded_metrics)
    del padded_output
    if phase == "in_memory":
        write_json(out / "sequence.json", asdict(example))
        write_json(out / "alignment.json", compare_alignment(sample, runtime.tokenizer, example.dialogue_frames,
                        runtime.codec.frame_rate, config.vietnamese_text_mode, request["reference"]))
        infer_user = runtime.codec.encode_conversation(sample.conversation_wav, sample.user_channel,
                                                       sample.window_start_sec, sample.window_end_sec)
        train_user = tuple(tuple(row[example.prompt_frames:]) for row in example.input_codes[9:])
        if train_user != infer_user:
            raise AssertionError("training and inference user codes differ")
        roundtrip = out / "roundtrip"
        roundtrip.mkdir()
        save_file(adapter_state_dict(runtime.model), str(roundtrip / "lora.safetensors"))
        from safetensors.torch import load_file
        original_state = load_file(str(adapter))
        saved_state = load_file(str(roundtrip / "lora.safetensors"))
        write_json(out / "adapter_roundtrip.json", {
            "keys_equal": original_state.keys() == saved_state.keys(),
            "tensors_equal": original_state.keys() == saved_state.keys() and all(
                torch.equal(original_state[k], saved_state[k]) for k in original_state),
            "scope": "original adapter tensors versus saved in-memory adapter"})
        del original_state, saved_state
        _, rank, alpha, _, prefixes = resolve_adapter_checkpoint(adapter)
        metadata = json.loads(adapter.with_name("adapter.json").read_text())
        metadata.update(rank=rank, alpha=alpha, scaling=alpha / rank, model_root=str(config.model_root.resolve()))
        write_json(roundtrip / "adapter.json", metadata)
        generate_observed(config, sample, out / "in_memory", runtime=runtime)
        if request.get("forced_agent_audio"):
            forced_audio = [list(row[example.prompt_frames:]) for row in example.input_codes[1:9]]
            generate_observed(config, sample, out / "forced_agent_audio", runtime=runtime,
                              forced_agent_audio=forced_audio)
        if request["forced_text"]:
            forced = list(example.input_codes[0][example.prompt_frames:])
            generate_observed(config, sample, out / "forced_text", runtime=runtime, forced_text=forced)
        if request["parity_frames"]:
            # Restrict only this expensive GT-history parity pass; never silently crop the primary inference.
            bounded = codes[:, :, :min(codes.shape[-1], request["parity_frames"])]
            args = SimpleNamespace(atol=request["atol"], rtol=request["rtol"], adapter=str(adapter))
            from contextlib import nullcontext
            with torch.autocast("cuda", dtype=torch.bfloat16) if codes.is_cuda else nullcontext():
                result, _ = parity(runtime, bounded, args)
            write_json(out / "streaming_parity.json", result)
    elif phase == "reload":
        write_json(out / "reload_comparison.json", output_comparison(load_output(out / "in_memory_logits.safetensors"),
                         load_output(out / "reload_logits.safetensors"), request["atol"], request["rtol"]))
    write_json(out / f"{phase}.json", {"completed": True,
               "provenance": "existing checkpoint loaded into memory; not original historical training process",
               "peak_cuda_bytes": torch.cuda.max_memory_allocated() if codes.is_cuda else None})


def report_logs(args):
    run = load_run(args.run)
    summary = summarize_run(run)
    result = compare_runs(load_run(args.control_run), run) if args.control_run else summary
    write_json(args.output_dir / "logs.json", result)
    lines = ["# PersonaPlex run audit", "", f"Failing run: `{args.run}`", "",
             "## Observed training losses by logical role", "", "| Role | Total loss | Real text loss | CB0 loss |", "|---|---:|---:|---:|"]
    for role, values in summary["role_loss_means"].items():
        lines.append(f"| {role} | {values.get('loss/total', 'unavailable')} | {values.get('loss/text_nonpadding', 'unavailable')} | {values.get('loss/audio_cb0', 'unavailable')} |")
    lines += ["", "## Free-running", "", "| Step | CER | WER | Empty samples |", "|---|---:|---:|---:|"]
    for row in summary["generation"]:
        lines.append(f"| {row['step']} | {row['cer']} | {row['wer']} | {row['empty_samples']} |")
    lines += ["", "## Evidence boundaries", "", f"- Reload checks: {summary['checkpoint_reload']}.",
              "- No observed step-to-sample identities: exact per-chunk/role update counts remain unavailable.",
              "- Role means pool different samples; compare the same retained sample/window before diagnosing.",
              "- The reported successful 10-sample run is in-training inference; fresh reload is unverified.",
              "- Reference alignment/LoRA/runtime differences require controlled probes, not wholesale replacement."]
    if not args.control_run:
        lines.append("- Successful run logs are missing here; no 10-versus-100 causal comparison has run.")
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    logs = subs.add_parser("logs", help="summarize recorded logs; optionally compare a successful control")
    logs.add_argument("--run", type=Path, required=True)
    logs.add_argument("--control-run", type=Path)
    alignment = subs.add_parser("alignment", help="real tokenizer/actual reference alignment; explicit legacy data bridge")
    alignment.add_argument("--sample-dir", type=Path, action="append", required=True)
    alignment.add_argument("--tokenizer", type=Path, required=True)
    alignment.add_argument("--text-mode", choices=["diacritics", "no_diacritics", "telex"], default="diacritics")
    alignment.add_argument("--start-sec", type=float, default=0)
    alignment.add_argument("--end-sec", type=float, default=10)
    alignment.add_argument("--encode", action="store_true", help="real Mimi/voice/sequence smoke using the explicit legacy bridge")
    alignment.add_argument("--model-root", type=Path)
    alignment.add_argument("--device", default="cpu")
    mini = subs.add_parser("mini", help="native small-model train/stream/reload on CPU, MPS or CUDA")
    mini.add_argument("--source-state", choices=["head", "worktree"], default="worktree")
    mini.add_argument("--device", default="cpu")
    mini.add_argument("--timeout", type=float, default=600)
    for name in ("data", "probe"):
        p = subs.add_parser(name, help="retained train sample audit" if name == "data" else "existing 7B checkpoint audit in sequential workers")
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--override", action="append", default=[])
        p.add_argument("--sample-id", required=True)
        p.add_argument("--start-sec", type=float, default=0)
        p.add_argument("--end-sec", type=float)
        p.add_argument("--role", choices=["left-agent", "right-agent"], default="left-agent")
        p.add_argument("--device", default="cuda")
        if name == "probe":
            p.add_argument("--adapter", type=Path, required=True)
            p.add_argument("--forced-text", action="store_true", help="GT text/audio diagnostic, never quality evidence")
            p.add_argument("--forced-agent-audio", action="store_true",
                           help="force GT agent audio via native moshi_tokens while text stays free; diagnostic only")
            p.add_argument("--parity-frames", type=int, default=0, help="explicit bounded GT-history parity prefix; 0 disables")
            p.add_argument("--atol", type=float, default=.05)
            p.add_argument("--rtol", type=float, default=.005)
            p.add_argument("--timeout", type=float, default=1800)
    worker = subs.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--request", type=Path, required=True)
    worker.add_argument("--phase", required=True)
    for p in (logs, alignment, mini, subs.choices["data"], subs.choices["probe"]):
        p.add_argument("--output-dir", type=Path, required=True)
        if p is not logs:
            p.add_argument("--reference", type=Path, default=PROJECT.parent / "refs/personaplex-finetune")
    args = parser.parse_args()
    if args.command == "_worker":
        request = json.loads(args.request.read_text())
        from audit_native import tiny_probe, tiny_reload
        if args.phase == "mini":
            tiny_probe(request["device"], request["reference"], request["output_dir"], request["source_state"])
        elif args.phase == "mini_reload":
            tiny_reload(request["output_dir"], request["device"])
        else:
            full_worker(request, args.phase)
        return
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "request.json").exists() or (args.output_dir / "report.json").exists():
        parser.error("output directory already contains an audit; choose a fresh directory to prevent stale results")
    write_json(args.output_dir / "invocation.json", {"argv": sys.argv, "python": sys.executable,
                                                    "git": git_identity(PROJECT)})
    try:
        if args.command == "logs":
            result = report_logs(args)
        elif args.command == "alignment":
            from personaplex_finetuning.runtime import SentencePieceTokenizer
            tokenizer = SentencePieceTokenizer(args.tokenizer)
            results = []
            for path in args.sample_dir:
                sample = legacy_sample(path.resolve(), args.start_sec, args.end_sec)
                frames = math.ceil((sample.window_end_sec - sample.window_start_sec) * 12.5 - 1e-6)
                row = compare_alignment(sample, tokenizer, frames, 12.5, args.text_mode, args.reference)
                write_json(args.output_dir / f"{sample.sample_id}.json", row)
                summary = {"sample_id": sample.sample_id, "local_lost": row["local"]["lost_tokens"],
                           "reference_lost": row["reference"]["lost_tokens"], "frame_mismatches": row["frame_mismatches"]}
                if args.encode:
                    if args.model_root is None:
                        raise ValueError("--encode requires explicit --model-root")
                    from personaplex_finetuning.config import Config
                    cfg = Config(path=path / "metadata.json", model_root=args.model_root.resolve(),
                                 personaplex_source=PROJECT / "src", prepared_dir=path.parent,
                                 output_dir=args.output_dir, device=args.device,
                                 duration_sec=sample.window_end_sec - sample.window_start_sec,
                                 vietnamese_text_mode=args.text_mode)
                    output = args.output_dir / sample.sample_id
                    output.mkdir()
                    summary["codec"] = encode_data(cfg, sample, tokenizer, args.reference, output)
                results.append(summary)
            result = {"scope": "explicit legacy schema bridge; not production loader compatibility or failing synthetic corpus",
                      "tokenizer_sha256": digest(args.tokenizer), "samples": results}
        elif args.command == "mini":
            write_json(args.output_dir / "request.json", vars(args))
            child(args, "mini")
            child(args, "mini_reload")
            result = json.loads((args.output_dir / "mini.json").read_text())
            result["fresh_process_reload"] = json.loads((args.output_dir / "reload.json").read_text())
        else:
            cfg, sample, tokenizer = selected_sample(args)
            if args.command == "data":
                result = encode_data(cfg, sample, tokenizer, args.reference, args.output_dir)
            else:
                from personaplex_finetuning.inference import resolve_adapter_checkpoint
                adapter, _, _, base, _ = resolve_adapter_checkpoint(args.adapter)
                if base is not None and base != cfg.model_root.resolve():
                    raise ValueError("adapter base path differs from config; explicit matching base identity required")
                request = {"config": asdict(cfg), "sample": asdict(sample), "adapter": str(adapter.resolve()),
                           "output_dir": str(args.output_dir), "forced_text": args.forced_text,
                           "forced_agent_audio": args.forced_agent_audio,
                           "reference": str(args.reference.resolve()),
                           "parity_frames": args.parity_frames, "atol": args.atol, "rtol": args.rtol}
                if args.parity_frames < 0:
                    raise ValueError("parity_frames must be non-negative")
                write_json(args.output_dir / "request.json", request)
                for phase in ("in_memory", "reload", "standalone", "base"):
                    if phase == "reload":
                        request["adapter"] = str(args.output_dir / "roundtrip/lora.safetensors")
                        write_json(args.output_dir / "request.json", request)
                    if phase == "standalone":
                        request["adapter"] = str(adapter.resolve())
                        write_json(args.output_dir / "request.json", request)
                    child(args, phase)
                from validation_generation import compare_outputs
                tokens_a = json.loads((args.output_dir / "in_memory/tokens.json").read_text())
                tokens_b = json.loads((args.output_dir / "standalone/tokens.json").read_text())
                result = {"scope": "reloaded existing checkpoint; not historical training in-memory snapshot",
                          "reload": json.loads((args.output_dir / "reload_comparison.json").read_text()),
                          "production_entrypoints": compare_outputs(args.output_dir / "in_memory", args.output_dir / "standalone"),
                          "raw_tokens_equal": tokens_a["returned"] == tokens_b["returned"]}
                if args.forced_agent_audio:
                    result["forced_agent_audio"] = json.loads(
                        (args.output_dir / "forced_agent_audio/generation.json").read_text())
                identities = [json.loads((args.output_dir / f"identity_{p}.json").read_text())
                              for p in ("in_memory", "reload")]
                result["fresh_worker_identity_equal"] = {
                    key: identities[0][key] == identities[1][key]
                    for key in ("source", "git", "config", "sample", "torch")}
                adapter_paths = {str(adapter.resolve()), str(adapter.with_name("adapter.json").resolve()),
                                 str(args.output_dir / "roundtrip/lora.safetensors"),
                                 str(args.output_dir / "roundtrip/adapter.json")}
                result["fresh_worker_base_data_assets_equal"] = (
                    {k: v for k, v in identities[0]["assets"].items() if k not in adapter_paths} ==
                    {k: v for k, v in identities[1]["assets"].items() if k not in adapter_paths})
                result["adapter_roundtrip"] = json.loads((args.output_dir / "adapter_roundtrip.json").read_text())
        write_json(args.output_dir / "report.json", result)
        print(f"Audit artifacts: {args.output_dir}")
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
