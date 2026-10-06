#!/usr/bin/env python3
"""Six ordered real-sample gates. Production modules are never modified.

The parent never loads a backbone. Each GPU phase runs in a child process,
which must exit before the next phase starts. See validation_one_sample.md.

python "$PROJECT/scripts/validate_one_sample.py" \
  --config "$PROJECT/configs/overfit-10-train-v2.yaml" \
  --override "model.root=$MODELS" \
  --override "model.source=$PROJECT/src" \
  --override "data.prepared_dir=$DATA" \
  --output-dir "$OUT" \
  --index 0 --chunk 0 --device cuda \
  --train-steps 300 --through 6

python /home/voice/code/VDT_02/baottn/personaplex-finetune-v11/scripts/validate_one_sample.py \
  --config /home/voice/code/VDT_02/baottn/personaplex-finetune-v11/configs/overfit-10-train-telex.yaml \
  --output-dir /home/voice/code/VDT_02/baottn/personaplex-finetune-v11/validation-v4-diagnostic \
  --index 0 --chunk 0 --device cuda --through 2
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("NO_TORCH_COMPILE", "1")


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(config, sample):
    from personaplex_finetuning.runtime import RuntimePaths
    paths = RuntimePaths(config.model_root, config.personaplex_source).validate()
    files = [config.path, config.manifest, paths.moshi_weight, paths.mimi_weight,
             paths.tokenizer, sample.conversation_wav, sample.voice_prompt_wav]
    return dict(config=asdict(config), sample=asdict(sample),
                files={str(p.resolve()): digest(p) for p in files},
                source={str(p.relative_to(config.personaplex_source)): digest(p)
                        for p in config.personaplex_source.rglob("*.py")})


def assert_supervision(rows):
    assert rows, "no real text token occurrences"
    bad = [r for r in rows if not r["supervised"] or r["label"] != r["token_id"]]
    assert not bad, f"{len(bad)} unsupervised/mismatched occurrences: {bad[:10]}"


def load_config_sample(args):
    from personaplex_finetuning.config import load_config
    from personaplex_finetuning.data import PreparedDataset, duration_chunks
    config = load_config(args.config, overrides=args.override)
    config = replace(config, device=args.device)
    dataset = PreparedDataset(config.manifest, config.window_seconds)
    try:
        samples = dataset.load()
    except Exception:
        write(Path(args.output_dir) / "dataset_rejections.json", asdict(dataset.load_report))
        raise
    selected = samples[args.index]
    sample = duration_chunks([selected], config.duration_sec)[args.chunk]
    return config, sample


def codec_runtime(config, device):
    """Real Mimi + real tokenizer without allocating the 7B backbone."""
    import importlib
    from types import SimpleNamespace
    from personaplex_finetuning.runtime import RuntimePaths, MimiCodec, SentencePieceTokenizer
    paths = RuntimePaths(config.model_root, config.personaplex_source).validate(require_model=False)
    sys.path.insert(0, str(paths.source))
    loaders = importlib.import_module("moshi.models.loaders")
    lm = importlib.import_module("moshi.models.lm")
    mimi = loaders.get_mimi(paths.mimi_weight, device=device)
    mimi.eval().requires_grad_(False)
    # Native IDs follow the loader's card/text_card, not dataset magic numbers.
    card = loaders._lm_kwargs["card"]
    text_card = loaders._lm_kwargs["text_card"]
    return SimpleNamespace(
        codec=MimiCodec(mimi, mimi.sample_rate, mimi.frame_rate, device, lm),
        tokenizer=SentencePieceTokenizer(paths.tokenizer),
        initial_tokens=(text_card,) + (card,) * 16, zero_token=-1,
        delays=tuple(loaders._lm_kwargs["delays"]),
    )


def supervision_dump(config, sample, runtime, example):
    from personaplex_finetuning.text_normalization import normalize_vietnamese_text
    rows, words = [], []
    alignments = [a for a in example.word_alignments if a.speaker == "agent"]
    agent_words = [w for w in sample.words if w.speaker == "agent"
                   and sample.window_start_sec <= w.start < sample.window_end_sec]
    assert len(alignments) == len(agent_words), "alignment lost a word"
    for wi, (word, alignment) in enumerate(zip(agent_words, alignments)):
        normalized = normalize_vietnamese_text(word.word, config.vietnamese_text_mode)
        ids = runtime.tokenizer.encode(normalized)
        assert len(ids) == len(alignment.token_frames), f"word {wi} lost subwords"
        words.append(dict(word_index=wi, raw=word.word, normalized=normalized,
                          token_ids=ids, frame_positions=list(alignment.token_frames)))
        for si, (token, frame) in enumerate(zip(ids, alignment.token_frames)):
            position = example.prompt_frames + frame
            # forward_train truncates delayed tail; text mask also rejects zero sentinel.
            model_valid = position + runtime.delays[0] < example.total_frames and token != runtime.zero_token
            rows.append(dict(word_index=wi, subword_index=si, token_id=token,
                             dialogue_frame=frame, sequence_frame=position,
                             label=example.labels[0][position],
                             builder_mask=example.loss_mask[0][position],
                             model_mask=model_valid,
                             supervised=bool(example.loss_mask[0][position] and model_valid)))
    return dict(raw_transcript=" ".join(w.word for w in agent_words),
                normalized_transcript=" ".join(w["normalized"] for w in words),
                words=words, token_occurrences=rows,
                loss_mask=example.loss_mask, prompt_frames=example.prompt_frames,
                dialogue_frames=example.dialogue_frames,
                effective_text_mask_source="native delay/zero-token semantics; verified against model in Test2")


def full_runtime(config, adapter=None, *, full_precision_model=False):
    from personaplex_finetuning.runtime import load_runtime, RuntimePaths
    from personaplex_finetuning.inference import resolve_adapter_checkpoint
    from personaplex_finetuning.lora import inject_lora, load_adapter
    if adapter:
        file, rank, alpha, base, prefixes = resolve_adapter_checkpoint(Path(adapter))
        assert base == config.model_root.resolve(), "checkpoint base differs from config"
    runtime = load_runtime(RuntimePaths(config.model_root, config.personaplex_source),
                           config.device, config.qlora, config.quant_type,
                           full_precision_model=full_precision_model)
    if adapter:
        inject_lora(runtime.model, rank, alpha, prefixes=prefixes)
        load_adapter(runtime.model, file)
    runtime.model.eval()
    return runtime


@contextmanager
def parity_backend(torch, backend, fp32):
    """Scope diagnostic kernel selection and restore global precision settings."""
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        if fp32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if backend == "math":
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                scope = sdpa_kernel(SDPBackend.MATH)
            except ImportError:
                # Compatibility with the older PyTorch installed on the Mac.
                scope = torch.backends.cuda.sdp_kernel(enable_flash=False,
                                                       enable_math=True,
                                                       enable_mem_efficient=False)
            with scope:
                yield
        else:
            yield
    finally:
        if fp32:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32


def compare_logits(torch, batch, stream, mask, atol, rtol):
    a, b = batch[mask].float(), stream[mask].float()
    assert a.numel(), "no valid logits"
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    diff = (a - b).abs()
    return dict(passed=finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
                finite=finite, max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                valid_positions=int(mask.sum()), atol=atol, rtol=rtol)


def comparison_details(torch, batch, stream, mask, atol, rtol):
    """Bounded-memory summaries on [B, streams, frames, classes] tensors."""
    result = compare_logits(torch, batch, stream, mask, atol, rtol)
    frames, codebooks = [], []
    first_failure = None
    for t in range(mask.shape[-1]):
        valid = mask[..., t]
        if not valid.any():
            frames.append(dict(frame=t, valid_positions=0))
            continue
        a, b = batch[..., t, :][valid].float(), stream[..., t, :][valid].float()
        close = torch.isclose(a, b, atol=atol, rtol=rtol)
        failed = int((~close).sum())
        if failed and first_failure is None:
            first_failure = t
        diff = (a - b).abs()
        frames.append(dict(frame=t, valid_positions=int(valid.sum()),
                           failed_elements=failed, total_elements=a.numel(),
                           max_abs=float(diff.max()), mean_abs=float(diff.mean())))
    for k in range(mask.shape[1]):
        valid = mask[:, k]
        if not valid.any():
            codebooks.append(dict(stream=k, valid_positions=0))
            continue
        row = compare_logits(torch, batch[:, k], stream[:, k], valid, atol, rtol)
        a, b = batch[:, k][valid].float(), stream[:, k][valid].float()
        row.update(stream=k, argmax_agreement=float((a.argmax(-1) == b.argmax(-1)).float().mean()))
        codebooks.append(row)
    result.update(first_failure_frame=first_failure,
                  failed_elements=sum(r.get("failed_elements", 0) for r in frames),
                  frames=frames, streams=codebooks)
    return result


def parity(runtime, codes, args):
    """Native causal temporal streaming + per-frame GT depth streaming.

    No LMGen sampling/cache overrides: directly exercise native model streaming
    APIs with exactly the delayed GT inputs used by forward_train.
    """
    import torch
    from moshi.models.lm import _delay_sequence, _undelay_sequence
    model = runtime.model
    initial = model._get_initial_token()
    delayed = torch.cat([initial, _delay_sequence(model.delays, codes, initial)], dim=2)
    texts, audios, hiddens, isolated = [], [], [], []
    with torch.inference_mode():
        batch = model.forward_train(codes)
        # Independent batched temporal pass, using identical delayed inputs.
        batch_hidden, _ = model.forward_codes(delayed[:, :, :-1])
        isolated_batch = model.forward_depformer_training(delayed[:, :, 1:], batch_hidden).cpu()
        # Only temporal transformer state persists across time. Depth resets every frame.
        with model.transformer.streaming(1):
            for t in range(codes.shape[-1]):
                hidden, text = model.forward_codes(delayed[:, :, t:t + 1])
                hiddens.append(hidden.cpu())
                texts.append(text.cpu())
                depth = []
                with model.depformer.streaming(1):
                    for k in range(model.dep_q):
                        previous_stream = 0 if k == 0 else model.audio_offset + k - 1
                        gt = delayed[:, previous_stream:previous_stream + 1, t + 1:t + 2]
                        depth.append(model.forward_depformer(k, gt, hidden).cpu())
                audios.append(torch.cat(depth, dim=1))
        # Same batch hidden and GT tokens: isolate depth from temporal error.
        for t in range(codes.shape[-1]):
            depth = []
            with model.depformer.streaming(1):
                for k in range(model.dep_q):
                    previous_stream = 0 if k == 0 else model.audio_offset + k - 1
                    gt = delayed[:, previous_stream:previous_stream + 1, t + 1:t + 2]
                    depth.append(model.forward_depformer(k, gt, batch_hidden[:, t:t + 1]).cpu())
            isolated.append(torch.cat(depth, dim=1))
        audio_delays = model.delays[model.audio_offset:model.audio_offset + model.dep_q]
        audio, am = _undelay_sequence(audio_delays, torch.cat(audios, dim=2), float("nan"))
        text, tm = _undelay_sequence(model.delays[:1], torch.cat(texts, dim=2), float("nan"))
        am &= codes[:, model.audio_offset:model.audio_offset + model.dep_q].cpu() != model.zero_token_id
        tm &= codes[:, :1].cpu() != model.zero_token_id
        masks_equal = torch.equal(am, batch.mask.cpu()) and torch.equal(tm, batch.text_mask.cpu())
        report = dict(masks_equal=masks_equal,
                      text=compare_logits(torch, batch.text_logits.cpu(), text, tm, args.atol, args.rtol),
                      audio=compare_logits(torch, batch.logits.cpu(), audio, am, args.atol, args.rtol),
                      scope="native temporal/depth streaming GT-history, not LMGen cache parity")
        report["passed"] = masks_equal and report["text"]["passed"] and report["audio"]["passed"]
        hidden_batch = batch_hidden.cpu().unsqueeze(1)
        hidden_stream = torch.cat(hiddens, dim=1).unsqueeze(1)
        hidden_mask = torch.ones(hidden_batch.shape[:-1], dtype=torch.bool)
        depth_mask = torch.ones(isolated_batch.shape[:-1], dtype=torch.bool)
        report["diagnostics"] = dict(
            frame_grid="text/audio: undelayed; hidden/depth_only: delayed model steps",
            temporal_hidden=comparison_details(torch, hidden_batch, hidden_stream, hidden_mask, args.atol, args.rtol),
            depth_only=comparison_details(torch, isolated_batch, torch.cat(isolated, dim=2), depth_mask, args.atol, args.rtol),
            text=comparison_details(torch, batch.text_logits.cpu(), text, tm, args.atol, args.rtol),
            audio=comparison_details(torch, batch.logits.cpu(), audio, am, args.atol, args.rtol))
        parameter = next(model.parameters())
        report["runtime"] = dict(dtype=str(parameter.dtype), device=str(parameter.device),
                                torch_version=torch.__version__, cuda_version=torch.version.cuda,
                                gpu=torch.cuda.get_device_name(parameter.device) if parameter.is_cuda else None,
                                total_frames=codes.shape[-1], dep_q=model.dep_q, delays=list(model.delays),
                                temporal_context=getattr(model.transformer.layers[0].self_attn, "context", None),
                                depth_context=getattr(model.depformer.layers[0].self_attn, "context", None),
                                flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                                mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                                math_sdp=torch.backends.cuda.math_sdp_enabled(),
                                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                                requested_dtype=getattr(args, "parity_dtype", "default"),
                                requested_sdp_backend=getattr(args, "parity_sdp_backend", "auto"),
                                adapter=str(getattr(args, "adapter", None)))
        return report, batch


def diagnostics(output, example, device):
    import torch
    import torch.nn.functional as F
    labels = torch.tensor(example.labels, device=device).unsqueeze(0)
    masks = torch.tensor(example.loss_mask, device=device).unsqueeze(0)
    results = []
    for k in range(8):
        valid = masks[:, k + 1] & output.mask[:, k]
        logits, target = output.logits[:, k][valid].float(), labels[:, k + 1][valid]
        assert target.numel() and torch.isfinite(logits).all(), f"invalid cb{k}"
        results.append(dict(codebook=k, count=target.numel(),
                            ce=float(F.cross_entropy(logits, target)),
                            accuracy=float((logits.argmax(-1) == target).float().mean())))
    return results


def phase(args):
    import torch
    from personaplex_finetuning.train import build_example, seed_everything
    config, sample = load_config_sample(args)
    current_identity = identity(config, sample)
    if args.phase == "1":
        write(Path(args.output_dir) / "identity.json", current_identity)
    else:
        prior = json.loads((Path(args.output_dir) / "identity.json").read_text())
        assert json.loads(json.dumps(current_identity, default=str)) == prior, "config/sample/assets/source changed between gates"
    seed_everything(config.seed, torch)
    out = Path(args.output_dir)
    if args.phase != "1" and not torch.cuda.is_available():
        raise RuntimeError("Tests 2–6 require CUDA here; no silent CPU fallback for 7B")
    if args.phase == "1":
        runtime = codec_runtime(config, args.device)
        with torch.inference_mode():
            example = build_example(config, sample, runtime)
        dump = supervision_dump(config, sample, runtime, example)
        dump.update(sample_id=sample.sample_id, window=[sample.window_start_sec, sample.window_end_sec])
        write(out / "test1_dump.json", dump)
        torch.save(dict(codes=torch.tensor(example.input_codes).unsqueeze(0),
                        labels=torch.tensor(example.labels).unsqueeze(0),
                        loss_mask=torch.tensor(example.loss_mask).unsqueeze(0)), out / "sample.pt")
        write(out / "example.json", asdict(example))
        assert_supervision(dump["token_occurrences"])
        return dict(passed=True, tokens=len(dump["token_occurrences"]))
    from personaplex_finetuning.sequence import TrainingExample
    example = TrainingExample(**json.loads((out / "example.json").read_text()))
    tensors = torch.load(out / "sample.pt", weights_only=True, map_location=config.device)
    codes = tensors["codes"]
    if args.phase in ("2", "5", "train"):
        adapter = args.adapter
        if args.phase == "5":
            adapter = json.loads((out / "trained.json").read_text())["adapter"]
        fp32 = args.phase == "2" and args.parity_dtype == "fp32"
        if fp32 and config.qlora:
            raise ValueError("FP32 parity diagnostic requires qlora=false")
        runtime = full_runtime(config, adapter, full_precision_model=fp32)
        if args.phase == "2":
            dtype = next(runtime.model.parameters()).dtype
            if args.parity_dtype != "default":
                expected_dtype = torch.float32 if fp32 else torch.bfloat16
                assert dtype == expected_dtype, f"requested {expected_dtype}, loaded {dtype}"
            if args.atol is None:
                args.atol = 1e-5 if dtype == torch.float32 else 1e-2 if dtype == torch.float16 else 5e-2
            if args.rtol is None:
                args.rtol = 1e-4 if dtype == torch.float32 else 1e-3 if dtype == torch.float16 else 5e-3
            with parity_backend(torch, args.parity_sdp_backend, fp32):
                result, output = parity(runtime, codes, args)
            result["diagnostic_only"] = args.parity_dtype != "default" or args.parity_sdp_backend != "auto"
            write(out / "phase_2_diagnostics.json", result.pop("diagnostics"))
            result["diagnostics_file"] = str(out / "phase_2_diagnostics.json")
            for row in json.loads((out / "test1_dump.json").read_text())["token_occurrences"]:
                assert bool(output.text_mask[0, 0, row["sequence_frame"]]), "Test1 analytical mask disagrees"
            return result
        if args.phase == "5":
            with torch.inference_mode():
                rows = diagnostics(runtime.model.forward_train(codes), example, config.device)
            return dict(passed=all(r["accuracy"] >= args.min_audio_accuracy and r["ce"] <= args.max_audio_ce for r in rows),
                        codebooks=rows, min_accuracy=args.min_audio_accuracy, max_ce=args.max_audio_ce)
        from personaplex_finetuning.lora import inject_lora
        from personaplex_finetuning.train import loss_components, save_adapter
        if not adapter:
            runtime.model.requires_grad_(False)
            prefixes = ("transformer.", "depformer.") if config.train_stage == "joint" else ("transformer.",)
            inject_lora(runtime.model, config.lora_rank, config.lora_alpha, prefixes=prefixes)
        else:
            for name, param in runtime.model.named_parameters():
                param.requires_grad_("lora_a" in name or "lora_b" in name)
        params = [p for p in runtime.model.parameters() if p.requires_grad]
        assert params, "no trainable adapter parameters"
        optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        runtime.model.train()
        history = []
        for step in range(args.train_steps):
            optimizer.zero_grad(set_to_none=True)
            output = runtime.model.forward_train(codes)
            loss, components = loss_components(output, codes, tensors, runtime.tokenizer.padding_id, torch,
                                     config.first_codebook_weight_multiplier, config.text_padding_weight,
                                     user_loss=config.user_loss)
            assert torch.isfinite(loss), "non-finite training loss"
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            assert torch.isfinite(norm), "non-finite gradients"
            optimizer.step()
            history.append(dict(step=step + 1, loss=float(loss.detach()), grad_norm=float(norm),
                                components={k: float(v.detach()) for k, v in components.items()},
                                lr=config.learning_rate, trainable_parameters=sum(p.numel() for p in params)))
            write(out / "training_history.json", history)
        checkpoint = save_adapter(out / "training", runtime.model, config, args.train_steps)
        # Existing adapters may have rank/prefixes different from config: do not misrecord identity.
        if adapter:
            from personaplex_finetuning.inference import resolve_adapter_checkpoint
            _, rank, alpha, _, prefixes = resolve_adapter_checkpoint(Path(adapter))
            metadata = json.loads(checkpoint.with_name("adapter.json").read_text())
            metadata.update(rank=rank, alpha=alpha, scaling=alpha / rank, prefixes=prefixes)
            write(checkpoint.with_name("adapter.json"), metadata)
        from personaplex_finetuning.inference import generate_text_with_runtime, write_generated_text
        from validation_generation import greedy_settings, verify_argmax
        runtime.model.eval()
        baseline = out / "test3" / "baseline"
        baseline.mkdir(parents=True, exist_ok=True)
        real_sample = sample.with_window(sample.window_start_sec, min(sample.window_end_sec, sample.audio.duration_sec))
        with verify_argmax() as counts:
            text = generate_text_with_runtime(runtime, real_sample, generation=greedy_settings(),
                                              seed=config.seed, output_wav=baseline / "generated.wav")
        write(baseline / "argmax_checks.json", counts)
        write_generated_text(baseline / "generated.txt", text, config.vietnamese_text_mode)
        write(out / "trained.json", dict(adapter=str(checkpoint.resolve()), pid=os.getpid(), steps=args.train_steps))
        return dict(passed=True, checkpoint=str(checkpoint), training_pid=os.getpid())
    from validation_generation import _request, _spawn, compare_outputs, run_test4, run_test6
    adapter = Path(json.loads((out / "trained.json").read_text())["adapter"])
    real_sample = sample.with_window(sample.window_start_sec, min(sample.window_end_sec, sample.audio.duration_sec))
    if args.phase == "3":
        request = _request(config, real_sample, adapter, out / "test3" / "fresh", config.seed)
        _spawn(request, args)
        result = compare_outputs(out / "test3" / "baseline", out / "test3" / "fresh")
        result.update(training_process_ended=True, training_pid=json.loads((out / "trained.json").read_text())["pid"], reload_pid=os.getpid())
        return result
    if args.phase == "4":
        result = run_test4(args, config, real_sample, trained_adapter=adapter)
        result["passed"] = result["text_metrics"]["wer"] <= args.max_wer
        result["max_wer"] = args.max_wer
        return result
    return run_test6(args, config, real_sample, trained_adapter=adapter, enabled=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--through", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--max-wer", type=float, default=0.05)
    parser.add_argument("--min-audio-accuracy", type=float, default=0.99)
    parser.add_argument("--max-audio-ce", type=float, default=0.1)
    parser.add_argument("--worker-timeout", type=int, default=7200)
    parser.add_argument("--parity-dtype", choices=["default", "bf16", "fp32"], default="default",
                        help="gate 2 diagnostic model dtype; non-default requires --through 2")
    parser.add_argument("--parity-sdp-backend", choices=["auto", "math"], default="auto",
                        help="gate 2 SDP backend; math requires --through 2")
    parser.add_argument("--phase", choices=["1", "2", "train", "3", "4", "5", "6"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    diagnostic = args.parity_dtype != "default" or args.parity_sdp_backend != "auto"
    if diagnostic and (args.through != 2 or args.phase not in (None, "1", "2")):
        parser.error("parity diagnostic overrides require --through 2 and cannot advance to training/gates 3–6")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase:
        try:
            result = phase(args)
            write(args.output_dir / f"phase_{args.phase}.json", result)
            if result.get("passed") is False:
                raise AssertionError(f"gate {args.phase} failed: {result}")
        except Exception as exc:
            write(args.output_dir / f"failure_{args.phase}.json", dict(error=str(exc), traceback=traceback.format_exc()))
            raise
        return
    if any(args.output_dir.iterdir()):
        parser.error("use a new empty output-dir to prevent stale gate reuse")
    if args.through >= 3 and not args.adapter and args.train_steps < 1:
        parser.error("Test3 needs --adapter or explicit --train-steps N")
    write(args.output_dir / "invocation.json", vars(args))
    phases = ["1", "2", "train", "3", "4", "5", "6"]
    selected = phases[:1] if args.through == 1 else phases[:args.through + 1] if args.through >= 3 else phases[:2]
    original = sys.argv[1:]
    for gate in selected:
        print(f"=== phase {gate} ===", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), *original, "--phase", gate]
        # subprocess.run returns only after the training child has actually exited.
        with (args.output_dir / f"phase_{gate}.log").open("w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=args.worker_timeout)
        print((args.output_dir / f"phase_{gate}.json").read_text(), flush=True)


if __name__ == "__main__":
    main()