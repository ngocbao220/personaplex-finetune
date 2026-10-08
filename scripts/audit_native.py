"""Native model probes used only by the audit CLI (not by the trainer)."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

from audit_core import digest, write_json


def require_device(device):
    import torch
    kind = torch.device(device).type
    if kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fallback")
    if kind == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; run outside the restricted sandbox")
    if kind not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {device}")


def tensor_comparison(a, b, mask, atol=1e-5, rtol=1e-4):
    import torch
    a, b, mask = a.detach().cpu(), b.detach().cpu(), mask.detach().cpu()
    if a.shape != b.shape or mask.shape != a.shape[:-1]:
        return {"passed": False, "reason": "shape mismatch", "left_shape": list(a.shape), "right_shape": list(b.shape)}
    valid_a, valid_b = a[mask].float(), b[mask].float()
    if not valid_a.numel():
        return {"passed": False, "reason": "no valid positions"}
    finite = bool(torch.isfinite(valid_a).all() and torch.isfinite(valid_b).all())
    close = torch.isclose(a.float(), b.float(), atol=atol, rtol=rtol).all(-1)
    failed = (~close & mask).nonzero()
    diff = (valid_a - valid_b).abs()
    return {"passed": finite and not len(failed), "finite": finite, "valid_positions": int(mask.sum()),
            "max_abs": float(diff.max()) if finite else None,
            "argmax_agreement": float((valid_a.argmax(-1) == valid_b.argmax(-1)).float().mean()),
            "first_failure_b_stream_frame": failed[0].tolist() if len(failed) else None,
            "atol": atol, "rtol": rtol}


def output_comparison(a, b, atol=1e-5, rtol=1e-4):
    import torch
    masks = torch.equal(a.mask.cpu(), b.mask.cpu()) and torch.equal(a.text_mask.cpu(), b.text_mask.cpu())
    return {"masks_equal": masks,
            "text": tensor_comparison(a.text_logits, b.text_logits, a.text_mask, atol, rtol),
            "audio": tensor_comparison(a.logits, b.logits, a.mask, atol, rtol)}


def output_metrics(config, runtime, example, output):
    import torch
    import torch.nn.functional as F
    from personaplex_finetuning.train import loss_components, tokenizer_text_padding_ids
    codes = torch.tensor(example.input_codes, device=config.device).unsqueeze(0)
    total, components = loss_components(output, codes, example, tokenizer_text_padding_ids(runtime.tokenizer), torch,
                                       config.first_codebook_weight_multiplier, config.text_padding_weight,
                                       user_loss=config.user_loss)
    labels = torch.tensor(example.labels, device=config.device).unsqueeze(0)
    mask = torch.tensor(example.loss_mask, device=config.device).unsqueeze(0)
    rows = []
    for stream in range(17):
        logits = output.text_logits[:, 0] if stream == 0 else output.logits[:, stream - 1]
        native_mask = output.text_mask[:, 0] if stream == 0 else output.mask[:, stream - 1]
        valid = mask[:, stream] & native_mask
        if stream == 0:
            for padding in tokenizer_text_padding_ids(runtime.tokenizer):
                valid &= labels[:, stream] != padding
        targets, selected = labels[:, stream][valid], logits[valid].float()
        if not targets.numel():
            rows.append({"stream": stream, "count": 0})
            continue
        if not torch.isfinite(selected).all():
            raise ValueError(f"non-finite supervised logits in stream {stream}")
        rows.append({"stream": stream, "count": int(targets.numel()),
                     "ce_sum": float(F.cross_entropy(selected, targets, reduction="sum")),
                     "ce_mean": float(F.cross_entropy(selected, targets)),
                     "correct": int((selected.argmax(-1) == targets).sum()),
                     "accuracy": float((selected.argmax(-1) == targets).float().mean())})
    return {"loss_total": float(total), "components": {k: float(v) for k, v in components.items()},
            "unweighted_streams": rows}


def save_output(path, output):
    from safetensors.torch import save_file
    save_file({"audio": output.logits.detach().cpu().contiguous(),
               "text": output.text_logits.detach().cpu().contiguous(),
               "audio_mask": output.mask.detach().cpu().contiguous(),
               "text_mask": output.text_mask.detach().cpu().contiguous()}, str(path))


def load_output(path):
    from safetensors.torch import load_file
    state = load_file(str(path))
    return SimpleNamespace(logits=state["audio"], text_logits=state["text"],
                           mask=state["audio_mask"], text_mask=state["text_mask"])


@contextmanager
def layer_outputs(model):
    """Observe responsible native layer boundaries on the same batched inputs."""
    values, handles = {}, []
    for name, module in model.named_modules():
        if name == "out_norm" or name == "text_linear" or (
            name.startswith(("transformer.layers.", "depformer.layers.")) and name.count(".") == 2
        ):
            def capture(mod, args, output, key=name):
                values[key] = output.detach().cpu().clone()
            handles.append(module.register_forward_hook(capture))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def ring_boundary_probe():
    import inspect
    import torch
    from moshi.modules.transformer import RingKVCache
    cache = RingKVCache(1, 1, 2, 16, "cpu", torch.float32)
    observed = []
    for t in range(16):
        x = torch.full((1, 1, 1, 2), float(t))
        result = cache.complete(x, x)
        observed.append(result.positions.tolist())
    # The entries were written in time order, before the first wrap.
    expected = list(range(16))
    actual = result.positions.flatten().tolist()
    return {"capacity": 16, "expected_positions_at_full_cache": expected, "actual_positions_at_full_cache": actual,
            "matches": actual == expected, "positions_each_step": observed,
            "source": inspect.getsourcefile(RingKVCache),
            "impact_boundary": "mini depformer cb15 differs; shared reference logic, not evidence of agent-generation cause"}


def tiny_generation(model, codes):
    import torch
    from moshi.models.lm import LMGen
    generator = LMGen(model, device=str(codes.device), use_sampling=False, temp=0, temp_text=0)
    tokens = []
    with torch.no_grad(), generator.streaming(1):
        for t in range(codes.shape[-1]):
            value = generator.step(input_tokens=codes[:, 9:17, t:t + 1])
            if value is not None:
                tokens.append(value.detach().cpu().flatten().tolist())
    return tokens


@contextmanager
def capture_generation(output_dir, *, forced_text=None, forced_agent_audio=None):
    """Observe native boundaries; optional GT text/audio are diagnostic-only."""
    import torch
    lm = importlib.import_module("moshi.models.lm")
    original = lm.LMGen
    steps, returned, text_frames = [], [], []
    completed = False
    if forced_agent_audio is not None and (
        len(forced_agent_audio) != 8 or not forced_agent_audio[0]
        or any(len(row) != len(forced_agent_audio[0]) for row in forced_agent_audio)
    ):
        raise ValueError("forced agent audio requires eight equal nonempty streams")

    class ObservedLMGen(original):
        in_dialogue = False
        dialogue_index = 0

        def step_system_prompts(self, mimi):
            value = super().step_system_prompts(mimi)
            self.in_dialogue = True
            return value

        def prepare_step_input(self, *args, **kwargs):
            value = super().prepare_step_input(*args, **kwargs)
            if value is not None:
                inp, provided, target, _, _ = value
                steps.append({"offset": self._streaming_state.offset, "dialogue": self.in_dialogue,
                              "input": inp.cpu().flatten().tolist(), "target": target.cpu().flatten().tolist(),
                              "provided": provided.cpu().flatten().tolist()})
            return value

        def process_transformer_output(self, transformer_out, text_logits, provided_, target_,
                                       model_input_position, target_position):
            # Observe before native torch.where replaces the predicted token with GT.
            # Never change logits, sampling, cache contents or returned outputs.
            if self.in_dialogue:
                logits = text_logits.detach()[0, 0, 0].float()
                if not torch.isfinite(logits).all():
                    raise ValueError("non-finite native text logits")
                log_probs = logits.log_softmax(-1)
                top = logits.topk(min(5, logits.numel())).indices.tolist()
                provided = bool(provided_[0, 0, 0])
                target = int(target_[0, 0, 0]) if provided else None
                if target is not None and not 0 <= target < logits.numel():
                    raise ValueError(f"provided native text target outside vocabulary: {target}")
                text_frames.append({
                    "dialogue_input_frame": self.dialogue_index,
                    "native_offset": int(self._streaming_state.offset),
                    "text_delay": int(self.lm_model.delays[0]),
                    "prediction": int(logits.argmax()), "top_tokens": top,
                    "pad_probability": float(log_probs[[0, 3]].exp().sum()),
                    "gt_target": target,
                    "target_log_probability": float(log_probs[target]) if target is not None else None,
                })
            return super().process_transformer_output(transformer_out, text_logits, provided_, target_,
                                                       model_input_position, target_position)

        def step(self, *args, **kwargs):
            if self.in_dialogue and forced_agent_audio is not None:
                if self.dialogue_index >= len(forced_agent_audio[0]):
                    raise ValueError("forced agent audio shorter than user input")
                kwargs["moshi_tokens"] = torch.tensor(
                    [[row[self.dialogue_index] for row in forced_agent_audio]],
                    dtype=torch.long, device=self.lm_model.device,
                ).unsqueeze(-1)
            if self.in_dialogue and forced_text is not None:
                if self.dialogue_index >= len(forced_text):
                    raise ValueError("forced text shorter than user input")
                kwargs["text_token"] = torch.tensor([forced_text[self.dialogue_index]], device=self.lm_model.device)
            value = super().step(*args, **kwargs)
            if self.in_dialogue:
                if value is not None:
                    returned.append({"input_frame": self.dialogue_index,
                                     "tokens": value.detach().cpu().flatten().tolist()})
                self.dialogue_index += 1
            return value

    lm.LMGen = ObservedLMGen
    try:
        yield
        completed = True
    finally:
        lm.LMGen = original
        write_json(Path(output_dir) / "tokens.json", {"diagnostic_forced_text": forced_text is not None,
                                                     "diagnostic_forced_agent_audio": forced_agent_audio is not None,
                                                     "steps": steps, "returned": returned,
                                                     "returned_frames": len(returned)})
        real = [r for r in text_frames if r["gt_target"] is not None and r["gt_target"] not in (0, 3)]
        gt = [r for r in text_frames if r["gt_target"] is not None]
        summary = {
            "logit_frames": len(text_frames), "provided_gt_frames": len(gt),
            "nonpadding_gt_frames": len(real),
            "prediction_pad_fraction": sum(r["prediction"] in (0, 3) for r in text_frames) / len(text_frames) if text_frames else None,
            "accuracy_on_nonpadding_gt": sum(r["prediction"] == r["gt_target"] for r in real) / len(real) if real else None,
            "ce_on_nonpadding_gt": -sum(r["target_log_probability"] for r in real) / len(real) if real else None,
            "pad_prediction_on_nonpadding_gt": sum(r["prediction"] in (0, 3) for r in real) / len(real) if real else None,
            "first_nonpadding_gt_failure": next((r for r in real if r["prediction"] != r["gt_target"]), None),
        }
        write_json(Path(output_dir) / "text_logits.json", {
            "completed": completed, "summary": summary, "frames": text_frames,
            "scope": "raw native text prediction before GT forcing; targets come from delayed provided cache",
            "diagnostic_forced_text": forced_text is not None,
            "diagnostic_forced_agent_audio": forced_agent_audio is not None,
            "NO_CUDA_GRAPH": os.environ.get("NO_CUDA_GRAPH"),
        })


def generate_observed(config, sample, out, *, runtime=None, adapter=None, forced_text=None,
                      forced_agent_audio=None):
    """Use actual in-training or standalone production entrypoint, with observation."""
    from personaplex_finetuning.inference import (generate, generate_text_with_runtime, write_generated_text,
                                                 export_original_audio_window, text_error_metrics)
    from personaplex_finetuning.text_normalization import normalize_vietnamese_text
    from validation_generation import greedy_settings, verify_argmax
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with capture_generation(out, forced_text=forced_text, forced_agent_audio=forced_agent_audio), verify_argmax() as checks:
        if runtime is None:
            generate(config, sample, out / "generated.wav", out / "generated.txt", adapter,
                     generation=greedy_settings(), seed=config.seed)
            text = (out / "generated.txt").read_text()
        else:
            text = generate_text_with_runtime(runtime, sample, generation=greedy_settings(), seed=config.seed,
                                              output_wav=out / "generated.wav")
            write_generated_text(out / "generated.txt", text, config.vietnamese_text_mode)
    export_original_audio_window(sample, out / "source.wav")
    reference = " ".join(normalize_vietnamese_text(w.word, config.vietnamese_text_mode) for w in sample.words
                         if w.speaker == "agent" and sample.window_start_sec <= w.start < sample.window_end_sec)
    (out / "reference.txt").write_text(reference)
    result = {"sample_id": sample.sample_id, "window": [sample.window_start_sec, sample.window_end_sec],
              "diagnostic_forced_text": forced_text is not None, "argmax_checks": checks,
              "diagnostic_forced_agent_audio": forced_agent_audio is not None,
              "native_text_predictions": json.loads((out / "text_logits.json").read_text())["summary"],
              "errors": text_error_metrics(reference, text, config.vietnamese_text_mode)}
    write_json(out / "generation.json", result)
    return result


def tiny_probe(device, reference, out, source_state="worktree"):
    """Real native model, synthetic 17-stream data; never a 7B quality claim."""
    import torch
    from moshi.models import loaders
    from moshi.models import lm
    from personaplex_finetuning.config import Config
    from personaplex_finetuning.sequence import TrainingExample
    from personaplex_finetuning.lora import inject_lora, load_adapter, adapter_state_dict
    from safetensors.torch import save_file
    from personaplex_finetuning.train import one_step
    from validate_one_sample import parity
    require_device(device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    kwargs = loaders._lm_kwargs.copy()
    kwargs.update(dim=32, text_card=128, card=64, num_heads=2, num_layers=2, hidden_scale=2,
                  depformer_dim=16, depformer_dim_feedforward=32, depformer_num_heads=2,
                  depformer_num_layers=1, dep_q=16)
    model = loaders.LMModel(device=device, dtype=torch.float32, **kwargs).to(device).eval()
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    codes = torch.randint(1, 63, (1, 17, 24), device=device)
    codes[:, 0] %= 128
    example = TrainingExample(tuple(tuple(row) for row in codes[0].tolist()),
                              tuple(tuple(row) for row in codes[0].tolist()),
                              tuple((False,) * 4 + (True,) * 20 for _ in range(17)),
                              ("text",) + tuple(f"audio_{k}" for k in range(16)), 4, 20)
    config = Config(out / "synthetic.yaml", out, Path(__file__).resolve().parents[1] / "src", out, out,
                    device=device, lora_rank=4, lora_alpha=8)
    tokenizer = SimpleNamespace(padding_id=3, end_padding_id=0)
    runtime = SimpleNamespace(model=model, tokenizer=tokenizer)
    args = SimpleNamespace(atol=1e-5, rtol=1e-4, adapter=None)
    with torch.no_grad(), layer_outputs(model) as base_layers:
        base = model.forward_train(codes)
    base_streaming, _ = parity(runtime, codes, args)
    if source_state == "head":
        import subprocess
        project = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(["git", "-C", str(project), "show", "HEAD:src/personaplex_finetuning/lora.py"], text=True)
        namespace = {}
        exec(compile(source, "HEAD:src/personaplex_finetuning/lora.py", "exec"), namespace)
        inject_lora, load_adapter, adapter_state_dict = (namespace[key] for key in ("inject_lora", "load_adapter", "adapter_state_dict"))
    targets = inject_lora(model, 4, 8, prefixes=("transformer", "depformer"))
    with torch.no_grad():
        zero = model.forward_train(codes)
    zero_streaming, _ = parity(runtime, codes, args)
    zero_report = output_comparison(base, zero)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003)
    history = []
    for _ in range(3):
        total, components, norm = one_step(config, runtime, example, optimizer)
        history.append({"loss": float(total), "grad_norm": norm})
    gradient_report = {name: float(p.grad.norm()) if p.grad is not None else None
                       for name, p in model.named_parameters() if p.requires_grad}
    frozen_unchanged = all(torch.equal(p, frozen[name]) and p.grad is None for name, p in model.named_parameters() if not p.requires_grad)
    with torch.no_grad():
        trained = model.forward_train(codes)
    delta = output_comparison(zero, trained)
    streaming, _ = parity(runtime, codes, args)
    adapter = out / "lora.safetensors"
    save_file(adapter_state_dict(model), str(adapter))
    save_output(out / "expected.safetensors", trained)
    write_json(out / "greedy_tokens.json", tiny_generation(model, codes))
    torch.save({"base_state": base_state, "kwargs": kwargs, "codes": codes.cpu(),
                "source_state": source_state}, out / "reload.pt")
    # Exercise actual reference LM source with shared native dependencies held fixed.
    ref_path = Path(reference) / "personaplex/moshi/moshi/models/lm.py"
    spec = importlib.util.spec_from_file_location("moshi.models.audit_reference_lm", ref_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ref_model = module.LMModel(device=device, dtype=torch.float32, **kwargs).to(device).eval()
    ref_model.load_state_dict(base_state)
    with torch.no_grad(), layer_outputs(ref_model) as ref_layers:
        ref_output = ref_model.forward_train(codes)
    reference_report = output_comparison(base, ref_output)
    layer_report = {name: tensor_comparison(value, ref_layers[name], torch.ones(value.shape[:-1], dtype=torch.bool))
                    for name, value in base_layers.items()}
    ref_runtime = SimpleNamespace(model=ref_model, tokenizer=tokenizer)
    reference_streaming, _ = parity(ref_runtime, codes, args)
    loss_path = Path(reference) / "moshi-finetune/finetune/loss.py"
    loss_spec = importlib.util.spec_from_file_location("audit_reference_loss", loss_path)
    loss_module = importlib.util.module_from_spec(loss_spec)
    loss_spec.loader.exec_module(loss_module)
    ref_text = loss_module.compute_loss_with_mask(ref_output.text_logits, codes[:, :1], ref_output.text_mask,
                     "text", text_padding_weight=config.text_padding_weight, text_padding_ids={0, 3}, prompt_lengths=[4])
    ref_audio = loss_module.compute_loss_with_mask(ref_output.logits, codes[:, 1:], ref_output.mask,
                     "audio", first_codebook_weight_multiplier=1, prompt_lengths=[4], return_per_codebook=True)
    local_metrics = output_metrics(config, SimpleNamespace(tokenizer=tokenizer), example, base)
    import os
    report = {"scope": "native miniature model; synthetic inputs; no Mimi or 7B quality claim",
              "device": device, "torch": torch.__version__, "source_state": source_state,
              "mps_operator_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
              "targets": targets, "zero_init": zero_report, "adapter_effect": delta,
              "frozen_unchanged": frozen_unchanged, "gradients": gradient_report, "training": history,
              "streaming_base": base_streaming, "streaming_zero_lora": zero_streaming,
              "streaming": streaming, "reference_lm": reference_report,
              "reference_layers": layer_report, "reference_streaming": reference_streaming,
              "ring_boundary": ring_boundary_probe(),
              "objective_same_logits": {"local": local_metrics,
                  "reference": {"text_loss": float(ref_text), "audio_loss": float(ref_audio[0]),
                                "per_codebook_ce": ref_audio[1]},
                  "difference": "local agent-only and acoustic 0.02; reference all 16 codebooks without acoustic downweight",
                  "reference_loss_sha256": digest(loss_path)},
              "reference_source": {"file": str(ref_path), "sha256": digest(ref_path),
                                   "scope": "reference LM source with local native dependencies held fixed"}}
    write_json(out / "mini.json", report)
    return report


def tiny_reload(out, device):
    """Called only in a fresh child after the miniature baseline process exits."""
    import torch
    from moshi.models import loaders
    from personaplex_finetuning.lora import inject_lora, load_adapter
    require_device(device)
    torch.set_num_threads(1)
    out = Path(out)
    state = torch.load(out / "reload.pt", map_location="cpu", weights_only=True)
    if state["source_state"] == "head":
        import subprocess
        source = subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]),
                                          "show", "HEAD:src/personaplex_finetuning/lora.py"], text=True)
        namespace = {}
        exec(compile(source, "HEAD:src/personaplex_finetuning/lora.py", "exec"), namespace)
        inject_lora, load_adapter = namespace["inject_lora"], namespace["load_adapter"]
    model = loaders.LMModel(device=device, dtype=torch.float32, **state["kwargs"]).to(device).eval()
    model.load_state_dict(state["base_state"])
    inject_lora(model, 4, 8, prefixes=("transformer", "depformer"))
    load_adapter(model, out / "lora.safetensors")
    with torch.no_grad():
        output = model.forward_train(state["codes"].to(device))
    report = output_comparison(load_output(out / "expected.safetensors"), output)
    import json
    expected_tokens = json.loads((out / "greedy_tokens.json").read_text())
    actual_tokens = tiny_generation(model, state["codes"].to(device))
    report["greedy_tokens_equal"] = expected_tokens == actual_tokens
    report["greedy_frames"] = len(actual_tokens)
    write_json(out / "reload.json", report)
    return report
