"""Read-only audit calculations. No production behavior is replaced here."""
from __future__ import annotations

import ast
from collections import deque
from dataclasses import asdict
from functools import reduce
import hashlib
import json
import math
from pathlib import Path
import subprocess


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n")


def git_identity(root):
    def run(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"root": str(Path(root).resolve()), "commit": run("rev-parse", "HEAD"),
            "status": run("status", "--short"), "diff": run("diff", "--", "src")}


def load_run(root):
    """Canonical log names or one unambiguous Finder-exported '(N)' copy."""
    root = Path(root)
    def read(stem, suffix, required=False):
        canonical = root / (stem + suffix)
        matches = [canonical] if canonical.is_file() else sorted(root.glob(f"{stem} (*){suffix}"))
        if len(matches) > 1:
            raise ValueError(f"ambiguous {stem} logs in {root}: {matches}")
        if not matches:
            if required:
                raise FileNotFoundError(canonical)
            return [] if suffix == ".jsonl" else {}
        text = matches[0].read_text()
        return [json.loads(row) for row in text.splitlines() if row.strip()] if suffix == ".jsonl" else json.loads(text)
    return {"path": str(root.resolve()), "config": read("config", ".json", True),
            "metrics": read("metrics", ".jsonl", True),
            "generation": read("free_running_metrics", ".jsonl"), "run": read("run", ".json")}


def means(rows):
    keys = ("loss/total", "loss/text_nonpadding", "loss/audio_cb0", "accuracy/text_nonpad", "accuracy/audio_cb0")
    return {key: sum(values) / len(values) for key in keys
            if (values := [r[key] for r in rows if isinstance(r.get(key), (int, float))])}


def summarize_run(run):
    cfg, rows, generation = run["config"], run["metrics"], run["generation"]
    contract = cfg.get("training_contract", {})
    swap = cfg.get("swap_roles_after_pass", contract.get("swap_roles_after_pass", False))
    role_rows = {"left-agent": [], "right-agent": [], "unknown": []}
    for row in rows:
        role = "unknown" if "epoch" not in row else "right-agent" if swap and row["epoch"] % 2 else "left-agent"
        role_rows[role].append(row)
    epochs = sorted({r["epoch"] for r in rows if "epoch" in r})
    return {"path": run.get("path"), "metric_rows": len(rows),
            "role_loss_means": {role: means(values) for role, values in role_rows.items() if values},
            "epoch_loss_means": [{"epoch": epoch, "role": "right-agent" if swap and epoch % 2 else "left-agent",
                                  "rows": len(values), **means(values)} for epoch in epochs
                                 if (values := [r for r in rows if r.get("epoch") == epoch])],
            "step_blocks": [{"first_step": values[0]["step"], "last_step": values[-1]["step"], **means(values)}
                            for offset in range(0, len(rows), 100) if (values := rows[offset:offset + 100])],
            "generation": [{"step": r["step"], "cer": r.get("val/generation_cer"),
                            "wer": r.get("val/generation_wer"), "samples": r.get("val/generation_samples"),
                            "empty_samples": r.get("val/generation_empty_samples")} for r in generation],
            "evaluated_windows": [{"step": r["step"], "sample_id": s.get("sample_id"),
                                   "start": s.get("window_start_sec"), "end": s.get("window_end_sec"),
                                   "cer": s.get("cer"), "wer": s.get("wer")}
                                  for r in generation for s in r.get("samples", [])],
            "checkpoint_reload": "recorded_inspect_results" if run["run"].get("reload_checks") else "not_recorded",
            "same_sample_teacher_forced_verified": False,
            "exact_per_chunk_updates": "unavailable_without_step_identities",
            "limitations": ["Step loss may describe another sample or the opposite role from generation.",
                            "A decreasing aggregate loss is not evidence of autoregressive memorization.",
                            "Current YAML and source do not establish historical server provenance."]}


def compare_runs(control, failing):
    a, b = control["config"], failing["config"]
    # Include all recorded settings; do not silently equate nested/flat configs.
    differences = {key: {"control": a.get(key), "failing": b.get(key)} for key in sorted(set(a) | set(b))
                   if a.get(key) != b.get(key)}
    def windows(run):
        return {(s.get("sample_id"), s.get("window_start_sec"), s.get("window_end_sec"))
                for row in run["generation"] for s in row.get("samples", [])}
    return {"control": summarize_run(control), "failing": summarize_run(failing),
            "config_differences": differences,
            "shared_evaluated_windows": sorted(windows(control) & windows(failing), key=str),
            "causal_status": "not_established; compare hashes, same-window metrics and controlled ablations"}


def replay_updates(config, chunks):
    """Replay the actual sampler, conditional on single-GPU fresh-run provenance.

    This is an estimate until its sequence is matched against observed server
    sample identities. A chunk key includes its crop and logical role.
    """
    from personaplex_finetuning.batching import RankStrideBatchSampler
    sampler = RankStrideBatchSampler(len(chunks), config.per_device_batch_size, 0, 1,
                                    seed=config.seed, shuffle=config.shuffle)
    if not len(sampler):
        raise ValueError("no complete training batches")
    counts = {}
    micro_steps = config.max_steps * config.gradient_accumulation_steps
    epoch, consumed = 0, 0
    while consumed < micro_steps:
        sampler.set_epoch(epoch)
        role = "right-agent" if config.swap_roles_after_pass and epoch % 2 else "left-agent"
        for indices in sampler:
            if consumed >= micro_steps:
                break
            for index in indices:
                s = chunks[index]
                key = (s.sample_id, s.window_start_sec, s.window_end_sec, role)
                counts[key] = counts.get(key, 0) + 1
            consumed += 1
        epoch += 1
    return {"status": "conditional_replay_not_observed", "assumptions": ["single GPU", "fresh step zero", "current sampler source"],
            "micro_steps": consumed,
            "chunks": [{"sample_id": key[0], "start": key[1], "end": key[2], "role": key[3], "updates": value}
                       for key, value in sorted(counts.items())]}


def load_reference_alignment(reference):
    """Compile the verbatim Interleaver/tokenize nodes, without importing its trainer.

    Local serving Moshi lacks moshi.conditioners. Extracting these self-contained
    nodes avoids mocking that dependency or modifying the reference. Hash and
    extraction scope are reported; this does NOT validate its full runtime.
    """
    import sentencepiece
    import torch
    path = Path(reference) / "moshi-finetune/finetune/data/interleaver.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in {"Interleaver", "tokenize"}]
    if {node.name for node in selected} != {"Interleaver", "tokenize"}:
        raise ValueError("reference Interleaver/tokenize not found")
    namespace = dict(sentencepiece=sentencepiece, torch=torch, math=math, deque=deque, reduce=reduce,
                     Alignment=tuple, TokenizedAlignment=tuple)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Interleaver"], {"file": str(path.resolve()), "sha256": digest(path),
                                      "scope": "verbatim AST Interleaver and tokenize; trainer dependencies not exercised"}


def compare_alignment(sample, tokenizer, frames, frame_rate, text_mode, reference):
    """Tag occurrences and execute actual algorithms twice to locate token loss.

    Tags distinguish repeated words/token IDs. No placement algorithm is copied.
    Reference trace is checked against its untagged output, including padding.
    """
    from personaplex_finetuning.sequence import align_dialogue_text_targets
    from personaplex_finetuning.text_normalization import normalize_vietnamese_text
    cls, source = load_reference_alignment(reference)
    words = [w for w in sample.words if w.speaker == "agent"
             and sample.window_start_sec <= w.start < sample.window_end_sec]
    rows, tagged_words, token_by_tag = [], [], {}
    for wi, word in enumerate(words):
        normalized = normalize_vietnamese_text(word.word, text_mode)
        tags = []
        for si, token in enumerate(tokenizer.encode(normalized)):
            tag = 1_000_000 + len(rows)
            tags.append(tag)
            token_by_tag[tag] = token
            rows.append({"word_index": wi, "subword_index": si, "word": word.word,
                         "normalized_word": normalized, "token_id": token, "tag": tag,
                         "start_frame": int((word.start - sample.window_start_sec) * frame_rate)})
        tagged_words.append((word, tags))

    class TaggedTokenizer:
        padding_id, end_padding_id = tokenizer.padding_id, tokenizer.end_padding_id
        def __init__(self):
            self.index = 0
        def encode(self, text):
            tags = tagged_words[self.index][1]
            self.index += 1
            return tags

    local = align_dialogue_text_targets(sample, frames, frame_rate, tokenizer, text_mode)
    tagged = align_dialogue_text_targets(sample, frames, frame_rate, TaggedTokenizer(), text_mode)
    interleaver = cls(tokenizer=tokenizer._processor, audio_frame_rate=frame_rate,
                      text_padding=tokenizer.padding_id, end_of_text_padding=tokenizer.end_padding_id,
                      zero_padding=-1, keep_main_only=True, main_speaker_label="SPEAKER_BROKER", device="cpu")
    raw = [(normalize_vietnamese_text(w.word, text_mode),
            (w.start - sample.window_start_sec, w.end - sample.window_start_sec), "SPEAKER_BROKER") for w in words]
    duration = frames / frame_rate
    reference_tokens = interleaver.prepare_item(raw, duration).flatten().tolist()
    ref_tagged = [(tags, (w.start - sample.window_start_sec, w.end - sample.window_start_sec), "SPEAKER_BROKER")
                  for w, tags in sorted(tagged_words, key=lambda pair: pair[0].start) if w.start < w.end]
    reference_tags = interleaver.build_token_stream(ref_tagged, duration).flatten().tolist()
    for label, actual, trace in (("local", list(local.tokens), tagged.tokens),
                                  ("reference", reference_tokens, reference_tags)):
        if list(actual) != [token_by_tag.get(t, t) for t in trace]:
            raise AssertionError(f"{label} occurrence trace differs from actual tokens")
    local_pos = {t: i for i, t in enumerate(tagged.tokens) if t in token_by_tag}
    ref_pos = {t: i for i, t in enumerate(reference_tags) if t in token_by_tag}
    for row in rows:
        tag = row.pop("tag")
        row.update(local_frame=local_pos.get(tag), reference_frame=ref_pos.get(tag))
        row["local_shift_frames"] = None if tag not in local_pos else local_pos[tag] - row["start_frame"]
        row["reference_shift_frames"] = None if tag not in ref_pos else ref_pos[tag] - row["start_frame"]
    filtered = sum(len(tags) for w, tags in tagged_words if w.start >= w.end)
    return {"sample_id": sample.sample_id, "window": [sample.window_start_sec, sample.window_end_sec],
            "text_mode": text_mode, "frames": frames, "reference_source": source,
            "local": {"requested_tokens": local.required_tokens, "placed_tokens": local.placed_tokens,
                      "lost_tokens": local.required_tokens - local.placed_tokens,
                      "overflow_word": asdict(local.overflow_word) if local.overflow_word else None,
                      "tokens": local.tokens},
            "reference": {"requested_tokens": len(rows), "placed_tokens": len(ref_pos),
                          "lost_tokens": len(rows) - len(ref_pos), "filtered_tokens": filtered,
                          "queue_or_boundary_lost_tokens": len(rows) - len(ref_pos) - filtered,
                          "tokens": reference_tokens},
            "frame_mismatches": sum(a != b for a, b in zip(local.tokens, reference_tokens)) + abs(len(local.tokens) - len(reference_tokens)),
            "occurrences": rows}


def compare_voice_encoding(sample, runtime, reference):
    """Actual reference voice method; only its hardcoded device is relocated."""
    import os
    import sphn
    import torch
    from types import SimpleNamespace
    from moshi.models.lm import LMGen
    path = Path(reference) / "moshi-finetune/finetune/data/interleaver.py"
    tree = ast.parse(path.read_text())
    method = next(n for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "InterleavedTokenizer"
                  for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_encode_voice_prompt")
    relocated = 0
    for node in ast.walk(method):
        if isinstance(node, ast.keyword) and node.arg == "device" and isinstance(node.value, ast.Constant) and node.value.value == "cuda":
            node.value = ast.Constant(runtime.codec.device)
            relocated += 1
    if relocated != 1:
        raise ValueError("reference voice method device layout changed; review the probe")
    namespace = dict(os=os, sphn=sphn, torch=torch)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    codec = runtime.codec
    with torch.inference_mode():
        local = torch.tensor(codec.encode_voice_prompt(sample.voice_prompt_wav))
        native = LMGen.__new__(LMGen)
        native._sample_rate = codec.sample_rate
        native._frame_size = int(codec.sample_rate / codec.frame_rate)
        native.load_voice_prompt(str(sample.voice_prompt_wav))
        with codec.mimi.streaming(1):
            native_codes = torch.cat(list(native._encode_voice_prompt_frames(codec.mimi)), dim=2)[0].cpu()
        ref_self = SimpleNamespace(mimi=codec.mimi, _voice_prompt_cache={})
        ref_codes = namespace["_encode_voice_prompt"](ref_self, str(sample.voice_prompt_wav.resolve()),
                                                     str(sample.conversation_wav.resolve()))[0].cpu()
    overlap = min(local.shape[-1], ref_codes.shape[-1])
    return {"local_native_equal": bool(torch.equal(local, native_codes)),
            "local_shape": list(local.shape), "reference_shape": list(ref_codes.shape),
            "reference_overlap_mismatches": int((local[:, :overlap] != ref_codes[:, :overlap]).sum()),
            "local_tokens": local.tolist(), "reference_tokens": ref_codes.tolist(),
            "reference_source_sha256": digest(path), "reference_device": codec.device,
            "reference_device_relocated": codec.device != "cuda",
            "scope": "same Mimi weights; local normalized streaming vs reference raw batch method"}
