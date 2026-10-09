"""PersonaPlex LoRA and full-parameter training with single-GPU or DDP execution."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import random
import shlex
import sys
import time
import traceback
import math
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm.auto import tqdm

from .config import Config, load_config
from .data import (
    PreparedDataset,
    conversation_group_keys,
    duration_chunks,
    limit_conversations,
    samples_for_role_pass,
)
from .batching import (
    RankStrideBatchSampler, RawAudioDataset, collate_raw_audio, post_encode_collate,
)
from .chunk_filter import ChunkFilterResult, expected_mimi_frames, filter_text_capacity_chunks
from .filter_cache import filter_fingerprint
from .lora import adapter_state_dict, inject_lora, load_adapter
from .full_checkpoint import load_full_weights, resolve_full_checkpoint, save_full_weights, write_full_metadata
from .generation import GenerationSettings
from .inference import (
    export_original_audio_window,
    generate_text_with_runtime,
    resolve_adapter_checkpoint,
    text_error_metrics,
)
from .objective import (
    normalize_text_padding_ids,
    stream_weights_torch,
    text_padding_mask_torch,
    torch_weighted_cross_entropy_stats,
)
from .runtime import PERSONAPLEX_MIMI_FRAME_RATE, MimiCodec, RuntimePaths, SentencePieceTokenizer, load_runtime
from .sequence import PersonaPlexTrainingExampleBuilder, pad_training_example
from .text_normalization import normalize_vietnamese_text


def limit_cpu_threads(torch_module) -> int:
    """Keep training processes from consuming every CPU core."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    torch_module.set_num_threads(1)
    torch_module.set_num_interop_threads(1)
    return 1


def seed_everything(seed: int, torch_module) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def resolve_training_device(requested: str, local_rank: int, world_size: int) -> torch.device:
    device = torch.device(requested)
    if world_size > 1:
        if device.type != "cuda":
            raise RuntimeError("multi-process training requires CUDA")
        return torch.device(f"cuda:{local_rank}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS device requested but unavailable to this process")
    return device


def device_memory_bytes(device: torch.device) -> int:
    """Report allocated accelerator memory; MPS exposes driver allocation."""
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    return 0


def create_run_dir(output_root: Path, smoke: bool) -> Path:
    """Create a unique, timestamped run directory below the configured root."""
    label = "smoke" if smoke else "train"
    run_dir = output_root / f"{label}_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_example(config: Config, sample, runtime, dialogue_codes=None):
    pause_frames = getattr(getattr(config, "generation_settings", None), "audio_silence_frame_cnt", 6)
    builder = PersonaPlexTrainingExampleBuilder(
        runtime.codec, runtime.tokenizer, runtime.initial_tokens, runtime.zero_token,
        pause_frames=pause_frames,
        vietnamese_text_mode=config.vietnamese_text_mode,
    )
    # PersonaPlex LMModel.forward_train applies its native per-stream delays.
    example = builder.build(sample, dialogue_codes=dialogue_codes)
    expected_frames = expected_mimi_frames(config.duration_sec, runtime.codec.frame_rate)
    if example.dialogue_frames > expected_frames:
        raise RuntimeError(
            f"{sample.sample_id}: Mimi returned {example.dialogue_frames} frames for a "
            f"{config.duration_sec:g}s chunk; maximum is {expected_frames}"
        )
    return example


def dataset_load_summary(label: str, report) -> str:
    return (
        f"[Dataset load] split={label} manifest_entries={report.manifest_entries} "
        f"loaded={report.loaded_samples} skipped_out_of_bounds_samples={report.skipped_out_of_bounds} "
        f"skipped_invalid_samples={report.skipped_invalid}"
    )


def chunk_filter_payload(label: str, candidate_count: int, result: ChunkFilterResult, max_kept: int | None = None) -> dict:
    return {
        "split": label,
        "candidate_chunks": candidate_count,
        "scanned_chunks": len(result.kept) + len(result.rejected),
        "requested_valid_chunks": max_kept,
        "quota_shortfall": max(0, max_kept - len(result.kept)) if max_kept is not None else 0,
        "kept_chunks": len(result.kept),
        "skipped_out_of_bounds_chunks": result.skipped_out_of_bounds,
        "skipped_text_overflow_chunks": result.skipped_text_overflow,
        "rejected": [
            {
                "sample_id": item.sample_id,
                "window_start_sec": item.window_start_sec,
                "window_end_sec": item.window_end_sec,
                "reason": item.reason,
                "roles": list(item.roles),
                "word": item.word.word,
                "word_start_sec": item.word.start,
            }
            for item in result.rejected
        ],
    }


def chunk_filter_summary(payload: dict) -> str:
    return (
        f"[Chunk filter] split={payload['split']} candidates={payload['candidate_chunks']} "
        f"kept={payload['kept_chunks']} "
        f"scanned={payload.get('scanned_chunks', payload['candidate_chunks'])} "
        f"requested={payload.get('requested_valid_chunks')} shortfall={payload.get('quota_shortfall', 0)} "
        f"skipped_out_of_bounds_chunks={payload['skipped_out_of_bounds_chunks']} "
        f"skipped_text_overflow_chunks={payload['skipped_text_overflow_chunks']}"
    )


def effective_global_batch_size(per_process_batch_size: int, num_processes: int, gradient_accumulation_steps: int) -> int:
    """Return the number of samples contributing to one optimizer update."""
    if min(per_process_batch_size, num_processes, gradient_accumulation_steps) < 1:
        raise ValueError("batch size, process count, and accumulation steps must all be positive")
    return per_process_batch_size * num_processes * gradient_accumulation_steps


def validate_resume_step(start_step: int, max_steps: int) -> None:
    if start_step > max_steps:
        raise RuntimeError(f"resume checkpoint step {start_step} exceeds train.max_steps={max_steps}")
    if start_step == max_steps:
        raise RuntimeError(
            f"resume checkpoint is already complete at train.max_steps={max_steps}; "
            "resume from an earlier checkpoint or start a new run with a new schedule"
        )


def validate_resume_checkpoint(
    config: Config, resume_from: str, prefixes: tuple[str, ...], train_conversations: list,
    train_chunks: list | None = None,
) -> tuple[Path, int]:
    """Require a complete checkpoint compatible with this exact training run."""
    if config.train_method == "full":
        adapter, metadata = resolve_full_checkpoint(Path(resume_from))
        metadata_file = adapter.parent / "checkpoint.json"
        model_root = Path(metadata["base_model_root"]).expanduser().resolve()
    else:
        adapter, rank, alpha, model_root, adapter_prefixes = resolve_adapter_checkpoint(Path(resume_from))
        metadata_file = adapter.parent / "adapter.json"
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    step = metadata.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"resume checkpoint has invalid optimizer step: {metadata_file}")
    if not (adapter.parent / "training_state.pt").is_file():
        raise RuntimeError(
            f"resume requires training_state.pt alongside {adapter}; "
            "weights alone cannot restore optimizer, scheduler, or data position"
        )
    if config.train_method == "lora" and (rank != config.lora_rank or alpha != config.lora_alpha or adapter_prefixes != prefixes):
        raise RuntimeError(
            f"resume LoRA configuration differs: checkpoint rank={rank}, alpha={alpha}, "
            f"prefixes={adapter_prefixes}; current rank={config.lora_rank}, "
            f"alpha={config.lora_alpha}, prefixes={prefixes}"
        )
    if model_root is not None and model_root != Path(config.model_root).expanduser().resolve():
        raise RuntimeError(f"resume base model differs: checkpoint={model_root}, current={config.model_root}")
    run_config_file = adapter.parent.parent.parent / "config.json"
    if not run_config_file.is_file():
        raise RuntimeError(f"resume checkpoint has no run config to verify training data/objective: {run_config_file}")
    saved_run = json.loads(run_config_file.read_text(encoding="utf-8"))
    expected_contract = training_contract(config, train_conversations, train_chunks)
    saved_contract = saved_run.get("training_contract")
    if saved_contract != expected_contract:
        differing = sorted(
            key for key in set(expected_contract) | set(saved_contract or {})
            if expected_contract.get(key) != (saved_contract or {}).get(key)
        )
        raise RuntimeError(
            f"resume training data/objective differs or cannot be verified: {differing}; "
            f"source={run_config_file}"
        )
    return adapter, step


def training_contract(config: Config, train_conversations: list, train_chunks: list | None = None) -> dict:
    """Inputs that must remain fixed to resume the same batch and loss trajectory."""
    manifest = Path(config.manifest).expanduser().resolve()
    digest = hashlib.sha256()
    with manifest.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    source_digest = hashlib.sha256()
    for sample in train_conversations:
        def file_stamp(path):
            if path is None:
                return None
            resolved = Path(path).expanduser().resolve()
            stat = resolved.stat()
            return [str(resolved), stat.st_size, stat.st_mtime_ns]

        item = {
            "sample_id": sample.sample_id,
            "conversation": file_stamp(sample.conversation_wav),
            "voice_prompt_left": file_stamp(sample.voice_prompt_wav),
            "voice_prompt_right": file_stamp(sample.voice_prompt_right_wav),
            "text_prompt_left": sample.text_prompt,
            "text_prompt_right": sample.text_prompt_right,
            "metadata": sample.metadata,
            "audio_duration_sec": sample.audio.duration_sec,
            "agent_channel": sample.agent_channel,
            "user_channel": sample.user_channel,
            "words": [(word.speaker, word.word, word.start, word.end) for word in sample.words],
        }
        source_digest.update(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        source_digest.update(b"\n")
    chunk_items = [
        (sample.sample_id, str(Path(sample.conversation_wav).expanduser().resolve()),
         sample.window_start_sec, sample.window_end_sec)
        for sample in (train_chunks or [])
    ]
    chunk_digest = hashlib.sha256(
        json.dumps(chunk_items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    contract = {
        "manifest": str(manifest), "manifest_sha256": digest.hexdigest(),
        "prepared_sources_sha256": source_digest.hexdigest(),
        "text_capacity_filter_version": 1,
        "sample_number_contract": "valid-chunks-v1",
        "mimi_encoding_contract": MimiCodec.conversation_encoding_contract,
        "kept_train_chunks_sha256": chunk_digest,
        "kept_train_chunk_count": len(chunk_items),
        "seed": config.seed, "duration_sec": config.duration_sec,
        "sample_number": config.sample_number, "shuffle": config.shuffle,
        "val_ratio": config.val_ratio,
        "val_manifest": str(config.val_manifest) if config.val_manifest else None,
        "eval_on_train_samples": config.eval_on_train_samples,
        "no_eval": config.no_eval,
        "swap_roles_after_pass": config.swap_roles_after_pass,
        "prompt_aug_prob": config.prompt_aug_prob,
        "vietnamese_text_mode": config.vietnamese_text_mode,
        "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
        "depformer_learning_rate": config.depformer_learning_rate,
        "pct_start": config.pct_start,
        "first_codebook_weight_multiplier": config.first_codebook_weight_multiplier,
        "text_padding_weight": config.text_padding_weight,
        "epad_as_padding": config.epad_as_padding,
        "user_loss": config.user_loss,
        "train_stage": config.train_stage,
        "qlora": config.qlora, "quant_type": config.quant_type if config.qlora else None,
        "gradient_checkpointing": config.gradient_checkpointing,
    }
    if config.train_method == "full":
        contract["train_method"] = "full"
    return contract


def rank_stride_indices(sample_count: int, rank: int, world_size: int) -> list[int]:
    if sample_count < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid sample count or distributed rank")
    return list(range(rank, sample_count, world_size))


def lora_prefixes_for_stage(config: Config) -> tuple[str, ...]:
    """Select the same LoRA target modules for training and adapter reload."""
    stage = getattr(config, "train_stage", "joint").lower()
    if stage in {"temporal_only", "freeze_depformer"}:
        return ("transformer",)
    if stage in {"depth_only", "freeze_tempformer"}:
        return ("depformer",)
    return ("transformer", "depformer")


def configure_trainable_parameters(model, config: Config, prefixes: tuple[str, ...]) -> list[str]:
    if config.train_method == "full":
        model.requires_grad_(True)
        if hasattr(model, "dep_q"):
            # The loss supervises agent codebooks 0..7. User codebooks 8..15
            # are conditioning only; their output-only parameters must not be
            # moved by AdamW weight decay. The main audio embeddings remain
            # trainable because both channels feed the shared transformer.
            if model.dep_q != 16 or len(model.linears) != 16 or len(model.depformer_emb) != 15:
                raise ValueError("full fine-tuning requires the 16-codebook PersonaPlex LM")
            if model.depformer_multi_linear:
                if len(model.depformer_in) != 16 or model.depformer_weights_per_step_schedule is not None:
                    raise ValueError("unsupported PersonaPlex depformer input layout for full fine-tuning")
                for layer in model.depformer_in[8:]:
                    layer.requires_grad_(False)
            for layer in model.depformer_emb[7:]:
                layer.requires_grad_(False)
            for layer in model.linears[8:]:
                layer.requires_grad_(False)
            for layer in model.depformer.layers:
                if len(layer.gating) != 16 or layer.self_attn.weights_per_step != 16:
                    raise ValueError("unsupported PersonaPlex depformer layer layout for full fine-tuning")
                for user_gate in layer.gating[8:]:
                    user_gate.requires_grad_(False)
        return []
    if not config.lora_enabled:
        raise ValueError("train.method=lora requires lora.enable=true")
    return inject_lora(model, config.lora_rank, config.lora_alpha, prefixes=prefixes)


def full_attention_no_decay_ids(model) -> set[int]:
    """Packed attention tensors include unsupervised user-step rows."""
    return {
        id(parameter)
        for layer in model.depformer.layers
        for parameter in (layer.self_attn.in_proj_weight, layer.self_attn.out_proj.weight)
    }


def unwrap_parallel_model(model):
    """Return the underlying model so checkpoints use inference-compatible keys."""
    while hasattr(model, "module"):
        model = model.module
    return model


def inspect_training_sample(
    sample, tokenizer, vietnamese_text_mode: str = "diacritics",
) -> dict[str, object]:
    """Validate and describe the actual agent-text target before optimization."""
    if sample.agent_channel != 0 or sample.user_channel != 1:
        raise ValueError(f"{sample.sample_id}: expected LEFT=agent and RIGHT=user")
    source_words = [
        word.word for word in sample.words
        if word.speaker == "agent" and sample.window_start_sec <= word.start < sample.window_end_sec
    ]
    if not source_words:
        raise ValueError(f"{sample.sample_id}: training window has no agent text target")
    target_words = [normalize_vietnamese_text(word, vietnamese_text_mode) for word in source_words]
    text = " ".join(target_words)
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    normalize = lambda value: " ".join(value.split())
    if normalize(decoded) != normalize(text):
        raise ValueError(
            f"{sample.sample_id}: tokenizer round-trip changed agent text; "
            f"source={text!r}, decoded={decoded!r}"
        )
    return {
        "sample_id": sample.sample_id,
        "window_start_sec": sample.window_start_sec,
        "window_end_sec": sample.window_end_sec,
        "agent_channel": sample.agent_channel,
        "user_channel": sample.user_channel,
        "agent_word_count": len(source_words),
        "token_count": len(tokens),
        "text_prompt": sample.text_prompt,
        "agent_text": text,
        "tokenizer_round_trip": decoded,
    }


def text_supervision_counts(batch, model_output, padding_id: int):
    """Count valid non-padding text targets and weighted padding positions."""
    labels = batch["labels"][:, 0, :]
    valid = batch["loss_mask"][:, 0, :] & model_output.text_mask[:, 0]
    is_padding = text_padding_mask_torch(labels, padding_id)
    return ((~is_padding) & valid).sum(), (is_padding & valid).sum()


def text_prediction_diagnostic_counts(batch, model_output, padding_id: int):
    """Count PAD prediction outcomes over the valid, loss-supervised text positions."""
    labels = batch["labels"][:, 0, :]
    valid = batch["loss_mask"][:, 0, :] & model_output.text_mask[:, 0]
    is_padding_target = text_padding_mask_torch(labels, padding_id)
    predictions = model_output.text_logits[:, 0].argmax(dim=-1)
    is_padding_prediction = text_padding_mask_torch(predictions, padding_id)
    nonpadding_target = valid & ~is_padding_target
    padding_target = valid & is_padding_target
    return torch.stack((
        ((predictions == labels) & padding_target).sum(),
        (is_padding_prediction & valid).sum(),
        (is_padding_prediction & nonpadding_target).sum(),
    ))


def pack_text_training_stats(
    target_tokens,
    padding_positions,
    target_ce_sum,
    target_ce_count,
    target_correct,
    prediction_counts,
):
    """Pack the eight scalar text metrics in the order consumed by the logger."""
    if prediction_counts.shape != (3,):
        raise ValueError(
            "text prediction diagnostics must contain three counts; "
            f"got shape {tuple(prediction_counts.shape)}"
        )
    return torch.stack((
        target_tokens.float(),
        padding_positions.float(),
        target_ce_sum,
        target_ce_count.float(),
        target_correct.float(),
        *(count.float() for count in prediction_counts.unbind()),
    ))


def tokenizer_text_padding_ids(tokenizer, include_end_padding: bool = False) -> tuple[int, ...]:
    """Text IDs treated as padding (down-weighted in loss, excluded from "real" text metrics).

    EPAD (end_padding_id) marks a word onset. Free-running generation can only start a
    word after the model itself emits EPAD, so by default EPAD is a full-weight target.
    Audit evidence: with EPAD down-weighted, the model never emitted EPAD and stayed on PAD.
    ``include_end_padding=True`` restores the legacy (and moshi-finetune reference) behavior.
    """
    ids = [int(tokenizer.padding_id)]
    end_padding_id = getattr(tokenizer, "end_padding_id", None)
    if include_end_padding and end_padding_id is not None and int(end_padding_id) not in ids:
        ids.append(int(end_padding_id))
    return normalize_text_padding_ids(ids)


def text_target_token_loss_stats(batch, model_output, padding_id: int):
    """Return diagnostic CE over valid non-padding text tokens only."""
    labels = batch["labels"][:, 0, :]
    valid_targets = (
        batch["loss_mask"][:, 0, :]
        & model_output.text_mask[:, 0]
        & ~text_padding_mask_torch(labels, padding_id)
    )
    with torch.no_grad():
        selected_logits = model_output.text_logits[:, 0][valid_targets].float()
        selected_targets = labels[valid_targets]
        loss_sum = torch.nn.functional.cross_entropy(
            selected_logits, selected_targets, reduction="sum"
        )
    return loss_sum, valid_targets.sum()


def codebook_diagnostic_stats(batch, model_output, padding_id: int):
    """Compute per-codebook loss and argmax accuracy on valid (unmasked) positions.

    Returns:
        text_correct, text_count, text_loss_sum,
        audio_correct (shape 16), audio_count (shape 16), audio_loss_sum (shape 16)
    """
    with torch.no_grad():
        # 1. Text stream
        text_labels = batch["labels"][:, 0, :]
        text_valid = (
            batch["loss_mask"][:, 0, :]
            & model_output.text_mask[:, 0]
            & ~text_padding_mask_torch(text_labels, padding_id)
        )
        if text_valid.any():
            text_logits = model_output.text_logits[:, 0][text_valid].float()
            text_targets = text_labels[text_valid]
            text_preds = text_logits.argmax(dim=-1)
            t_correct = (text_preds == text_targets).sum()
            t_count = text_valid.sum()
            t_loss = torch.nn.functional.cross_entropy(text_logits, text_targets, reduction="sum")
        else:
            t_correct = torch.zeros((), dtype=torch.int64, device=text_labels.device)
            t_count = torch.zeros((), dtype=torch.int64, device=text_labels.device)
            t_loss = torch.zeros((), dtype=torch.float32, device=text_labels.device)

        # Native depformer predicts both eight-codebook speaker streams.
        audio_labels = batch["labels"][:, 1:17, :]
        audio_mask = batch["loss_mask"][:, 1:17, :] & model_output.mask

        cb_correct = []
        cb_count = []
        cb_loss = []
        for i in range(16):
            valid_i = audio_mask[:, i, :]
            if valid_i.any():
                logits_i = model_output.logits[:, i][valid_i].float()  # [N, vocab]
                targets_i = audio_labels[:, i][valid_i]                # [N]
                preds_i = logits_i.argmax(dim=-1)
                cb_correct.append((preds_i == targets_i).sum())
                cb_count.append(valid_i.sum())
                cb_loss.append(torch.nn.functional.cross_entropy(logits_i, targets_i, reduction="sum"))
            else:
                cb_correct.append(torch.zeros((), dtype=torch.int64, device=audio_labels.device))
                cb_count.append(torch.zeros((), dtype=torch.int64, device=audio_labels.device))
                cb_loss.append(torch.zeros((), dtype=torch.float32, device=audio_labels.device))

        return (
            t_correct, t_count, t_loss,
            torch.stack(cb_correct), torch.stack(cb_count), torch.stack(cb_loss)
        )


def step_optimizer_if_ready(sync_state, optimizer, scheduler, trainable, model=None, max_norm=1.0) -> float:
    if not sync_state.sync_gradients:
        return 0.0

    if model is not None and hasattr(model, "clip_grad_norm_"):
        # Distributed wrappers may provide a model-aware global norm operation.
        grad_norm = model.clip_grad_norm_(max_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm)

    grad_norm = float(grad_norm)
    if not math.isfinite(grad_norm):
        raise RuntimeError(f"non-finite gradient norm before optimizer step: {grad_norm}")
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm

def deterministic_crop_rng(seed: int, process_index: int, micro_step: int) -> random.Random:
    """Make dynamic crops reproducible across resume without saving Python RNG state."""
    return random.Random(seed + (process_index * 1_000_003) + micro_step)


def _initialize_audio_worker(_worker_id: int) -> None:
    """Keep each spawned audio decoder worker from oversubscribing CPU threads."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)


def log_batch_phase(rank: int, phase: str, status: str, **details) -> None:
    print(json.dumps({"event": "training_batch_phase", "rank": rank,
                      "phase": phase, "status": status, **details}), flush=True)


@contextmanager
def trace_batch_preparation(rank: int, micro_step: int, enabled: bool):
    """Dump the blocked rank's Python stack before NCCL's 600-second timeout."""
    if not enabled:
        yield
        return
    started = time.monotonic()
    log_batch_phase(rank, "batch_prepare", "start", micro_step=micro_step)
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        yield
    except BaseException as exc:
        log_batch_phase(rank, "batch_prepare", "error", micro_step=micro_step,
                        error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        log_batch_phase(rank, "batch_prepare", "complete", micro_step=micro_step,
                        elapsed_sec=time.monotonic() - started)
    finally:
        faulthandler.cancel_dump_traceback_later()


def iter_training_batches(config, chunks, runtime, device, rank: int, world_size: int, smoke: bool, skip_batches: int = 0):
    """Iterate the prefiltered fixed chunks with rank stride and small batches."""
    epoch = 0
    samples = list(chunks)
    if not samples:
        raise ValueError("no training chunks remain after dataset filtering")
    use_prefetch = hasattr(runtime.codec, "encode_conversation_stereo_batch")
    trace_first_batch = world_size > 1
    loader = sampler = None
    if use_prefetch:
        sampler = RankStrideBatchSampler(
            len(samples), config.per_device_batch_size, rank, world_size,
            seed=config.seed, shuffle=config.shuffle and not smoke,
        )
        dataset = RawAudioDataset(samples, runtime.codec.sample_rate)
        worker_count = 0 if smoke else config.num_workers
        loader_options = {
            "dataset": dataset,
            "batch_sampler": sampler,
            "collate_fn": collate_raw_audio,
            "num_workers": worker_count,
            "pin_memory": config.pin_memory and str(device).startswith("cuda"),
            "persistent_workers": config.persistent_workers and worker_count > 0,
            "worker_init_fn": _initialize_audio_worker if worker_count > 0 else None,
            # Worker stalls must raise here, before peers hit the NCCL watchdog.
            # PyTorch requires timeout=0 for in-process loading.
            "timeout": 120 if worker_count > 0 else 0,
        }
        if worker_count > 0:
            loader_options["prefetch_factor"] = config.prefetch_factor
            # The model has already initialized CUDA in the parent. Spawn keeps
            # decoder workers isolated from that CUDA runtime state.
            loader_options["multiprocessing_context"] = "spawn"
        loader = torch.utils.data.DataLoader(**loader_options)

    while True:
        if use_prefetch:
            sampler.set_epoch(epoch)
            if trace_first_batch:
                log_batch_phase(rank, "audio_loader", "start", workers=worker_count,
                                epoch=epoch, batch_indices=next(iter(sampler), []))
            batches = enumerate(loader)
        else:
            ordered_samples = list(samples)
            if config.shuffle and not smoke:
                random.Random(config.seed + epoch).shuffle(ordered_samples)
            rank_samples = [ordered_samples[index] for index in rank_stride_indices(len(ordered_samples), rank, world_size)]
            common_batch_count = min(
                len(rank_stride_indices(len(ordered_samples), worker, world_size)) // config.per_device_batch_size
                for worker in range(world_size)
            )
            batches = enumerate(
                rank_samples[index * config.per_device_batch_size:(index + 1) * config.per_device_batch_size]
                for index in range(common_batch_count)
            )

        common_batch_count = len(loader) if use_prefetch else common_batch_count
        if common_batch_count == 0:
            raise ValueError(
                f"only {len(samples)} filtered duration chunks for world_size={world_size} and "
                f"batch_size_per_gpu={config.per_device_batch_size}; use sample_number=null/full data "
                "or run a one-GPU overfit config with batch_size=1"
            )
        for batch_index, batch_value in batches:
            if skip_batches:
                skip_batches -= 1
                continue
            if use_prefetch:
                raw_audio = batch_value
                batch_samples = raw_audio["samples"]
                if trace_first_batch:
                    log_batch_phase(rank, "audio_loader", "complete",
                                    sample_ids=[s.sample_id for s in batch_samples])
            else:
                raw_audio = None
                batch_samples = batch_value
            if getattr(config, "swap_roles_after_pass", False) and epoch % 2:
                # Keep the same audio chunks and timestamps; swap the speaker
                # interpretation before channel encoding and target alignment.
                # The prepared prompt assets must describe each corresponding
                # side (voice_prompt_left/right.wav and text_prompt_left/right).
                batch_samples = samples_for_role_pass(batch_samples, epoch, True)
            prepared_batch = []
            for sample_index, sample in enumerate(batch_samples):
                if config.prompt_aug_prob and not smoke:
                    sample = sample.with_window(
                        sample.window_start_sec, sample.window_end_sec,
                        sample.get_augmented_prompt(
                            config.prompt_aug_prob,
                            rng=random.Random(
                                config.seed + epoch * 1_000_003
                                + batch_index * config.per_device_batch_size + sample_index
                            ),
                        ),
                    )
                prepared_batch.append(sample)
            if hasattr(runtime.codec, "encode_conversation_stereo_batch"):
                if trace_first_batch:
                    log_batch_phase(rank, "mimi_dialogue_encode", "start",
                                    sample_ids=[s.sample_id for s in prepared_batch])
                dialogue_codes = runtime.codec.encode_conversation_stereo_batch([
                    (
                        sample.conversation_wav, sample.agent_channel, sample.user_channel,
                        sample.window_start_sec, sample.window_end_sec,
                    )
                    for sample in prepared_batch
                ], raw_audio=raw_audio)
                if trace_first_batch:
                    log_batch_phase(rank, "mimi_dialogue_encode", "complete")
                    log_batch_phase(rank, "prompt_and_targets", "start")
                examples = [
                    build_example(config, sample, runtime, dialogue_codes=codes)
                    for sample, codes in zip(prepared_batch, dialogue_codes, strict=True)
                ]
            else:
                if trace_first_batch:
                    log_batch_phase(rank, "prompt_and_targets", "start", includes_dialogue=True)
                examples = [build_example(config, sample, runtime) for sample in prepared_batch]
            if trace_first_batch:
                log_batch_phase(rank, "prompt_and_targets", "complete")
                log_batch_phase(rank, "collate_to_device", "start")
            fixed_frames = expected_mimi_frames(config.duration_sec, runtime.codec.frame_rate) + max(
                (example.prompt_frames for example in examples), default=0
            )
            examples = [
                pad_training_example(example, fixed_frames, runtime.tokenizer.padding_id, runtime.zero_token)
                for example in examples
            ]
            if any(
                len(example.input_codes) != 17
                or any(len(stream) != fixed_frames for stream in example.input_codes)
                or any(len(stream) != fixed_frames for stream in example.loss_mask)
                for example in examples
            ):
                raise AssertionError("batch padding produced inconsistent PersonaPlex stream shapes")
            batch = post_encode_collate(
                examples, runtime.tokenizer.padding_id, runtime.zero_token, device
            )
            if trace_first_batch:
                log_batch_phase(rank, "collate_to_device", "complete")
                trace_first_batch = False
            audio_seconds = sum(
                max(0.0, min(item.window_end_sec, item.audio.duration_sec) - item.window_start_sec)
                for item in batch_samples
            )
            audio_frames = sum(item.dialogue_frames for item in examples)
            yield batch, epoch, len(examples), audio_seconds, audio_frames, batch_samples
        epoch += 1


def sample_index_for_rank(micro_step: int, process_index: int, num_processes: int, sample_count: int) -> int:
    """Select a disjoint rank-local sample before wrapping around the dataset."""
    if micro_step < 0 or process_index < 0 or process_index >= num_processes or sample_count < 1:
        raise ValueError("invalid distributed sample selection inputs")
    return (micro_step * num_processes + process_index) % sample_count


def write_rank_info(run_dir: Path, process_index: int, num_processes: int, device, sample_count: int, peak_gpu_bytes: int | None = None) -> Path:
    """Emit one non-contended runtime record per DDP rank for GPU verification."""
    visible_devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    physical_gpu = visible_devices[process_index] if process_index < len(visible_devices) else None
    info = {
        "rank": process_index,
        "world_size": num_processes,
        "device": str(device),
        "physical_gpu": physical_gpu,
        "first_sample_index": sample_index_for_rank(0, process_index, num_processes, sample_count),
    }
    if peak_gpu_bytes is not None:
        info["peak_gpu_bytes"] = peak_gpu_bytes
    path = run_dir / "ranks" / f"rank_{process_index:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return path


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    import math
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Enable activation checkpointing on StreamingTransformer layers without breaking streaming inference."""
    import torch.utils.checkpoint
    for module in model.modules():
        if module.__class__.__name__ == "StreamingTransformer":
            orig_forward = module.forward
            def make_checkpointed_forward(mod, orig_fwd):
                def checkpointed_forward(x: torch.Tensor, *args, **kwargs):
                    if not mod.training:
                        return orig_fwd(x, *args, **kwargs)
                    B, T, C = x.shape
                    state = mod._streaming_state
                    if state is None:
                        offset = torch.zeros(1, dtype=torch.long, device=x.device)
                    else:
                        offset = state.offset
                    if mod.positional_embedding in {"sin", "sin_rope"}:
                        from moshi.modules.transformer import create_sin_embedding
                        positions = torch.arange(T, device=x.device).view(1, -1, 1)
                        positions = positions + offset.view(-1, 1, 1)
                        pos_emb = create_sin_embedding(
                            positions, C, max_period=mod.max_period, dtype=x.dtype
                        )
                        x = x + mod.positional_scale * pos_emb
                    for layer in mod.layers:
                        x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
                    if state is not None:
                        state.offset.add_(T)
                    return x
                return checkpointed_forward
            module.forward = make_checkpointed_forward(module, orig_forward)


def model_forward_train(model, codes: torch.Tensor):
    """Run Moshi's training forward with the runtime's CUDA BF16 precision."""
    from torch.nn.parallel import DistributedDataParallel

    # Some Moshi activations are produced in FP32 while the PersonaPlex model
    # weights are BF16. Autocast keeps all projections dtype-compatible.
    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if getattr(getattr(codes, "device", None), "type", None) == "cuda"
        else nullcontext()
    )
    with precision_context:
        if isinstance(model, DistributedDataParallel):
            # Calling forward_train directly bypasses distributed wrapper hooks.
            output = model(codes)
        elif hasattr(model, "forward_train"):
            output = model.forward_train(codes)
        else:
            output = model(codes)
    if hasattr(output, "logits") and hasattr(output, "text_logits"):
        if output.logits.shape[0] != codes.shape[0] or output.logits.shape[2] != codes.shape[2]:
            raise AssertionError(f"audio logits must align to [B,K,T] codes; got {output.logits.shape} vs {codes.shape}")
        if output.text_logits.shape[0] != codes.shape[0] or output.text_logits.shape[2] != codes.shape[2]:
            raise AssertionError(f"text logits must align to [B,1,T] codes; got {output.text_logits.shape} vs {codes.shape}")
    return output


def write_tensorboard_scalars(writer, record: dict[str, float | int], trainable_parameters: int, cpu_threads: int) -> None:
    step = int(record["step"])
    for name, value in (
        ("loss/total", record["loss/total"]),
        ("loss/text", record["loss/text"]),
        ("loss/text_real", record.get("loss/text_real", record["loss/text_nonpadding"])),
        ("loss/text_nonpadding", record["loss/text_nonpadding"]),
        ("loss/audio_total", record.get("loss/audio_total", 0.0)),
        ("loss/agent_semantic", record.get("loss/agent_semantic", record["loss/audio_semantic"])),
        ("loss/agent_acoustic", record.get("loss/agent_acoustic", record["loss/audio_nonsemantic"])),
        ("loss/user_semantic", record.get("loss/user_semantic", 0.0)),
        ("loss/user_acoustic", record.get("loss/user_acoustic", 0.0)),
        ("loss/audio_semantic", record["loss/audio_semantic"]),
        ("loss/audio_nonsemantic", record["loss/audio_nonsemantic"]),
        ("accuracy/text", record.get("accuracy/text", 0.0)),
        ("accuracy/text_nonpad", record.get("accuracy/text_nonpad", 0.0)),
        ("accuracy/text_pad", record.get("accuracy/text_pad", 0.0)),
        ("target_pad_pct", record.get("target_pad_pct", 0.0)),
        ("predicted_pad_pct", record.get("predicted_pad_pct", 0.0)),
        ("probability/pad_given_nonpad_target", record.get("probability/pad_given_nonpad_target", 0.0)),
        ("probability/correct_given_nonpad_target", record.get("probability/correct_given_nonpad_target", 0.0)),
        ("accuracy/audio_total", record.get("accuracy/audio_total", 0.0)),
        ("train/valid_token_pct", record.get("valid_token_pct", 0.0)),
        ("train/learning_rate", record["lr"]),
        ("train/gradient_norm", record["grad_norm"]),
        ("system/gpu_peak_bytes", record["gpu_peak_bytes"]),
        ("system/trainable_parameters", trainable_parameters),
        ("system/cpu_threads", cpu_threads),
        *((name, value) for name, value in record.items() if name.startswith("timing/")),
        *((name, value) for name, value in record.items() if name.startswith("accuracy/audio_cb")),
        *((name, value) for name, value in record.items() if name.startswith("loss/audio_cb")),
    ):
        writer.add_scalar(name, value, step)


OPTIMIZED_LOSS_COMPONENTS = (
    "text", "agent_semantic", "agent_acoustic", "user_semantic", "user_acoustic",
)


def _add_loss_compatibility_aliases(components):
    if not all(name in components for name in OPTIMIZED_LOSS_COMPONENTS[1:]):
        return components
    components["audio_semantic"] = components["agent_semantic"] + components["user_semantic"]
    components["audio_nonsemantic"] = components["agent_acoustic"] + components["user_acoustic"]
    components["audio_total"] = components["audio_semantic"] + components["audio_nonsemantic"]
    return components


def loss_components(
    model_output, codes, example, text_padding_id, torch_module,
    first_codebook_weight_multiplier=1.0, text_padding_weight=0.3,
    *, user_loss=False, distributed=False,
):
    """Compute per-stream losses using GPU-native vectorized weights."""
    if isinstance(example, dict):
        labels_tensor = example["labels"]
        mask_tensor = example["loss_mask"]
    else:
        labels_tensor = torch_module.tensor(example.labels, dtype=torch_module.long, device=codes.device).unsqueeze(0)
        mask_tensor = torch_module.tensor(example.loss_mask, dtype=torch_module.bool, device=codes.device).unsqueeze(0)
    weights = stream_weights_torch(
        labels_tensor, mask_tensor, text_padding_id,
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
        text_padding_weight=text_padding_weight,
        user_loss=user_loss,
    )

    text_target = labels_tensor[:, 0, :]
    text_weight = weights[:, 0, :] * model_output.text_mask[:, 0].to(weights.dtype)
    text_stats = torch_weighted_cross_entropy_stats(
        model_output.text_logits.reshape(-1, model_output.text_logits.shape[-1]),
        text_target.reshape(-1),
        text_weight.reshape(-1),
    )
    text_real_weight = (
        mask_tensor[:, 0].to(weights.dtype)
        * model_output.text_mask[:, 0].to(weights.dtype)
        * (~text_padding_mask_torch(text_target, text_padding_id)).to(weights.dtype)
    )
    text_real_stats = torch_weighted_cross_entropy_stats(
        model_output.text_logits.reshape(-1, model_output.text_logits.shape[-1]),
        text_target.reshape(-1),
        text_real_weight.reshape(-1),
    )

    audio_target = labels_tensor[:, 1:17, :]
    audio_weights = weights[:, 1:17, :] * model_output.mask.to(weights.dtype)

    def audio_group_stats(start: int, end: int):
        return torch_weighted_cross_entropy_stats(
            model_output.logits[:, start:end].reshape(-1, model_output.logits.shape[-1]),
            audio_target[:, start:end].reshape(-1),
            audio_weights[:, start:end].reshape(-1),
        )

    agent_semantic = audio_group_stats(0, 1)
    user_semantic = audio_group_stats(8, 9)
    agent_nonsemantic = audio_group_stats(1, 8)
    user_nonsemantic = audio_group_stats(9, 16)
    audio_denominator = sum(
        stats[1]
        for stats in (agent_semantic, agent_nonsemantic, user_semantic, user_nonsemantic)
    )
    audio_denominator = audio_denominator.clamp_min(1e-12)
    audio_stats = {
        "agent_semantic": (agent_semantic[0], audio_denominator),
        "agent_acoustic": (agent_nonsemantic[0], audio_denominator),
        "user_semantic": (user_semantic[0], audio_denominator),
        "user_acoustic": (user_nonsemantic[0], audio_denominator),
    }
    if distributed:
        # Each logged component is a contribution to the same weighted audio
        # mean. Sharing its denominator keeps 0.02 and first-codebook weights
        # effective after the cross-rank reduction.
        stats = {
            "text": text_stats,
            "text_real": text_real_stats,
            **audio_stats,
        }
        return None, stats
    text_loss = text_stats[0] / text_stats[1].clamp_min(1e-12)
    components = {
        "text": text_loss,
        "text_real": text_real_stats[0] / text_real_stats[1].clamp_min(1e-12),
        **{name: numerator / denominator for name, (numerator, denominator) in audio_stats.items()},
    }
    total = sum(components[name] for name in OPTIMIZED_LOSS_COMPONENTS)
    return total, _add_loss_compatibility_aliases(components)


def reduce_distributed_loss(stats, all_reduce_sum, *, preserve_grad: bool = False, world_size: int = 1):
    """Reduce weighted CE sums/weights in one collective and return global means."""
    names = tuple(stats)
    packed = torch.stack([
        value
        for name in names
        for value in stats[name]
    ]).detach()
    global_values = all_reduce_sum(packed)
    components = {}
    total = None
    split_objective = all(name in stats for name in OPTIMIZED_LOSS_COMPONENTS)
    for index, name in enumerate(names):
        numerator, denominator = stats[name]
        global_numerator = global_values[index * 2]
        global_denominator = global_values[index * 2 + 1].clamp_min(1e-12)
        global_mean = global_numerator / global_denominator
        if preserve_grad:
            # DDP averages gradients across ranks; scale each local numerator
            # so the averaged gradient equals the global weighted-mean gradient.
            local_gradient = numerator * world_size / global_denominator
            loss = local_gradient + (global_mean - local_gradient.detach())
        else:
            loss = global_mean
        components[name] = loss
        if name in OPTIMIZED_LOSS_COMPONENTS or not split_objective:
            total = loss if total is None else total + loss
    return _add_loss_compatibility_aliases(components), total


def one_step(config: Config, runtime, example, optimizer=None):
    codes = torch.tensor(example.input_codes, dtype=torch.long, device=config.device).unsqueeze(0)
    output = model_forward_train(runtime.model, codes)
    total, components = loss_components(
        output, codes, example, tokenizer_text_padding_ids(runtime.tokenizer, config.epad_as_padding), torch,
        config.first_codebook_weight_multiplier, config.text_padding_weight,
        user_loss=config.user_loss,
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        trainable = [p for p in runtime.model.parameters() if p.requires_grad]
        if not any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in trainable):
            raise RuntimeError("LoRA gradients are all zero")
        if any(p.grad is not None for p in runtime.model.parameters() if not p.requires_grad):
            raise RuntimeError("frozen base parameter received a gradient")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    else:
        grad_norm = torch.tensor(0.0)
    return total, components, float(grad_norm)


def save_training_state(
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    optimizer_step: int,
    gradient_accumulation_steps: int,
    num_processes: int,
    per_device_batch_size: int = 1,
) -> Path:
    """Persist enough state to continue the optimizer trajectory exactly."""
    path = checkpoint_dir / "training_state.pt"
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    if isinstance(scheduler_state, dict) and callable(scheduler_state.get("anneal_func")):
        # OneCycleLR stores a Python function in its state. Persist its known
        # function name so the checkpoint remains loadable with weights_only=True.
        scheduler_state["anneal_func"] = scheduler_state["anneal_func"].__name__
    torch.save(
        {
            "format_version": 1,
            "optimizer_step": optimizer_step,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_processes": num_processes,
            "per_device_batch_size": per_device_batch_size,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler_state,
        },
        path,
    )
    return path


def load_training_state(
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    gradient_accumulation_steps: int,
    num_processes: int,
    per_device_batch_size: int = 1,
    total_steps: int | None = None,
) -> int | None:
    """Restore an optimizer checkpoint, rejecting incompatible DDP topology."""
    path = checkpoint_dir / "training_state.pt"
    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("gradient_accumulation_steps") != gradient_accumulation_steps:
        raise RuntimeError("resume checkpoint gradient_accumulation_steps differs from this run")
    if state.get("num_processes") != num_processes:
        raise RuntimeError("resume checkpoint num_processes differs from this run")
    if state.get("per_device_batch_size", 1) != per_device_batch_size:
        raise RuntimeError("resume checkpoint per_device_batch_size differs from this run")
    saved_scheduler = state.get("scheduler")
    saved_total_steps = saved_scheduler.get("total_steps") if isinstance(saved_scheduler, dict) else None
    if total_steps is not None and saved_total_steps != total_steps:
        raise RuntimeError(
            f"resume checkpoint max_steps ({saved_total_steps}) differs from this run ({total_steps}); "
            "start a fresh run to use a different OneCycleLR schedule"
        )
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is None and saved_scheduler is not None:
        raise RuntimeError("resume checkpoint has scheduler state but this run has no scheduler")
    if scheduler is not None and saved_scheduler is None:
        raise RuntimeError("resume checkpoint has no scheduler state but this run requires one")
    if scheduler is not None and saved_scheduler is not None:
        anneal_name = saved_scheduler.get("anneal_func")
        if isinstance(anneal_name, str):
            supported_annealers = {
                "_annealing_cos": scheduler._annealing_cos,
                "_annealing_linear": scheduler._annealing_linear,
            }
            if anneal_name not in supported_annealers:
                raise RuntimeError(f"unsupported scheduler anneal function in checkpoint: {anneal_name}")
            saved_scheduler["anneal_func"] = supported_annealers[anneal_name]
        scheduler.load_state_dict(saved_scheduler)
    return int(state["optimizer_step"])


def save_adapter(run_dir: Path, model, config: Config, step: int, optimizer=None, scheduler=None, gradient_accumulation_steps: int = 1, num_processes: int = 1) -> Path:
    return save_adapter_state(
        run_dir, adapter_state_dict(model), config, step, optimizer, scheduler,
        gradient_accumulation_steps, num_processes,
    )


def save_full_checkpoint(
    run_dir: Path, model, config: Config, step: int, optimizer, scheduler,
    gradient_accumulation_steps: int, num_processes: int, checkpoint_name: str | None = None,
) -> Path:
    """Save a complete FP32 LM and resumable optimizer state on the main DDP rank."""
    name = checkpoint_name or f"checkpoint_{step:06d}"
    path = run_dir / "checkpoints" / name
    save_full_weights(model, path)
    save_training_state(
        path, optimizer, scheduler, step, gradient_accumulation_steps,
        num_processes, config.per_device_batch_size,
    )
    write_full_metadata(path, base_model_root=config.model_root, step=step)
    return path


def save_adapter_state(run_dir: Path, state_dict, config: Config, step: int, optimizer=None, scheduler=None, gradient_accumulation_steps: int = 1, num_processes: int = 1) -> Path:
    path = run_dir / "checkpoints" / f"checkpoint_{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    adapter = path / "lora.safetensors"
    save_file(state_dict, str(adapter))
    (path / "adapter.json").write_text(
        json.dumps(
            {"step": step, "model_root": str(config.model_root), "rank": config.lora_rank, "alpha": config.lora_alpha,
             "scaling": config.lora_scaling,
             "gradient_accumulation_steps": gradient_accumulation_steps, "num_processes": num_processes,
             "per_device_batch_size": config.per_device_batch_size},
            indent=2,
        )
    )
    if optimizer is not None:
        save_training_state(path, optimizer, scheduler, step, gradient_accumulation_steps, num_processes, config.per_device_batch_size)
    return adapter


def save_best_adapter(
    run_dir: Path, model, config: Config, step: int, val_loss: float,
    optimizer=None, scheduler=None, gradient_accumulation_steps: int = 1, num_processes: int = 1,
) -> Path:
    return save_best_adapter_state(
        run_dir, adapter_state_dict(model), config, step, val_loss, optimizer, scheduler,
        gradient_accumulation_steps, num_processes,
    )


def save_best_adapter_state(
    run_dir: Path, state_dict, config: Config, step: int, val_loss: float,
    optimizer=None, scheduler=None, gradient_accumulation_steps: int = 1, num_processes: int = 1,
    checkpoint_name: str = "best",
) -> Path:
    path = run_dir / "checkpoints" / checkpoint_name
    path.mkdir(parents=True, exist_ok=True)
    adapter = path / "lora.safetensors"
    save_file(state_dict, str(adapter))
    (path / "adapter.json").write_text(
        json.dumps(
            {"step": step, "val_loss": val_loss, "model_root": str(config.model_root), "rank": config.lora_rank, "alpha": config.lora_alpha,
             "scaling": config.lora_scaling,
             "gradient_accumulation_steps": gradient_accumulation_steps, "num_processes": num_processes,
             "per_device_batch_size": config.per_device_batch_size},
            indent=2,
        )
    )
    if optimizer is not None:
        save_training_state(path, optimizer, scheduler, step, gradient_accumulation_steps, num_processes, config.per_device_batch_size)
    return adapter


def evaluate_validation(config: Config, runtime, val_samples: list, rank: int, world_size: int, device) -> dict[str, float]:
    if not val_samples:
        return {}
    unwrapped = runtime.model
    unwrapped.eval()
    loss_names = (
        "total", "text", "agent_semantic", "agent_acoustic", "user_semantic", "user_acoustic",
    )
    totals = {name: 0.0 for name in loss_names}
    nonpadding_text_loss_sum = 0.0
    nonpadding_text_token_count = 0
    count = 0
    with torch.no_grad():
        selected_samples = evenly_spaced_validation_samples(val_samples, config.validation_max_samples)
        local_samples = selected_samples[rank::world_size]
        for sample in local_samples:
            example = build_example(config, sample, runtime)
            codes = torch.tensor(example.input_codes, dtype=torch.long, device=device).unsqueeze(0)
            output = model_forward_train(unwrapped, codes)
            total, comps = loss_components(
                output, codes, example, tokenizer_text_padding_ids(runtime.tokenizer, config.epad_as_padding), torch,
                config.first_codebook_weight_multiplier, config.text_padding_weight,
                user_loss=config.user_loss,
            )
            totals["total"] += float(total.detach())
            totals["text"] += float(comps["text"].detach())
            for name in loss_names[2:]:
                totals[name] += float(comps[name].detach())
            labels = torch.tensor(example.labels, dtype=torch.long, device=device).unsqueeze(0)
            loss_mask = torch.tensor(example.loss_mask, dtype=torch.bool, device=device).unsqueeze(0)
            text_loss_sum, text_token_count = text_target_token_loss_stats(
                {"labels": labels, "loss_mask": loss_mask}, output, tokenizer_text_padding_ids(runtime.tokenizer, config.epad_as_padding),
            )
            nonpadding_text_loss_sum += float(text_loss_sum)
            nonpadding_text_token_count += int(text_token_count)
            count += 1
    unwrapped.train()
    reduced = torch.tensor(
        [*(totals[name] for name in loss_names), count, nonpadding_text_loss_sum, nonpadding_text_token_count],
        device=device,
    )
    if world_size > 1:
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
    count_index = len(loss_names)
    total_count = float(reduced[count_index].item())
    if total_count == 0:
        raise RuntimeError("validation has no samples after rank partitioning")
    text_token_count = float(reduced[count_index + 2].item())
    text_real_loss = float(reduced[count_index + 1].item()) / text_token_count if text_token_count else float("inf")
    averaged = {name: float(reduced[index].item()) / total_count for index, name in enumerate(loss_names)}
    metrics = {
        "val/loss_total": averaged["total"],
        "val/loss_text": averaged["text"],
        "val/loss_text_real": text_real_loss,
        "val/loss_text_nonpadding": text_real_loss,
        "val/text_nonpadding_tokens": text_token_count,
        **{f"val/loss_{name}": averaged[name] for name in loss_names[2:]},
    }
    metrics["val/loss_audio_semantic"] = averaged["agent_semantic"] + averaged["user_semantic"]
    metrics["val/loss_audio_nonsemantic"] = averaged["agent_acoustic"] + averaged["user_acoustic"]
    metrics["val/loss_audio_total"] = metrics["val/loss_audio_semantic"] + metrics["val/loss_audio_nonsemantic"]
    metrics["val/loss_selection"] = validation_selection_loss(metrics)
    return metrics


def validation_selection_loss(metrics: dict[str, float]) -> float:
    """Choose checkpoints using real text plus every enabled audio contribution."""
    keys = (
        "val/loss_text_real",
        "val/loss_agent_semantic",
        "val/loss_agent_acoustic",
        "val/loss_user_semantic",
        "val/loss_user_acoustic",
    )
    if any(key not in metrics for key in keys):
        return float("inf")
    return sum(metrics[key] for key in keys)


def generation_checkpoint_score(metrics: dict, baseline_cer: float | None = None) -> float:
    """Return CER only for complete, non-empty generation that beats the starting adapter."""
    if metrics.get("val/generation_empty_samples", 0):
        return float("inf")
    cer = float(metrics.get("val/generation_cer", float("inf")))
    if baseline_cer is not None and cer >= baseline_cer:
        return float("inf")
    return cer


def inference_config_snapshot(config: Config, adapter_path: Path, output_dir: Path) -> dict:
    """Create a standalone inference config matching the effective training inputs."""
    data = {
        "prepared_dir": str(config.prepared_dir),
        "window_seconds": config.window_seconds,
        "val_ratio": config.val_ratio,
        "vietnamese_text_mode": config.vietnamese_text_mode,
    }
    if config.val_manifest is not None:
        data["val_manifest"] = str(config.val_manifest)
    if config.test_manifest is not None:
        data["test_manifest"] = str(config.test_manifest)
    snapshot = {
        "model": {
            "root": str(config.model_root),
            "source": str(config.personaplex_source),
            "device": config.device,
        },
        "data": data,
        "lora": {"qlora": config.qlora, "quant_type": config.quant_type},
        "seed": config.seed,
        "generation": config.generation_settings.as_dict(),
        "inference": {
            "output_dir": str(output_dir),
            "sample_id": None,
            "input_file": None,
            "voice_prompt": None,
            "text_prompt": None,
        },
    }
    snapshot["checkpoint" if config.train_method == "full" else "adapter"] = {"path": str(adapter_path)}
    return snapshot


def evenly_spaced_validation_samples(samples: list, limit: int) -> list:
    """Choose a deterministic spread over a conversation-ordered validation set."""
    if limit < 1:
        raise ValueError("validation sample limit must be positive")
    if len(samples) <= limit:
        return list(samples)
    if limit == 1:
        return [samples[len(samples) // 2]]
    indices = [round(index * (len(samples) - 1) / (limit - 1)) for index in range(limit)]
    return [samples[index] for index in indices]


def evaluate_free_running(
    runtime, samples: list, config: Config, audio_output_dir: Path | None = None,
    *, step: int = 0, baseline_dir: Path | None = None,
) -> dict:
    """Score autoregressive text and export per-sample stereo comparisons."""
    import shutil

    evaluated = []
    manifests = {}
    unique_conversations = []
    seen_groups = set()
    for sample in samples:
        keys = conversation_group_keys(sample)
        if any(key in seen_groups for key in keys):
            continue
        seen_groups.update(keys)
        unique_conversations.append(sample)
    candidates = evenly_spaced_validation_samples(
        unique_conversations, max(config.free_running_eval_samples * 4, config.free_running_eval_samples),
    )
    for sample in candidates:
        agent_words = [
            word for word in sample.words
            if word.speaker == "agent" and sample.window_start_sec <= word.start < sample.window_end_sec
        ]
        if not agent_words:
            continue
        clipped_end = min(sample.window_end_sec, sample.audio.duration_sec)
        latest_start = max(sample.window_start_sec, clipped_end - config.free_running_eval_window_seconds)
        window_start = max(sample.window_start_sec, min(min(word.start for word in agent_words), latest_start))
        end_sec = min(
            sample.window_end_sec,
            window_start + config.free_running_eval_window_seconds,
            sample.audio.duration_sec,
        )
        if end_sec <= window_start:
            continue
        window = sample.with_window(window_start, end_sec)
        reference = " ".join(
            word.word for word in window.words
            if word.speaker == "agent" and window.window_start_sec <= word.start < window.window_end_sec
        )
        if not reference.strip():
            continue
        if audio_output_dir is not None and (
            sample.sample_id in {"", ".", ".."}
            or "/" in sample.sample_id or "\\" in sample.sample_id
        ):
            raise ValueError(f"unsafe free-running sample_id: {sample.sample_id!r}")
        sample_dir = audio_output_dir / sample.sample_id if audio_output_dir is not None else None
        audio_path = sample_dir / "dialogue_step.wav" if sample_dir is not None else None
        original_audio_path = (
            sample_dir / "dialogue_original.wav" if sample_dir is not None else None
        )
        generation_kwargs = {
            "generation": getattr(config, "generation_settings", GenerationSettings()),
            "seed": config.seed + len(evaluated),
        }
        if audio_path is not None:
            generation_kwargs["output_wav"] = audio_path
            generation_kwargs["output_stereo"] = True
        hypothesis = generate_text_with_runtime(runtime, window, **generation_kwargs)
        if audio_path is not None and not audio_path.is_file():
            raise RuntimeError(f"free-running validation did not write generated audio: {audio_path}")
        if original_audio_path is not None:
            export_original_audio_window(window, original_audio_path)
            if not original_audio_path.is_file():
                raise RuntimeError(
                    f"free-running validation did not write original audio: {original_audio_path}"
                )
        text_mode = getattr(config, "vietnamese_text_mode", "diacritics")
        metrics = text_error_metrics(
            reference, hypothesis, vietnamese_text_mode=text_mode,
        )
        if metrics is not None:
            metrics_reference = normalize_vietnamese_text(reference, text_mode)
            if sample_dir is not None:
                base_path = sample_dir / "dialogue_base.wav"
                if baseline_dir is None:
                    if step != 0:
                        raise ValueError("nonzero free-running step requires a base-model baseline")
                    shutil.copyfile(audio_path, base_path)
                else:
                    base_sample_dir = baseline_dir / sample.sample_id
                    base_manifest = json.loads((base_sample_dir / "manifest.json").read_text(encoding="utf-8"))
                    for key, value in {
                        "window_start_sec": window.window_start_sec,
                        "window_end_sec": window.window_end_sec,
                        "vietnamese_text_mode": text_mode,
                        "seed": generation_kwargs["seed"],
                        "generation": generation_kwargs["generation"].as_dict(),
                    }.items():
                        if base_manifest[key] != value:
                            raise ValueError(f"base-model comparison mismatch for {sample.sample_id}: {key}")
                    shutil.copyfile(base_sample_dir / "dialogue_base.wav", base_path)
                manifest = {
                    "sample_id": sample.sample_id, "step": step,
                    "vietnamese_text_mode": text_mode,
                    "hypothesis": hypothesis, "transcript": hypothesis,
                    "reference": metrics_reference, "raw_reference": reference,
                    "cer": metrics["cer"], "wer": metrics["wer"],
                    "window_start_sec": window.window_start_sec,
                    "window_end_sec": window.window_end_sec,
                    "source_duration_sec": sample.audio.duration_sec,
                    "sample_rate": 24000, "channels": {"left": "agent", "right": "user"},
                    "seed": generation_kwargs["seed"],
                    "generation": generation_kwargs["generation"].as_dict(),
                    "audio_files": {"original": "dialogue_original.wav",
                                    "base": "dialogue_base.wav", "current_step": "dialogue_step.wav"},
                }
                (sample_dir / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                )
                manifests[sample.sample_id] = {
                    "base_audio_path": str(base_path.resolve()),
                    "manifest_path": str((sample_dir / "manifest.json").resolve()),
                    "vietnamese_text_mode": text_mode, "raw_reference": reference,
                }
            evaluated.append((metrics, sample.sample_id, metrics_reference, hypothesis,
                              window.window_start_sec, window.window_end_sec,
                              audio_path, original_audio_path))
        if len(evaluated) >= config.free_running_eval_samples:
            break
    if not evaluated:
        raise RuntimeError("free-running validation found no samples with agent text references")
    empty_hypotheses = sum(not hypothesis.strip() for _, _, _, hypothesis, _, _, _, _ in evaluated)
    result = {
        "val/generation_cer": sum(float(item[0]["cer"]) for item in evaluated) / len(evaluated),
        "val/generation_wer": sum(float(item[0]["wer"]) for item in evaluated) / len(evaluated),
        "val/generation_samples": len(evaluated),
        "val/generation_empty_samples": empty_hypotheses,
        "samples": [
            {"sample_id": sample_id, "reference": reference, "hypothesis": hypothesis,
             **manifests.get(sample_id, {}),
             "cer": metrics["cer"], "wer": metrics["wer"],
             "audio_path": str(audio_path.resolve()) if audio_path is not None else None,
             "original_audio_path": (
                 str(original_audio_path.resolve()) if original_audio_path is not None else None
             ),
             "window_start_sec": start_sec, "window_end_sec": end_sec,
             "source_duration_sec": next(
                 sample.audio.duration_sec for sample in unique_conversations
                 if sample.sample_id == sample_id
             )}
            for (
                metrics, sample_id, reference, hypothesis, start_sec, end_sec,
                audio_path, original_audio_path,
            ) in evaluated
        ],
    }
    if audio_output_dir is not None:
        update_free_running_report(audio_output_dir.parent, step, result, text_mode)
    return result


def update_free_running_report(output_dir: Path, step: int, metrics: dict, text_mode: str) -> None:
    """Atomically publish completed evaluations; retain the JSONL log separately."""
    report_path = output_dir / "free-running-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"evaluations": []}
    rows = [row for row in report["evaluations"] if row["step"] != step]
    rows.append({"step": step, "val/generation_baseline": step == 0, **metrics})
    rows.sort(key=lambda row: row["step"])
    baseline = next((row["val/generation_cer"] for row in rows if row["step"] == 0), None)
    eligible = [row for row in rows if row["step"] > 0 and baseline is not None
                and generation_checkpoint_score(row, baseline_cer=baseline) < float("inf")]
    best = min(eligible, key=lambda row: row["val/generation_cer"], default=None)
    report.update(vietnamese_text_mode=text_mode, evaluations=rows, baseline_cer=baseline,
                  best_step=best["step"] if best else None,
                  best_cer=best["val/generation_cer"] if best else None)
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)


def verify_reloaded_adapter(config: Config, sample, adapter: Path, runtime) -> float:
    """Round-trip saved LoRA weights in-place without allocating a second 7B base."""
    parallel_model = runtime.model
    model = unwrap_parallel_model(parallel_model)
    was_training = parallel_model.training
    runtime.model = model
    try:
        model.eval()
        example = build_example(config, sample, runtime)
        with torch.no_grad():
            reference_loss, _, _ = one_step(config, runtime, example)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".lora_a." in name or ".lora_b." in name:
                    parameter.add_(1.0)
        load_adapter(model, adapter)
        with torch.no_grad():
            reloaded_loss, _, _ = one_step(config, runtime, example)
        reference_loss = float(reference_loss)
        reloaded_loss = float(reloaded_loss)
        if not math.isfinite(reference_loss) or not math.isfinite(reloaded_loss):
            raise RuntimeError(f"non-finite adapter reload loss: {reloaded_loss} vs {reference_loss}")
        if abs(reloaded_loss - reference_loss) > 1e-5:
            raise RuntimeError(
                f"reloaded adapter loss drifted: {reloaded_loss} vs {reference_loss}"
            )
        return reloaded_loss
    finally:
        runtime.model = parallel_model
        parallel_model.train(was_training)


def verify_reloaded_full(config: Config, sample, checkpoint: Path, runtime) -> float:
    """Verify full-weight reload in place without allocating another PersonaPlex LM."""
    parallel_model = runtime.model
    model = unwrap_parallel_model(parallel_model)
    was_training = parallel_model.training
    runtime.model = model
    try:
        model.eval()
        example = build_example(config, sample, runtime)
        with torch.no_grad():
            reference_loss, _, _ = one_step(config, runtime, example)
            parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
            parameter.view(-1)[0].add_(1.0)
            load_full_weights(model, checkpoint)
            reloaded_loss, _, _ = one_step(config, runtime, example)
        if abs(float(reloaded_loss) - float(reference_loss)) > 1e-5:
            raise RuntimeError(f"reloaded full checkpoint loss drifted: {reloaded_loss} vs {reference_loss}")
        return float(reloaded_loss)
    finally:
        runtime.model = parallel_model
        parallel_model.train(was_training)


def run(
    config: Config,
    smoke: bool = False,
    resume_from: str | None = None,
    force_filter: bool = False,
) -> Path | None:
    if config.train_method == "lora" and config.ft_embed:
        raise ValueError("lora.ft_embed=true is not implemented by this LoRA-only trainer")
    if config.train_method == "full":
        if config.qlora:
            raise ValueError("train.method=full cannot use lora.qlora=true")
        if config.train_stage != "joint":
            raise ValueError("train.method=full requires train.stage=joint")
    if config.randomize_train:
        raise ValueError("data.randomize_train=true is not implemented; fixed chunks may be shuffled with data.shuffle")
    if config.mixed_precision.lower() != "bf16":
        raise ValueError("train.mixed_precision must be bf16; this trainer uses BF16 CUDA autocast")
    # Moshi's lazy compile wrappers can trigger graph/compile shape issues on
    # the fixed, padded sequences used by distributed training.
    os.environ.setdefault("NO_TORCH_COMPILE", "1")
    # Mimi's streaming encoder uses CUDA Graph capture for voice prompts; this
    # capture can be invalidated on the training CUDA stack before the LM step.
    os.environ.setdefault("NO_CUDA_GRAPH", "1")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required; install the project requirements before training") from exc

    if smoke and config.shuffle:
        raise ValueError("overfit/smoke configuration must set data.shuffle: false")
    accum_steps = max(1, config.gradient_accumulation_steps)
    if "LOCAL_RANK" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun distributed mode requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            timeout=timedelta(hours=2) if config.train_method == "full" else timedelta(minutes=10),
        )
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank, world_size, local_rank = 0, 1, 0
    distributed = world_size > 1
    if distributed and not torch.cuda.is_available():
        raise RuntimeError("multi-process DDP training requires CUDA")
    device = resolve_training_device(config.device, local_rank, world_size)
    global_batch_size = effective_global_batch_size(config.per_device_batch_size, world_size, accum_steps)
    cpu_threads = limit_cpu_threads(torch)
    seed_everything(config.seed, torch)
    main_process = rank == 0
    def barrier():
        if distributed:
            torch.distributed.barrier(device_ids=[local_rank])
    def all_reduce_sum(value):
        if distributed:
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        return value
    # Dataset loading
    test_samples = []
    dataset_load_reports = {}

    def read_dataset(manifest, label: str, split_by_conversation: bool = False):
        dataset = PreparedDataset(
            manifest, config.window_seconds,
            filter_num_workers=config.filter_num_workers if main_process else 1,
            force_filter=force_filter and main_process,
        )

        def load_local():
            try:
                if split_by_conversation:
                    return dataset.split(val_ratio=config.val_ratio, seed=config.seed)
                return dataset.load()
            finally:
                report = dataset.load_report
                dataset_load_reports[label] = report
                if main_process and report.manifest_entries:
                    print(dataset_load_summary(label, report), flush=True)

        if distributed:
            state = [None]
            result = None
            if main_process:
                try:
                    result = load_local()
                except Exception as exc:
                    state[0] = f"{type(exc).__name__}: {exc}"
            torch.distributed.broadcast_object_list(state, src=0, device=device)
            if state[0] is not None:
                raise RuntimeError(f"prepared-sample filtering failed on rank 0: {state[0]}")
            if main_process:
                return result
            return load_local()  # Rank 0 has completed the shared validated-sample cache.
        return load_local()

    if config.eval_on_train_samples:
        if config.no_eval or config.sample_number is None:
            raise ValueError("eval_on_train_samples requires evaluation enabled and a finite sample_number")
        if config.val_manifest_path is not None:
            raise ValueError("eval_on_train_samples cannot be combined with a separate validation manifest")
        train_samples = read_dataset(config.manifest, "train")
        val_samples = []
    elif config.val_manifest:
        train_samples = read_dataset(config.manifest, "train")
        val_samples = read_dataset(config.val_manifest, "validation")
    elif (config.eval_every_steps > 0 or config.free_running_eval_every_steps > 0) and not config.no_eval:
        train_samples, val_samples = read_dataset(
            config.manifest, "train+validation", split_by_conversation=True,
        )
    else:
        train_samples = read_dataset(config.manifest, "train")
        val_samples = []

    if config.test_manifest:
        test_samples = read_dataset(config.test_manifest, "test")
    train_groups = {key for sample in train_samples for key in conversation_group_keys(sample)}
    val_groups = {key for sample in val_samples for key in conversation_group_keys(sample)}
    test_groups = {key for sample in test_samples for key in conversation_group_keys(sample)}
    if (
        (not config.eval_on_train_samples and train_groups & val_groups)
        or train_groups & test_groups
        or val_groups & test_groups
    ):
        raise ValueError("train/validation/test manifests contain overlapping conversation groups")

    train_conversations = limit_conversations(
        train_samples,
        sample_index=config.sample_index,
    )
    train_samples = duration_chunks(train_conversations, config.duration_sec)
    if config.sample_index is not None:
        # User specified an exact sample index to overfit: train on exactly the first 100s window of that sample
        train_samples = train_samples[:1]
    val_samples = duration_chunks(val_samples, config.duration_sec) if val_samples else []
    test_samples = duration_chunks(test_samples, config.duration_sec) if test_samples else []

    resolved = RuntimePaths(config.model_root, config.personaplex_source).validate(require_model=False)
    filter_tokenizer = SentencePieceTokenizer(resolved.tokenizer)
    filter_payloads = {}

    def filter_split(label: str, chunks: list, check_role_swap: bool = False) -> list:
        max_kept = config.sample_number if label == "train" and config.sample_index is None else None
        result = None
        filter_error = None
        filter_state = [None]
        if main_process:
            try:
                result = filter_text_capacity_chunks(
                    chunks, filter_tokenizer, PERSONAPLEX_MIMI_FRAME_RATE,
                    vietnamese_text_mode=config.vietnamese_text_mode,
                    swap_roles=check_role_swap,
                    num_workers=config.filter_num_workers,
                    max_kept=max_kept,
                    cache_path=config.prepared_dir / ".filter-cache" / f"training-{label}.jsonl",
                    cache_fingerprint=filter_fingerprint(
                        [path for path in (config.manifest, config.val_manifest, config.test_manifest) if path],
                        {
                            "kind": "training-chunks", "split": label,
                            "sample_number_contract": "valid-chunks-v1",
                            "duration_sec": config.duration_sec,
                            "window_seconds": config.window_seconds,
                            "sample_number": config.sample_number,
                            "sample_index": config.sample_index,
                            "val_ratio": config.val_ratio, "seed": config.seed,
                            "eval_on_train_samples": config.eval_on_train_samples,
                            "vietnamese_text_mode": config.vietnamese_text_mode,
                            "swap_roles": check_role_swap,
                            "chunks": [(s.sample_id, s.window_start_sec, s.window_end_sec) for s in chunks],
                        },
                        resolved.tokenizer,
                    ),
                    force_filter=force_filter,
                )
                kept_keys = {
                    (sample.sample_id, sample.window_start_sec, sample.window_end_sec)
                    for sample in result.kept
                }
                kept_indices = [
                    index for index, sample in enumerate(chunks)
                    if (sample.sample_id, sample.window_start_sec, sample.window_end_sec) in kept_keys
                ]
                filter_state[0] = (kept_indices, result.rejected, None)
            except Exception as exc:
                filter_error = exc
                filter_state[0] = (None, None, f"{type(exc).__name__}: {exc}")

        # All ranks construct the same chunks locally. Rank 0 owns filtering so
        # distributed runs do not repeat CPU work or spawn workers per GPU.
        if distributed:
            torch.distributed.broadcast_object_list(filter_state, src=0, device=device)
        kept_indices, rejected, remote_error = filter_state[0]
        if remote_error is not None:
            if filter_error is not None:
                raise filter_error
            raise RuntimeError(f"chunk filtering failed on rank 0: {remote_error}")
        if not main_process:
            result = ChunkFilterResult(
                kept=tuple(chunks[index] for index in kept_indices),
                rejected=tuple(rejected),
            )
        assert result is not None
        payload = chunk_filter_payload(label, len(chunks), result, max_kept)
        filter_payloads[label] = payload
        if main_process:
            print(chunk_filter_summary(payload), flush=True)
            for item in payload["rejected"]:
                print(
                    f"[Skipped chunk] split={label} sample={item['sample_id']} "
                    f"chunk={item['window_start_sec']:.3f}-{item['window_end_sec']:.3f}s "
                    f"reason={item['reason']} roles={','.join(item['roles'])} "
                    f"word={item['word']!r}@{item['word_start_sec']:.3f}s",
                    flush=True,
                )
        return list(result.kept)

    train_candidate_count = len(train_samples)
    train_samples = filter_split("train", train_samples, config.swap_roles_after_pass)
    retained_groups = {key for sample in train_samples for key in conversation_group_keys(sample)}
    train_conversations = [
        sample for sample in train_conversations
        if retained_groups.intersection(conversation_group_keys(sample))
    ]
    if config.eval_on_train_samples:
        # Reuse the exact kept training chunks; do not classify them twice.
        val_samples = list(train_samples)
        filter_payloads["validation"] = {
            "split": "validation", "candidate_chunks": 0,
            "scanned_chunks": 0, "requested_valid_chunks": None, "quota_shortfall": 0,
            "kept_chunks": len(val_samples), "skipped_out_of_bounds_chunks": 0,
            "skipped_text_overflow_chunks": 0, "rejected": [],
            "reused_from": "train",
        }
    else:
        val_samples = filter_split("validation", val_samples)
    test_samples = filter_split("test", test_samples)
    minimum_chunks = world_size * config.per_device_batch_size
    if len(train_samples) < minimum_chunks:
        raise ValueError(
            f"only {len(train_samples)} train chunks remain after filtering "
            f"(candidates={train_candidate_count}, skipped_out_of_bounds_chunks="
            f"{filter_payloads['train']['skipped_out_of_bounds_chunks']}, skipped_text_overflow_chunks="
            f"{filter_payloads['train']['skipped_text_overflow_chunks']}); at least {minimum_chunks} "
            f"are required for world_size={world_size}, batch_size_per_gpu={config.per_device_batch_size}"
        )

    resume_adapter_file = None
    adapter_resume_step = 0
    if resume_from:
        resume_adapter_file, adapter_resume_step = validate_resume_checkpoint(
            config, resume_from, lora_prefixes_for_stage(config), train_conversations,
            train_chunks=train_samples,
        )
        validate_resume_step(adapter_resume_step, 1 if smoke else config.max_steps)

    # Run directory creation (coordinated across ranks)
    run_dir = None
    if main_process:
        run_dir = create_run_dir(config.output_dir, smoke)
        config_record = {
            "event": "configuration",
            "seed": config.seed,
            "model_root": str(config.model_root),
            "personaplex_source": str(config.personaplex_source),
            "manifest": str(config.manifest),
            "training_contract": training_contract(config, train_conversations, train_samples),
            "dataset_load": {
                label: {
                    "manifest_entries": report.manifest_entries,
                    "loaded_samples": report.loaded_samples,
                    "skipped_out_of_bounds_samples": report.skipped_out_of_bounds,
                    "skipped_invalid_samples": report.skipped_invalid,
                }
                for label, report in dataset_load_reports.items()
            },
            "chunk_filter": {
                label: {key: value for key, value in payload.items() if key != "rejected"}
                for label, payload in filter_payloads.items()
            },
            "codec_cache_dir": str(config.codec_cache_dir) if config.codec_cache_dir else None,
            "output_dir": str(run_dir),
            "duration_sec": config.duration_sec,
            "sample_number": config.sample_number,
            "profile_steps": config.profile_steps,
            "max_steps": 1 if smoke else config.max_steps,
            "learning_rate": config.learning_rate,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_scaling": config.lora_scaling,
            "lora_enabled": config.lora_enabled,
            "train_method": config.train_method,
            "ft_embed": config.ft_embed,
            "weight_decay": config.weight_decay,
            "pct_start": config.pct_start,
            "first_codebook_weight_multiplier": config.first_codebook_weight_multiplier,
            "text_padding_weight": config.text_padding_weight,
            "epad_as_padding": config.epad_as_padding,
            "user_loss": config.user_loss,
            "log_freq": config.log_freq,
            "no_eval": config.no_eval,
            "eval_on_train_samples": config.eval_on_train_samples,
            "ckpt_freq": config.ckpt_freq,
            "torch_compile_disabled": os.environ.get("NO_TORCH_COMPILE", ""),
            "qlora": config.qlora,
            "quant_type": config.quant_type if config.qlora else None,
            "device": str(device),
            "num_processes": world_size,
            "cpu_threads": cpu_threads,
            "gradient_accumulation_steps": accum_steps,
            "per_device_batch_size": config.per_device_batch_size,
            "num_workers": 0 if smoke else config.num_workers,
            "filter_num_workers": config.filter_num_workers,
            "force_filter": force_filter,
            "prefetch_factor": config.prefetch_factor,
            "pin_memory": config.pin_memory,
            "persistent_workers": config.persistent_workers and not smoke and config.num_workers > 0,
            "global_batch_size": global_batch_size,
            "warmup_steps": config.warmup_steps,
            "eval_every_steps": config.eval_every_steps,
            "free_running_eval_every_steps": config.free_running_eval_every_steps,
            "free_running_eval_samples": config.free_running_eval_samples,
            "free_running_eval_window_seconds": config.free_running_eval_window_seconds,
            "validation_max_samples": config.validation_max_samples,
            "validation_generation_settings": config.generation_settings.as_dict(),
            "save_every_steps": config.save_every_steps,
            "randomize_train": config.randomize_train,
            "prompt_aug_prob": config.prompt_aug_prob,
            "vietnamese_text_mode": config.vietnamese_text_mode,
            "static_chunking": config.static_chunking,
            "swap_roles_after_pass": config.swap_roles_after_pass,
            "gradient_checkpointing": config.gradient_checkpointing,
            "mixed_precision": config.mixed_precision,
            "num_train_samples": len(train_samples),
            "num_train_candidate_chunks": train_candidate_count,
            "num_val_samples": len(val_samples),
            "num_test_samples": len(test_samples),
            "num_train_conversations": len(train_conversations),
            "num_train_conversations_with_kept_chunks": len({sample.sample_id for sample in train_samples}),
            "num_train_role_views": len(train_samples),
        }
        (run_dir / "config.json").write_text(json.dumps(config_record, indent=2) + "\n", encoding="utf-8")
        filter_report = {
            "dataset_load": {
                label: {
                    "manifest_entries": report.manifest_entries,
                    "loaded_samples": report.loaded_samples,
                    "skipped_out_of_bounds_samples": report.skipped_out_of_bounds,
                    "skipped_invalid_samples": report.skipped_invalid,
                    "rejected_entries": list(report.rejected_entries),
                }
                for label, report in dataset_load_reports.items()
            },
            "chunk_filter": filter_payloads,
        }
        (run_dir / "data_filter_report.json").write_text(
            json.dumps(filter_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        print(json.dumps(config_record))
        print(f"[Main Process] Active across {world_size} process(es) on device {device}")

    if world_size > 1:
        run_dir_list = [str(run_dir) if main_process else ""]
        torch.distributed.broadcast_object_list(run_dir_list, src=0)
        run_dir = Path(run_dir_list[0])

    if run_dir is not None:
        write_rank_info(run_dir, rank, world_size, device, len(train_samples))
    if distributed:
        barrier()

    # DDP keeps a complete frozen base model on every device. Each rank loads
    # the same explicit local PersonaPlex checkpoint before DDP synchronizes it.
    runtime = load_runtime(
        RuntimePaths(config.model_root, config.personaplex_source),
        str(device), config.qlora, config.quant_type,
        model_device=str(device),
        load_model_weights=True,
        codec_cache_dir=config.codec_cache_dir,
        full_precision_model=config.train_method == "full",
    )
    if not math.isclose(runtime.codec.frame_rate, PERSONAPLEX_MIMI_FRAME_RATE, rel_tol=0, abs_tol=1e-6):
        raise RuntimeError(
            f"loaded Mimi frame_rate={runtime.codec.frame_rate:g} does not match "
            f"PersonaPlex frame grid {PERSONAPLEX_MIMI_FRAME_RATE:g}"
        )
    inspection = None
    for sample in train_samples:
        try:
            inspection = inspect_training_sample(
                sample, runtime.tokenizer, config.vietnamese_text_mode,
            )
            break
        except ValueError as exc:
            if "no agent text target" not in str(exc):
                raise
    if inspection is None:
        raise ValueError("training chunks contain no agent text targets")
    if main_process:
        print("[Training sample inspection] " + json.dumps(inspection, ensure_ascii=False))

    # Inject LoRA with Stage-Wise Freezing support
    stage = getattr(config, "train_stage", "joint").lower()
    lora_prefixes = lora_prefixes_for_stage(config)
    if stage in {"temporal_only", "freeze_depformer"}:
        if hasattr(runtime.model, "depformer"):
            runtime.model.depformer.requires_grad_(False)
        if main_process:
            print("[Stage-Wise Training] Active Stage: TEMPORAL ONLY (Depth Transformer / Depformer is 100% frozen).")
    elif stage in {"depth_only", "freeze_tempformer"}:
        if hasattr(runtime.model, "transformer"):
            runtime.model.transformer.requires_grad_(False)
        if main_process:
            print("[Stage-Wise Training] Active Stage: DEPTH ONLY (Temporal 7B Transformer is 100% frozen).")
    else:
        if main_process:
            print("[Stage-Wise Training] Active Stage: JOINT (Both Temporal & Depth active).")

    # Capture the true base before LoRA injection or resumed full weights.
    if val_samples and not config.no_eval and config.free_running_eval_every_steps > 0:
        barrier()
        baseline_result = [None]
        if main_process:
            try:
                baseline_result[0] = evaluate_free_running(
                    runtime, val_samples, config,
                    audio_output_dir=run_dir / "free-running" / "step_000000",
                )
            except Exception as exc:
                traceback.print_exc()
                baseline_result[0] = {"error": f"{type(exc).__name__}: {exc}"}
        if distributed:
            torch.distributed.broadcast_object_list(baseline_result, src=0, device=device)
        if isinstance(baseline_result[0], dict) and "error" in baseline_result[0]:
            raise RuntimeError(f"base-model free-running validation failed: {baseline_result[0]['error']}")
        barrier()

    targets = configure_trainable_parameters(runtime.model, config, lora_prefixes)

    # Optional gradient checkpointing
    if config.gradient_checkpointing:
        enable_gradient_checkpointing(runtime.model)
        if main_process:
            print("Gradient checkpointing enabled on transformer layers.")

    # Load adapter weights before DDP wrapping.
    resume_checkpoint_dir = None
    if resume_adapter_file is not None:
        resume_checkpoint_dir = resume_adapter_file.parent
        if main_process:
            print(f"Resuming {config.train_method} weights and optimizer state from {resume_adapter_file}")
        if config.train_method == "full":
            load_full_weights(runtime.model, resume_adapter_file)
        else:
            load_adapter(runtime.model, resume_adapter_file)
        if main_process:
            print(f"Checkpoint is at optimizer step {adapter_resume_step}")

    # Alias LMModel.forward to forward_train for training execution
    from moshi.models.lm import LMModel
    LMModel.forward = LMModel.forward_train

    trainable = [p for p in runtime.model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("model has no trainable parameters")
    unexpected_trainable = [
        name for name, parameter in runtime.model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if config.train_method == "lora" and unexpected_trainable:
        raise RuntimeError(f"unexpected non-LoRA trainable parameters: {unexpected_trainable[:5]}")
    if config.train_method == "full" and device.type == "cuda":
        # Parameters already occupy GPU memory. FP32 gradients and both AdamW
        # moment tensors are allocated later, mostly at the first optimizer step.
        free_bytes, _ = torch.cuda.mem_get_info(device)
        future_state_bytes = sum(parameter.numel() for parameter in trainable) * (16 if distributed else 12)
        activation_reserve_bytes = 16 * 1024**3
        if free_bytes < future_state_bytes + activation_reserve_bytes:
            raise RuntimeError(
                "insufficient free GPU memory for full DDP fine-tuning: "
                f"free={free_bytes / 1024**3:.1f} GiB, estimated gradients/AdamW/DDP buffers="
                f"{future_state_bytes / 1024**3:.1f} GiB plus "
                f"{activation_reserve_bytes / 1024**3:.0f} GiB activation reserve; "
                "DDP keeps a complete model and optimizer on every GPU"
            )
    if main_process:
        print(f"Training method: {config.train_method}; LoRA targets: {len(targets)}; trainable parameters: {sum(p.numel() for p in trainable):,}")

    temp_lr = config.learning_rate
    dep_lr = config.depformer_learning_rate
    if config.train_method == "full":
        no_decay = full_attention_no_decay_ids(runtime.model)
        groups = {}
        for name, parameter in runtime.model.named_parameters():
            if not parameter.requires_grad:
                continue
            lr = dep_lr if dep_lr is not None and "depformer" in name else temp_lr
            decay = 0.0 if id(parameter) in no_decay else config.weight_decay
            groups.setdefault((lr, decay), []).append(parameter)
        optimizer = torch.optim.AdamW(
            [{"params": params, "lr": lr, "weight_decay": decay} for (lr, decay), params in groups.items()],
            fused=device.type == "cuda",
        )
    elif dep_lr is not None and dep_lr != temp_lr:
        temp_params = [p for n, p in runtime.model.named_parameters() if p.requires_grad and "depformer" not in n]
        dep_params = [p for n, p in runtime.model.named_parameters() if p.requires_grad and "depformer" in n]
        param_groups = []
        if temp_params:
            param_groups.append({"params": temp_params, "lr": temp_lr})
        if dep_params:
            param_groups.append({"params": dep_params, "lr": dep_lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay, fused=device.type == "cuda")
        if main_process:
            print(f"Using Dual Learning Rates -> Temporal Transformer: {temp_lr:.2e}, Depth Transformer: {dep_lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(trainable, lr=temp_lr, weight_decay=config.weight_decay, fused=device.type == "cuda")
    max_steps = 1 if smoke else config.max_steps
    scheduler = None
    if max_steps > 1:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            total_steps=max_steps,
            pct_start=config.pct_start,
        )
    if distributed:
        runtime.model = torch.nn.parallel.DistributedDataParallel(
            runtime.model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=config.train_method == "full",
        )
        barrier()

    start_step = adapter_resume_step
    if resume_checkpoint_dir is not None:
        restored_step = load_training_state(
            resume_checkpoint_dir,
            optimizer,
            scheduler,
            accum_steps,
            world_size,
            config.per_device_batch_size,
            total_steps=max_steps if max_steps > 1 else None,
        )
        if restored_step is not None:
            if restored_step != adapter_resume_step:
                raise RuntimeError(
                    f"resume checkpoint step mismatch: adapter.json={adapter_resume_step}, "
                    f"training_state.pt={restored_step}"
                )
            start_step = restored_step
            if main_process:
                print(f"Restored optimizer and scheduler state at optimizer step {start_step}")
    start_micro_step = start_step * accum_steps
    validate_resume_step(start_step, max_steps)

    writer = None
    log_file = None
    if main_process and run_dir is not None:
        log_path = run_dir / "metrics.jsonl"
        log_file = log_path.open("w", encoding="utf-8")
        tensorboard_dir = run_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        writer.add_text("configuration", json.dumps(config_record, indent=2), 0)

    saved = None
    best_saved = None
    best_val_loss = float("inf")
    best_inference_saved = None
    best_generation_cer = float("inf")
    baseline_generation_cer = None
    best_generation_sample = None
    last_record = None
    reload_checks: list[dict[str, float | int]] = []
    started = time.monotonic()

    if val_samples and not config.no_eval and config.free_running_eval_every_steps > 0:
        # Establish the pre-training generation baseline with the same samples,
        # seed, and LMGen settings used for later checkpoint selection.
        barrier()
        baseline_generation_cer = float(baseline_result[0]["val/generation_cer"])
        if main_process and run_dir is not None:
            baseline_row = {
                "step": 0, "val/generation_baseline": True, **baseline_result[0],
            }
            with (run_dir / "free_running_metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(baseline_row, ensure_ascii=False) + "\n")
            if writer:
                writer.add_scalar("val/generation_cer_base", baseline_generation_cer, 0)
                writer.add_scalar(
                    "val/generation_empty_samples_base",
                    baseline_result[0]["val/generation_empty_samples"], 0,
                )
            print(json.dumps({
                "event": "starting_generation_baseline",
                "cer": baseline_generation_cer,
                "wer": baseline_result[0]["val/generation_wer"],
                "empty_samples": baseline_result[0]["val/generation_empty_samples"],
            }, ensure_ascii=False))
        barrier()

    try:
        max_micro_steps = max_steps * accum_steps
        progress = tqdm(total=max_steps, initial=start_step, desc="training (DDP)" if distributed else "training", unit="update", dynamic_ncols=True) if main_process else None
        update_started_at = None
        pending_train_seconds = 0.0
        pending_train_updates = 0
        batch_iterator = iter(iter_training_batches(
            config, train_samples, runtime, device, rank,
            world_size, smoke, skip_batches=start_micro_step,
        ))
        samples_seen = 0
        audio_seconds_seen = 0.0
        audio_frames_seen = 0
        text_target_tokens_seen = 0
        text_padding_positions_seen = 0
        pending_text_target_tokens = torch.zeros((), dtype=torch.int64, device=device)
        pending_text_padding_positions = torch.zeros((), dtype=torch.int64, device=device)
        pending_text_target_ce_sum = torch.zeros((), dtype=torch.float32, device=device)
        pending_text_target_ce_count = torch.zeros((), dtype=torch.int64, device=device)
        pending_text_target_correct = torch.zeros((), dtype=torch.int64, device=device)
        pending_text_prediction_counts = torch.zeros(3, dtype=torch.int64, device=device)
        pending_audio_cb_correct = torch.zeros(16, dtype=torch.int64, device=device)
        pending_audio_cb_count = torch.zeros(16, dtype=torch.int64, device=device)
        pending_audio_cb_loss_sum = torch.zeros(16, dtype=torch.float32, device=device)
        pending_profile_times = torch.zeros(3, dtype=torch.float32, device=device)

        def profile_sync() -> None:
            if config.profile_steps and device.type == "cuda":
                torch.cuda.synchronize(device)

        optimizer_step = start_step
        for micro_step in range(start_micro_step, max_micro_steps):
            if update_started_at is None:
                # Start after the previous update's validation/checkpoint work.
                update_started_at = time.monotonic()
            profile_sync()
            data_started = time.monotonic() if config.profile_steps else 0.0
            try:
                with trace_batch_preparation(rank, micro_step,
                                             enabled=distributed and micro_step == start_micro_step):
                    batch, epoch, local_samples, local_audio_seconds, local_audio_frames, batch_samples = next(batch_iterator)
            except StopIteration as exc:
                raise RuntimeError("training batch iterator stopped before max_steps") from exc
            profile_sync()
            if config.profile_steps:
                pending_profile_times[0] += time.monotonic() - data_started
            codes = batch["codes"]
            samples_seen += local_samples * world_size
            if distributed and micro_step == start_micro_step:
                log_batch_phase(rank, "audio_stats_allreduce", "start")
            global_audio_seconds = all_reduce_sum(
                torch.tensor([local_audio_seconds, local_audio_frames], dtype=torch.float32, device=device)
            )
            if distributed and micro_step == start_micro_step:
                log_batch_phase(rank, "audio_stats_allreduce", "complete")
            audio_seconds_seen += float(global_audio_seconds[0])
            audio_frames_seen += int(global_audio_seconds[1])

            sync_gradients = (micro_step + 1 - start_micro_step) % accum_steps == 0 or micro_step + 1 == max_micro_steps
            sync_context = runtime.model.no_sync() if distributed and not sync_gradients else nullcontext()
            with sync_context:
                profile_sync()
                forward_started = time.monotonic() if config.profile_steps else 0.0
                output = model_forward_train(runtime.model, codes)
                text_targets, text_padding = text_supervision_counts(
                    batch, output, tokenizer_text_padding_ids(runtime.tokenizer, config.epad_as_padding)
                )
                pending_text_target_tokens += text_targets.detach()
                pending_text_padding_positions += text_padding.detach()
                target_ce_sum, target_ce_count = text_target_token_loss_stats(
                    batch, output, tokenizer_text_padding_ids(runtime.tokenizer, config.epad_as_padding)
                )
                pending_text_target_ce_sum += target_ce_sum
                pending_text_target_ce_count += target_ce_count.detach()
                diag_t_corr, diag_t_cnt, diag_t_loss, diag_cb_corr, diag_cb_cnt, diag_cb_loss = codebook_diagnostic_stats(
                    batch, output, tokenizer_text_padding_ids(runtime.tokenizer, config.epad_as_padding)
                )
                pending_text_target_correct += diag_t_corr
                pending_text_prediction_counts += text_prediction_diagnostic_counts(
                    batch, output, tokenizer_text_padding_ids(runtime.tokenizer, config.epad_as_padding)
                )
                pending_audio_cb_correct += diag_cb_corr
                pending_audio_cb_count += diag_cb_cnt
                pending_audio_cb_loss_sum += diag_cb_loss

                loss_result = loss_components(
                    output, codes, batch, tokenizer_text_padding_ids(runtime.tokenizer, config.epad_as_padding), torch,
                    config.first_codebook_weight_multiplier, config.text_padding_weight,
                    user_loss=config.user_loss,
                    distributed=distributed,
                )
                if distributed:
                    components, total = reduce_distributed_loss(
                        loss_result[1], all_reduce_sum, preserve_grad=True, world_size=world_size
                    )
                else:
                    total = loss_result[0]
                    components = loss_result[1]
                profile_sync()
                if config.profile_steps:
                    pending_profile_times[1] += time.monotonic() - forward_started
                profile_sync()
                backward_started = time.monotonic() if config.profile_steps else 0.0
                (total / accum_steps).backward()
                profile_sync()
                if config.profile_steps:
                    pending_profile_times[2] += time.monotonic() - backward_started
            profile_sync()
            optimizer_started = time.monotonic() if config.profile_steps else 0.0
            if config.train_method == "full" and optimizer_step == start_step and sync_gradients:
                missing_gradients = [
                    name for name, parameter in unwrap_parallel_model(runtime.model).named_parameters()
                    if parameter.requires_grad and parameter.grad is None
                ]
                if missing_gradients:
                    raise RuntimeError(
                        "full fine-tuning has LM parameters without gradients: "
                        f"{missing_gradients[:8]} ({len(missing_gradients)} total)"
                    )
            sync_state = type("SyncState", (), {"sync_gradients": sync_gradients})()
            grad_norm = step_optimizer_if_ready(sync_state, optimizer, scheduler, trainable, runtime.model)
            profile_sync()
            optimizer_seconds = time.monotonic() - optimizer_started if config.profile_steps else 0.0

            if not sync_gradients:
                continue

            global_text_stats = all_reduce_sum(pack_text_training_stats(
                pending_text_target_tokens,
                pending_text_padding_positions,
                pending_text_target_ce_sum,
                pending_text_target_ce_count,
                pending_text_target_correct,
                pending_text_prediction_counts,
            ))
            global_audio_cb_stats = all_reduce_sum(torch.stack([
                pending_audio_cb_correct.float(),
                pending_audio_cb_count.float(),
                pending_audio_cb_loss_sum,
            ]))  # shape [3, 16]
            text_target_tokens_seen += int(global_text_stats[0])
            text_padding_positions_seen += int(global_text_stats[1])
            text_nonpadding_loss = float(
                global_text_stats[2] / global_text_stats[3].clamp_min(1.0)
            )
            text_token_acc = float(
                global_text_stats[4] / global_text_stats[3].clamp_min(1.0)
            ) if global_text_stats[3] > 0 else 0.0
            text_nonpad_count = global_text_stats[0]
            text_pad_count = global_text_stats[1]
            total_text_count = text_nonpad_count + text_pad_count
            text_pad_correct = global_text_stats[5]
            predicted_pad_count = global_text_stats[6]
            predicted_pad_nonpad_count = global_text_stats[7]
            text_nonpad_accuracy = float(
                global_text_stats[4] / text_nonpad_count.clamp_min(1.0)
            ) if text_nonpad_count > 0 else 0.0
            text_pad_accuracy = float(
                text_pad_correct / text_pad_count.clamp_min(1.0)
            ) if text_pad_count > 0 else 0.0
            target_pad_pct = float(text_pad_count / total_text_count.clamp_min(1.0) * 100.0)
            predicted_pad_pct = float(predicted_pad_count / total_text_count.clamp_min(1.0) * 100.0)
            pad_given_nonpad_target = float(
                predicted_pad_nonpad_count / text_nonpad_count.clamp_min(1.0)
            ) if text_nonpad_count > 0 else 0.0

            audio_cb_corr_reduced = global_audio_cb_stats[0]
            audio_cb_cnt_reduced = global_audio_cb_stats[1]
            audio_cb_loss_reduced = global_audio_cb_stats[2]

            cb_losses = [
                float(audio_cb_loss_reduced[i] / audio_cb_cnt_reduced[i].clamp_min(1.0))
                for i in range(16)
            ]
            cb_accuracies = [
                float(audio_cb_corr_reduced[i] / audio_cb_cnt_reduced[i].clamp_min(1.0))
                for i in range(16)
            ]
            mean_audio_loss = float((components["audio_semantic"] + components["audio_nonsemantic"]).detach())
            mean_audio_acc = (
                float(audio_cb_corr_reduced.sum() / audio_cb_cnt_reduced.sum().clamp_min(1.0))
                if audio_cb_cnt_reduced.sum() > 0 else 0.0
            )

            pending_text_target_tokens.zero_()
            pending_text_padding_positions.zero_()
            pending_text_target_ce_sum.zero_()
            pending_text_target_ce_count.zero_()
            pending_text_target_correct.zero_()
            pending_text_prediction_counts.zero_()
            pending_audio_cb_correct.zero_()
            pending_audio_cb_count.zero_()
            pending_audio_cb_loss_sum.zero_()

            pending_train_seconds += time.monotonic() - update_started_at
            pending_train_updates += 1

            profile_times = None
            if config.profile_steps:
                profile_times = torch.cat((
                    pending_profile_times,
                    torch.tensor([optimizer_seconds], dtype=torch.float32, device=device),
                ))
                if distributed:
                    torch.distributed.all_reduce(profile_times, op=torch.distributed.ReduceOp.MAX)
                profile_times = profile_times.cpu().tolist()
                pending_profile_times.zero_()

            optimizer_step += 1

            reduced_total = total.detach()
            reduced_text = components["text"].detach()
            reduced_agent_semantic = components["agent_semantic"].detach()
            reduced_agent_acoustic = components["agent_acoustic"].detach()
            reduced_user_semantic = components["user_semantic"].detach()
            reduced_user_acoustic = components["user_acoustic"].detach()
            reduced_sem = components["audio_semantic"].detach()
            reduced_nonsem = components["audio_nonsemantic"].detach()
            reduced_audio_total = components["audio_total"].detach()

            if main_process and (optimizer_step % config.log_freq == 0 or optimizer_step == max_steps):
                total_target_positions = text_target_tokens_seen + text_padding_positions_seen
                valid_token_pct = (text_target_tokens_seen / max(1, total_target_positions)) * 100.0

                record = {
                    "step": optimizer_step,
                    "micro_step": micro_step + 1,
                    "epoch": epoch,
                    "samples_seen": samples_seen,
                    "audio_seconds_seen": audio_seconds_seen,
                    "audio_frames_seen": audio_frames_seen,
                    "text_target_tokens_seen": text_target_tokens_seen,
                    "text_padding_positions_seen": text_padding_positions_seen,
                    "valid_token_pct": valid_token_pct,
                    "global_batch_size": global_batch_size,
                    "loss/total": float(reduced_total.detach()),
                    "loss/text": float(reduced_text.detach()),
                    "loss/text_real": text_nonpadding_loss,
                    "loss/text_nonpadding": text_nonpadding_loss,
                    "loss/audio_total": float(reduced_audio_total),
                    "loss/agent_semantic": float(reduced_agent_semantic),
                    "loss/agent_acoustic": float(reduced_agent_acoustic),
                    "loss/user_semantic": float(reduced_user_semantic),
                    "loss/user_acoustic": float(reduced_user_acoustic),
                    "loss/audio_semantic": float(reduced_sem.detach()),
                    "loss/audio_nonsemantic": float(reduced_nonsem.detach()),
                    "accuracy/text": text_token_acc,
                    "accuracy/text_nonpad": text_nonpad_accuracy,
                    "accuracy/text_pad": text_pad_accuracy,
                    "target_pad_pct": target_pad_pct,
                    "predicted_pad_pct": predicted_pad_pct,
                    "probability/pad_given_nonpad_target": pad_given_nonpad_target,
                    "probability/correct_given_nonpad_target": text_nonpad_accuracy,
                    "accuracy/audio_total": mean_audio_acc,
                    **{f"loss/audio_cb{i}": cb_losses[i] for i in range(16)},
                    **{f"accuracy/audio_cb{i}": cb_accuracies[i] for i in range(16)},
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                    "samples_per_second": (
                        global_batch_size * pending_train_updates
                        / max(pending_train_seconds, 1e-9)
                    ),
                    "timing/train_update_sec_mean": pending_train_seconds / pending_train_updates,
                    "gpu_peak_bytes": device_memory_bytes(device),
                }
                if profile_times is not None:
                    record.update({
                        "timing/data_sec": profile_times[0],
                        "timing/forward_loss_sec": profile_times[1],
                        "timing/backward_sec": profile_times[2],
                        "timing/optimizer_sec": profile_times[3],
                        "timing/profiled_phase_sum_sec": sum(profile_times),
                    })
                pending_train_seconds = 0.0
                pending_train_updates = 0
                last_record = record
                if log_file:
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                if writer:
                    write_tensorboard_scalars(writer, record, sum(p.numel() for p in trainable), cpu_threads)
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        loss=f"{record['loss/total']:.3f}",
                        t_acc=f"{text_token_acc * 100:.1f}%",
                        a_acc=f"{mean_audio_acc * 100:.1f}%",
                        cb0=f"{cb_accuracies[0] * 100:.1f}%",
                        grad=f"{grad_norm:.2f}",
                    )
                if max_steps == 1:
                    tqdm.write(json.dumps({"event": "training_step", **record}))

            # Reuse a full checkpoint when several save reasons coincide at
            # this optimizer step. Best references point to immutable step dirs.
            step_full_saved = None

            def save_full_step_once():
                nonlocal step_full_saved
                if step_full_saved is None:
                    step_full_saved = save_full_checkpoint(
                        run_dir, unwrap_parallel_model(runtime.model), config, optimizer_step,
                        optimizer, scheduler, accum_steps, world_size,
                    )
                return step_full_saved

            # Validation evaluation
            teacher_eval_due = (
                config.eval_every_steps > 0
                and (optimizer_step % config.eval_every_steps == 0 or optimizer_step == max_steps)
            )
            generation_eval_due = (
                config.free_running_eval_every_steps > 0
                and (optimizer_step % config.free_running_eval_every_steps == 0 or optimizer_step == max_steps)
            )
            if val_samples and not config.no_eval and (teacher_eval_due or generation_eval_due):
                barrier()
                val_metrics = evaluate_validation(config, runtime, val_samples, rank, world_size, device)
                barrier()
                best_state = (
                    adapter_state_dict(unwrap_parallel_model(runtime.model))
                    if config.train_method == "lora" else None
                )
                generation_metrics = None
                if generation_eval_due:
                    barrier()
                    generation_result = [None]
                    if main_process:
                        try:
                            generation_result[0] = evaluate_free_running(
                                runtime, val_samples, config,
                                audio_output_dir=(
                                    run_dir / "free-running" / f"step_{optimizer_step:06d}"
                                ),
                                step=optimizer_step,
                                baseline_dir=run_dir / "free-running" / "step_000000",
                            )
                        except Exception as exc:
                            traceback.print_exc()
                            generation_result[0] = {"error": f"{type(exc).__name__}: {exc}"}
                    if distributed:
                        torch.distributed.broadcast_object_list(generation_result, src=0, device=device)
                    if isinstance(generation_result[0], dict) and "error" in generation_result[0]:
                        raise RuntimeError(f"free-running validation failed: {generation_result[0]['error']}")
                    generation_metrics = generation_result[0]
                    if main_process:
                        generation_row = {"step": optimizer_step, **generation_metrics}
                        with (run_dir / "free_running_metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
                            metrics_file.write(json.dumps(generation_row, ensure_ascii=False) + "\n")
                        if writer:
                            writer.add_scalar("val/generation_cer", generation_metrics["val/generation_cer"], optimizer_step)
                            writer.add_scalar("val/generation_wer", generation_metrics["val/generation_wer"], optimizer_step)
                        # A blank transcript is a failed generation even when its
                        # normalized CER happens to beat earlier checkpoints.
                        generation_score = generation_checkpoint_score(
                            generation_metrics, baseline_cer=baseline_generation_cer,
                        )
                        if generation_score < best_generation_cer:
                            best_generation_cer = generation_score
                            best_generation_sample = generation_metrics["samples"][0]
                            if config.train_method == "full":
                                best_inference_saved = save_full_step_once()
                            else:
                                best_inference_saved = save_best_adapter_state(
                                    run_dir, best_state, config, optimizer_step,
                                    val_metrics.get("val/loss_selection", float("inf")), optimizer, scheduler,
                                    gradient_accumulation_steps=accum_steps, num_processes=world_size,
                                    checkpoint_name="best_inference",
                                )
                            tqdm.write(json.dumps({
                                "event": "new_best_free_running_checkpoint", "step": optimizer_step,
                                "cer": best_generation_cer,
                                "wer": generation_metrics["val/generation_wer"],
                                "audio_path": best_generation_sample["audio_path"],
                                "original_audio_path": best_generation_sample["original_audio_path"],
                                "adapter": str(best_inference_saved),
                            }, ensure_ascii=False))
                    barrier()
                if main_process:
                    for k, v in val_metrics.items():
                        if writer:
                            writer.add_scalar(k, v, optimizer_step)
                    val_loss = validation_selection_loss(val_metrics)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if config.train_method == "full":
                            best_saved = save_full_step_once()
                        else:
                            best_saved = save_best_adapter_state(
                                run_dir, best_state, config, optimizer_step, val_loss, optimizer, scheduler,
                                gradient_accumulation_steps=accum_steps, num_processes=world_size,
                                checkpoint_name="best_loss",
                            )
                        tqdm.write(json.dumps({
                            "event": "new_best_val_checkpoint", "step": optimizer_step,
                            "selection_loss": val_loss,
                            "text_real_loss": val_metrics.get("val/loss_text_real"),
                            "agent_semantic_loss": val_metrics.get("val/loss_agent_semantic"),
                            "agent_acoustic_loss": val_metrics.get("val/loss_agent_acoustic"),
                            "user_semantic_loss": val_metrics.get("val/loss_user_semantic"),
                            "user_acoustic_loss": val_metrics.get("val/loss_user_acoustic"),
                            "audio_semantic_loss": val_metrics.get("val/loss_audio_semantic"),
                            "audio_nonsemantic_loss": val_metrics.get("val/loss_audio_nonsemantic"),
                        }))

            # Periodic saving & smoke reload verification
            save_interval = 1 if smoke else config.ckpt_freq
            if optimizer_step % save_interval == 0 or optimizer_step == max_steps:
                barrier()
                if main_process:
                    if config.train_method == "full":
                        saved = save_full_step_once()
                    else:
                        checkpoint_state = adapter_state_dict(unwrap_parallel_model(runtime.model))
                        saved = save_adapter_state(
                            run_dir, checkpoint_state, config, optimizer_step, optimizer, scheduler,
                            accum_steps, world_size,
                        )
                    if smoke:
                        reload_loss = (
                            verify_reloaded_full(config, batch_samples[0], saved, runtime)
                            if config.train_method == "full"
                            else verify_reloaded_adapter(config, batch_samples[0], saved, runtime)
                        )
                        reload_checks.append({"step": optimizer_step, "loss": reload_loss})
                barrier()

            # Validation, checkpoint writes, and reload checks are not training
            # work. Start the next timing window only after they have completed.
            update_started_at = None

        if log_file:
            log_file.close()
        if progress is not None:
            progress.close()
    finally:
        if writer:
            writer.close()

    if run_dir is not None:
        rank_peak = device_memory_bytes(device)
        write_rank_info(run_dir, rank, world_size, device, len(train_samples), rank_peak)
    barrier()

    if main_process and run_dir is not None:
        peak = device_memory_bytes(device)
        run_info = {
            "seconds": time.monotonic() - started,
            "peak_gpu_bytes": peak,
            "checkpoint": str(saved) if saved else None,
            "best_checkpoint": str(best_saved) if best_saved else None,
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "best_inference_checkpoint": str(best_inference_saved) if best_inference_saved else None,
            "best_generation_cer": best_generation_cer if best_generation_cer < float("inf") else None,
            "starting_generation_cer": baseline_generation_cer,
            "inference_checkpoint_status": (
                "validated_generation_improves_base"
                if best_inference_saved is not None else "no_checkpoint_improved_base_generation"
            ),
            "best_inference_sample": best_generation_sample,
            "reload_checks": reload_checks,
        }
        report = run_dir / "training_report.md"
        inference_adapter = best_inference_saved or saved
        inference_config_path = run_dir / "inference_config.json"
        inference_output_dir = run_dir / "inference_smoke"
        inference_config_path.write_text(
            json.dumps(
                inference_config_snapshot(config, inference_adapter, inference_output_dir),
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        run_info["inference_config"] = str(inference_config_path)
        (run_dir / "run.json").write_text(json.dumps(run_info, indent=2))
        inference_args = [
            sys.executable, "-m", "tools.inference_smoke", "--config", str(inference_config_path),
            "--checkpoint" if config.train_method == "full" else "--adapter", str(inference_adapter),
            "--output-dir", str(inference_output_dir),
        ]
        if best_generation_sample is not None:
            inference_args.extend([
                "--split", "train" if config.eval_on_train_samples else "validation",
                "--sample-id", str(best_generation_sample["sample_id"]),
                "--start", str(best_generation_sample["window_start_sec"]),
                "--window-seconds", str(best_generation_sample["window_end_sec"] - best_generation_sample["window_start_sec"]),
            ])
        training_duration_inference_args = None
        if (
            best_generation_sample is not None
            and best_generation_sample["source_duration_sec"]
            >= best_generation_sample["window_start_sec"] + config.duration_sec
        ):
            training_duration_inference_args = list(inference_args)
            duration_index = training_duration_inference_args.index("--window-seconds") + 1
            training_duration_inference_args[duration_index] = str(config.duration_sec)
            training_duration_inference_args.extend([
                "--output-dir", str(run_dir / "inference_training_duration"),
            ])
        report.write_text(
            "# PersonaPlex Training Report\n\n"
            f"Status: {'GENERATION-VALIDATED' if best_inference_saved is not None else 'NO GENERATION-VALID CHECKPOINT'} — "
            "training loss alone does not establish usable inference.\n\n"
            f"- Command: `{shlex.join([sys.executable, *sys.argv])}`\n"
            f"- Dataset: {config.manifest}\n- Model: {config.model_root}\n- Method: {config.train_method}\n"
            f"- LoRA: {f'rank={config.lora_rank}, alpha={config.lora_alpha}' if config.train_method == 'lora' else 'disabled'}\n"
            f"- Processes (GPUs): {world_size}\n"
            f"- Steps: {max_steps}\n- Final loss: {last_record['loss/total'] if last_record else 'n/a'}\n"
            f"- Text targets/padding positions: "
            f"{last_record['text_target_tokens_seen'] if last_record else 'n/a'}/"
            f"{last_record['text_padding_positions_seen'] if last_record else 'n/a'}\n"
            f"- Non-padding text loss: {last_record['loss/text_nonpadding'] if last_record else 'n/a'}\n"
            f"- Best val loss: {best_val_loss if best_val_loss < float('inf') else 'n/a'}\n"
            f"- Peak GPU bytes: {peak}\n- Checkpoint: {saved}\n"
            f"- Best checkpoint: {best_saved}\n"
            f"- Best inference checkpoint: {best_inference_saved}\n"
            f"- Starting free-running CER: {baseline_generation_cer if baseline_generation_cer is not None else 'n/a'}\n"
            f"- Best free-running CER: {best_generation_cer if best_generation_cer < float('inf') else 'n/a'}\n"
            f"- Best inference sample: {best_generation_sample['sample_id'] if best_generation_sample else 'n/a'}\n"
            f"- Inference command: `{shlex.join(inference_args)}`\n",
            encoding="utf-8",
        )
        if training_duration_inference_args is not None:
            with report.open("a", encoding="utf-8") as report_file:
                report_file.write(
                    f"- Same checkpoint/sample with training-duration input ({config.duration_sec:g}s): "
                    f"`{shlex.join(training_duration_inference_args)}`\n"
                )

    barrier()
    return best_inference_saved or best_saved or saved


def main() -> int:
    parser = argparse.ArgumentParser(description="PersonaPlex Fine-Tuning with Moshi-style training")
    parser.add_argument("config_file", nargs="?", help="Path to Hydra YAML configuration file")
    parser.add_argument("--config", type=str, default=None, help="Path to Hydra YAML configuration file")
    parser.add_argument("--config-name", type=str, default=None, help="Name of config in configs/ directory (Hydra style)")
    parser.add_argument("--smoke", action="store_true", help="Run 1-step smoke test with verification")
    parser.add_argument("--qlora", action="store_true", default=None, help="Enable 4-bit QLoRA")
    parser.add_argument("--no-qlora", dest="qlora", action="store_false", help="Disable QLoRA")
    parser.add_argument("--user-loss", action="store_true", default=None, help="Supervise user audio stream codebooks (ablation)")
    parser.add_argument("--no-user-loss", dest="user_loss", action="store_false", help="Disable user audio supervision")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--force-filter", action="store_true", help="Revalidate prepared samples and rebuild chunk-filter caches")

    args, unknown = parser.parse_known_args()
    overrides = [arg for arg in unknown if "=" in arg]

    if args.config and args.config_file:
        parser.error("pass the config either positionally or with --config, not both")
    config_path = args.config or args.config_file
    if config_path is None:
        if args.config_name:
            config_path = Path("configs") / f"{args.config_name}.yaml"
        else:
            config_path = Path("configs/config.yaml")

    config = load_config(config_path, overrides=overrides)
    if args.qlora is not None:
        config = config.replace(qlora=args.qlora)
    if args.user_loss is not None:
        config = config.replace(user_loss=args.user_loss)

    run(
        config,
        smoke=args.smoke,
        resume_from=args.resume_from,
        force_filter=args.force_filter,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
