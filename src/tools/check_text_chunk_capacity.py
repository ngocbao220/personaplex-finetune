"""Check text-token placement against real Mimi frames before fine-tuning."""

from __future__ import annotations

import argparse
import importlib
import json
import sys

from personaplex_finetuning.config import load_config
from personaplex_finetuning.chunk_filter import expected_mimi_frames
from personaplex_finetuning.data import PreparedDataset, duration_chunks, limit_conversations
from personaplex_finetuning.runtime import MimiCodec, RuntimePaths, SentencePieceTokenizer
from personaplex_finetuning.sequence import align_dialogue_text_targets


def _load_text_and_audio_tools(config):
    """Load only the local tokenizer and Mimi codec; do not initialize the 7B LM."""
    resolved = RuntimePaths(config.model_root, config.personaplex_source).validate(require_model=False)
    source = str(resolved.source)
    if source not in sys.path:
        sys.path.insert(0, source)

    loaders = importlib.import_module("moshi.models.loaders")
    lm_helpers = importlib.import_module("moshi.models.lm")
    mimi = loaders.get_mimi(resolved.mimi_weight, device=config.device)
    mimi.eval()
    mimi.requires_grad_(False)

    stat = resolved.mimi_weight.stat()
    cache_namespace = json.dumps(
        {"path": str(resolved.mimi_weight), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        sort_keys=True,
        separators=(",", ":"),
    )
    codec = MimiCodec(
        mimi, mimi.sample_rate, mimi.frame_rate, config.device, lm_helpers,
        cache_dir=config.codec_cache_dir, cache_namespace=cache_namespace,
    )
    return codec, SentencePieceTokenizer(resolved.tokenizer)


def _select_conversations(samples, sample_id: str | None, scan_all: bool):
    if sample_id is not None:
        selected = [sample for sample in samples if sample.sample_id == sample_id]
        if not selected:
            raise ValueError(f"sample_id not found in configured manifest: {sample_id}")
        return selected
    if scan_all:
        return samples
    return limit_conversations(samples, sample_number=10)


def _frame_count(codebooks, label: str) -> int:
    if len(codebooks) != 8 or not codebooks or len({len(book) for book in codebooks}) != 1:
        raise ValueError(f"Mimi returned invalid {label} codebooks")
    return len(codebooks[0])


def scan(config, conversations, codec, tokenizer) -> tuple[int, int]:
    chunks = duration_chunks(conversations, config.duration_sec)
    checked = overflows = 0
    expected_frames = expected_mimi_frames(config.duration_sec, codec.frame_rate)
    print(
        f"Conversations={len(conversations)} chunks={len(chunks)} "
        f"duration_sec={config.duration_sec:g} Mimi={codec.frame_rate:g}Hz "
        f"expected_frames={expected_frames} "
        f"vietnamese_text_mode={config.vietnamese_text_mode}",
        flush=True,
    )

    for chunk in chunks:
        agent_codes, user_codes = codec.encode_conversation_stereo_cached(
            chunk.conversation_wav, chunk.agent_channel, chunk.user_channel,
            chunk.window_start_sec, chunk.window_end_sec,
        )
        agent_frames = _frame_count(agent_codes, "agent")
        user_frames = _frame_count(user_codes, "user")
        frame_mismatch = agent_frames != user_frames or agent_frames != expected_frames
        roles = [("left-agent", chunk)]
        if config.swap_roles_after_pass:
            roles.append(("right-agent", chunk.swapped_roles()))

        for role, role_chunk in roles:
            result = align_dialogue_text_targets(
                role_chunk, agent_frames, codec.frame_rate, tokenizer,
                config.vietnamese_text_mode,
            )
            checked += 1
            failed = result.overflow_word is not None or frame_mismatch
            overflows += int(failed)
            overflow_detail = ""
            if result.overflow_word is not None:
                word = result.overflow_word
                overflow_detail = f" first_overflow={word.word!r}@{word.start:.3f}s"
            if frame_mismatch:
                overflow_detail += (
                    f" frame_mismatch=agent:{agent_frames},user:{user_frames},"
                    f"expected:{expected_frames}"
                )
            print(
                f"{'FAIL' if failed else 'OK'} sample={chunk.sample_id} "
                f"role={role} chunk={chunk.window_start_sec:.3f}-{chunk.window_end_sec:.3f}s "
                f"frames={agent_frames} text_tokens={result.placed_tokens}/{result.required_tokens}"
                f"{overflow_detail}",
                flush=True,
            )

    print(f"Checked role-views={checked}; failed={overflows}", flush=True)
    return checked, overflows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check transcript token placement against actual Mimi frames before training."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Training YAML config")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all", action="store_true", help="Scan every conversation in the manifest")
    scope.add_argument("--sample-id", help="Scan one conversation by sample_id")
    args, unknown = parser.parse_known_args()
    overrides = [value for value in unknown if "=" in value]

    try:
        config = load_config(args.config, overrides=overrides)
        samples = PreparedDataset(config.manifest, config.window_seconds).load()
        conversations = _select_conversations(samples, args.sample_id, args.all)
        codec, tokenizer = _load_text_and_audio_tools(config)
        _, failures = scan(config, conversations, codec, tokenizer)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
