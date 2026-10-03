from __future__ import annotations

import argparse
import json
from pathlib import Path

from personaplex_finetuning.config import load_config
from personaplex_finetuning.data import PreparedDataset
from personaplex_finetuning.objective import stream_weights
from personaplex_finetuning.runtime import RuntimePaths, load_runtime
from personaplex_finetuning.sequence import PersonaPlexTrainingExampleBuilder, TrainingExample


def native_debug_delay(example: TrainingExample, initial_tokens, delays, zero_token: int) -> TrainingExample:
    """Render native delay positions for inspection only; training uses canonical codes."""
    import torch
    from moshi.models.lm import _delay_sequence

    if len(initial_tokens) != 17 or len(delays) != 17:
        raise ValueError("debug export requires 17 initial tokens and delays")
    maximum = max(delays)
    canonical = torch.tensor(example.input_codes, dtype=torch.long)[None]
    mask = torch.tensor(example.loss_mask, dtype=torch.bool)[None]
    initial = torch.tensor(initial_tokens, dtype=torch.long)[None, :, None]
    tail = torch.full((1, 17, maximum), zero_token, dtype=torch.long)
    delayed = _delay_sequence(delays, torch.cat([canonical, tail], dim=2), initial)
    delayed = torch.cat([initial, delayed], dim=2)[0]
    delayed_mask = _delay_sequence(
        delays, torch.nn.functional.pad(mask, (0, maximum), value=False),
        torch.zeros((1, 17, 1), dtype=torch.bool),
    )
    delayed_mask = torch.nn.functional.pad(delayed_mask, (1, 0), value=False)[0]
    streams = tuple(tuple(int(token) for token in row) for row in delayed.tolist())
    masks = tuple(tuple(bool(value) for value in row) for row in delayed_mask.tolist())
    return TrainingExample(
        streams, streams, masks, example.stream_names, example.prompt_frames,
        example.dialogue_frames, example.voice_prompt_frames, example.text_prompt_frames,
        example.word_alignments,
    )


def _token_piece(tokenizer, token_id: int) -> str:
    if token_id == tokenizer.padding_id:
        return "PAD"
    if token_id == tokenizer.end_padding_id:
        return "END_PAD"
    if token_id < 0:
        return "DELAY_PAD"
    processor = getattr(tokenizer, "_processor", None)
    if processor is not None:
        piece_size = getattr(processor, "get_piece_size", lambda: token_id + 1)()
        if token_id >= piece_size:
            return f"SPECIAL_{token_id}"
        return processor.id_to_piece(token_id)
    return str(token_id)


def build_debug_payload(
    sample, example: TrainingExample, tokenizer, frame_rate: float, delays,
    *, first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 0.3,
    user_loss: bool = False,
):
    if len(delays) != 17:
        raise ValueError("debug export requires 17 stream delays")
    text_offset = 1 + delays[0]
    voice_start = text_offset
    voice_end = voice_start + example.voice_prompt_frames
    text_start = voice_end + PersonaPlexTrainingExampleBuilder.pause_frames
    text_end = text_start + example.text_prompt_frames
    dialogue_start = text_offset + example.prompt_frames
    dialogue_end = dialogue_start + example.dialogue_frames
    if not voice_end < text_start or not text_end < dialogue_start:
        raise AssertionError("hybrid prompt segments are not ordered with pauses")

    labels = example.labels
    text_padding_ids = (tokenizer.padding_id, tokenizer.end_padding_id)
    weights = stream_weights(
        labels, example.loss_mask, text_padding_ids,
        text_padding_weight=text_padding_weight,
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
        user_loss=user_loss,
    )
    max_delay = max(delays)
    content_frames = 1 + max_delay + example.prompt_frames + example.dialogue_frames
    word_at_frame: dict[int, str] = {}
    word_events = []
    for alignment in example.word_alignments:
        token_frames = [dialogue_start + frame for frame in alignment.token_frames]
        for frame in token_frames:
            word_at_frame[frame] = alignment.word
        word_events.append({
            "speaker": alignment.speaker,
            "word": alignment.word,
            "start_sec": alignment.start_sec,
            "end_sec": alignment.end_sec,
            "dialogue_frame": alignment.start_frame,
            "token_frames": token_frames,
        })

    regions = {
        "voice_prompt": {"start": voice_start, "end": voice_end},
        "pause_after_voice": {"start": voice_end, "end": text_start},
        "text_prompt": {"start": text_start, "end": text_end},
        "pause_after_text": {"start": text_end, "end": dialogue_start},
        "dialogue": {"start": dialogue_start, "end": dialogue_end},
    }
    stream_regions = {}
    for stream_index, stream_name in enumerate(example.stream_names):
        offset = 1 + delays[stream_index]
        stream_regions[stream_name] = {
            name: {"start": region["start"] - text_offset + offset,
                   "end": region["end"] - text_offset + offset}
            for name, region in regions.items()
        }
    frames = []
    for index, token_id in enumerate(labels[0]):
        if index < text_offset:
            segment = "initial/delay padding"
        elif voice_start <= index < voice_end:
            segment = "voice prompt"
        elif voice_end <= index < text_start:
            segment = "pause after voice"
        elif text_start <= index < text_end:
            segment = "text prompt"
        elif text_end <= index < dialogue_start:
            segment = "pause after text"
        elif dialogue_start <= index < dialogue_end:
            segment = "dialogue"
        else:
            segment = "delay padding"
        audio_index = index - text_offset
        audio_codes = [
            int(labels[stream][audio_index + 1 + delays[stream]])
            if 0 <= audio_index < content_frames - (1 + max_delay) else None
            for stream in range(1, 17)
        ]
        frames.append({
            "frame": index,
            "time_sec": round(index / frame_rate, 6),
            "source_time_sec": round(
                sample.window_start_sec + (index - dialogue_start) / frame_rate, 6
            ) if dialogue_start <= index < dialogue_end else None,
            "segment": segment,
            "word": word_at_frame.get(index),
            "token_id": int(token_id),
            "token_piece": _token_piece(tokenizer, int(token_id)),
            "loss_mask": bool(example.loss_mask[0][index]),
            "loss_weight": float(weights[0][index]),
            "agent_audio_codes": audio_codes[:8],
            "user_audio_codes": audio_codes[8:],
        })

    stream_dump = []
    for index, name in enumerate(example.stream_names):
        stream_dump.append({
            "stream": name,
            "mask": [bool(value) for value in example.loss_mask[index]],
            "weights": [float(value) for value in weights[index]],
        })
    payload = {
        "sample_id": sample.sample_id,
        "window": {"start_sec": sample.window_start_sec, "end_sec": sample.window_end_sec},
        "channel_map": {
            "left": "agent target",
            "right": "user input and target" if user_loss else "user conditioning",
        },
        "user_loss": user_loss,
        "frame_rate": frame_rate,
        "frame_count": example.total_frames,
        "regions": regions,
        "stream_regions": stream_regions,
        "word_events": word_events,
        "frames": frames,
        "streams": stream_dump,
        "notes": {
            "dialogue_text_target": "agent only",
            "user_audio": "conditioning only",
            "batch_padding": "not present in this single-sample inspection",
        },
    }
    return payload


def render_debug_text(payload) -> str:
    rows = [
        f"sample_id: {payload['sample_id']}",
        f"window: {payload['window']['start_sec']:.3f}-{payload['window']['end_sec']:.3f} sec",
        f"Mimi frame rate: {payload['frame_rate']:.3f} Hz",
        f"stream shape: 17 x {payload['frame_count']}",
        "",
        "Prompt regions (end frame is exclusive):",
    ]
    for name, region in payload["regions"].items():
        rows.append(f"  {name}: [{region['start']}, {region['end']}) ({region['end'] - region['start']} frames)")
    rows += ["", "Text frame visualization:", "frame  seq_time  source_time  word       token_id  token_piece              segment"]
    for frame in payload["frames"]:
        rows.append(
            f"{frame['frame']:5d}  {frame['time_sec']:8.3f}  "
            f"{frame['source_time_sec'] if frame['source_time_sec'] is not None else '-':>11}  {(frame['word'] or '-'):10.10s} "
            f"{frame['token_id']:8d}  {frame['token_piece'][:24]:24s}  {frame['segment']}"
        )
    rows += ["", "Word timestamps:", "speaker  start(s)  end(s)  dialogue_frame  token_frames  word"]
    for event in payload["word_events"]:
        rows.append(
            f"{event['speaker']:7s}  {event['start_sec']:8.3f}  {event['end_sec']:6.3f}  "
            f"{event['dialogue_frame']:14d}  {event['token_frames']!s:12s}  {event['word']}"
        )
    rows += ["", "Actual loss masks and weights (0 = OFF, 1 = ON):"]
    for stream in payload["streams"]:
        name = stream["stream"]
        regions = payload["stream_regions"][name]
        prompt_region = regions["voice_prompt"]["start"], regions["pause_after_text"]["end"]
        dialogue_region = regions["dialogue"]["start"], regions["dialogue"]["end"]
        prompt_values = list(zip(
            stream["mask"][prompt_region[0]:prompt_region[1]],
            stream["weights"][prompt_region[0]:prompt_region[1]],
        ))
        dialogue_values = list(zip(
            stream["mask"][dialogue_region[0]:dialogue_region[1]],
            stream["weights"][dialogue_region[0]:dialogue_region[1]],
        ))
        prompt_mask = "".join("1" if mask else "0" for mask, _ in prompt_values)
        dialogue_mask = "".join("1" if mask else "0" for mask, _ in dialogue_values)
        prompt_weights = [round(weight, 6) for _, weight in prompt_values]
        dialogue_weights = [round(weight, 6) for _, weight in dialogue_values]
        rows.extend((
            f"{name}:",
            f"  Prompt   mask={prompt_mask} weights={prompt_weights}",
            f"  Dialogue mask={dialogue_mask} weights={dialogue_weights}",
        ))
    rows += ["", "Per-frame weights are in sequence_debug.json (including zero-weight delay positions)."]
    return "\n".join(rows) + "\n"


def write_debug_artifacts(output_dir: Path, payload) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sequence_debug.json"
    text_path = output_dir / "sequence_debug.txt"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(render_debug_text(payload), encoding="utf-8")
    return json_path, text_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a human-readable PersonaPlex training sequence.")
    parser.add_argument("sample_id", nargs="?", help="Sample ID from the prepared manifest")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--index", type=int, default=None, help="Sample index (default: first sample)")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults to outputs/inspect/<sample_id>")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args, unknown = parser.parse_known_args()
    if args.sample_id is not None and args.index is not None:
        parser.error("provide sample_id or --index, not both")
    overrides = [arg for arg in unknown if "=" in arg]
    config = load_config(args.config, overrides=overrides)
    import torch

    device = args.device or config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    samples = PreparedDataset(config.manifest, config.window_seconds).load()
    if args.sample_id is not None:
        sample = next((candidate for candidate in samples if candidate.sample_id == args.sample_id), None)
        if sample is None:
            raise ValueError(f"sample_id not found in manifest: {args.sample_id}")
    else:
        sample = samples[0 if args.index is None else args.index]
    runtime = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), device)
    builder = PersonaPlexTrainingExampleBuilder(
        runtime.codec, runtime.tokenizer, runtime.initial_tokens, runtime.zero_token,
        vietnamese_text_mode=config.vietnamese_text_mode,
    )
    example = native_debug_delay(builder.build(sample), runtime.initial_tokens, runtime.delays, runtime.zero_token)
    payload = build_debug_payload(
        sample, example, runtime.tokenizer, runtime.codec.frame_rate, runtime.delays,
        first_codebook_weight_multiplier=getattr(config, "first_codebook_weight_multiplier", 1.0),
        text_padding_weight=getattr(config, "text_padding_weight", 0.3),
        user_loss=getattr(config, "user_loss", False),
    )
    output_dir = args.output_dir or Path("outputs/inspect") / sample.sample_id
    json_path, text_path = write_debug_artifacts(output_dir, payload)
    print(f"wrote {json_path}\nwrote {text_path}")
    print(f"stream shape: 17 x {example.total_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
