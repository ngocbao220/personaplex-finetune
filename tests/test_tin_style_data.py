import json
import wave
from pathlib import Path

import pytest

from tin_style.data import load_sidecar, native_manifest, chunk_rejections, validate_chunk_manifest


def wav(path, channels):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\0\0" * channels * 24000)


@pytest.fixture
def prepared(tmp_path):
    sample = tmp_path / "samples" / "one"
    sample.mkdir(parents=True)
    wav(sample / "conversation.wav", 2)
    wav(sample / "voice_prompt.wav", 1)
    (sample / "metadata.json").write_text(json.dumps({
        "agent_channel": "left", "user_channel": "right", "text_prompt": "Be helpful.",
    }))
    (sample / "words.json").write_text(json.dumps([
        {"speaker": "user", "word": "Hi", "start": .5, "end": .7},
        {"speaker": "agent", "word": "Hello", "start": .1, "end": .4},
    ]))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"sample_id": "one", "sample_dir": "samples/one"}) + "\n")
    return sample, manifest


def test_prepared_mapping_and_native_manifest(prepared):
    sample, manifest = prepared
    before = {p.name: p.read_bytes() for p in sample.iterdir()}
    data = load_sidecar(str(sample / "conversation.wav"))
    assert data["text_prompt"] == "Be helpful."
    assert data["voice_prompt"] == str(sample / "voice_prompt.wav")
    assert data["alignments"] == [
        ["Hello", [.1, .4], "SPEAKER_BROKER"], ["Hi", [.5, .7], "SPEAKER_CLIENT"],
    ]
    rows = [json.loads(row) for row in Path(native_manifest(str(manifest))).read_text().splitlines()]
    assert rows == [{"path": str(sample / "conversation.wav"), "duration": 1.0}]
    assert {p.name: p.read_bytes() for p in sample.iterdir()} == before


def test_flat_prepared_schema(prepared):
    sample, manifest = prepared
    (sample / "voice_prompt.wav").rename(sample / "voice_prompt_left.wav")
    metadata = {"layout": "flat", "text_prompt_left": "Left prompt",
                "text_prompt_right": "Right prompt",
                "voice_prompt_left": "same_conversation"}
    (sample / "metadata.json").write_text(json.dumps(metadata))
    before = {p.name: p.read_bytes() for p in sample.iterdir()}
    data = load_sidecar(str(sample / "conversation.wav"))
    assert data["text_prompt"] == "Left prompt"
    assert data["voice_prompt"] == str(sample / "voice_prompt_left.wav")
    native_manifest(str(manifest))
    assert {p.name: p.read_bytes() for p in sample.iterdir()} == before


def test_flat_rejects_explicit_reversed_mapping(prepared):
    sample, _ = prepared
    (sample / "metadata.json").write_text(json.dumps({
        "layout": "flat", "agent_channel": "right", "user_channel": "left",
        "text_prompt_left": "Left prompt",
    }))
    with pytest.raises(ValueError, match="require LEFT=agent"):
        load_sidecar(str(sample / "conversation.wav"))


@pytest.mark.parametrize("change", ["channels", "speaker", "prompt", "voice"])
def test_invalid_prepared_assets_fail(prepared, change):
    sample, manifest = prepared
    if change in ("channels", "prompt"):
        path = sample / "metadata.json"
        data = json.loads(path.read_text())
        data["agent_channel" if change == "channels" else "text_prompt"] = "right" if change == "channels" else ""
        path.write_text(json.dumps(data))
    elif change == "voice":
        (sample / "voice_prompt.wav").unlink()
    else:
        path = sample / "words.json"
        data = json.loads(path.read_text())
        data[0]["end" if change == "timestamp" else "speaker"] = 2.0 if change == "timestamp" else "other"
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        native_manifest(str(manifest))


def test_native_format_passthrough(tmp_path):
    audio = tmp_path / "native.wav"
    sidecar = {"alignments": [], "text_prompt": "Native"}
    audio.with_suffix(".json").write_text(json.dumps(sidecar))
    manifest = tmp_path / "native.jsonl"
    manifest.write_text(json.dumps({"path": str(audio), "duration": 3.0}))
    assert native_manifest(str(manifest)) == str(manifest)
    assert load_sidecar(str(audio)) == sidecar


@pytest.mark.parametrize("start,end,rejected", [
    (.8, 2.0, [False, True]),
    (.4, 2.0, [True, True]),
    (2.0, 3.0, [False, True]),
    (-1.0, -.1, [True, False]),
    (.8, .6, [False, True]),
    (.5, .5, [False, True]),
    (.8, 1.022, [False, False]),
])
def test_invalid_words_reject_only_affected_windows(prepared, start, end, rejected):
    sample, manifest = prepared
    words = [{"speaker": "agent", "word": "bad", "start": start, "end": end}]
    (sample / "words.json").write_text(json.dumps(words))
    before = {p.name: p.read_bytes() for p in sample.iterdir()}
    path = str(sample / "conversation.wav")
    native_manifest(str(manifest))
    assert [bool(chunk_rejections(path, t, .5)) for t in (0.0, .5)] == rejected
    assert {p.name: p.read_bytes() for p in sample.iterdir()} == before


def test_filter_preflight_and_real_sphn_windows(prepared, caplog):
    sphn = pytest.importorskip("sphn")
    sample, manifest = prepared
    words = json.loads((sample / "words.json").read_text())
    words[0]["end"] = 2.0
    (sample / "words.json").write_text(json.dumps(words))
    converted = native_manifest(str(manifest))
    validate_chunk_manifest(converted, .5)
    assert "Skipping invalid chunk" in caplog.text
    dataset = sphn.dataset_jsonl(converted, duration_sec=.5, num_threads=1,
                                sample_rate=24000, pad_last_segment=True).seq(skip=0, step_by=1)
    kept = [s for s in dataset if not chunk_rejections(s["path"], s["start_time_sec"], .5)]
    assert len(kept) == 1
    assert kept[0]["start_time_sec"] == 0
    assert kept[0]["data"].shape == (2, 12000)
    with pytest.raises(ValueError, match="no valid chunks"):
        validate_chunk_manifest(converted, 1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_unlocatable_timestamp_still_fails(prepared, value):
    sample, manifest = prepared
    (sample / "words.json").write_text(json.dumps([
        {"speaker": "agent", "word": "bad", "start": value, "end": 2.0},
    ]))
    with pytest.raises(ValueError, match="invalid timestamps"):
        native_manifest(str(manifest))


def test_prepared_manifest_reads_through_reference_sphn(prepared):
    sphn = pytest.importorskip("sphn")
    _, manifest = prepared
    dataset = sphn.dataset_jsonl(
        native_manifest(str(manifest)), duration_sec=.5, num_threads=1,
        sample_rate=24000, pad_last_segment=True,
    ).seq(skip=0, step_by=1)
    chunks = list(dataset)
    assert len(chunks) == 2
    assert chunks[0]["data"].shape == (2, 12000)
    assert chunks[1]["start_time_sec"] == .5
    assert load_sidecar(chunks[0]["path"])["voice_prompt"].endswith("voice_prompt.wav")