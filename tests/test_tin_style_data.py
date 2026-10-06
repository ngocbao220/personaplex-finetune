import json
import wave
from pathlib import Path

import pytest

from tin_style.data import load_sidecar, native_manifest


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


@pytest.mark.parametrize("change", ["channels", "timestamp", "speaker", "prompt", "voice"])
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