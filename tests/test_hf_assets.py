import json
import tempfile
import unittest
import wave
from pathlib import Path

from personaplex_finetuning.hf_assets import (
    MODEL_FILES,
    download_assets,
    publish_prepared,
)


def write_wav(path: Path, channels: int = 2) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\0\0" * channels * 24000)


def make_prepared(root: Path) -> Path:
    sample = root / "samples" / "conv_0001"
    sample.mkdir(parents=True)
    write_wav(sample / "conversation.wav")
    write_wav(sample / "voice_prompt.wav")
    (sample / "metadata.json").write_text(json.dumps({"text_prompt": "Helpful."}))
    (sample / "words.json").write_text(json.dumps([
        {"speaker": "agent", "word": "Hello", "start": 0.0, "end": 0.2},
    ]))
    (root / "train.jsonl").write_text(json.dumps({"sample_id": "conv_0001", "sample_dir": "samples/conv_0001"}) + "\n")
    (root / ".DS_Store").write_bytes(b"ignored")
    return root


class FakeHub:
    def __init__(self) -> None:
        self.created = []
        self.uploaded = []
        self.downloads = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)
        return type("Commit", (), {"oid": "dataset-revision"})()

    def repo_info(self, **kwargs):
        return type("Info", (), {"sha": f"{kwargs['repo_type']}-revision", "private": True})()

    def snapshot_download(self, **kwargs):
        self.downloads.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        if kwargs["repo_type"] == "dataset":
            make_prepared(local_dir)
        else:
            for name in MODEL_FILES:
                (local_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (local_dir / name).write_bytes(b"model")
        return str(local_dir)


class HuggingFaceAssetsTest(unittest.TestCase):
    def test_publish_uses_private_dataset_and_excludes_os_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub = FakeHub()
            result = publish_prepared(hub, make_prepared(Path(tmp) / "prepared"), "ngocbao220/personaplex-otospeech-prepared")

        self.assertEqual(result.revision, "dataset-revision")
        self.assertEqual(hub.created[0]["repo_type"], "dataset")
        self.assertTrue(hub.created[0]["private"])
        self.assertEqual(hub.uploaded[0]["ignore_patterns"], [".DS_Store", "**/.DS_Store"])

    def test_publish_refuses_an_existing_public_dataset(self) -> None:
        class PublicHub(FakeHub):
            def repo_info(self, **kwargs):
                return type("Info", (), {"sha": "dataset-revision", "private": False})()

        with tempfile.TemporaryDirectory() as tmp:
            hub = PublicHub()
            with self.assertRaisesRegex(RuntimeError, "must be private"):
                publish_prepared(hub, make_prepared(Path(tmp) / "prepared"), "ngocbao220/personaplex-otospeech-prepared")

    def test_downloads_only_required_assets_then_validates_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub = FakeHub()
            assets = download_assets(hub, Path(tmp) / "assets", "ngocbao220/personaplex-otospeech-prepared")

            self.assertTrue((assets.prepared_dir / "train.jsonl").is_file())
            self.assertEqual(tuple(hub.downloads[1]["allow_patterns"]), MODEL_FILES)
            self.assertTrue((assets.model_dir / "model.safetensors").is_file())
            provenance = json.loads((Path(tmp) / "assets" / "hf-assets.json").read_text())
            self.assertEqual(provenance["dataset_repo"], "ngocbao220/personaplex-otospeech-prepared")
