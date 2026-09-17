"""Explicit Hugging Face transfer boundary for the online server recipe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .data import PreparedDataset


DATASET_REPO = "ngocbao220/personaplex-otospeech-prepared"
MODEL_REPO = "nvidia/personaplex-7b-v1"
MODEL_FILES = (
    "model.safetensors",
    "tokenizer-e351c8d8-checkpoint125.safetensors",
    "tokenizer_spm_32k_3.model",
)


class HubClient(Protocol):
    def create_repo(self, **kwargs): ...
    def upload_folder(self, **kwargs): ...
    def snapshot_download(self, **kwargs) -> str: ...
    def repo_info(self, **kwargs): ...


@dataclass(frozen=True)
class PublishedDataset:
    repo_id: str
    revision: str


@dataclass(frozen=True)
class DownloadedAssets:
    prepared_dir: Path
    model_dir: Path
    dataset_revision: str
    model_revision: str


def authenticated_hub() -> HubClient:
    """Create the only online client; authentication comes from ``hf auth login``."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the online environment before using Hugging Face asset commands") from exc

    api = HfApi()

    class Client:
        def create_repo(self, **kwargs):
            return api.create_repo(**kwargs)

        def upload_folder(self, **kwargs):
            return api.upload_folder(**kwargs)

        def repo_info(self, **kwargs):
            return api.repo_info(**kwargs)

        def snapshot_download(self, **kwargs) -> str:
            return snapshot_download(**kwargs)

    return Client()


def publish_prepared(hub: HubClient, prepared_dir: str | Path, repo_id: str = DATASET_REPO) -> PublishedDataset:
    """Validate and upload a private prepared-data tree without OS metadata."""
    root = Path(prepared_dir).resolve()
    PreparedDataset(root / "train.jsonl").load()
    hub.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = hub.repo_info(repo_id=repo_id, repo_type="dataset")
    if getattr(info, "private", None) is not True:
        raise RuntimeError(f"refusing upload: Hugging Face dataset {repo_id} must be private")
    commit = hub.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(root),
        path_in_repo=".",
        ignore_patterns=[".DS_Store", "**/.DS_Store"],
        commit_message="Upload validated PersonaPlex prepared conversations",
    )
    revision = str(getattr(commit, "oid", ""))
    if not revision:
        raise RuntimeError("Hugging Face upload completed without a commit revision")
    return PublishedDataset(repo_id, revision)


def _revision(hub: HubClient, repo_id: str, repo_type: str, revision: str) -> str:
    info = hub.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
    sha = str(getattr(info, "sha", ""))
    if not sha:
        raise RuntimeError(f"could not resolve Hugging Face revision for {repo_type} repo {repo_id}")
    return sha


def download_assets(
    hub: HubClient,
    assets_dir: str | Path,
    dataset_repo: str = DATASET_REPO,
    model_repo: str = MODEL_REPO,
    revision: str = "main",
) -> DownloadedAssets:
    """Download immutable revisions, validate data, then persist provenance."""
    root = Path(assets_dir).resolve()
    prepared_dir = root / "prepared"
    model_dir = root / "personaplex-7b-v1"
    dataset_revision = _revision(hub, dataset_repo, "dataset", revision)
    model_revision = _revision(hub, model_repo, "model", revision)
    hub.snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        revision=dataset_revision,
        local_dir=str(prepared_dir),
    )
    hub.snapshot_download(
        repo_id=model_repo,
        repo_type="model",
        revision=model_revision,
        local_dir=str(model_dir),
        allow_patterns=list(MODEL_FILES),
    )
    PreparedDataset(prepared_dir / "train.jsonl").load()
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Hugging Face model download is incomplete: {', '.join(missing)}")
    provenance = {
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "model_repo": model_repo,
        "model_revision": model_revision,
        "model_files": list(MODEL_FILES),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "hf-assets.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return DownloadedAssets(prepared_dir, model_dir, dataset_revision, model_revision)
