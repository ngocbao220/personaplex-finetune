from __future__ import annotations

import argparse

from personaplex_finetuning.hf_assets import DATASET_REPO, MODEL_REPO, authenticated_hub, download_assets


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and validate PersonaPlex data and checkpoint assets from Hugging Face.")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    assets = download_assets(authenticated_hub(), args.assets_dir, args.dataset_repo, args.model_repo, args.revision)
    print(f"Prepared data: {assets.prepared_dir} ({assets.dataset_revision})")
    print(f"Checkpoint: {assets.model_dir} ({assets.model_revision})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
