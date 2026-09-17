from __future__ import annotations

import argparse

from personaplex_finetuning.data import PreparedDataset
from personaplex_finetuning.hf_assets import DATASET_REPO, authenticated_hub, publish_prepared


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish validated prepared data to a private Hugging Face dataset.")
    parser.add_argument("--prepared-dir", default="../prepared")
    parser.add_argument("--repo-id", default=DATASET_REPO)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    samples = PreparedDataset(args.prepared_dir + "/train.jsonl").load()
    if args.dry_run:
        print(f"DRY RUN: {len(samples)} validated samples would upload to private dataset {args.repo_id}")
        return 0
    result = publish_prepared(authenticated_hub(), args.prepared_dir, args.repo_id)
    print(f"Published private dataset {result.repo_id} at revision {result.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
