import sys

from personaplex_finetuning.train import main

if __name__ == "__main__":
    if "--smoke" not in sys.argv:
        sys.argv.append("--smoke")
    raise SystemExit(main())
