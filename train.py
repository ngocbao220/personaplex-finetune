"""Support the Moshi-style ``python -m train CONFIG`` entrypoint."""

from pathlib import Path
import sys


_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from personaplex_finetuning.train import main


if __name__ == "__main__":
    raise SystemExit(main())
