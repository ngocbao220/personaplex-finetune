"""Support the Moshi-style ``python -m train CONFIG`` entrypoint."""

from pathlib import Path
import sys


_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

if __name__ == "__main__":
    # Import inside the guard: multiprocessing children re-import this file as
    # __mp_main__ and must not pull in the whole trainer stack.
    from personaplex_finetuning.train import main

    raise SystemExit(main())
