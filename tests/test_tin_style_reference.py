"""Guard against accidental algorithm changes while adapting the boundaries."""
import hashlib
import json
from pathlib import Path


def test_reference_algorithms_match_recorded_source():
    root = Path(__file__).resolve().parents[1] / "tin_style"
    provenance = json.loads((root / "provenance.json").read_text())
    changed = set(provenance["data_boundary_changes"])
    assert changed == {
        "moshi-finetune/finetune/data/dataset.py",
        "moshi-finetune/finetune/data/interleaver.py",
    }
    for name, digest in provenance["sha256"].items():
        if name not in changed:
            assert hashlib.sha256((root / "reference" / name).read_bytes()).hexdigest() == digest, name


def test_data_boundary_patch_does_not_change_interleaving():
    root = Path(__file__).resolve().parents[1] / "tin_style"
    provenance = json.loads((root / "provenance.json").read_text())
    path = "moshi-finetune/finetune/data/interleaver.py"
    code = (root / "reference" / path).read_text()
    restored = code.replace("from tin_style.data import load_sidecar\n", "").replace(
        "            data = load_sidecar(path)\n",
        '            info_file = os.path.splitext(path)[0] + ".json"\n'
        '            with open(info_file) as f:\n'
        '                data = json.load(f)\n',
    )
    assert hashlib.sha256(restored.encode()).hexdigest() == provenance["sha256"][path]
    path = "moshi-finetune/finetune/data/dataset.py"
    code = (root / "reference" / path).read_text()
    restored = code.replace("from tin_style.data import native_manifest\n", "").replace(
        "native_manifest(str(jsonl_file)),", "str(jsonl_file),"
    )
    assert hashlib.sha256(restored.encode()).hexdigest() == provenance["sha256"][path]