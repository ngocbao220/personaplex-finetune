import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tin_style.data import filter_audio_chunks


def sample(length):
    return {"data": np.zeros((2, 1280)), "unpadded_len": length,
            "path": "/sample.wav", "start_time_sec": 0.0}


def test_filter_checks_unpadded_length_and_keeps_original(caplog):
    mimi = SimpleNamespace(encoder=SimpleNamespace(hop_length=320))
    good = sample(960)
    kept = list(filter_audio_chunks([sample(959), good, sample(0)], mimi))
    assert len(kept) == 1 and kept[0] is good
    assert "samples=959 encoder_hop=320 remainder=319" in caplog.text
    assert "samples=0" in caplog.text


@pytest.mark.parametrize("hop", [None, 0, -1, 1.5, True])
def test_invalid_hop_fails_without_guessing(hop):
    with pytest.raises(ValueError, match="hop_length"):
        list(filter_audio_chunks([], SimpleNamespace(encoder=SimpleNamespace(hop_length=hop))))


def test_real_iterator_guard_and_filter_without_gpu():
    # Execute the actual iterator function without importing GPU/distributed dependencies.
    path = Path(__file__).resolve().parents[1] / "tin_style/reference/moshi-finetune/finetune/data/dataset.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_dataset_iterator")
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None
    class Dataset:
        def __init__(self, samples):
            self.samples = samples
        def seq(self, **kwargs):
            return self.samples
    calls = []
    class Tokenizer:
        mimi = SimpleNamespace(encoder=SimpleNamespace(hop_length=320), sample_rate=24000)
        chunk_step_sec = 1.0
        def __call__(self, wav, start, path):
            calls.append(wav.shape[-1])
            return wav.shape[-1]
    samples = [sample(959), sample(960)]
    namespace = {
        "sphn": SimpleNamespace(dataset_jsonl=lambda *a, **kw: Dataset(samples)),
        "native_manifest": lambda path: str(path),
        "validate_chunk_manifest": lambda *a: None,
        "chunk_rejections": lambda *a: [],
        "filter_audio_chunks": filter_audio_chunks,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    source = SimpleNamespace(jsonl_files=[Path("/manifest.jsonl")])
    def iterator():
        return namespace["get_dataset_iterator"](source, Tokenizer(), 0, 1, False, 0, False)
    stream = iterator()
    assert next(stream) == 960
    stream.close()
    assert calls == [960]
    samples[:] = [sample(959)]
    with pytest.raises(ValueError, match="no valid chunks remain"):
        next(iterator())
    samples[:] = [sample(960)]
    namespace["chunk_rejections"] = lambda *a: ["invalid timestamp"]
    with pytest.raises(ValueError, match="no valid chunks remain"):
        next(iterator())