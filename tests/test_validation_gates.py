import importlib.util
from pathlib import Path

import pytest
import sys
from types import SimpleNamespace


def harness():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_one_sample.py"
    spec = importlib.util.spec_from_file_location("validation_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_token_occurrence_fails_even_if_token_id_is_repeated():
    with pytest.raises(AssertionError):
        harness().assert_supervision([
            {"token_id": 42, "label": 42, "supervised": True},
            {"token_id": 42, "label": 42, "supervised": False},
        ])


def test_wrong_label_fails():
    with pytest.raises(AssertionError):
        harness().assert_supervision([{"token_id": 42, "label": 43, "supervised": True}])


def test_empty_transcript_is_not_a_pass():
    with pytest.raises(AssertionError):
        harness().assert_supervision([])


def test_all_occurrences_supervised():
    harness().assert_supervision([{"token_id": 42, "label": 42, "supervised": True}])


def test_native_small_model_parity_detects_depth_mismatch():
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from moshi.models.lm import LMModel
    torch.manual_seed(42)
    model = LMModel(dim=16, text_card=32, card=8, n_q=16, dep_q=16,
                    num_heads=2, num_layers=1, causal=True, context=32,
                    depformer_dim=16, depformer_num_heads=2, depformer_num_layers=1,
                    depformer_causal=True, depformer_multi_linear=True,
                    depformer_weights_per_step=True, depformer_pos_emb="none", gating="silu",
                    delays=[0, 0] + [1] * 7 + [0] + [1] * 7)
    model.eval()
    codes = torch.randint(0, 8, (1, 17, 4))
    codes[:, 0] = torch.randint(0, 32, (1, 4))
    result, _ = harness().parity(SimpleNamespace(model=model), codes,
                               SimpleNamespace(atol=1e-5, rtol=1e-4))
    # Current native implementation disagrees at the final depth step on CPU.
    # Preserve this finding; do not loosen FP32 tolerance to manufacture a pass.
    assert result["masks_equal"]
    assert result["text"]["passed"]
    assert result["audio"]["finite"]
    assert not result["passed"]
    assert result["audio"]["max_abs"] > 1e-3
    assert result["diagnostics"]["temporal_hidden"]["passed"]
    assert not result["diagnostics"]["depth_only"]["passed"]
    assert result["runtime"]["dep_q"] == 16
    assert result["runtime"]["total_frames"] == 4


def test_invalid_masked_logits_do_not_pollute_comparison():
    torch = pytest.importorskip("torch")
    a = torch.tensor([[[[1., 2.], [float("nan"), float("nan")]]]])
    mask = torch.tensor([[[True, False]]])
    assert harness().compare_logits(torch, a, a.clone(), mask, 0, 0)["passed"]


def test_comparison_details_localizes_failure_and_ignores_masked_nan():
    import torch
    a = torch.tensor([[[[1., 2.], [3., 4.], [float("nan"), float("nan")]]]])
    b = a.clone()
    b[0, 0, 1, 0] = 5.
    result = harness().comparison_details(torch, a, b, torch.tensor([[[True, True, False]]]), 0, 0)
    assert result["finite"]
    assert not result["passed"]
    assert result["first_failure_frame"] == 1
    assert result["failed_elements"] == 1
    assert result["streams"][0]["argmax_agreement"] == 0.5
    assert result["frames"][2]["valid_positions"] == 0


def test_acoustic_diagnostics_are_unweighted_and_exclude_padding():
    torch = pytest.importorskip("torch")
    logits = torch.zeros(1, 16, 2, 2)
    logits[:, :, 0, 0] = 5
    logits[:, :, 1] = float("nan")
    output = SimpleNamespace(logits=logits, mask=torch.ones(1, 16, 2, dtype=torch.bool))
    example = SimpleNamespace(labels=[[0, 0]] * 17, loss_mask=[[True, False]] * 17)
    rows = harness().diagnostics(output, example, "cpu")
    assert len(rows) == 8
    assert all(r["accuracy"] == 1 and r["count"] == 1 and r["ce"] > 0 for r in rows)


def test_reload_pcm_and_text_comparison(tmp_path):
    import numpy as np
    import sphn
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from validation_generation import compare_outputs
    baseline, fresh = tmp_path / "baseline", tmp_path / "fresh"
    for directory in (baseline, fresh):
        directory.mkdir()
        sphn.write_wav(str(directory / "generated.wav"), np.zeros(800, dtype=np.float32), 8000)
        (directory / "generated.txt").write_text("hello")
    assert compare_outputs(baseline, fresh)["passed"]
    (fresh / "generated.txt").write_text("different")
    assert not compare_outputs(baseline, fresh)["passed"]


def test_native_argmax_instrumentation_restores_function():
    import torch
    from moshi.models import lm
    from validation_generation import verify_argmax
    original = lm.sample_token
    with verify_argmax() as counts:
        assert lm.sample_token(torch.tensor([[1., 3.]]), False, 0).item() == 1
    assert counts["calls"] == 1
    assert lm.sample_token is original


def test_math_backend_and_fp32_controls_restore_on_failure():
    import torch
    prior = (torch.backends.cuda.flash_sdp_enabled(),
             torch.backends.cuda.mem_efficient_sdp_enabled(),
             torch.backends.cuda.math_sdp_enabled(),
             torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError, match="probe"):
        with harness().parity_backend(torch, "math", True):
            assert not torch.backends.cuda.flash_sdp_enabled()
            assert not torch.backends.cuda.mem_efficient_sdp_enabled()
            assert torch.backends.cuda.math_sdp_enabled()
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            raise RuntimeError("probe")
    assert prior == (torch.backends.cuda.flash_sdp_enabled(),
                     torch.backends.cuda.mem_efficient_sdp_enabled(),
                     torch.backends.cuda.math_sdp_enabled(),
                     torch.backends.cuda.matmul.allow_tf32,
                     torch.backends.cudnn.allow_tf32)


def test_diagnostic_flags_cannot_advance_to_training(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["validate_one_sample.py", "--config", "unused.yaml",
                                    "--output-dir", str(tmp_path / "run"), "--through", "6",
                                    "--train-steps", "300", "--parity-sdp-backend", "math"])
    with pytest.raises(SystemExit) as exc:
        harness().main()
    assert exc.value.code == 2
    assert not (tmp_path / "run").exists()


def test_full_runtime_requests_fp32_at_load(monkeypatch):
    from personaplex_finetuning import runtime as runtime_module
    import torch
    received = {}

    def fake_load(*args, **kwargs):
        received.update(kwargs)
        return SimpleNamespace(model=torch.nn.Linear(2, 2))

    monkeypatch.setattr(runtime_module, "load_runtime", fake_load)
    config = SimpleNamespace(model_root=Path("/models"), personaplex_source=Path("/source"),
                             device="cpu", qlora=False, quant_type="nf4")
    loaded = harness().full_runtime(config, full_precision_model=True)
    assert received["full_precision_model"] is True
    assert not loaded.model.training