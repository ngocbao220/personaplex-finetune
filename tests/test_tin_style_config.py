"""CPU-only config boundary smoke tests (no reference or CUDA imports)."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from tin_style.config import ConfigError, load_config, to_train_args


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    for name in ("model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer_spm_32k_3.model"):
        (root / name).touch()
    (root / "config.json").write_text(json.dumps({"dep_q": 16}))
    (tmp_path / "train.jsonl").write_text("{}\n")
    return {"model": {"root": "model"}, "data": {"train_data": "train.jsonl"},
            "train": {"output_dir": "run"}, "lora": {"enable": True}, "user_loss": True}


def test_only_supplied_values_preserve_reference_defaults(tmp_path, config):
    args = to_train_args(config, tmp_path)
    assert args["moshi_paths"]["hf_repo_id"] is None
    assert args["data"]["train_data"] == str(tmp_path / "train.jsonl")
    assert args["lora"] == {"enable": True}
    for name in ("text_padding_weight", "first_codebook_weight_multiplier", "seed", "duration_sec", "param_dtype"):
        assert name not in args
    assert json.loads((tmp_path / "model/config.json").read_text()) == {"dep_q": 16}


def test_native_values_and_mapping(tmp_path, config):
    config["reference"] = {"text_padding_weight": 0.5, "optim": {"weight_decay": 0.1}}
    config["train"].update(learning_rate=0.0003, gradient_accumulation_steps=4, mixed_precision="bf16")
    config["lora"].update(rank=16, alpha=32)
    args = to_train_args(config, tmp_path)
    assert args["optim"] == {"weight_decay": 0.1, "lr": 0.0003}
    assert args["num_microbatches"] == 4
    assert args["lora"]["scaling"] == 2
    assert args["param_dtype"] == "bfloat16"


@pytest.mark.parametrize("section,key,value", [
    (None, "user_loss", False), (None, "dep_q", 8),
    ("train", "depformer_learning_rate", 1e-5), ("train", "num_workers", 4),
    ("data", "swap_roles_after_pass", True), ("data", "val_ratio", 0.05),
    ("lora", "qlora", True), ("reference", "world_size", 1),
])
def test_reject_unsupported(tmp_path, config, section, key, value):
    target = config.setdefault(section, {}) if section else config
    target[key] = value
    with pytest.raises(ConfigError):
        to_train_args(config, tmp_path)


def test_all_local_assets_required(tmp_path, config):
    (tmp_path / "model/config.json").unlink()
    with pytest.raises(ConfigError, match="Required local asset missing"):
        to_train_args(config, tmp_path)


def test_explicit_assets_no_root(tmp_path, config):
    config["model"] = {"lm_checkpoint": "model/model.safetensors",
                       "mimi_checkpoint": "model/tokenizer-e351c8d8-checkpoint125.safetensors",
                       "tokenizer": "model/tokenizer_spm_32k_3.model", "config": "model/config.json"}
    assert to_train_args(config, tmp_path)["moshi_paths"]["config_path"].endswith("/config.json")


def test_conflicting_native_alias_rejected(tmp_path, config):
    config["reference"] = {"run_dir": "other"}
    with pytest.raises(ConfigError, match="Conflicting"):
        to_train_args(config, tmp_path)


def test_hydra_groups_and_override(tmp_path, config):
    (tmp_path / "model").mkdir(exist_ok=True)
    OmegaConf.save(OmegaConf.create(config.pop("model")), tmp_path / "model/server.yaml")
    config["defaults"] = [{"model": "server"}, "_self_"]
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    args = load_config(path, ["model=server", "+train.max_steps=7"])
    assert args["max_steps"] == 7


def test_check_config_no_gpu_import_and_offline(tmp_path, config):
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    code = (
        "import os,sys; from tin_style.__main__ import main; "
        f"main([{str(path)!r}, '--check-config']); "
        "assert 'torch' not in sys.modules; assert 'finetune' not in sys.modules; "
        "assert 'moshi' not in sys.modules; assert os.environ['HF_HUB_OFFLINE']=='1'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "HF_HUB_OFFLINE": "0"})
    assert result.returncode == 0, result.stderr
    assert "hf_repo_id: null" in result.stdout


def test_resume_passed_to_reference(tmp_path, config, capsys):
    from tin_style.__main__ import main
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    assert main([str(path), "--resume-from", str(checkpoint), "--check-config"]) == 0
    assert f"resume_from: {checkpoint}" in capsys.readouterr().out


def test_actual_native_schema_passthrough(tmp_path, config):
    config["reference"] = {"audio_loss_weight": 1.0, "lora_l2_weight": 0.01,
                           "skip_zero_eval": True, "system_prompt": {"enable": True},
                           "lora": {"skip_depformer": False}, "puppeteer": {"enable": False},
                           "gen_eval": {"enable": False}}
    args = to_train_args(config, tmp_path)
    assert args["system_prompt"] == {"enable": True}
    assert args["lora"]["skip_depformer"] is False


def test_training_receives_temp_native_yaml(tmp_path, config, monkeypatch):
    import tin_style.__main__ as cli
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    captured = []

    def fake_train(resolved):
        captured.append(resolved)
        args = OmegaConf.to_container(OmegaConf.load(resolved))
        assert "reference" not in args
        assert args["moshi_paths"]["hf_repo_id"] is None

    monkeypatch.setattr(cli, "run_reference", fake_train)
    assert cli.main([str(path)]) == 0
    assert not Path(captured[0]).exists()