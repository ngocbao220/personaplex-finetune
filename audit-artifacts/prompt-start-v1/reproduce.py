"""CPU token-history audit; no production changes or 7B quality claims."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from moshi.models.lm import LMGen

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("contract_helpers", ROOT / "tests/test_native_train_contract.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def run(prime):
    model = helpers._RecordingLM().eval()
    sample = helpers.PreparedSample(
        "short", Path("stereo.wav"), Path("voice.wav"),
        (helpers.Word("agent", "hello", 0.1, 0.5),), "You are a helpful assistant.",
        {}, helpers.AudioInfo(24000, 2, 3.04), 0.0, 3.04,
    )
    example = helpers.PersonaPlexTrainingExampleBuilder(
        helpers._Codec(), helpers._Tokenizer(), model._get_initial_token()[0, :, 0].tolist(),
        model.zero_token_id,
    ).build(sample)
    canonical = torch.tensor(example.input_codes)[None]
    model.forward_train(canonical)
    batch_input = model.inputs.pop()
    model.targets.clear()

    generator = LMGen(model, device="cpu", use_sampling=False, check=True,
                      audio_silence_frame_cnt=6,
                      text_prompt_tokens=helpers._Tokenizer().encode(f"<system> {sample.text_prompt} <system>"))
    # Native streaming entry only installs this state; it does not call step().
    generator._streaming_state = generator._init_streaming_state(1)
    # Neural depth computation is replaced, because all 17 streams are forced.
    # Native prepare_step_input / step / process_transformer_output remain intact.
    def forced_depth(_text, _hidden, target, provided):
        assert provided.all()
        return target.clone()
    generator._streaming_state.graphed_depth = forced_depth
    generator.voice_prompt_audio = torch.zeros(1, 1)
    voice = torch.tensor(helpers._Codec().encode_voice_prompt(None))[None]
    generator._encode_voice_prompt_frames = lambda _mimi: (
        voice[:, :, t:t + 1] for t in range(voice.shape[-1])
    )
    if prime:
        assert generator.step() is None
    generator.step_system_prompts(None)
    prompt_forward_count = len(model.inputs)
    for t in range(example.prompt_frames, canonical.shape[-1]):
        generator.step(input_tokens=canonical[:, 9:17, t:t + 1],
                       moshi_tokens=canonical[:, 1:9, t:t + 1], text_token=canonical[:, 0, t])
    native = torch.cat(model.inputs, dim=-1)
    first_dialogue_input = native[:, :, prompt_forward_count]
    expected = batch_input[:, :, example.prompt_frames]
    overlap = min(native.shape[-1], batch_input.shape[-1])
    return {
        "primed": prime, "prompt_frames": example.prompt_frames,
        "prompt_temporal_forwards": prompt_forward_count,
        "batch_temporal_forwards": batch_input.shape[-1],
        "native_temporal_forwards": native.shape[-1],
        "same_index_input_mismatches": int((native[:, :, :overlap] != batch_input[:, :, :overlap]).sum()),
        "first_dialogue_input_mismatches": int((first_dialogue_input != expected).sum()),
        "first_dialogue_input_native": first_dialogue_input.flatten().tolist(),
        "first_dialogue_input_training": expected.flatten().tolist(),
        "first_voice_agent_cb0": int(voice[0, 0, 0]),
        "voice_agent_cb0_in_native_history": native[0, 1, :4].tolist(),
        "voice_agent_cb0_in_batch_history": batch_input[0, 1, :4].tolist(),
    }


report = {"scope": "synthetic CPU token history; actual native prompt/step/cache methods; stub neural depth and codec",
          "unprimed": run(False), "primed_control": run(True)}
assert report["primed_control"]["same_index_input_mismatches"] == 0
assert report["unprimed"]["native_temporal_forwards"] == report["unprimed"]["batch_temporal_forwards"] - 1
report["source_sha256"] = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [ROOT / "src/moshi/models/lm.py", ROOT / "src/personaplex_finetuning/inference.py",
                 ROOT / "src/personaplex_finetuning/sequence.py"]
}
Path(__file__).with_name("report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
