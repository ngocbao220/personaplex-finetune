from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from personaplex_finetuning.data import AudioInfo, PreparedSample, Word
from tools.check_text_chunk_capacity import _select_conversations, scan


def _sample(sample_id: str, duration: float = 1.0) -> PreparedSample:
    return PreparedSample(
        sample_id=sample_id,
        conversation_wav=Path(f"{sample_id}.wav"),
        voice_prompt_wav=Path("voice.wav"),
        words=(Word("agent", "hello", 0.0, 0.2),),
        text_prompt="prompt",
        metadata={"conversation_id": sample_id},
        audio=AudioInfo(24000, 2, duration),
        window_start_sec=0.0,
        window_end_sec=duration,
        voice_prompt_right_wav=Path("voice_right.wav"),
        text_prompt_right="right prompt",
    )


class FakeTokenizer:
    padding_id = 3
    end_padding_id = 0

    def encode(self, text):
        return [1, 2] if text else []


class FakeCodec:
    frame_rate = 10.0

    def encode_conversation_stereo_cached(self, *_args):
        stream = tuple(tuple(range(10)) for _ in range(8))
        return stream, stream


def test_default_selection_is_first_ten_unique_conversations():
    samples = [_sample(f"s{i}") for i in range(12)]
    samples.insert(1, replace(samples[0], sample_id="duplicate-row"))

    selected = _select_conversations(samples, sample_id=None, scan_all=False)

    assert [sample.sample_id for sample in selected] == ["s0", *[f"s{i}" for i in range(1, 10)]]


def test_selection_can_scan_all_or_one_sample_id():
    samples = [_sample(f"s{i}") for i in range(12)]

    assert len(_select_conversations(samples, sample_id=None, scan_all=True)) == 12
    assert _select_conversations(samples, sample_id="s7", scan_all=False) == [samples[7]]


def test_scan_checks_contiguous_duration_chunks_and_both_role_views(capsys):
    short = _sample("conversation", duration=2.2)
    short = replace(short, words=(Word("agent", "hello", 0.0, 0.2),))
    config = SimpleNamespace(
        duration_sec=1.0,
        swap_roles_after_pass=True,
        vietnamese_text_mode="diacritics",
    )

    checked, failures = scan(config, [short], FakeCodec(), FakeTokenizer())

    assert checked == 6  # three fixed chunks times two logical roles
    assert failures == 0
    output = capsys.readouterr().out
    assert "chunks=3" in output
    assert "chunk=2.000-3.000s" in output
    assert "role=right-agent" in output
