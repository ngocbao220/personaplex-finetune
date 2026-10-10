# Định dạng dữ liệu đầu vào (prepared dir)

Tài liệu này mô tả **bước cuối** của pipeline tạo data stereo: xuất ra thư mục mà trainer
đọc trực tiếp. Trainer không chạy alignment, không chọn voice prompt, không sửa transcript.
Mọi thứ phải có sẵn đúng định dạng dưới đây. Nguồn kiểm tra là
`PreparedDataset._load_entry` trong `src/personaplex_finetuning/data.py`.

## 1. Cấu trúc thư mục

```
<prepared_dir>/
├── train.jsonl                    # manifest: mỗi dòng 1 hội thoại
├── val.jsonl                      # (tuỳ chọn) manifest validation riêng
└── samples/
    └── <sample_id>/
        ├── conversation.wav       # BẮT BUỘC: stereo, LEFT = agent, RIGHT = user
        ├── voice_prompt_left.wav  # BẮT BUỘC: giọng mẫu của agent (kênh trái)
        ├── voice_prompt_right.wav # tuỳ chọn: giọng mẫu của kênh phải (cần khi swap_roles)
        ├── words.json             # BẮT BUỘC: transcript cấp từ có timestamp
        └── metadata.json          # BẮT BUỘC: text prompt + thông tin mẫu
```

Config trỏ vào thư mục này bằng `data.prepared_dir`. Manifest mặc định là `<prepared_dir>/train.jsonl`.

## 2. `train.jsonl` (manifest)

Định dạng JSON Lines: mỗi dòng là một object, không có dấu phẩy giữa các dòng.

```json
{"sample_id": "conv_000001", "sample_dir": "samples/conv_000001", "duration": 312.48}
{"sample_id": "conv_000002", "sample_dir": "samples/conv_000002", "duration": 187.20}
```

| Trường | Bắt buộc | Mô tả |
|---|---|---|
| `sample_id` | có | ID duy nhất, khác rỗng. Nên trùng tên thư mục. |
| `sample_dir` | có | Đường dẫn **tương đối** so với thư mục chứa `train.jsonl`. Không được là đường dẫn tuyệt đối và không được trỏ ra ngoài prepared dir. |
| `duration` | không, nhưng nên có | Độ dài `conversation.wav` tính bằng giây (số dương). Nếu có, trainer so với độ dài WAV thật và từ chối mẫu nếu lệch quá 0.05 s. Dùng để thống kê số giờ, cân batch và phát hiện file WAV bị cắt cụt. |

Dòng trống được bỏ qua. Mẫu không hợp lệ bị bỏ qua và ghi lý do vào báo cáo lọc. Nếu **tất cả**
mẫu đều lỗi, trainer dừng và in lý do của từng dòng.

## 3. `conversation.wav`

- WAV PCM (16-bit khuyến nghị). Module `wave` của Python phải đọc được, nên **không dùng WAV float32**.
- **Đúng 2 kênh**: kênh 0 (LEFT) = agent, tức là giọng PersonaPlex sẽ học nói; kênh 1 (RIGHT) = user.
- Sample rate: khuyến nghị **24000 Hz** (Mimi). Rate khác vẫn được resample khi đọc, nhưng tốn CPU hơn.
- Hai kênh tách biệt hoàn toàn: không trộn giọng user vào kênh trái và ngược lại. Khi một bên im, kênh đó là silence, không phải nhiễu của bên kia.
- Header WAV phải khớp dữ liệu. File bị cắt cụt sẽ bị từ chối.

## 4. `voice_prompt_left.wav` / `voice_prompt_right.wav`

- Một đoạn giọng sạch của người nói kênh trái (hoặc phải), khoảng 5–10 s, mono 24 kHz.
- `voice_prompt_left.wav` là **bắt buộc** và không được rỗng.
- `voice_prompt_right.wav` là tuỳ chọn. Cần có khi bật `data.swap_roles_after_pass=true`: lượt train thứ hai đảo vai, kênh phải thành agent và dùng prompt này.
- Có thể cắt từ chính hội thoại, nhưng phải là đoạn chỉ có một người nói, không bị chồng tiếng.

## 5. `words.json`

Mảng JSON gồm các từ, **sắp xếp tăng dần theo `start`**, gộp cả hai người nói:

```json
[
  {"speaker": "agent", "word": "Xin", "start": 0.00, "end": 0.21},
  {"speaker": "agent", "word": "chào,", "start": 0.21, "end": 0.55},
  {"speaker": "user",  "word": "Chào", "start": 0.80, "end": 1.02},
  {"speaker": "user",  "word": "bạn.", "start": 1.02, "end": 1.40}
]
```

| Trường | Quy tắc |
|---|---|
| `speaker` | `agent` / `left` / `a` → agent; `user` / `right` / `b` → user (không phân biệt hoa thường). |
| `word` | Một từ (âm tiết với tiếng Việt), khác rỗng sau khi strip. Giữ dấu câu và dấu tiếng Việt. Chế độ `data.vietnamese_text_mode` tự chuyển đổi lúc train, không cần làm trước. |
| `start`, `end` | Giây, tính từ đầu `conversation.wav`. Phải có `0 ≤ start < end ≤ duration + 0.05`. |

- Phải có **ít nhất một từ của agent**.
- Timestamp vượt độ dài audio → mẫu bị đếm vào `out_of_bounds` và loại.
- Timestamp nên lấy từ forced alignment trên đúng kênh của người nói. Text agent được đặt lên lưới Mimi 12.5 frame/s theo `start`, nên lệch timestamp sẽ làm lệch text so với audio.
- Các trường thừa (ví dụ `alignment_status`) được bỏ qua.

## 6. `metadata.json`

```json
{
  "sample_id": "conv_000001",
  "conversation_id": "conv_000001",
  "language": "vi",
  "agent_channel": "left",
  "user_channel": "right",
  "text_prompt_left": "Bạn là nhân viên chăm sóc khách hàng, trả lời ngắn gọn và lịch sự.",
  "text_prompt_right": "Bạn là khách hàng đang hỏi về đơn hàng bị giao trễ."
}
```

| Trường | Bắt buộc | Mô tả |
|---|---|---|
| `text_prompt_left` | có | System/text prompt của agent (kênh trái). Chỉ dùng làm conditioning, không tính loss. |
| `text_prompt_right` | không | Prompt cho kênh phải khi swap roles. |
| `agent_channel` / `user_channel` | không | Nếu có, phải là `left` / `right`. Các giá trị khác bị từ chối. |
| `conversation_id` | không | Các mẫu cùng `conversation_id` luôn nằm chung một phía khi chia train/val, tránh rò dữ liệu khi cùng một hội thoại có nhiều biến thể (`__var00`, `__var01`, ...). |
| `language` | không | `vi` / `vietnamese` để augmentation prompt dùng câu tiếng Việt. |

Các trường khác (`layout`, `transcript_source`, ...) được giữ nguyên trong metadata nhưng trainer không đọc.

## 7. Script xuất mẫu (bước cuối của pipeline)

Ví dụ nhận đầu vào từ pipeline tạo stereo (audio 2 kênh, danh sách từ, voice prompt, prompt) rồi ghi ra
đúng định dạng và tạo `train.jsonl` có `duration`:

```python
import json, wave
from pathlib import Path
import numpy as np
import soundfile as sf

def write_pcm16(path: Path, audio: np.ndarray, sample_rate: int = 24000) -> None:
    """audio: [T] mono hoặc [T, 2] stereo, float trong [-1, 1]."""
    sf.write(path, np.clip(audio, -1, 1), sample_rate, subtype="PCM_16")

def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()

def export_sample(out_root: Path, sample_id: str, stereo, voice_left, voice_right,
                  words: list[dict], prompt_left: str, prompt_right: str | None,
                  conversation_id: str, language: str = "vi") -> dict:
    d = out_root / "samples" / sample_id
    d.mkdir(parents=True, exist_ok=True)
    write_pcm16(d / "conversation.wav", stereo)            # [T, 2]: cột 0 = agent, cột 1 = user
    write_pcm16(d / "voice_prompt_left.wav", voice_left)
    if voice_right is not None:
        write_pcm16(d / "voice_prompt_right.wav", voice_right)
    duration = wav_duration(d / "conversation.wav")
    words = sorted(
        ({"speaker": w["speaker"], "word": w["word"].strip(),
          "start": round(float(w["start"]), 3), "end": round(min(float(w["end"]), duration), 3)}
         for w in words if w["word"].strip() and float(w["end"]) > float(w["start"])),
        key=lambda w: w["start"],
    )
    assert any(w["speaker"] == "agent" for w in words), f"{sample_id}: no agent words"
    (d / "words.json").write_text(json.dumps(words, ensure_ascii=False, indent=1), encoding="utf-8")
    metadata = {"sample_id": sample_id, "conversation_id": conversation_id, "language": language,
                "agent_channel": "left", "user_channel": "right", "text_prompt_left": prompt_left}
    if prompt_right:
        metadata["text_prompt_right"] = prompt_right
    (d / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"sample_id": sample_id, "sample_dir": f"samples/{sample_id}", "duration": round(duration, 3)}

def write_manifest(out_root: Path, entries: list[dict], name: str = "train.jsonl") -> None:
    with open(out_root / name, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

Dùng:

```python
entries = [export_sample(Path("data_ready/my_set"), **item) for item in pipeline_outputs]
write_manifest(Path("data_ready/my_set"), entries)
```

Nếu đã có sẵn prepared dir mà manifest chưa có `duration`, bổ sung bằng:

```bash
python - <<'EOF'
import json, wave, sys
from pathlib import Path
root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
lines = [json.loads(l) for l in (root / "train.jsonl").read_text().splitlines() if l.strip()]
for e in lines:
    with wave.open(str(root / e["sample_dir"] / "conversation.wav"), "rb") as w:
        e["duration"] = round(w.getnframes() / w.getframerate(), 3)
(root / "train.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in lines))
print(f"{len(lines)} entries, {sum(e['duration'] for e in lines) / 3600:.2f} h")
EOF
```

(chạy từ trong prepared dir, hoặc truyền đường dẫn: `python - /path/prepared_dir <<'EOF' ...`).

## 8. Kiểm tra trước khi train

```bash
cd personaplex-finetuning
# Kiểm tra contract: file, số kênh, timestamp, prompt, duration
python -m tools.validate_dataset --config configs/config.yaml data.prepared_dir=/path/prepared_dir
# Xem một mẫu đã được dựng thành chuỗi PersonaPlex (stream, mask loss)
python -m tools.inspect_sample --config configs/config.yaml data.prepared_dir=/path/prepared_dir --index 0
# Đo tokenizer và số chunk bị overflow text (xem README, mục Benchmark tokenizer)
PYTHONPATH=src python -m tools.benchmark_vi_tokenizer --config configs/train_vi_synthetic.yaml \
  data.prepared_dir=/path/prepared_dir
```

Checklist nhanh:
- [ ] `conversation.wav` 2 kênh, PCM, 24 kHz; agent ở kênh trái.
- [ ] Mỗi mẫu có `voice_prompt_left.wav`, `words.json`, `metadata.json` với `text_prompt_left`.
- [ ] `words.json` sắp xếp theo `start`, mọi `end ≤ duration`, có từ của agent.
- [ ] `train.jsonl`: `sample_dir` tương đối, `duration` khớp WAV.
- [ ] Biến thể của cùng một hội thoại có chung `conversation_id`.
- [ ] `validate_dataset` không báo lỗi; `data_filter_report.json` sau bước precompute có tỉ lệ loại thấp.

Mẫu tham chiếu: `../synthetic_samples` đã đúng định dạng (chưa có `duration`; bổ sung bằng script ở mục 7).
`../otospeech-prepared` là định dạng cũ (`voice_prompt.wav`) nên bị từ chối: cần đổi tên thành
`voice_prompt_left.wav` và kiểm tra lại metadata.
