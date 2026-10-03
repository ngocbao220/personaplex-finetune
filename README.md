# PersonaPlex fine-tuning

Các lệnh dưới đây chạy từ thư mục gốc repo, trong môi trường đã cài `requirements.txt`. Model phải có sẵn tại đường dẫn trong config; chương trình không tự tải model.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Kiểm tra dữ liệu

```bash
python -m tools.validate_dataset --config configs/config.yaml
python -m tools.inspect_sample --config configs/config.yaml --index 0
python -m tools.check_text_chunk_capacity --config configs/config.yaml
```

- `validate_dataset`: kiểm tra manifest, WAV stereo, transcript và prompt.
- `inspect_sample`: xem sequence PersonaPlex và loss mask của một mẫu.
- `check_text_chunk_capacity`: mã hóa Mimi thực tế và kiểm tra transcript có đặt đủ token lên chunk `duration_sec` không; mặc định quét 10 hội thoại đầu.
- Thêm `--all` để quét toàn bộ manifest hoặc `--sample-id ID` để chỉ quét một hội thoại. Tool chỉ nạp Mimi và tokenizer, không nạp model ngôn ngữ 7B.
- Khi train, log riêng số manifest entry bị bỏ vì timestamp ngoài audio và số chunk bị bỏ vì out-of-bounds/text-token overflow. Chi tiết nằm trong `data_filter_report.json` của run; overflow làm bỏ cả chunk, và cả hai role view nếu bật role swap.
- `--config`: chọn YAML; `--index`: chọn mẫu theo thứ tự.

## Chạy thử và huấn luyện

Chạy kiểm tra một bước trên một GPU:

```bash
python -m train configs/moshi_overfit_10.yaml model=server --smoke
```

Overfit 10 hội thoại trên một GPU:

```bash
python -m train configs/moshi_overfit_10.yaml model=server
```

Huấn luyện đa GPU với 8 GPU, batch toàn cục 16:

```bash
torchrun --nproc-per-node 8 -m train configs/moshi_code_style.yaml \
  model=server batch_size=1 train.gradient_accumulation_steps=2
```

Tiếp tục từ checkpoint:

```bash
torchrun --nproc-per-node 8 -m train configs/moshi_code_style.yaml \
  model=server batch_size=1 train.gradient_accumulation_steps=2 \
  --resume-from runs/moshi-code-style/<run>/checkpoints/checkpoint_000500
```

Thay `<run>` bằng thư mục run thực tế; đường dẫn resume phải trỏ tới thư mục checkpoint có `lora.safetensors` và `training_state.pt`.

Các override dạng `key=value` thay YAML cho lần chạy đó. Tham số chính:

| Tham số | Ý nghĩa |
| --- | --- |
| `model=server` | Chọn cấu hình đường dẫn model; sửa `configs/model/server.yaml` theo máy. |
| `duration_sec` | Độ dài chunk hội thoại. |
| `sample_number=10` / `sample_number=null` | Giới hạn 10 hội thoại hoặc dùng toàn bộ dữ liệu. |
| `batch_size` | Số chunk mỗi GPU trong một microbatch. |
| `train.gradient_accumulation_steps` | Số microbatch tích lũy trước mỗi cập nhật. Batch toàn cục = batch mỗi GPU × accumulation × số GPU. |
| `max_steps` | Số lần cập nhật optimizer. |
| `lora.rank`, `lora.scaling` | Kích thước và hệ số LoRA. |
| `optim.lr` | Learning rate. |
| `gradient_checkpointing` | Giảm bộ nhớ GPU khi huấn luyện. |
| `data.swap_roles_after_pass` | Luân phiên LEFT/RIGHT làm agent giữa các epoch; cần voice prompt và text prompt cho cả hai phía. |
| `data.vietnamese_text_mode` | Dạng text target: `diacritics` (mặc định), `no_diacritics`, hoặc `telex`. |

Chọn Telex khi chạy training bằng Hydra override `data.vietnamese_text_mode=telex`.
| `--resume-from` | Tiếp tục adapter và trạng thái optimizer từ checkpoint. |
| `--force-filter` | Bỏ cache và chạy lại cả validate sample lẫn lọc chunk overflow. Mặc định cache tự tái sử dụng và tự tạo lại nếu manifest, asset hoặc thiết lập liên quan thay đổi. |

## Inference

```bash
python -m tools.inference_smoke --config configs/infer.yaml \
  --adapter runs/moshi-code-style/<checkpoint> \
  --window-seconds 30 --start 0 \
  --output-dir outputs/inference
```

`--sample-id` tìm trực tiếp trên các manifest train/validation/test đã cấu hình và không phụ thuộc `--split`; `--split` chỉ dùng khi chọn bằng `--index`. Với WAV/MP3 bên ngoài manifest, có thể truyền trực tiếp cả hai prompt qua CLI hoặc cấu hình sẵn trong `configs/infer.yaml` (`inference.voice_prompt`, `inference.text_prompt`, `inference.input_file`). Khi có đủ voice/text prompt riêng, chế độ standalone không nạp manifest. Khi chỉ có `input_file` và `sample_id`, file ngoài cung cấp user audio còn sample cung cấp voice/text conditioning:

```bash
python -m tools.inference_smoke --config configs/infer.yaml \
  --adapter /path/to/checkpoint \
  --input-file /path/to/user.wav \
  --voice-prompt /path/to/agent_voice.wav \
  --text-prompt "Bạn đang trò chuyện tự nhiên." \
  --window-seconds 30 --output-dir outputs/inference
```

Thêm `--force-filter` để xác thực manifest và lọc chunk lại thay vì dùng cache.

- `--adapter`: checkpoint LoRA cần nạp.
- `--window-seconds`, `--start`: độ dài và thời điểm bắt đầu đoạn audio.
- `--output-dir`: thư mục gốc; mỗi lần chạy tạo `infer_<YYYYMMDD_HHMMSS_microseconds>/` riêng, gồm WAV, transcript, `config.json`, `run.json` và `inference.log`. Config `inference.output_dir` cũng được hiểu là thư mục gốc.

Với `data.vietnamese_text_mode=telex`, model học và xuất text Telex; CER/WER được tính trên dạng Telex. Inference cũng ghi thêm `base_unicode.txt` và `finetuned_unicode.txt` để đọc transcript có dấu.

## Full fine-tuning

`configs/full-finetuning.yaml` chọn `train.method=full`, batch 1 mỗi GPU và gradient checkpointing. Trước khi chạy dài trên nhiều B200, chạy một bước với mẫu cố định:

```bash
python train.py --config configs/full-finetuning.yaml --smoke
```

Sau đó chạy DDP bằng `torchrun` với số GPU thực tế:

```bash
torchrun --nproc-per-node 4 train.py --config configs/full-finetuning.yaml
```

Đổi `4` thành số GPU được cấp. `train.method=full` cập nhật LM PersonaPlex, gồm embedding của cả hai kênh. Full checkpoint nằm trong `runs/full-finetuning/<run>/checkpoints/checkpoint_N/`, gồm `model.safetensors`, `training_state.pt` và `checkpoint.json`. Mỗi GPU DDP giữ một bản đầy đủ của model và AdamW; cần kiểm tra bộ nhớ GPU và dung lượng đĩa trước khi train dài. Dùng `--resume-from <checkpoint_dir>` để tiếp tục và `--checkpoint <checkpoint_dir>` với `tools.inference_smoke` để infer. `configs/config.yaml` và các lệnh LoRA cũ giữ nguyên mặc định.

### Trọng số loss

Mặc định huấn luyện là `user_loss: false`: user audio chỉ đóng vai trò làm input context/conditioning và không bị tính loss/gradient. Khi cần chạy thử nghiệm ablation study để so sánh, bật lại bằng cờ `--user-loss` (CLI) hoặc override `user_loss=true` (Hydra). Layout forward luôn giữ nguyên 17 streams. Text token thật có weight `1.0`, PAD/END_PAD là `0.3`; semantic codebook của agent là `1.0`, bảy acoustic codebook là `0.02` (khi bật `user_loss: true`, user codebook cũng nhận trọng số tương ứng). Log tách riêng `loss/text_real`, `loss/agent_semantic`, `loss/agent_acoustic`, `loss/user_semantic` và `loss/user_acoustic`.

## Log huấn luyện

```bash
tensorboard --logdir runs/moshi-code-style --port 6006
```

`runs/moshi-code-style` là thư mục log/checkpoint; thay bằng `train.output_dir` trong config nếu dùng đường dẫn khác.

Mỗi lần free-running validation lưu kết quả theo cấu trúc:

```text
<run_dir>/
├── checkpoints/
├── ranks/
├── free_running_metrics.jsonl
└── free-running/
    ├── free-running-report.json
    └── step_XXXXXX/
        └── <sample_id>/
            ├── dialogue_original.wav
            ├── dialogue_base.wav
            ├── dialogue_step.wav
            └── manifest.json
```

Ba WAV đều stereo 24 kHz: LEFT=agent, RIGHT=user. File `dialogue_original.wav`
là cửa sổ hội thoại gốc; hai file còn lại ghép agent do model sinh với user gốc.
Giữ nguyên độ dài và timeline user; phần agent được thêm silence/cắt để khớp
độ dài nguồn, kể cả các frame đầu chưa có output từ generator.

`step_000000` chạy base trước khi inject LoRA hoặc nạp checkpoint resume.
Các step sau tái sử dụng `dialogue_base.wav` từ step 0, không nạp thêm model 7B.
Tại step 0, `dialogue_step.wav` bằng kết quả base. Cửa sổ, seed, text mode và
generation settings phải khớp baseline để so sánh hợp lệ.

`manifest.json` chứa transcript thô của model (`transcript`/`hypothesis`),
`reference` chuẩn hóa theo `vietnamese_text_mode`, `raw_reference`, CER/WER,
cửa sổ, seed, generation settings và tên các WAV. Transcript là text tokens
do model sinh, không phải kết quả ASR của WAV. `free-running-report.json`
tổng hợp các lần eval hoàn tất, baseline CER và best step cải thiện baseline
(không chấp nhận transcript rỗng). `free_running_metrics.jsonl` ở gốc run
vẫn được duy trì cho công cụ tổng hợp cũ; các trường đường dẫn trỏ tới layout mới.
