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
| `data.normalize_vietnamese_diacritics` | Bỏ dấu khỏi text target và CER/WER; mặc định `false`, không sửa transcript nguồn hoặc prompt. |
| `--resume-from` | Tiếp tục adapter và trạng thái optimizer từ checkpoint. |
| `--force-filter` | Bỏ cache và chạy lại cả validate sample lẫn lọc chunk overflow. Mặc định cache tự tái sử dụng và tự tạo lại nếu manifest, asset hoặc thiết lập liên quan thay đổi. |

## Inference

```bash
python -m tools.inference_smoke --config configs/infer.yaml \
  --adapter runs/moshi-code-style/<checkpoint> \
  --window-seconds 30 --start 0 \
  --output-dir outputs/inference
```

Thêm `--force-filter` để xác thực manifest và lọc chunk lại thay vì dùng cache.

- `--adapter`: checkpoint LoRA cần nạp.
- `--window-seconds`, `--start`: độ dài và thời điểm bắt đầu đoạn audio.
- `--output-dir`: thư mục gốc; mỗi lần chạy tạo `infer_<YYYYMMDD_HHMMSS_microseconds>/` riêng, gồm WAV, transcript, `config.json`, `run.json` và `inference.log`. Config `inference.output_dir` cũng được hiểu là thư mục gốc.

## Log huấn luyện

```bash
tensorboard --logdir runs/moshi-code-style --port 6006
```

`runs/moshi-code-style` là thư mục log/checkpoint; thay bằng `train.output_dir` trong config nếu dùng đường dẫn khác.
