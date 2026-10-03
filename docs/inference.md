# Suy Luận (Inference)

Tài liệu này trình bày cách chạy mô hình PersonaPlex sau khi huấn luyện, cũng như các lưu ý về lỗi môi trường phổ biến.

## 1. Chạy Suy Luận (Inference Smoke Test)

Công cụ `tools.inference_smoke` dùng để đánh giá mô hình bằng cách nạp adapter LoRA hoặc checkpoint.

**Sử dụng trực tiếp các mẫu trong tập Manifest:**
```bash
python -m tools.inference_smoke --config configs/infer.yaml \
  --adapter runs/moshi-code-style/<checkpoint> \
  --window-seconds 30 --start 0 \
  --output-dir outputs/inference
```

**Sử dụng tệp Audio độc lập (Standalone):**
Trong trường hợp chạy trực tiếp trên file Audio người dùng cung cấp và file prompt, chương trình sẽ không tải tập dữ liệu gốc:

```bash
python -m tools.inference_smoke --config configs/infer.yaml \
  --adapter /path/to/checkpoint \
  --input-file /path/to/user.wav \
  --voice-prompt /path/to/agent_voice.wav \
  --text-prompt "Bạn đang trò chuyện tự nhiên." \
  --window-seconds 30 --output-dir outputs/inference
```

Kết quả sinh ra (bao gồm file WAV, Transcript, các file cấu hình lúc suy luận) sẽ nằm trong một thư mục có mốc thời gian riêng bên dưới `outputs/inference/`.

## 2. Giải quyết lỗi CUDA/cuDNN trong quá trình tạo Audio

Trong môi trường chứa GPU, hàm giải mã âm thanh của Mimi (`mimi.decode` sử dụng ConvTranspose1d) thường gặp lỗi kernel từ **cuDNN** nếu tensor có cấu trúc dạng nhỏ hoặc ngắn:
`RuntimeError: GET was unable to find an engine to execute this computation`

**Giải pháp đã áp dụng trong Source Code:**

1.  **Vô hiệu hoá cuDNN tạm thời**: Module `inference.py` chứa một context manager/wrapper `decode_generated_audio` vô hiệu hóa `cuDNN` và `autocast` riêng cho bước decode. Điều này cho phép `mimi.decode` chạy fallback bằng kernel Pytorch mặc định.
    ```python
    import torch
    with torch.autocast(device_type="cuda", enabled=False), torch.backends.cudnn.flags(enabled=False):
        # ... gọi mimi.decode ...
    ```

2.  **Vô hiệu hóa `torch.compile`**: Sự tối ưu hóa này đôi khi kết xuất sai graph trên Mimi codebook. Khi khởi chạy lệnh, môi trường phải dùng cờ `NO_TORCH_COMPILE=1` nếu luồng inference vẫn bị crash.

## 3. Đầu ra Văn Bản

Nếu huấn luyện bằng chế độ Telex (`data.vietnamese_text_mode=telex`), mô hình sẽ nhận thức và sinh ra văn bản kiểu Telex. Quá trình sinh sẽ cung cấp cả hai tệp:
*   `base_unicode.txt`
*   `finetuned_unicode.txt` (Dạng chữ quốc ngữ có dấu bình thường sau khi chuyển đổi từ telex/no_diacritics ngược lại).
