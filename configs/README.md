# Thư mục Cấu hình (`configs/`)

Thư mục chứa các tệp cấu hình phục vụ huấn luyện và suy luận PersonaPlex. Hệ thống hỗ trợ Hydra và OmegaConf cho phép ghép nối modular và ghi đè (override) linh hoạt qua dòng lệnh.

---

## 1. Cấu hình Cấp gốc (Root Configs)

| Tên File | Mục Đích & Ý Nghĩa |
| :--- | :--- |
| **`config.yaml`** | File cấu hình tổng thể (Master Config). Khai báo các module mặc định (`defaults`) gồm `model: server`, `data: otospeech`, `lora: default`, `train: 104h`. Dùng với `python -m personaplex_finetuning.train`, có thể khởi chạy đa GPU qua `python -m accelerate.commands.launch`. |
| **`infer.yaml`** | File cấu hình chuyên dụng cho quá trình suy luận (Inference Smoke Test). Tích hợp cấu hình mô hình, đường dẫn adapter checkpoint LoRA, mẫu giọng nói (voice prompt), system text prompt và siêu tham số lấy mẫu (`generation`). Mỗi lần chạy lưu vào thư mục `infer_<date>` bên dưới `inference.output_dir`. |

---

## 2. Các Thư mục Con (Modular Sub-configs)

### `configs/model/` (Đường dẫn mô hình & Thiết bị)
- **`server.yaml`**: Đường dẫn tới checkpoint PersonaPlex-7B trên máy chủ GPU lưu trữ (`/storage-voice/voice/vdt/baottn/personaplex-7b-v1`), thiết bị mặc định `cuda`.
- **`local.yaml`**: Đường dẫn tới checkpoint mô hình khi chạy trên máy trạm cục bộ (`../models`).

### `configs/data/` (Tập dữ liệu & Cửa sổ thời gian)
- **`otospeech.yaml`**: Cấu hình nạp dữ liệu chuẩn bị sẵn OtoSpeech (`window_seconds: 30.0`, đường dẫn manifest `train.jsonl`, hỗ trợ hoán đổi vai trò và chunking tĩnh).
- **`vietnamese.yaml`**: Cấu hình tổng quát cho các tập dữ liệu hội thoại tiếng Việt khác.

### `configs/lora/` (Cấu hình LoRA Adapter)
- **`default.yaml`**: LoRA tiêu chuẩn với `rank: 16`, `alpha: 32`, huấn luyện trên precision FP16/BF16.
- **`qlora.yaml`**: Kích hoạt 4-bit Quantization (NF4) kết hợp LoRA (QLoRA) giúp giảm thiểu tiêu thụ VRAM khi chạy trên GPU yếu.

### `configs/train/` (Siêu tham số Huấn luyện)
- **`overfit.yaml`**: Thiết lập chạy ngắn (300 bước, save/eval mỗi 50 bước) để kiểm thử overfit 10 mẫu nhanh.
- **`full.yaml`**: Thiết lập huấn luyện sản xuất quy mô đầy đủ với learning rate, scheduler cosine decay và gradient accumulation.
