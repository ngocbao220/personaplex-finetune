# Thư mục Kiểm thử Tự động (`tests/`)

Thư mục chứa toàn bộ bộ kiểm thử tự động (Unit Tests & Integration Smoke Tests) nhằm đảm bảo mọi thành phần kiến trúc của PersonaPlex fine-tuning hoạt động chính xác và không bị hồi quy (regression).

Để chạy toàn bộ kiểm thử:
```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
```

---

## Danh sách và Ý nghĩa từng tệp kiểm thử

| Tên File Test | Thành Phần Được Kiểm Thử & Mục Đích |
| :--- | :--- |
| **`test_sequence.py`** | Kiểm tra logic dựng ma trận 17 stream, padding, căn chỉnh delay 8 frame audio/text và tạo target tensor đúng quy cách. |
| **`test_golden_sample.py`** | Golden invariant test: Xác minh tuyệt đối kênh LEFT = Agent (target), RIGHT = User (conditioning), kiểm tra loss mask prompt và các trọng số mất mát chuẩn trên mẫu kiểm chứng. |
| **`test_objective.py`** | Kiểm tra tính toán hàm loss đa luồng có trọng số: xác nhận codebook 0 có trọng số 1.0, codebook 1-7 có trọng số 0.02, text padding có trọng số 0.3 và mask = 0 trên prompt conditioning. |
| **`test_lora.py`** | Kiểm tra việc chèn (inject) các tầng LoRA vào đúng module, kiểm tra tham số gốc được đóng băng (`requires_grad=False`), và kiểm tra việc lưu/nạp file `lora.safetensors`. |
| **`test_config.py`** | Kiểm tra bộ phân giải cấu hình: nạp file YAML, ghép nối Hydra modular configs, xử lý ghi đè (overrides) và phân giải đường dẫn tương đối. |
| **`test_contract.py`** | Kiểm tra tính tương thích với đặc tả dữ liệu (Data Contract): cấu trúc metadata, định dạng âm thanh stereo WAV, thông tin căn chỉnh từ `words.json`. |
| **`test_train.py`** | Kiểm tra vòng lặp huấn luyện đơn GPU: các bước forward, backward, cập nhật trọng số LoRA và bảo toàn base weights. |
| **`test_inference.py`** | Kiểm tra quy trình suy luận: nạp adapter, khởi chạy generator streaming theo từng khung 80ms, giữ mimi streaming decoder liên tục và giao diện CLI `inference_smoke`. |
| **`test_production_features.py`** | Kiểm tra các tính năng nâng cao: kích hoạt Gradient Checkpointing, Cosine Learning Rate Scheduler với Warmup và khôi phục trạng thái huấn luyện (Stateful Resume). |
| **`test_fsdp.py`** | Kiểm tra thiết lập phân tán FSDP: bfloat16 mixed-precision, auto-wrapping policy và khởi tạo sharded model. |
| **`test_gating.py`** | Kiểm tra cơ chế gating và kích hoạt mạng nơ-ron trong các khối Transformer. |
| **`test_runtime_paths.py`** | Kiểm tra khả năng phát hiện và phân giải đường dẫn các file trọng số cục bộ (`model.safetensors`, `tokenizer-*.safetensors`, `tokenizer_spm_32k_3.model`). |
| **`test_hf_assets.py`** | Kiểm tra logic tải và quản lý file từ Hugging Face Hub. |
| **`test_validate_dataset.py`** | Kiểm tra công cụ kiểm định dữ liệu `validate_dataset.py` đối với các trường hợp dữ liệu hợp lệ và không hợp lệ. |
| **`test_vietnamese_tokenizer.py`** | Kiểm tra bộ từ vựng SentencePiece tokenizer trong việc mã hóa và giải mã đầy đủ các thanh điệu tiếng Việt (ngã, hỏi, nặng, sắc, huyền). |
| **`test_benchmark_metrics.py`** | Kiểm tra tính chính xác của các thuật toán đo đạc benchmark (Latency, Turn Overlap Rate, Backchanneling ICC). |
| **`test_benchmark_runner.py`** | Kiểm tra luồng thực thi chấm điểm tự động trên các task của Full-Duplex-Bench. |
| **`test_pipeline_smoke.py`** | Kiểm thử tích hợp toàn diện (End-to-End): Chạy liên hoàn từ nạp dữ liệu $\to$ dựng chuỗi $\to$ huấn luyện LoRA $\to$ lưu adapter $\to$ nạp lại suy luận. |
