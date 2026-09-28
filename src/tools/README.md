# Thư mục Công cụ Bổ trợ (`src/tools/`)

Thư mục chứa các công cụ CLI độc lập phục vụ kiểm tra dữ liệu, đánh giá codec, giám định stream sequence, tải trọng số và chạy suy luận kiểm thử.

---

## Danh sách và Ý nghĩa từng công cụ

Chạy từ thư mục gốc repo. Thiết lập import path một lần trong shell:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

| Tên File | Mục Đích & Chức Năng | Ví Dụ Lệnh Thực Thi |
| :--- | :--- | :--- |
| **`validate_dataset.py`** | Kiểm tra toàn bộ tập dữ liệu đã chuẩn bị: xác thực kênh stereo (LEFT = Agent, RIGHT = User), tần số 24kHz, cấu trúc `words.json`, `metadata.json`, mẫu giọng và văn bản prompt. | `python -m tools.validate_dataset data=otospeech` |
| **`inspect_sample.py`** | Giám định chi tiết 1 mẫu hội thoại: dựng ma trận 17 stream PersonaPlex, đối chiếu frame audio với từ vựng text, kiểm tra mask loss và xuất báo cáo `sequence_debug.txt`/`.json`. | `python -m tools.inspect_sample conv_0001 --config configs/config.yaml model=local` |
| **`inference_smoke.py`** | Thực hiện nạp base model + adapter LoRA, áp dụng Hybrid System Prompt và sinh âm thanh phản hồi cùng transcript cho 1 mẫu hội thoại hoặc file audio ngoài. Mặc định đọc cấu hình từ `configs/infer.yaml`. | `python -m tools.inference_smoke --config configs/infer.yaml` |
| **`test_mimi.py`** | Đo lường độ trung thực của Mimi Neural Audio Codec (tính SI-SDR và SNR theo dB) qua chu trình Encode $\to$ Decode trên các mẫu âm thanh (hỗ trợ quét đệ quy và chọn ngẫu nhiên). | `python -m tools.test_mimi --input ../prepared/samples --num-samples 50` |
| **`train_mimi.py`** | Huấn luyện fine-tune Mimi Codec (Stage 0) nhằm thích ứng đặc trưng âm học và thanh điệu tiếng Việt trước khi fine-tune LLM 7B. | `python -m tools.train_mimi --config configs/config.yaml` |
| **`compare_mimi_asr.py`** | Đánh giá chất lượng âm thanh sau nén của Mimi bằng cách chạy nhận dạng tiếng nói ASR (so sánh độ chính xác transcript trước và sau Mimi Encode-Decode). | `python -m tools.compare_mimi_asr --audio-dir ../prepared/samples` |
| **`download_hf_assets.py`** | Script Python tải tự động các file trọng số PersonaPlex và SentencePiece tokenizer từ Hugging Face Hub về thư mục cục bộ mà không cần phụ thuộc mạng trong lúc train. | `python -m tools.download_hf_assets --output-dir ../models` |
| **`train_smoke.py`** | Chạy smoke test nhanh cho forward/backward pass trên mô hình giả lập nhẹ (mock/lightweight) để đảm bảo code logic không bị lỗi runtime. | `python -m tools.train_smoke` |
| **`publish_prepared.py`** | Đóng gói và xuất bản thư mục dữ liệu đã chuẩn bị sang định dạng manifest sẵn sàng phục vụ huấn luyện phân tán. | `python -m tools.publish_prepared --input-dir ../prepared` |
