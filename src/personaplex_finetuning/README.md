# Gói Mã Nguồn Cốt Lõi (`src/personaplex_finetuning/`)

Thư mục chứa gói thư viện Python chính thực hiện toàn bộ kiến trúc huấn luyện, xây dựng chuỗi đa luồng (sequence building), nạp mô hình runtime, áp dụng LoRA và tính hàm mất mát (loss objective) cho PersonaPlex 7B.

---

## Danh sách và Ý nghĩa từng module

| Tên Module | Trách Nhiệm & Chức Năng Chính |
| :--- | :--- |
| **`config.py`** | Định nghĩa đối tượng cấu hình bất biến `Config` (dataclass). Quản lý đọc tệp cấu hình YAML/JSON, hỗ trợ nạp cấu trúc modular Hydra, phân giải đường dẫn tuyệt đối/tương đối và kiểm tra tính hợp lệ của siêu tham số. |
| **`data.py`** | Lớp `PreparedDataset` và cấu trúc `PreparedSample`. Đọc file manifest `train.jsonl`, kiểm tra sự tồn tại của file audio/json, trích xuất cửa sổ thời gian (`window_seconds`), hỗ trợ static chunking và hoán đổi vai trò người nói (swap roles). |
| **`sequence.py`** | Module then chốt xây dựng ma trận **17 stream** PersonaPlex (1 stream Text Agent + 8 stream Audio Codebooks Agent + 8 stream Audio Codebooks User). Tích hợp cấu trúc Hybrid System Prompt (Voice Prompt + Pause + Text Prompt + Pause + Dialogue) và tính toán mặt nạ trọng số loss mask. |
| **`objective.py`** | Định nghĩa hàm tính loss trọng số PersonaPlex (`torch_weighted_cross_entropy`): Audio codebook đầu tiên (semantic) trọng số 1.0, các codebook acoustic 1-7 trọng số 0.02, text padding token trọng số 0.3; hoàn toàn triệt tiêu loss (mask = 0.0) trên phần system prompt conditioning và user input. |
| **`lora.py`** | Quản lý Parameter-Efficient Fine-Tuning qua LoRA: Inject các ma trận phân rã hạng thấp ($A$ và $B$) vào Temporal Transformer và Depformer (tập trung `q_proj`, `k_proj`, `v_proj`, `o_proj`), đóng băng toàn bộ base weights 7B và cung cấp cơ chế lưu/nạp file `lora.safetensors`. |
| **`runtime.py`** | Nạp toàn bộ mô hình vào bộ nhớ (`load_runtime`): Khởi tạo 7B Language Model từ checkpoint safetensors, nạp Mimi Audio Codec 24kHz, cấu hình bộ từ vựng SentencePiece tokenizer; hỗ trợ lượng tử hóa 4-bit (NF4) cho QLoRA. |
| **`train.py`** | Vòng lặp huấn luyện chính (Training Engine): Điều phối forward pass, backward pass, gradient clipping, gradient accumulation, cosine learning rate scheduler với warmup, lưu trữ checkpoint an toàn và hỗ trợ khôi phục trạng thái huấn luyện (stateful resume). |
| **`fsdp.py`** | Các tiện ích FSDP tương thích ngược; trainer hiện dùng DDP khi chạy nhiều GPU và không import module này. |
| **`inference.py`** | Động cơ suy luận kiểm thử (`generate`, `smoke`): Nạp adapter LoRA trên base model nguyên bản, chạy generator tự hồi quy theo từng khung 80ms, giải mã audio qua Mimi streaming decoder và giải mã text tiếng Việt qua SentencePiece processor. |
| **`hf_assets.py`** | Tiện ích hỗ trợ quản lý, đối chiếu mã hash và tải các asset từ Hugging Face Hub về môi trường lưu trữ cục bộ. |
