# Thư mục Scripts (`scripts/`)

Thư mục chứa các shell scripts phục vụ cài đặt môi trường và tải trọng số. Huấn luyện được chạy trực tiếp bằng Python; xem lệnh DDP trong README chính.

---

## Danh sách và Ý nghĩa từng file

| Tên File | Mục Đích & Chức Năng | Cách Dùng Điển Hình |
| :--- | :--- | :--- |
| **`setup_server_env.sh`** | Script cài đặt môi trường Linux GPU (CUDA 12.1/12.4, PyTorch, FlashAttention/triton và các phụ thuộc bắt buộc qua `pip`/`conda`). | `bash scripts/setup_server_env.sh` |
| **`setup_hf_server_env.sh`** | Cài đặt các công cụ tối ưu tải từ Hugging Face (`hf-transfer`, `huggingface-hub[cli]`) để tăng tốc độ download trọng số trên máy chủ. | `bash scripts/setup_hf_server_env.sh` |
| **`download_hf_assets.sh`** | Tự động tải checkpoint chính thức `nvidia/personaplex-7b-v1` và bộ từ vựng `kyutai/moshiko-pytorch-bf16` về thư mục cục bộ `../models/`. | `bash scripts/download_hf_assets.sh` |
| **`train_overfit.sh`** | Chạy kiểm thử overfit nhanh trên 1 GPU (Milestone 1) với 10 mẫu hội thoại chuẩn bị sẵn để kiểm chứng pipeline forward, backward, LoRA gradients. | `bash scripts/train_overfit.sh` |
