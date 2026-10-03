# Tài liệu Dự án PersonaPlex Fine-Tuning

Chào mừng bạn đến với tài liệu của dự án fine-tuning mô hình PersonaPlex. Dự án này tập trung vào việc fine-tune mô hình `nvidia/personaplex-7b-v1` bằng các đoạn hội thoại OtoSpeech đã được chuẩn bị sẵn, bao gồm cả hỗ trợ LoRA và Full Fine-tuning.

## Cấu trúc Tài liệu

*   **[Kiến trúc (Architecture)](architecture.md)**: Chi tiết về luồng dữ liệu 17 stream (text, 8 agent audio, 8 user audio), `MimiCodec`, `LMGen`, và các quy tắc xếp luồng (delay).
*   **[Huấn luyện (Training)](training.md)**: Hướng dẫn chi tiết cách cấu hình và chạy quá trình huấn luyện (LoRA & Full Fine-Tuning), các trọng số loss, tính toán memory, mask loss cho system prompt.
*   **[Suy luận (Inference)](inference.md)**: Cách chạy inference với adapter, giải quyết các lỗi liên quan đến CUDA engine/cuDNN khi chạy `mimi.decode`, xử lý text Telex.
*   **[Dữ liệu (Data Pipeline)](data.md)**: Chi tiết về cấu trúc Manifest, WAV stereo, các tool kiểm tra dữ liệu (`validate_dataset`, `inspect_sample`).

*(Lưu ý: Các liên kết trên hiện đang được xây dựng.)*
