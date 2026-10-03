# Dữ liệu (Data Pipeline)

Tài liệu này trình bày luồng xử lý dữ liệu và các công cụ kiểm tra dữ liệu trong dự án.

## 1. Yêu cầu Đầu vào (Data Requirements)

Dữ liệu đầu vào cho quá trình huấn luyện bắt buộc đã được xử lý (prepared) thông qua pipeline OtoSpeech trước khi đưa vào thư mục huấn luyện. Dữ liệu huấn luyện **không** thực hiện bất kỳ bước forced alignment, ghép prompt tĩnh, hay sửa chữa raw SRT nào bên trong mã huấn luyện. 

Đầu vào cho mỗi hội thoại cần:
*   File WAV âm thanh Stereo (Kênh Trái: Agent, Kênh Phải: User).
*   Transcript dạng Word-level.
*   WAV âm thanh giọng của Agent (Voice Prompt).
*   Text/System prompt của Agent.
*   Metadata.

## 2. Công cụ Đánh giá & Lọc dữ liệu

Trước khi train, người dùng cần chạy các module nằm trong `tools` để xác thực:

**Kiểm tra tính hợp lệ của Dataset:**
```bash
python -m tools.validate_dataset --config configs/config.yaml
```
Công cụ này sẽ kiểm tra cấu trúc thư mục, tệp manifest, file WAV stereo, transcript, và tính toàn vẹn của các file prompt.

**Tính toán Capacity của Codebook (Overflow text-token check):**
```bash
python -m tools.check_text_chunk_capacity --config configs/config.yaml
```
Sử dụng mã hoá Mimi để đảm bảo độ dài văn bản không vượt quá dung lượng cho phép của mỗi block/chunk kích thước `duration_sec`. Việc tràn văn bản (overflow) sẽ khiến cả đoạn chunk đó (kể cả phần đổi chiều role-swap) bị loại bỏ khỏi luồng train.

**Kiểm tra Trực quan (Inspect Sample):**
```bash
python -m tools.inspect_sample --config configs/config.yaml --index 0
```
Xem dữ liệu thô, trình tự chuỗi PersonaPlex và ma trận mask loss của mẫu đầu tiên (index = 0). Có thể dùng cờ `--sample-id ID` để kiểm tra theo ID hội thoại.

## 3. Cache và Lọc dữ liệu lúc Train

Trong quá trình huấn luyện, tập dataset sẽ tự động lưu lại cache danh sách những mục bị vứt bỏ (do timestamp vượt khỏi audio, text overflow) vào trong file log tên là `data_filter_report.json`.

Để ép hệ thống bỏ qua file cache và quét lại toàn bộ lỗi overflow/cấu trúc, hãy thêm cờ `--force-filter` vào dòng lệnh khi chạy train hoặc inference.
