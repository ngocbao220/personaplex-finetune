# Kiến trúc PersonaPlex (Architecture)

Tài liệu này mô tả chi tiết về cách biểu diễn dữ liệu và kiến trúc của model PersonaPlex trong quá trình huấn luyện và suy luận.

## 1. Cấu trúc 17 Luồng (Streams)

Mô hình LM của PersonaPlex dự đoán và nhận đầu vào qua **17 luồng dữ liệu song song** tại mỗi frame âm thanh (80ms).

*   **Luồng 0**: Text (văn bản) của Agent.
*   **Luồng 1 - 8**: 8 Codebook của Agent (Mimi Audio Encoder/Decoder).
*   **Luồng 9 - 16**: 8 Codebook của User (Mimi Audio Encoder/Decoder).

### Cài đặt Từ vựng (Vocabulary & Tokens)

*   `card=2048`: Số lượng token cho âm thanh (Mimi tokens).
*   `text_card=32000`: Kích thước từ vựng của Text.
*   **Text PAD**: 3
*   **Text END_PAD**: 0
*   **Zero Token**: -1
*   **Audio Offset**: 1 (Các token audio bắt đầu sau index này để tránh xung đột với các token đặc biệt, nếu có).

### Kênh Âm Thanh (Channel Mapping)

*   **Kênh Trái (LEFT)**: Agent (Luồng 1-8).
*   **Kênh Phải (RIGHT)**: User (Luồng 9-16).

## 2. Quản lý Độ trễ (Delays)

PersonaPlex sử dụng một cấu trúc delay giữa các luồng để tạo điều kiện tự hồi quy cho việc sinh văn bản trước khi sinh âm thanh, cũng như các phụ thuộc nhân quả giữa semantic tokens và acoustic tokens.

Danh sách delay cấu hình cho 17 luồng:
`[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]`

*   `delay=0` ở stream 0 (Text), stream 1 (Agent Codebook 0 / Semantic), và stream 9 (User Codebook 0 / Semantic).
*   `delay=1` ở các stream còn lại (Acoustic codebooks).

## 3. Quá trình Huấn Luyện (LMModel.forward_train)

Trong quá trình huấn luyện, `LMModel.forward_train` (trong `models/lm.py`):
1.  Nhận tensors chứa `[B, 17, T]` mã token.
2.  Áp dụng initial frame đặc biệt ở đầu chuỗi.
3.  Áp dụng các delay tương ứng cho từng luồng.
4.  Sử dụng `delayed[:, :, :-1]` làm input cho transformer và `delayed[:, :, 1:]` cho targets.
5.  Kết quả logits của Transformer (Text/Audio) sau đó được loại bỏ delay (undelayed) để tính Cross Entropy Loss (CE) với chuỗi gốc mà không cần dịch chuyển bù thêm lần nữa.

## 4. Thiết lập System Prompt & Loss Masking

`LMGen.step_system_prompts` được dùng để thiết lập prompt điều kiện tại đầu hội thoại:
*   Trình tự: Agent Voice Prompt -> Khoảng lặng (silence) -> Text System Prompt -> Khoảng lặng.
*   Trong chuỗi dữ liệu huấn luyện, các tokens của phần System Prompt này vẫn tồn tại nhưng **bị mask loss về 0** (chỉ có tác dụng tạo ngữ cảnh, model không bị phạt nếu không dự đoán đúng phần này).
