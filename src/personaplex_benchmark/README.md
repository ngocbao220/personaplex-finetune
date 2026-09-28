# Bộ Đánh Giá Full-Duplex-Bench (`src/personaplex_benchmark/`)

Thư mục chứa bộ công cụ benchmark đàm thoại song công toàn phần (Full-Duplex Speech Benchmark) dành cho mô hình PersonaPlex theo chuẩn **Full-Duplex-Bench (FDB v1.0)**.

---

## 1. Danh sách các module chính

| Tên Module | Chức Năng & Ý Nghĩa |
| :--- | :--- |
| **`cli.py`** | Điểm truy cập dòng lệnh (CLI Interface): Nhận tham số thư mục dữ liệu FDB, checkpoint LoRA, voice prompt; điều phối quá trình chấm điểm và xuất bảng kết quả. |
| **`contract.py`** | Định nghĩa cấu trúc mẫu kiểm thử `BenchmarkSample` (hỗ trợ các task: `pause_synthetic`, `pause_natural`, `backchannel`, `turn_taking`, `interruption`) và các hàm nạp dữ liệu từ dataset chính thức hoặc thư mục in-domain. |
| **`runner.py`** | Bộ điều khiển thực thi (`PersonaPlexStreamingRunner`): Truyền từng khung âm thanh 80ms của người dùng vào mô hình và quan sát hành vi phản hồi của Agent (thời điểm nói, thời điểm im lặng, ngắt lời). Có sẵn `MockBenchmarkRunner` cho dry-run. |
| **`reporter.py`** | Tổng hợp kết quả đo đạc từ các sample, tính toán giá trị trung bình/trung vị theo từng task và xuất bảng Markdown/Rich Console cùng file báo cáo JSON. |

---

## 2. Thư mục Metric Chuyên dụng (`metrics/`)

- **`backchannel.py`**: Đo lường khả năng chêm xen tự nhiên (như "ừm", "vâng", "dạ") bằng cách so khớp với phân phối xác suất Inter-Chunk Correlation (ICC) từ dữ liệu người thật.
- **`latency.py`**: Tính toán độ trễ phản hồi (Response Latency tính theo mili-giây) từ lúc người dùng dứt lời đến khi mô hình bắt đầu phát âm thanh đầu tiên.
- **`tor.py`**: Đo lường Turn Overlap Rate (tỉ lệ nói đè/chen ngang trái phép) và thời gian xử lý khi người dùng cắt ngang lời mô hình (Interruption handling).
- **`response_quality.py`**: Đánh giá tính mạch lạc và phù hợp ngữ cảnh của chuỗi phản hồi.
