# PersonaPlex LoRA Fine-Tuning & Live Interaction — Full Guide

Tài liệu hướng dẫn thiết lập môi trường, toàn bộ **các lệnh kiểm tra (pre-flight checks & validation)** trước khi thực thi, và **các lệnh chạy chính** (huấn luyện LoRA, đàm thoại trực tiếp) cho mô hình `nvidia/personaplex-7b-v1`.

---

## 1. Thiết lập môi trường (Environment Setup)

> **Lưu ý quan trọng**:
> - Sử dụng Conda env hoặc virtualenv (`.venv`) cục bộ; không cài đè lên môi trường hệ thống.
> - Sử dụng **Python 3.10 hoặc 3.11**.
> - Máy GPU thông thường dùng PyTorch **2.4.x** (CUDA 12.1/12.4). NVIDIA B200 cần build PyTorch có hỗ trợ Blackwell: dùng **PyTorch 2.8.x + CUDA 12.8** theo lệnh riêng bên dưới. macOS dùng bản PyTorch phù hợp với MPS/CPU.
> - Bắt buộc cài đặt `ffmpeg` để xử lý audio 24kHz / stereophonic.

### Cách A: Sử dụng Conda (Khuyên dùng)

```bash
# 1. Tạo và kích hoạt môi trường conda với Python 3.11
conda create -n personaplex python=3.11 -y
conda activate personaplex

# 2. Cài đặt ffmpeg qua conda-forge
conda install -c conda-forge ffmpeg -y

# 3. Cài đặt PyTorch cho GPU thông thường (CUDA 12.4). Với B200, thay lệnh này bằng lệnh CUDA 12.8 ở phần lưu ý bên dưới.
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# 4. Cài đặt các phụ thuộc dự án từ thư mục personaplex-finetuning
pip install -r requirements.txt

# 5. Thiết lập import path để chạy trực tiếp từ mã nguồn
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

### Cách B: Sử dụng Python venv cục bộ

```bash
# 1. Tạo và kích hoạt virtualenv
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Cập nhật pip và cài đặt PyTorch với CUDA 12.4 (B200: dùng lệnh CUDA 12.8 bên dưới)
pip install --upgrade pip
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Cài đặt các thư viện dự án
pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

#### PyTorch cho NVIDIA B200

Trong env Python 3.10/3.11 trên máy B200, cài PyTorch Blackwell trước khi cài requirements:

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

### Chạy các module Python

Sau khi kích hoạt môi trường và cài `requirements.txt`, chạy các lệnh `python -m ...` từ thư mục gốc repo. Không cần `pip install -e .`.

---

## 2. Các lệnh kiểm tra trước khi chạy lệnh chính (Pre-flight Checks)

Chạy tuần tự các lệnh kiểm tra sau để đảm bảo 100% môi trường, phần cứng, file trọng số và dữ liệu đã sẵn sàng trước khi bắt đầu huấn luyện hoặc demo.

### 2.1. Kiểm tra Python, PyTorch & Thư viện bắt buộc
Xác minh phiên bản PyTorch, khả năng tăng tốc phần cứng (CUDA trên Linux GPU, MPS trên Apple Silicon) và các thư viện cốt lõi:

```bash
python -c "
import torch
print('=== Kiểm tra môi trường ===')
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('MPS available (macOS):', torch.backends.mps.is_available())
for pkg in ['sphn', 'sentencepiece', 'safetensors', 'einops', 'accelerate']:
    __import__(pkg)
    print(f'{pkg}: OK')
print('===========================')
"
```

---

### 2.2. Kiểm tra file Checkpoint Model PersonaPlex cục bộ
Đảm bảo thư mục model chứa đủ 3 file trọng số bắt buộc (tổng dung lượng ~17GB):

```bash
ls -lh ../models
```
**Yêu cầu tối thiểu:**
- `model.safetensors` (~16.7 GB): Trọng số 7B Transformer của PersonaPlex.
- `tokenizer-e351c8d8-checkpoint125.safetensors` (~384 MB): Trọng số Mimi Audio Codec.
- `tokenizer_spm_32k_3.model` (~552 KB): Bộ từ vựng SentencePiece text tokenizer.

*(Nếu chưa có model, xem mục 3 bên dưới để tải tự động từ Hugging Face).*

---

### 2.4. Kiểm tra & Xác thực Dữ liệu huấn luyện (Validate Dataset)
Kiểm tra cấu trúc file, định dạng stereo WAV (kênh LEFT: Agent, kênh RIGHT: User), file căn chỉnh từ (`words.json`), file voice prompt và file text prompt:

```bash
# Kiểm tra tập mẫu overfit 10 hội thoại
python -m tools.validate_dataset data=overfit

# Hoặc kiểm tra tập dữ liệu quy mô lớn (104 giờ)
python -m tools.validate_dataset data=otospeech
```

---

### 2.5. Kiểm tra chi tiết cấu trúc frame một mẫu (Inspect Sample)
Kiểm tra cấu trúc frame Mimi, số frame Hybrid System Prompt (voice prompt + silence + text prompt + silence), số frame dialogue, và số vị trí token được tính loss mask:

```bash
python -m tools.inspect_sample data=overfit model=local --index 0
```

**Chi tiết các đối số (arguments):**
- `data=<preset>`: Chọn bộ dữ liệu (`overfit`, `otospeech`, `vietnamese`, `kaggle`).
- `model=<preset>`: Chọn đường dẫn model (`local`, `server`, `kaggle`).
- `--index` **[Optional]**: Thứ tự index của mẫu trong danh sách dataset cần kiểm tra (mặc định: `0`).

---

### 2.6. Chạy bộ Unit Tests tự động
Chạy toàn bộ các bài test tự động của dự án (kiểm tra phân giải LoRA adapter, cấu hình, objective loss weights, session frame stepping):

```bash
python -m unittest discover tests
```

---

### 2.7. Chạy 1-Step Smoke Test (Kiểm tra Forward & Backward trên GPU)
Kiểm tra nhanh xem pipeline mô hình, forward, backward, LoRA gradient và freeze base parameters có hoạt động đúng trên GPU hay không:

```bash
python -m personaplex_finetuning.train data=overfit train=overfit model=local --smoke
```
*(hoặc dùng lệnh rút gọn: `python -m tools.train_smoke data=overfit train=overfit model=local`)*

**Chi tiết các đối số (arguments):**
- `--smoke` **[Required cho smoke test]**: Chạy duy nhất 1 step (`max_steps=1`), xác thực gradient của các lớp LoRA khác 0, kiểm tra tham số gốc hoàn toàn đóng băng, lưu adapter và reload lại để kiểm tra sai số suy luận.

---

### 2.8. Đánh giá độ tương thích của Mimi Audio Codec (Đặc biệt cho tiếng Việt)
Kiểm tra chất lượng tái tạo âm thanh qua Mimi Codec (Mimi Encode -> Decode) để đo đạc chỉ số SNR (dB), SI-SDR (dB) và nghe thử các dấu thanh điệu tiếng Việt:

```bash
# Đánh giá 1 file âm thanh cụ thể
python -m tools.test_mimi --input path/to/sample.wav --model-root ../models

# Hoặc đánh giá toàn bộ thư mục âm thanh tiếng Việt
python -m tools.test_mimi --input path/to/vietnamese_wavs/ --num-codebooks 8 --output-dir outputs/mimi_vi_test
```

**Các tiêu chí đánh giá:**
- `SI-SDR >= 12 dB`: Xuất sắc, bảo toàn hoàn hảo âm vị và thanh điệu.
- `SI-SDR 8 - 12 dB`: Khá tốt, âm thanh rõ ràng, nghe rõ ngữ nghĩa.
- `SI-SDR < 8 dB`: Cảnh báo, có nguy cơ mất dấu hoặc biến dạng cao độ ($F_0$) -> cần Fine-tune Mimi trước (Stage 1).

---

## 3. Chuẩn bị Model Checkpoint & Dữ liệu (Nếu chưa có sẵn)

Nếu bạn thiết lập máy mới chưa có sẵn weights mô hình hoặc dataset:

```bash
# 1. Bật tăng tốc tải đa luồng qua Rust backend (hf_transfer)
export HF_HUB_ENABLE_HF_TRANSFER=1

# 2. Đăng nhập Hugging Face (cần chấp thuận điều khoản tại https://huggingface.co/nvidia/personaplex-7b-v1)
hf auth login

# Cách 1: Tải trực tiếp bằng lệnh hf (Khuyên dùng - Nhanh nhất)
hf download nvidia/personaplex-7b-v1 \
  model.safetensors \
  tokenizer-e351c8d8-checkpoint125.safetensors \
  tokenizer_spm_32k_3.model \
  --local-dir models/personaplex-7b-v1

hf download ngocbao220/personaplex-otospeech-prepared \
  --repo-type dataset \
  --local-dir prepared

# Cách 2: Tải tự động qua tool Python có sẵn trong repo
python -m tools.download_hf_assets \
  --assets-dir assets \
  --dataset-repo ngocbao220/personaplex-otospeech-prepared \
  --model-repo nvidia/personaplex-7b-v1 \
  --revision main
```

**Chi tiết các đối số (arguments cho Cách 2):**
- `--assets-dir` **[Optional]**: Thư mục lưu weights và dataset tải về (mặc định: `assets`).
- `--dataset-repo` **[Optional]**: Tên repository dataset trên Hugging Face Hub.
- `--model-repo` **[Optional]**: Tên repository chứa checkpoint PersonaPlex 7B.
- `--revision` **[Optional]**: Nhánh hoặc commit hash muốn tải về (mặc định: `main`).

---

## 4. Các lệnh chạy chính (Main Execution Commands)

### 4.1. Huấn luyện LoRA Fine-Tuning (Single GPU)

```bash
# Huấn luyện overfit trên 10 mẫu với model local
python -m personaplex_finetuning.train \
  data=overfit \
  train=overfit \
  model=local

# Hoặc tùy biến trực tiếp các siêu tham số
python -m personaplex_finetuning.train \
  data=overfit \
  train=overfit \
  model=local \
  train.learning_rate=2.0e-5 \
  train.max_steps=300 \
  lora.rank=16 \
  lora.alpha=32
```

**Chi tiết các cờ CLI (flags):**
- `data=<preset>`: Chọn dataset (`overfit`, `otospeech`, `vietnamese`, `kaggle`).
- `train=<preset>`: Chọn chế độ train (`overfit`, `full`).
- `model=<preset>`: Chọn đường dẫn model (`local`, `server`, `kaggle`).
- `lora=<preset>`: Chọn cấu hình LoRA (`default`, `qlora`, `kaggle`).
- `--qlora` / `--no-qlora` **[Optional]**: Bật hoặc tắt lượng tử hóa 4-bit QLoRA (`nf4`) để giảm dung lượng VRAM.
- `--resume-from` **[Optional]**: Đường dẫn checkpoint directory hoặc `lora.safetensors` để tiếp tục huấn luyện. Checkpoint mới khôi phục adapter, optimizer và scheduler; phải giữ nguyên số GPU và `train.gradient_accumulation_steps`. Checkpoint cũ chỉ có adapter vẫn nạp được với optimizer mới.

**Các tham số override trực tiếp (dotlist overrides) [Optional]:**
- `train.learning_rate`: Tốc độ học (Learning Rate). Thường dùng `2.0e-5` cho 10 mẫu overfit, `1.0e-5` cho tập lớn.
- `train.max_steps`: Tổng số bước huấn luyện (steps).
- `lora.rank`: Thứ hạng ma trận LoRA (Rank, mặc định `16`).
- `lora.alpha`: Hệ số tỉ lệ LoRA alpha (thường đặt bằng `2 * rank`, tức `32`).
- `data.window_seconds`: Độ dài cửa sổ thời gian (giây) lấy từ hội thoại để đưa vào huấn luyện (ví dụ: `30`).
- `train.output_dir`: Thư mục lưu checkpoint adapter và log metric.
- `train.gradient_accumulation_steps`: Số bước tích lũy gradient để tăng effective batch size.
- `train.gradient_checkpointing`: Đặt `true` để tiết kiệm bộ nhớ GPU.
- `train.mixed_precision`: Chế độ precision (`bf16` hoặc `fp16`).

---

### 4.2. Moshi-style fixed-duration training

Run the deterministic 10-conversation overfit on one GPU first:

```bash
python -m train configs/moshi_overfit_10.yaml
```

After verifying decreasing loss, adapter reload, and inference, run FSDP `FULL_SHARD` on eight GPUs with the full training split:

```bash
torchrun --nproc-per-node 8 -m train configs/moshi_code_style.yaml sample_number=null
```

`configs/moshi_code_style.yaml` retains the requested `duration_sec=100`, `sample_number=10`, `batch_size=16`, and `max_steps=2000`. This batch size cannot form batches from only ten chunks across eight ranks, so the full-run command explicitly sets `sample_number=null`. Use the same FSDP topology when resuming a distributed checkpoint. Training sets `NO_TORCH_COMPILE=1` by default to avoid Moshi compile failures reported with FSDP padding; opt in to compile by setting it to `0`.

## 5. Giám Sát Quá Trình Huấn Luyện (TensorBoard)

Khởi động giao diện trực quan hóa loss (`loss/total`, `loss/text`, `loss/audio_semantic`, `loss/audio_nonsemantic`), gradient norm và GPU memory:

```bash
tensorboard \
  --logdir runs/hf_overfit_10 \
  --host 0.0.0.0 \
  --port 6006
```

**Chi tiết các đối số (arguments):**
- `--logdir` **[Required]**: Thư mục chứa log sự kiện huấn luyện (event logs).
- `--host` **[Optional]**: Địa chỉ IP bind socket (`0.0.0.0` để cho phép truy cập từ xa qua mạng).
- `--port` **[Optional]**: Cổng mở giao diện web (mặc định: `6006`).

---

## 6. Đánh Giá Khả Năng Hội Thoại Song Công (Full-Duplex-Bench)

Hệ thống đánh giá tự động dựa trên chuẩn của **Full-Duplex-Bench (FDB v1.0 & v1.5)** để đo lường 4 trục tương tác cốt lõi:
1. **Pause Handling**: Khả năng kiên nhẫn, không cướp lời khi người dùng ngập ngừng/nghỉ giữa câu (**TOR ↓**).
2. **Backchanneling**: Khả năng chèn âm đệm ngắn tự nhiên khi người dùng nói dài (**TOR ↓, Freq ↑, JSD ↓**).
3. **Smooth Turn-Taking**: Tốc độ và độ nhạy bắt lời khi người dùng dứt câu (**TOR ↑, Latency ↓**).
4. **User Interruption**: Xử lý nhường microphone và trả lời nội dung mới khi bị chen ngang (**TOR ↑, Response Quality ↑, Latency ↓**).

> **Điểm ưu việt**: Tận dụng trực tiếp luồng native 12.5 Hz frame tokens của PersonaPlex và mốc thời gian có sẵn từ data preparation; **hoàn toàn offline, không cần cài đặt thêm mô hình ASR cồng kềnh**.

### 6.1. Chạy Thử Nghiệm Mock / So Sánh Nhanh (Dry-run Demo)
In bảng đối sánh mô phỏng giữa Base model và LoRA model dạng Rich Table trên Terminal:
```bash
python ../run_fdp_benchmark.py --compare_demo
```

### 6.2. Chạy Benchmark Thật Trên Checkpoint PersonaPlex Base
Đánh giá trên tập dữ liệu chuẩn Full-Duplex-Bench v1.0 (727 mẫu tiếng Anh) để đối chuẩn với Table 2 của paper:

```bash
# Chạy trên GPU / Linux CUDA:
python ../run_fdp_benchmark.py \
  --model_root ../models \
  --model_name "PersonaPlex Base" \
  --fdb_dir ../benchmarks/datasets/fdb_v1/v1.0/extracted \
  --max_samples_per_task 10 \
  --output_dir ../benchmarks/results_fdb

# Chạy trên macOS Apple Silicon (MPS):
PYTORCH_ENABLE_MPS_FALLBACK=1 python ../run_fdp_benchmark.py \
  --model_root ../models \
  --model_name "PersonaPlex Base" \
  --fdb_dir ../benchmarks/datasets/fdb_v1/v1.0/extracted \
  --max_samples_per_task 10 \
  --output_dir ../benchmarks/results_fdb
```

### 6.3. Chạy Benchmark Đánh Giá Checkpoint LoRA (Sau Khi Fine-tune)
Kiểm tra xem mô hình sau fine-tune có bảo toàn hoặc cải thiện năng lực full-duplex hay không:

```bash
python ../run_fdp_benchmark.py \
  --model_root ../models \
  --adapter ../checkpoints/lora_adapter.pt \
  --model_name "PersonaPlex LoRA" \
  --fdb_dir ../benchmarks/datasets/fdb_v1/v1.0/extracted \
  --max_samples_per_task 10 \
  --output_dir ../benchmarks/results_fdb
```

### 6.4. Chạy Benchmark Trên Dữ Liệu In-Domain / Tiếng Việt (Tương Lai)
Khi có tập dữ liệu hội thoại tiếng Việt (hoặc tập OtoSpeech trong `prepared/samples`):
```bash
python ../run_fdp_benchmark.py \
  --model_root ../models \
  --adapter ../checkpoints/lora_adapter.pt \
  --model_name "PersonaPlex-VI LoRA" \
  --prepared_dir ../prepared/samples \
  --max_samples_per_task 10 \
  --output_dir ../benchmarks/results_vi
```

### 6.5. Bảng Giải Thích Các Tham Số (Arguments)

| Tham số CLI | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--model_root` | `models` | Đường dẫn thư mục chứa base model (`model.safetensors`, tokenizer, mimi). |
| `--adapter` | `None` | Đường dẫn file trọng số LoRA (`lora.safetensors` hoặc `.pt`). Nếu bỏ trống, chạy base model gốc. |
| `--model_name` | `PersonaPlex Base` | Tên mô hình hiển thị trên bảng kết quả. |
| `--fdb_dir` | `benchmarks/datasets/fdb_v1/v1.0/extracted` | Thư mục chứa dataset chuẩn Full-Duplex-Bench v1.0. |
| `--prepared_dir` | `None` | Thư mục chứa các mẫu hội thoại tự chuẩn bị (`words.json`, `conversation.wav`). |
| `--max_samples_per_task` | `10` | Số lượng mẫu tối đa đánh giá cho mỗi nhóm tác vụ (giúp chạy nhanh hoặc toàn diện). |
| `--device` | `auto` | Thiết bị tính toán: `auto`, `cuda`, `mps`, hoặc `cpu`. |
| `--output_dir` | `benchmarks/results` | Thư mục lưu bảng báo cáo Markdown và JSON. |
| `--mock` | `False` | Bật chế độ chạy giả lập nhanh (dry-run). |
| `--compare_demo` | `False` | Chạy demo so sánh trực quan Base vs LoRA trên Terminal. |

### 6.6. File Kết Quả Đầu Ra
Kết quả sau khi chạy được tự động:
1. In bảng Rich Table màu sắc trực quan ngay trên Terminal.
2. Xuất bảng Markdown: `{output_dir}/fdp-benchmark-result.md`.
3. Xuất file JSON chi tiết: `{output_dir}/fdp-benchmark-result.json`.
