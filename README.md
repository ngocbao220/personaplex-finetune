# PersonaPlex LoRA Fine-Tuning & Live Interaction — Full Guide

Tài liệu hướng dẫn thiết lập môi trường, toàn bộ **các lệnh kiểm tra (pre-flight checks & validation)** trước khi thực thi, và **các lệnh chạy chính** (huấn luyện LoRA, đàm thoại trực tiếp) cho mô hình `nvidia/personaplex-7b-v1`.

---

## 1. Thiết lập môi trường (Environment Setup)

> **Lưu ý quan trọng**:
> - Sử dụng Conda env hoặc virtualenv (`.venv`) cục bộ; không cài đè lên môi trường hệ thống.
> - Sử dụng **Python 3.10 hoặc 3.11**.
> - PyTorch yêu cầu phiên bản **Torch 2.4.x** (CUDA 12.1 hoặc 12.4 trên Linux/Windows, hoặc MPS/CPU trên macOS); **không dùng Torch 2.8**.
> - Bắt buộc cài đặt `ffmpeg` để xử lý audio 24kHz / stereophonic.

### Cách A: Sử dụng Conda (Khuyên dùng)

```bash
# 1. Tạo và kích hoạt môi trường conda với Python 3.11
conda create -n personaplex python=3.11 -y
conda activate personaplex

# 2. Cài đặt ffmpeg qua conda-forge
conda install -c conda-forge ffmpeg -y

# 3. Cài đặt PyTorch 2.4.1 tương thích CUDA 12.4 (Nếu dùng Mac: pip install torch==2.4.1 torchaudio==2.4.1)
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# 4. Cài đặt các phụ thuộc dự án từ thư mục personaplex-finetuning
pip install -r requirements.txt

# 5. Cài đặt gói ở chế độ editable và thiết lập PYTHONPATH
pip install -e .
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

### Cách B: Sử dụng Python venv cục bộ

```bash
# 1. Tạo và kích hoạt virtualenv
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Cập nhật pip và cài đặt PyTorch với CUDA 12.4
pip install --upgrade pip
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Cài đặt các thư viện dự án
pip install -r requirements.txt
pip install -e .
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

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
for pkg in ['sphn', 'sounddevice', 'sentencepiece', 'safetensors', 'einops', 'accelerate']:
    __import__(pkg)
    print(f'{pkg}: OK')
print('===========================')
"
```

### 2.2. Kiểm tra phần cứng âm thanh (Microphone & Loa)
Liệt kê danh sách các thiết bị âm thanh đầu vào/đầu ra trên máy để phục vụ live demo:

```bash
python -m tools.interactive_cli --list-devices
```
**Ý nghĩa các cờ (flags):**
- `--list-devices` **[Optional]**: Truy vấn CoreAudio/ALSA và in bảng ID, tên thiết bị micro và loa có sẵn rồi thoát.

---

### 2.3. Kiểm tra file Checkpoint Model PersonaPlex cục bộ
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
python -m tools.validate_dataset \
  --config configs/test_overfit.yaml  # [Required] Đường dẫn file config
```

**Chi tiết các đối số (arguments):**
- `--config` **[Required]**: Đường dẫn tới file cấu hình YAML/JSON định nghĩa dataset (`data.prepared_dir` hoặc `data.manifest`) và thời lượng cửa sổ cắt `data.window_seconds`.

**Cách thay đổi tham số:**
- Thay đổi `--config configs/train_104h.yaml` để kiểm tra tập dữ liệu quy mô lớn (104 giờ).

---

### 2.5. Kiểm tra chi tiết cấu trúc frame một mẫu (Inspect Sample)
Kiểm tra cấu trúc frame Mimi, số frame Hybrid System Prompt (voice prompt + silence + text prompt + silence), số frame dialogue, và số vị trí token được tính loss mask:

```bash
python -m tools.inspect_sample \
  --config configs/test_overfit.yaml \
  --index 0
```

**Chi tiết các đối số (arguments):**
- `--config` **[Required]**: Đường dẫn tới file cấu hình.
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
python -m personaplex_finetuning.train \
  --config configs/test_overfit.yaml \
  --smoke
```
*(hoặc dùng lệnh rút gọn: `python -m tools.train_smoke --config configs/test_overfit.yaml`)*

**Chi tiết các đối số (arguments):**
- `--config` **[Required]**: Đường dẫn file cấu hình.
- `--smoke` **[Required cho smoke test]**: Chạy duy nhất 1 step (`max_steps=1`), xác thực gradient của các lớp LoRA khác 0, kiểm tra tham số gốc hoàn toàn đóng băng, lưu adapter và reload lại để kiểm tra sai số suy luận.

---

## 3. Chuẩn bị Model Checkpoint & Dữ liệu (Nếu chưa có sẵn)

Nếu bạn thiết lập máy mới chưa có sẵn weights mô hình hoặc dataset:

```bash
# 1. Đăng nhập Hugging Face (cần chấp thuận điều khoản tại https://huggingface.co/nvidia/personaplex-7b-v1)
hf auth login

# 2. Tải toàn bộ checkpoint PersonaPlex 7B và dataset mẫu
python -m tools.download_hf_assets \
  --assets-dir assets \
  --dataset-repo ngocbao220/personaplex-otospeech-prepared \
  --model-repo nvidia/personaplex-7b-v1 \
  --revision main
```

**Chi tiết các đối số (arguments):**
- `--assets-dir` **[Optional]**: Thư mục lưu weights và dataset tải về (mặc định: `assets`).
- `--dataset-repo` **[Optional]**: Tên repository dataset trên Hugging Face Hub.
- `--model-repo` **[Optional]**: Tên repository chứa checkpoint PersonaPlex 7B.
- `--revision` **[Optional]**: Nhánh hoặc commit hash muốn tải về (mặc định: `main`).

---

## 4. Các lệnh chạy chính (Main Execution Commands)

### 4.1. Huấn luyện LoRA Fine-Tuning (Single GPU)

```bash
python -m personaplex_finetuning.train \
  --config configs/test_overfit.yaml \
  --no-qlora \
  train.learning_rate=2.0e-5 \
  train.max_steps=300 \
  lora.rank=16 \
  lora.alpha=32 \
  data.window_seconds=30 \
  train.output_dir=../runs/hf_overfit_10
```

**Chi tiết các cờ CLI (flags):**
- `--config` **[Required]**: Đường dẫn file cấu hình YAML/JSON chính.
- `--qlora` / `--no-qlora` **[Optional]**: Bật hoặc tắt lượng tử hóa 4-bit QLoRA (`nf4`) để giảm dung lượng VRAM.
- `--resume-from` **[Optional]**: Đường dẫn tới checkpoint trước đó để tiếp tục huấn luyện (ví dụ: `--resume-from ../runs/hf_overfit_10/checkpoints/checkpoint_000100`).

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

### 4.2. Huấn luyện Multi-GPU với DDP (Accelerate)

#### Cách 1: Sử dụng launcher script `scripts/train_gpus.sh`

```bash
bash scripts/train_gpus.sh \
  --device_ids 0,1,2,3 \
  --num_processes 4 \
  --config configs/train_104h.yaml \
  train.learning_rate=2.0e-5 \
  train.max_steps=10000 \
  train.gradient_accumulation_steps=8
```

**Chi tiết các đối số (arguments):**
- `--device_ids` / `--gpu_ids` **[Optional nhưng khuyến nghị]**: Danh sách chỉ số GPU vật lý sử dụng (ví dụ: `0,1` hoặc `0,1,2,3`). Script tự động thiết lập `CUDA_VISIBLE_DEVICES`.
- `--num_processes` / `-n` **[Optional]**: Số lượng tiến trình DDP (mặc định khớp theo số GPU ở `--device_ids`).
- `--config` **[Optional]**: Đường dẫn file cấu hình (mặc định: `configs/train_104h.yaml`).
- Các tham số override sau cờ được chuyển trực tiếp vào chương trình.

#### Cách 2: Gọi trực tiếp qua lệnh `accelerate launch`

```bash
accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --mixed_precision bf16 \
  -m personaplex_finetuning.train \
  --config configs/train_104h.yaml \
  train.learning_rate=2.0e-5
```

---

### 4.3. Chạy Interactive Live Demo qua Terminal CLI (Nói chuyện Micro & Loa trực tiếp)

Tương tác đàm thoại 2 chiều thời gian thực (full-duplex) với PersonaPlex qua microphone và loa máy tính. Hỗ trợ trỏ vào base checkpoint cục bộ và nạp adapter LoRA đã fine-tune.

```bash
# Cách 1: Chạy trực tiếp với file cấu hình demo.yaml (Khuyên dùng)
python -m tools.interactive_cli \
  --config configs/demo.yaml

# Cách 2: Truyền đầy đủ flags hoặc override tham số từ dòng lệnh
python -m tools.interactive_cli \
  --model-root ../models \
  --adapter ../runs/hf_overfit_10/checkpoints/checkpoint_000300 \
  --voice-prompt ../prepared/samples/conv_0001/voice_prompt.wav \
  --text-prompt "You enjoy having a good conversation. You are a helpful and friendly assistant." \
  --device cuda \
  --save-session-dir outputs/live_session
```
*(Hoặc sử dụng launcher script: `bash scripts/run_cli_demo.sh --config configs/demo.yaml`)*

**Chi tiết các cờ (flags) và đối số:**
- `--config` **[Optional]**: Đường dẫn tới file cấu hình YAML/JSON (ví dụ: `configs/demo.yaml`).
- `--model-root` **[Required nếu không có config]**: Thư mục chứa base model checkpoint cục bộ (`model.safetensors`, `tokenizer-*.safetensors`, `tokenizer_spm_32k_3.model`).
- `--voice-prompt` **[Required nếu không có config]**: Đường dẫn tới file âm thanh mẫu giọng nói (WAV hoặc `.pt`).
- `--adapter` **[Optional]**: Thư mục checkpoint LoRA hoặc đường dẫn file `lora.safetensors`. Nếu bỏ cờ này, hệ thống tự động chạy PersonaPlex base nguyên bản.
- `--text-prompt` **[Optional]**: Câu prompt quy định vai trò/tính cách (chuỗi text hoặc đường dẫn file `.txt`), tự động bọc thẻ `<system> ... <system>`.
- `--device` **[Optional]**: Thiết bị chạy model (`cuda` hoặc `cpu`, mặc định: `cuda`).
- `--qlora` **[Optional]**: Bật lượng tử hóa 4-bit NF4 để tiết kiệm VRAM.
- `--lora-rank`, `--lora-alpha` **[Optional]**: Chỉ định rank/alpha nếu file `adapter.json` không tồn tại.
- `--greedy` **[Optional]**: Bật greedy decoding thay cho sampling ngẫu nhiên.
- `--temp`, `--temp-text`, `--top-k`, `--top-k-text` **[Optional]**: Siêu tham số điều khiển tính ngẫu nhiên khi sinh audio/text.
- `--input-device`, `--output-device` **[Optional]**: ID hoặc tên thiết bị micro/loa cụ thể.
- `--save-session-dir` **[Optional]**: Thư mục lưu lại bản ghi âm toàn bộ cuộc trò chuyện (`user.wav`, `agent.wav`, `dialogue_stereo.wav` và `transcript.txt`).
- `--input-wav` **[Optional]**: Chế độ file: nạp file WAV người dùng thay vì dùng micro, model sinh ra file `--output-wav`.

---

### 4.4. Chạy Thử Nghiệm Suy Luận File (Inference Smoke Test)

Thực hiện nạp lại adapter LoRA trên base model gốc, tái tạo luồng Hybrid System Prompt (Voice prompt + Text prompt) và sinh phản hồi âm thanh/văn bản từ 1 mẫu trong dataset:

```bash
python -m tools.inference_smoke \
  --config configs/test_overfit.yaml \
  --adapter runs/hf_overfit_10/checkpoints/checkpoint_000300/lora.safetensors \
  --index 0 \
  --output-dir outputs/smoke
```

**Chi tiết các đối số (arguments):**
- `--config` **[Required]**: File cấu hình chứa đường dẫn base checkpoint (`model.root`) và source code PersonaPlex.
- `--adapter` **[Required]**: Đường dẫn tới file trọng số LoRA đã huấn luyện (`lora.safetensors` hoặc thư mục checkpoint).
- `--index` **[Optional]**: Index của mẫu hội thoại trong dataset dùng làm ngữ cảnh giọng nói, prompt và input user (mặc định: `0`).
- `--output-dir` **[Optional]**: Thư mục lưu kết quả sinh (mặc định: `outputs/smoke`).

---

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
