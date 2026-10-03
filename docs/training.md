# Huấn Luyện (Training)

Tài liệu này hướng dẫn cách cấu hình và thực thi quá trình fine-tuning mô hình PersonaPlex. Mọi lệnh huấn luyện được chạy từ thư mục gốc của repository.

## 1. Yêu cầu Trước khi Huấn luyện

Mô hình gốc `nvidia/personaplex-7b-v1` phải tồn tại trong máy. Dự án **không** tự động tải checkpoint từ HuggingFace trong quá trình huấn luyện nhằm tránh rủi ro kết nối internet trong môi trường server khép kín.

Thiết lập môi trường:
```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## 2. Các Bước Kiểm Tra Overfit (Smoke Test)

Trước khi tiến hành huấn luyện toàn bộ dữ liệu (scale-up), dự án bắt buộc trải qua bước kiểm thử overfit trên một lượng nhỏ dữ liệu để xác nhận Data Pipeline và Forward/Backward pass đang hoạt động bình thường.

**Overfit 1 sample (Smoke Test):**
```bash
python -m train configs/overfit.yaml \
  model.root=/absolute/path/to/personaplex-7b-v1 \
  data.prepared_dir=/absolute/path/to/otospeech-prepared \
  sample_number=1 sample_index=0 duration_sec=3.04 \
  data.swap_roles_after_pass=false data.shuffle=false \
  data.prompt_aug_prob=0.0 \
  --smoke
```

**Overfit 10 samples (1 GPU):**
```bash
python -m train configs/moshi_overfit_10.yaml model=server
```

*(Lưu ý: Bạn cần chỉnh sửa đường dẫn đến model và tập dữ liệu trong file `configs/model/server.yaml` hoặc override trực tiếp qua dòng lệnh)*

## 3. Huấn luyện LoRA

Sử dụng LoRA để tiết kiệm bộ nhớ GPU khi huấn luyện. Chạy huấn luyện trên nhiều GPU (ví dụ: 8 GPUs):

```bash
torchrun --nproc-per-node 8 -m train configs/moshi_code_style.yaml \
  model=server batch_size=1 train.gradient_accumulation_steps=2
```

### Tiếp tục từ Checkpoint (Resume)

```bash
torchrun --nproc-per-node 8 -m train configs/moshi_code_style.yaml \
  model=server batch_size=1 train.gradient_accumulation_steps=2 \
  --resume-from runs/moshi-code-style/<run>/checkpoints/checkpoint_000500
```
Thư mục resume phải chứa `lora.safetensors` và `training_state.pt`.

## 4. Huấn luyện Đầy đủ (Full Fine-tuning)

Thay vì dùng LoRA, bạn có thể cập nhật toàn bộ tham số của LM (bao gồm cả embedding kênh trái/phải). Phương pháp này yêu cầu nhiều VRAM hơn.

Kiểm tra 1 bước (Smoke Test):
```bash
python train.py --config configs/full-finetuning.yaml --smoke
```

Huấn luyện DDP trên 4 GPU:
```bash
torchrun --nproc-per-node 4 train.py --config configs/full-finetuning.yaml
```

## 5. Trọng Số Loss (Loss Weights)

Loss (Cross Entropy) được tính sau khi so khớp chuẩn đầu ra của logits với chuỗi gốc:

*   **Văn bản (Text)**:
    *   Token văn bản thật: `1.0`
    *   Padding tokens (PAD / END_PAD): `0.3` (Hoặc `0.5` tuỳ cấu hình cũ trong test, tham chiếu theo contract hiện tại).
*   **Âm thanh (Audio)**:
    *   Codebook Semantic (Codebook 0): `1.0` (Áp dụng cho cả User và Agent nếu bật `user_loss: true`).
    *   Codebook Acoustic (Codebook 1-7): `0.02`

Các trường log trên Tensorboard (`tensorboard --logdir runs/moshi-code-style`):
- `loss/text_real`
- `loss/agent_semantic`
- `loss/agent_acoustic`
- `loss/user_semantic`
- `loss/user_acoustic`

## 6. Các Tham số Cấu hình Chính

| Tham số Override | Mô tả |
| :--- | :--- |
| `batch_size` | Số chunk hội thoại trên mỗi GPU cho một bước. |
| `train.gradient_accumulation_steps` | Số bước gom gradient trước khi cập nhật optimizer. |
| `lora.rank`, `lora.scaling` | Các thông số cơ bản cho LoRA. |
| `gradient_checkpointing` | Tiết kiệm bộ nhớ kích hoạt (activations), bật bằng `true`. |
| `data.swap_roles_after_pass` | `true`/`false`. Luân phiên vai trò Kênh Trái/Phải giữa mỗi Epoch. |
| `data.vietnamese_text_mode` | Cấu hình mã hoá văn bản: `diacritics`, `no_diacritics`, hoặc `telex`. |
| `--force-filter` | Bỏ qua các file cache và quét lại toàn bộ dữ liệu lỗi, tràn token. |
