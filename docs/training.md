# Huấn Luyện (Training)

Tài liệu này hướng dẫn cách cấu hình và thực thi quá trình fine-tuning mô hình PersonaPlex. Mọi lệnh huấn luyện được chạy từ thư mục gốc của repository.

## 1. Yêu cầu Trước khi Huấn luyện

### Mimi train/inference encoding contract

Training và inference encode từng kênh riêng bằng cùng helper Mimi: input `[1,1,T]`,
waveform float32, autocast tắt. LM vẫn có thể train theo batch; chỉ Mimi encoding
chạy riêng từng kênh để tránh codes thay đổi theo batch size hoặc cache hits.
Crop, padding của window và mapping agent/user được giữ nguyên.

Conversation codec cache dùng format version 2, kèm encoding contract và device
trong identity. Cache version 1 từ cách encode gộp kênh không được dùng lại; các file
cũ được giữ nguyên và cache mới được tạo khi cần. Chi phí encode lần đầu tăng vì
Mimi chạy riêng cho agent/user; cache mới vẫn được tái sử dụng.

Sửa đổi này thay đổi codes cho training mới, không sửa lại lịch sử supervision của
checkpoint cũ. Kiểm chứng trên GPU bằng `audit_personaplex.py data`, cùng mẫu/crop/
config đã dùng, không dùng `--match-user-conditioning`: `user_encoding.json` phải
có `training_vs_inference.equal=true`, mismatch và unmatched đều bằng 0.
Kiểm tra equality này không chứng minh lỗi free-running collapse đã được giải quyết.
Run mới ghi `mimi_encoding_contract` trong training contract. Exact resume từ run cũ
thiếu contract này bị từ chối vì không còn cùng supervision; adapter cũ vẫn có thể
load để inference/audit. Không chỉnh config.json cũ để giả lập contract mới.

Mô hình gốc `nvidia/personaplex-7b-v1` phải tồn tại trong máy. Dự án **không** tự động tải checkpoint từ HuggingFace trong quá trình huấn luyện nhằm tránh rủi ro kết nối internet trong môi trường server khép kín.

Thiết lập môi trường:
```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## 2. Các Bước Kiểm Tra Overfit (Smoke Test)

`sample_number=100` giữ **100 chunk hợp lệ**, theo thứ tự hội thoại rồi thứ tự
window trong mỗi hội thoại. Chunk bị loại không tiêu tốn quota: bộ lọc tiếp tục
đến khi đủ 100 hoặc hết nguồn. Khi bật đổi vai, chunk phải hợp lệ ở cả hai vai;
một chunk vẫn chỉ tính một lần. `sample_number=null` giữ tất cả chunk hợp lệ.
`sample_index` vẫn chọn đúng hội thoại và chỉ xét window đầu tiên, không tự chọn
hội thoại khác nếu window đó bị loại.

Log `[Chunk filter]` và `config.json` ghi số candidates, scanned, kept, requested
và shortfall. Nếu hết nguồn trước khi đủ quota, training dùng số chunk còn lại
và báo shortfall; vẫn lỗi nếu không đủ một batch. Cache quota mới không dùng lại
kết quả chọn theo số hội thoại. Run mới ghi `sample_number_contract=valid-chunks-v1`;
exact resume từ run cũ thiếu contract này bị từ chối.

Audit trực tiếp mặc định dùng quota chunk mới. `audit_vi.py` đọc contract của run:
run cũ thiếu trường này được audit theo quy tắc số hội thoại cũ
(`--sample-number-contract conversations-v1`), để không thay đổi danh sách mẫu
đã train khi kiểm chứng checkpoint lịch sử.

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

### DDP timeout trước update đầu tiên

Nếu một rank chờ ALLREDUCE có `NumelIn=2`, còn rank khác chưa enqueue operation
tương ứng, đó là collective thống kê audio ngay sau chuẩn bị batch. Kiểm tra log
`training_batch_phase` trên từng rank: `audio_loader`, `mimi_dialogue_encode`,
`prompt_and_targets`, `collate_to_device`, rồi `audio_stats_allreduce`.
Các phase chi tiết chỉ log ở batch đầu; không thêm collective vào luồng training.
Nếu chuẩn bị batch đầu quá 120 giây, Python stack của rank bị kẹt được in ra stderr.
DataLoader có worker cũng timeout sau 120 giây chờ batch để báo lỗi đọc dữ liệu
trước NCCL timeout 600 giây; khi `train.num_workers=0`, timeout worker tắt.

Để tách lỗi worker khỏi encode, chạy lại cùng lệnh/config với override
`train.num_workers=0`. Nếu vẫn kẹt, phase cuối và stack sẽ cho biết chỗ cần kiểm
tra tiếp. Override này là phép kiểm chứng; timeout NCCL riêng lẻ chưa chứng minh
lỗi gradient checkpointing hay lỗi kết nối GPU.

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
