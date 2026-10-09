# Overfit 8 mẫu với tokenizer ViT5 trên 1× A100 40GB

Config: `configs/overfit-8-vit5.yaml`. Mọi lệnh chạy từ thư mục `personaplex-finetuning/`.

## 0. Chuẩn bị (một lần)

```bash
# Code + 8 mẫu (từ máy local)
rsync -av --exclude __pycache__ --exclude .filter-cache --exclude .mimi-cache \
  personaplex-finetuning synthetic_samples <user>@<server>:<workdir>/

# Tokenizer ViT5 (server offline -> tải ở máy có mạng rồi copy lên)
curl -L -o spiece.model https://huggingface.co/VietAI/vit5-large/resolve/main/spiece.model
scp spiece.model <user>@<server>:/home/voice/data/voice/vit5-large/spiece.model
```

Sửa 3 dòng `# EDIT` trong config nếu đường dẫn khác: `model.root`, `model.text_tokenizer_path`,
`data.prepared_dir` (mặc định `../synthetic_samples`, tức thư mục cạnh `personaplex-finetuning/`).

Kiểm tra nhanh:
```bash
nvidia-smi                                   # A100 40GB, còn trống
python -c "import sentencepiece, torch; print(torch.__version__, torch.cuda.is_available())"
PYTHONPATH=src python -m pytest -q tests/test_text_vocab.py
PYTHONPATH=src python -m tools.benchmark_vi_tokenizer --config configs/overfit-8-vit5.yaml \
  --tokenizer /home/voice/data/voice/personaplex-7b-v1/tokenizer_spm_32k_3.model \
  --translated /home/voice/data/voice/vit5-large/spiece.model --modes diacritics
```
Kỳ vọng: ViT5 ~1 tok/âm tiết, `overflow 0/8`. (2 chunk `oob` là lỗi timestamp của dữ liệu, bị loại sẵn.)

## 1. Encode Mimi cache (nhanh, 8 mẫu)

```bash
python train.py configs/overfit-8-vit5.yaml --precompute-codec-cache
```

## 2. Train overfit

```bash
CUDA_VISIBLE_DEVICES=0 python train.py configs/overfit-8-vit5.yaml 2>&1 | tee ../runs/overfit-8-vit5.log
```

Log cần thấy lúc khởi động:
- `[Text vocab] tokenizer=vit5 ... text_card=36001 head_init=decomposition`
- `[Text vocab] trainable modules ['depformer_text_emb', 'text_emb', 'text_linear'] lr=1.00e-04 params=...`
- `[Chunk filter] ... kept=6` (8 chunk, 2 bị loại out_of_bounds)

Trong lúc train theo dõi: `text loss` phải giảm rõ (về gần 0 sau vài trăm step), `semantic audio loss` giảm,
`grad_norm` hữu hạn. Free-running eval mỗi 100 step ghi vào `../runs/overfit-8-vit5/free-running/`.

A/B khởi tạo `text_linear` (chạy sau, cùng điều kiện):
```bash
python train.py configs/overfit-8-vit5.yaml model.text_head_init=random \
  train.output_dir=../runs/overfit-8-vit5-random
```

## 3. Inference trên checkpoint

```bash
python -m tools.inference_smoke --config configs/overfit-8-vit5.yaml \
  --checkpoint ../runs/overfit-8-vit5/<run>/checkpoints/checkpoint_000600 \
  --split prepared --index 0
```
Checkpoint có `text_vocab.json`; nếu `model.text_tokenizer_path` trỏ sai file tokenizer, lệnh dừng với lỗi
`text vocabulary does not match`. Kỳ vọng: text sinh ra là tiếng Việt **có dấu**, gần như trùng transcript.

## Khi gặp lỗi

| Triệu chứng | Xử lý |
|---|---|
| CUDA OOM | `duration_sec=60`, rồi `lora.rank=32 lora.alpha=64`; giữ `train.gradient_checkpointing=true` |
| text loss đứng yên / NaN | thử `train.text_vocab_learning_rate=3e-5`, hoặc `model.text_head_init=random` |
| `requires model.text_tokenizer_path` | đường dẫn spiece.model sai/không tồn tại |
| `only N train chunks remain` | kiểm tra `data.prepared_dir`, xem log `[Skipped ...]` |
| `QLoRA` error | ViT5 chưa hỗ trợ QLoRA; dùng `lora: default` |

Gửi lại: file log, `../runs/overfit-8-vit5/<run>/` (metrics + free-running text), output inference.
