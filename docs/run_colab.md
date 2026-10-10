# Chạy trên Colab (terminal) với dữ liệu và checkpoint qua Hugging Face Hub

Hướng dẫn train PersonaPlex trên một máy GPU tạm thời (Colab terminal hoặc VM thuê theo giờ),
truy cập bằng terminal. Hugging Face Hub là nơi lưu trữ data và checkpoint. Cache Mimi được tạo lại trên máy GPU ở mỗi phiên.

**Vì sao dùng HF Hub thay vì Google Drive**
- Drive mount qua FUSE: đọc tuần tự từng file, chậm với file lớn và rất chậm với hàng nghìn file nhỏ.
- HF Hub tải qua CDN, song song nhiều file cùng lúc. Với `hf_xet`, file được chia chunk, tải song song và khử trùng lặp, nên thường đạt hàng trăm MB/s trên máy cloud.
- Upload checkpoint cũng nhanh: file không đổi giữa các lần push không bị upload lại.

**Nguyên tắc**
1. Data đóng gói thành **shard tar khoảng 2–4 GB**, không upload hàng nghìn file WAV rời. Lý do: tải song song hiệu quả, giải nén nhanh, và tránh giới hạn số file của một repo HF.
2. Train luôn đọc từ đĩa local (`/content/work`), không bao giờ đọc thẳng từ Hub. Trainer vẫn chạy offline với đường dẫn local; việc tải xuống là một bước riêng, chạy trước khi train.
3. Checkpoint ghi ra đĩa local, một tiến trình nền push từng checkpoint mới lên repo model private.

## 0. Yêu cầu

| Hạng mục | Yêu cầu |
|---|---|
| GPU | **A100 40/80GB** (hoặc H100). L4 24GB chưa kiểm chứng. T4/V100 không dùng được vì không hỗ trợ bf16. |
| Đĩa local | Model khoảng 17 GB, cộng data × 2 (shard và bản giải nén; xoá shard sau khi giải nén), cộng cache Mimi và checkpoint. |
| HF | Tài khoản có token **write**. Đã chấp nhận license của `nvidia/personaplex-7b-v1` trên web. |

Repo HF dùng trong hướng dẫn (tạo **private**):

| Repo | Loại | Nội dung |
|---|---|---|
| `<user>/personaplex-data` | dataset | `synthetic_500h/data.tar.000…` |
| `<user>/personaplex-runs` | model | `<run_name>/checkpoints/checkpoint_xxxxxx/…` |

Model gốc tải thẳng từ `nvidia/personaplex-7b-v1`. ViT5 lấy từ `VietAI/vit5-large`, chỉ cần `spiece.model`.

## 1. Cài CLI Hugging Face (máy local và máy GPU)

```bash
pip install -U "huggingface_hub[hf_xet,hf_transfer]"
hf auth login                 # dán token write; hoặc: export HF_TOKEN=hf_xxx (không ghi token vào file trong repo)
hf auth whoami
export HF_USER=<tên-tài-khoản-hoặc-org>   # đúng tên hiện trong `hf auth whoami`
```

Với `huggingface_hub < 0.34`, lệnh tương ứng là `huggingface-cli login / upload / download`. Nếu không có `hf_xet`, đặt
`export HF_HUB_ENABLE_HF_TRANSFER=1` để tải nhanh.

## 2. Upload data (một lần, từ máy có data)

```bash
cd /path/to/parent_of_prepared_dir           # chứa thư mục synthetic_500h/
NAME=synthetic_500h
mkdir -p hf_upload/$NAME
# Đóng gói 1 luồng tar rồi cắt thành shard 4 GB; không nén (WAV hầu như không nén được thêm)
tar -cf - --exclude .filter-cache --exclude .mimi-cache $NAME \
  | split -b 4G -d -a 3 - hf_upload/$NAME/data.tar.
( cd hf_upload/$NAME && sha256sum data.tar.* > data.sha256 )

hf repo create $HF_USER/personaplex-data --repo-type dataset --private
hf upload-large-folder $HF_USER/personaplex-data hf_upload --repo-type dataset --num-workers 16
```

- `upload-large-folder` chia nhiều luồng, tự resume khi mạng rớt (chạy lại đúng lệnh đó).
- Khi data thay đổi, đóng gói lại vào thư mục mới (ví dụ `synthetic_500h_v2`). Không ghi đè shard cũ, để các run cũ vẫn tái lập được.
- Định dạng data xem `docs/prepared_data_format.md`. Code thì `git clone` hoặc upload tương tự (một file `.tgz`).

## 3. Thiết lập máy GPU mỗi phiên

```bash
export WORK=/content/work                    # đĩa local; trên VM khác dùng đĩa NVMe local
export HF_HOME=$WORK/.hf                     # cache HF trên đĩa local, không trên /root nhỏ
mkdir -p $WORK && cd $WORK
nvidia-smi --query-gpu=name,memory.total --format=csv && df -h $WORK
```

### 3.1 Code và môi trường

```bash
cd $WORK
git clone <repo-url> personaplex-finetuning     # hoặc: hf download ... code.tgz rồi tar -xzf
cd personaplex-finetuning
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Giữ torch có sẵn; cài phần còn lại. Nếu lỗi tương thích, cài torch==2.8.0 torchaudio==2.8.0 đúng CUDA.
grep -v -E '^(torch|torchaudio)==' requirements.txt > /tmp/req.txt && pip install -q -r /tmp/req.txt
pip install -q -U "huggingface_hub[hf_xet,hf_transfer]"
export PYTHONPATH=$WORK/personaplex-finetuning/src
python -c "import torch, sentencepiece, sphn; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
python -m pytest -q tests/test_contract.py tests/test_text_vocab.py
```

### 3.2 Model (song song với data, xem 3.3)

```bash
hf download nvidia/personaplex-7b-v1 --local-dir $WORK/models/personaplex-7b-v1 \
  --include "model.safetensors" "tokenizer-e351c8d8-checkpoint125.safetensors" "tokenizer_spm_32k_3.model" "*.json"
# Chỉ khi dùng tokenizer ViT5:
hf download VietAI/vit5-large spiece.model --local-dir $WORK/models/vit5-large
```

`--local-dir` ghi file thật vào thư mục, không phải symlink vào cache. Nhờ vậy đường dẫn `model.root` cố định giữa các phiên, điều mà resume yêu cầu.

### 3.3 Data: tải shard rồi giải nén theo luồng

```bash
NAME=synthetic_500h
hf download $HF_USER/personaplex-data --repo-type dataset --include "$NAME/data.*" --local-dir $WORK/hf_data
( cd $WORK/hf_data/$NAME && sha256sum -c --quiet data.sha256 )
mkdir -p $WORK/data
cat $WORK/hf_data/$NAME/data.tar.* | tar -xf - -C $WORK/data
rm -rf $WORK/hf_data/$NAME/data.tar.*          # giải phóng đĩa
ls $WORK/data/$NAME | head && du -sh $WORK/data/$NAME
```

Chạy 3.2 và 3.3 song song bằng hai cửa sổ `tmux`, hoặc thêm `&` rồi `wait`, để tận dụng băng thông.

## 4. Config qua override

Không sửa YAML, chỉ truyền đường dẫn local:

```bash
cd $WORK/personaplex-finetuning
export CFG=configs/overfit-8-vit5.yaml           # hoặc configs/train_vi_synthetic.yaml
export RUN=overfit-8-vit5-a100                   # tên run = thư mục trên repo HF
export OVR="model.root=$WORK/models/personaplex-7b-v1 \
 data.prepared_dir=$WORK/data/$NAME data.codec_cache_dir=auto \
 train.output_dir=$WORK/runs/$RUN \
 batch_size=1 train.gradient_checkpointing=true train.num_workers=4 train.filter_num_workers=8"
# ViT5: thêm
# OVR="$OVR model.text_tokenizer=vit5 model.text_tokenizer_path=$WORK/models/vit5-large/spiece.model data.vietnamese_text_mode=diacritics"
```

`train_vi_synthetic.yaml` viết cho 2×B200. Trên một GPU, override `batch_size=1`, bật gradient checkpointing, tăng
`train.gradient_accumulation_steps` để giữ global batch, và chạy `python train.py` (không dùng `torchrun`).

**Luôn chạy trong `tmux`** để lệnh không chết khi terminal hoặc trình duyệt mất kết nối:

```bash
tmux new -s train        # thoát tạm: Ctrl-b d ; vào lại: tmux attach -t train
```

## 5. Precompute cache Mimi (mỗi phiên)

Cache Mimi chỉ nằm trên đĩa local và không upload lên HF. Mỗi phiên mới chạy lại bước này trước khi train:

```bash
python train.py $CFG $OVR --precompute-codec-cache 2>&1 | tee $WORK/precompute.log
du -sh $WORK/data/$NAME/.mimi-cache
```

- Cache được ghi vào `<prepared_dir>/.mimi-cache` (`data.codec_cache_dir=auto`).
- Chạy lại an toàn: các mẫu đã encode trong cùng phiên sẽ được bỏ qua.
- Cache không phụ thuộc tokenizer, nên một lần precompute dùng được cho cả run tokenizer gốc lẫn ViT5 trong cùng phiên.
- Thời gian precompute tỉ lệ với số giờ audio, vì Mimi encode từng mẫu (batch 1). Đọc audio đã chạy song song theo `train.filter_num_workers`. Hãy đo ở phiên đầu bằng `precompute.log`.

## 6. Train và tự động push checkpoint lên HF

Tạo repo một lần: `hf repo create $HF_USER/personaplex-runs --private`.

Uploader chạy nền. Một checkpoint chỉ được push khi thư mục của nó đã ngừng thay đổi 2 phút, để tránh upload file đang ghi dở. Mỗi checkpoint chỉ push một lần:

```bash
cat > $WORK/push_ckpt.sh <<'EOF'
#!/usr/bin/env bash
# usage: push_ckpt.sh <local_run_dir> <hf_repo> <path_in_repo>
RUN_DIR=$1; REPO=$2; DEST=$3; DONE=$RUN_DIR/.pushed
touch "$DONE"
while true; do
  for ck in $(find "$RUN_DIR" -mindepth 2 -maxdepth 3 -type d -path "*/checkpoints/*" | sort); do
    grep -qxF "$ck" "$DONE" && continue
    [ -n "$(find "$ck" -newermt '-2 minutes' -print -quit)" ] && continue   # còn đang ghi
    rel=${ck#$RUN_DIR/}
    if hf upload "$REPO" "$ck" "$DEST/$rel" --commit-message "$rel"; then
      echo "$ck" >> "$DONE"; echo "$(date +%T) pushed $rel"
    fi
  done
  # Log/config nhỏ: đẩy kèm để theo dõi từ xa
  hf upload "$REPO" "$RUN_DIR" "$DEST" --include "*.json" "*.log" "events.*" --commit-message logs >/dev/null 2>&1
  sleep 120
done
EOF
chmod +x $WORK/push_ckpt.sh
mkdir -p $WORK/runs/$RUN
nohup $WORK/push_ckpt.sh $WORK/runs/$RUN $HF_USER/personaplex-runs $RUN > $WORK/push.log 2>&1 &
```

Train:

```bash
python train.py $CFG $OVR 2>&1 | tee -a $WORK/runs/$RUN/train.log
```

Theo dõi:
- `tail -f $WORK/push.log`: các checkpoint đã push;
- `nvidia-smi -l 5`;
- dòng `[Text vocab] ... text_card=36001` khi dùng ViT5.

Sau khi train xong, đợi uploader push nốt checkpoint cuối (khoảng 2–4 phút), xem log rồi mới tắt máy:

```bash
tail -3 $WORK/push.log; kill %1 2>/dev/null || pkill -f push_ckpt.sh
```

Lưu ý dung lượng:
- Adapter LoRA khoảng vài trăm MB. Thêm ViT5 thì nặng hơn khoảng 0.6 GB, vì `text_emb`, `depformer_text_emb`, `text_linear` được lưu đầy đủ.
- `training_state.pt` (trạng thái optimizer) lớn hơn adapter, nhưng cần nó để resume chính xác.
- Chọn `train.save_every_steps` hợp lý. Nếu chỉ cần trọng số để infer, có thể xoá checkpoint cũ trên Hub bằng `hf repo-files delete`.

## 7. Phiên mới: resume

Chạy lại mục 3 (code, model, data), mục 4 và mục 5 (precompute Mimi) (cùng `$CFG`, `$OVR`, `$RUN`). Sau đó tải đúng checkpoint cần resume:

```bash
CK=checkpoints/checkpoint_000500          # xem danh sách checkpoint trên trang repo (tab Files)
hf download $HF_USER/personaplex-runs --include "$RUN/*/$CK/*" "$RUN/$CK/*" --local-dir $WORK/runs_hf
SRC=$(find $WORK/runs_hf/$RUN -type d -path "*$CK" | head -1); echo $SRC
mkdir -p $WORK/runs/$RUN && cp -r $WORK/runs_hf/$RUN/. $WORK/runs/$RUN/
# Đánh dấu các checkpoint đã có trên Hub để uploader không push lại
find $WORK/runs/$RUN -mindepth 2 -maxdepth 3 -type d -path "*/checkpoints/*" > $WORK/runs/$RUN/.pushed
nohup $WORK/push_ckpt.sh $WORK/runs/$RUN $HF_USER/personaplex-runs $RUN > $WORK/push.log 2>&1 &
python train.py $CFG $OVR --resume-from ${SRC/$WORK\/runs_hf/$WORK\/runs} 2>&1 | tee -a $WORK/runs/$RUN/train.log
```

- `--resume-from` khôi phục optimizer, lịch LR và vị trí batch. Config và data phải giống lần train trước.
- `--init-from` chỉ nạp trọng số rồi bắt đầu run mới (được đổi data hoặc epochs).
- Checkpoint ghi lại `model.root` lúc train, nên phải giữ cùng `$WORK/models/personaplex-7b-v1`. Nếu khác, trainer báo `base model differs`.
- Với ViT5, `spiece.model` phải giống hệt file lúc train (trainer kiểm tra sha256). Tải lại từ `VietAI/vit5-large` thì ra cùng file.

## 8. Inference (máy GPU khác hoặc máy local)

```bash
hf download $HF_USER/personaplex-runs --include "$RUN/*checkpoint_000600/*" --local-dir $WORK/runs_hf
CKPT=$(find $WORK/runs_hf/$RUN -type d -name checkpoint_000600 | head -1)
python -m tools.inference_smoke --config $CFG $OVR --adapter $CKPT \
  --split prepared --index 0 --window-seconds 30 --output-dir $WORK/inference
hf upload $HF_USER/personaplex-runs $WORK/inference $RUN/inference   # lưu WAV/transcript để nghe ở máy khác
```

## 9. So sánh tốc độ và xử lý sự cố

Ước lượng thời gian chuẩn bị mỗi phiên với 500 h data (khoảng 170 GB WAV 24 kHz stereo 16-bit), cộng 16 GB model:

| Cách | Hành vi | Ghi chú |
|---|---|---|
| Drive FUSE `cp -r` thư mục | Hàng chục MB/s, chậm hẳn khi file nhỏ | Có thể mất vài giờ |
| Drive, 1 file tar | Khá hơn, vẫn đơn luồng | |
| **HF Hub shard + xet** | Nhiều luồng song song qua CDN | Thường nhanh nhất; đo bằng `time` ở lần chạy đầu |

Đây là ước lượng; tốc độ thật tuỳ vùng máy và giờ cao điểm. Hãy đo bằng `time hf download ...` ở lần đầu.

| Triệu chứng | Xử lý |
|---|---|
| `401/403` khi tải `nvidia/personaplex-7b-v1` | Chấp nhận license trên trang model; token phải có quyền đọc gated repo. |
| Tải chậm | Kiểm tra `python -c "import hf_xet"`; hoặc `export HF_HUB_ENABLE_HF_TRANSFER=1`; tăng `--max-workers` cho `hf download`. |
| `sha256sum -c` báo lỗi | Shard tải hỏng: xoá shard đó rồi chạy lại `hf download` (chỉ tải phần thiếu). |
| Đầy đĩa | Xoá shard sau khi giải nén; đặt `HF_HOME` trên đĩa lớn; xoá `$WORK/runs_hf` sau khi copy. |
| Uploader không push | `tail $WORK/push.log`; kiểm tra `hf auth whoami` và quyền write; thư mục checkpoint phải nằm dưới `checkpoints/`. |
| Commit bị rate-limit | Tăng `sleep` trong `push_ckpt.sh` hoặc tăng `train.save_every_steps`. |
| Train chậm, GPU util thấp | `data.prepared_dir` phải trỏ vào đĩa local `$WORK/data`, và đã chạy precompute Mimi (mục 5) trước khi train. |
| `CUDA out of memory` | `batch_size=1`, gradient checkpointing, giảm `duration_sec` hoặc `lora.rank`. |
| `base model differs` / `text vocabulary does not match` | Dùng đúng đường dẫn model và đúng `spiece.model` như lúc train. |
