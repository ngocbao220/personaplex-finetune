# PersonaPlex fine-tuning

Các lệnh dưới đây chạy từ thư mục gốc repo, trong môi trường đã cài `requirements.txt`. Model phải có sẵn tại đường dẫn trong config; chương trình không tự tải model.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Kiểm tra dữ liệu

```bash
python -m tools.validate_dataset --config configs/config.yaml
python -m tools.inspect_sample --config configs/config.yaml --index 0
python -m tools.check_text_chunk_capacity --config configs/config.yaml
```

- `validate_dataset`: kiểm tra manifest, WAV stereo, transcript và prompt.
- `inspect_sample`: xem sequence PersonaPlex và loss mask của một mẫu.
- `check_text_chunk_capacity`: mã hóa Mimi thực tế và kiểm tra transcript có đặt đủ token lên chunk `duration_sec` không; mặc định quét 10 hội thoại đầu.
- Thêm `--all` để quét toàn bộ manifest hoặc `--sample-id ID` để chỉ quét một hội thoại. Tool chỉ nạp Mimi và tokenizer, không nạp model ngôn ngữ 7B.
- Khi train, log riêng số manifest entry bị bỏ vì timestamp ngoài audio và số chunk bị bỏ vì out-of-bounds/text-token overflow. Chi tiết nằm trong `data_filter_report.json` của run; overflow làm bỏ cả chunk, và cả hai role view nếu bật role swap.
- `--config`: chọn YAML; `--index`: chọn mẫu theo thứ tự.

## Huấn luyện

Kiểm tra nhanh trên một GPU:

```bash
python -m train configs/overfit.yaml --smoke   # 1 bước
python -m train configs/overfit.yaml           # overfit 10 chunk
```

Huấn luyện 2×B200 (ví dụ `configs/train_vi_synthetic.yaml`), luôn dùng **cùng config và override dữ liệu** ở mọi bước:

```bash
CFG=configs/train_vi_synthetic.yaml
# 1. Lọc dữ liệu + encode sẵn Mimi (chunk train và voice prompt), chạy lại an toàn
torchrun --nproc-per-node 2 -m train $CFG --precompute-codec-cache
# 2. (Tuỳ chọn) profile ngắn: bộ nhớ và thời gian data/forward/backward
torchrun --nproc-per-node 2 -m train $CFG profile_steps=true epochs=null max_steps=50
# 3. Train
torchrun --nproc-per-node 2 -m train $CFG
```

Tiếp tục:

```bash
# Chạy tiếp đúng run bị ngắt (cùng dữ liệu, lịch LR, vị trí batch)
torchrun --nproc-per-node 2 -m train $CFG --resume-from runs/<run>/checkpoints/checkpoint_000500
# Run mới từ trọng số cũ: được đổi dữ liệu/epochs; optimizer và LR bắt đầu lại
torchrun --nproc-per-node 2 -m train $CFG --init-from runs/<run>/checkpoints/best \
  data.prepared_dir=/path/new-data epochs=1
```

Override `key=value` thay YAML cho lần chạy đó. Tham số chính:

| Tham số | Ý nghĩa |
| --- | --- |
| `train=b200` / `train=b200-fast` | Preset 2×B200, batch toàn cục 32: an toàn (batch 1, checkpointing) / nhanh (batch 4, không checkpointing). |
| `data.prepared_dir` | Thư mục dữ liệu đã chuẩn bị. |
| `data.codec_cache_dir` | Cache Mimi: `auto` = `<prepared_dir>/.mimi-cache`, đường dẫn riêng, hoặc `null` để tắt. |
| `epochs` | Số epoch; tự tính `max_steps` từ số chunk sau lọc (log `epochs_to_steps`). `null` để dùng `max_steps`. |
| `max_steps` | Số lần cập nhật optimizer khi `epochs: null`. |
| `batch_size`, `train.gradient_accumulation_steps` | Batch toàn cục = batch mỗi GPU × accumulation × số GPU. |
| `optim.lr`, `lora.rank`, `lora.scaling` | Learning rate và LoRA. |
| `text_padding_weight`, `epad_weight` | Trọng số loss của PAD và EPAD (token bắt đầu từ). |
| `user_loss` | Tính loss cả user audio. |
| `data.vietnamese_text_mode` | `diacritics`, `no_diacritics` hoặc `telex`. |
| `data.swap_roles_after_pass` | Đổi vai LEFT/RIGHT giữa các epoch; cần prompt cho cả hai phía. |
| `--force-filter` | Bỏ cache, lọc lại sample và chunk. |

### Benchmark tokenizer tiếng Việt

`tools.benchmark_vi_tokenizer` đo tokenizer đúng như trainer đặt text agent lên lưới Mimi 12.5 frame/s. Tool chỉ chạy CPU, không load LM hay Mimi. Các chỉ số đo:
- round-trip: `decode(encode(x)) == x`;
- số token trên mỗi âm tiết (tok/syl) và token/giây nói;
- tỉ lệ byte-fallback;
- số chunk bị loại do quá tải text (`overflow`) hoặc lệch timestamp (`oob`).

```bash
# Tokenizer gốc (mặc định <model.root>/tokenizer_spm_32k_3.model) so với ViT5, cả 3 chế độ text
PYTHONPATH=src python -m tools.benchmark_vi_tokenizer --config configs/train_vi_synthetic.yaml \
  data.prepared_dir=../synthetic_samples \
  --tokenizer /path/personaplex-7b-v1/tokenizer_spm_32k_3.model \
  --translated /path/vit5-large/spiece.model \
  --modes diacritics,no_diacritics,telex \
  --output outputs/tokenizer_benchmark/report.json
```

Các tuỳ chọn:
- `--tokenizer`: một file `.model` SentencePiece đọc trực tiếp. Lặp lại flag để so sánh nhiều tokenizer.
- `--translated`: một file `.model` đọc qua lớp dịch ID của PersonaPlex. Lớp dịch này là cách `model.text_tokenizer=vit5` dùng tokenizer.
- `--speakers agent|all`: mặc định `agent`, là phần text được train.
- `--max-conversations N`: chỉ dùng N hội thoại.
- `--num-workers`: số worker CPU.
- Các override Hydra (`key=value`) được truyền thẳng vào config.

Tool in bảng ra màn hình và ghi JSON chi tiết vào `--output`, gồm cả ví dụ round-trip lỗi và các từ bị byte-fallback nhiều nhất. Nếu `tok/s` vượt 12.5 thì chunk dễ bị loại vì overflow.

Kết quả trên 8 mẫu `../synthetic_samples`:

| Tokenizer | Chế độ | tok/âm tiết | tok/s | byte-fallback |
|---|---|---|---|---|
| gốc 32k | diacritics | 4.07 | 16.5 | 46% |
| gốc 32k | no_diacritics | 1.81 | — | 0% |
| ViT5 | diacritics | 1.01 | 4.1 | 0% |

### Tokenizer text (tuỳ chọn ViT5)

Mặc định trainer dùng SentencePiece 32k gốc (`model.text_tokenizer: personaplex`), nên không có gì thay đổi. ViT5 phù hợp khi muốn train tiếng Việt có dấu (`data.vietnamese_text_mode=diacritics`).

**1. Chuẩn bị** (server offline, nên copy sẵn file lên): file `spiece.model` của `VietAI/vit5-large`, ví dụ `/home/voice/data/voice/vit5-large/spiece.model`.

**2. Config.** Thêm vào file YAML hoặc truyền dưới dạng override:
```yaml
model:
  text_tokenizer: vit5                       # personaplex | vit5
  text_tokenizer_path: /home/voice/data/voice/vit5-large/spiece.model
  text_head_init: decomposition              # decomposition | random (A/B cho text_linear)
data:
  vietnamese_text_mode: diacritics
train:
  text_vocab_learning_rate: 1.0e-4           # LR riêng cho 3 module text, weight decay 0
```
Có sẵn config mẫu `configs/overfit-8-vit5.yaml` để overfit 8 mẫu trên A100 40GB. Hướng dẫn chi tiết nằm ở `docs/run_vit5_overfit_a100.md`.

**3. Lọc chunk, precompute Mimi rồi train.** Hai lệnh dùng cùng config:
```bash
python train.py configs/overfit-8-vit5.yaml --precompute-codec-cache
python train.py configs/overfit-8-vit5.yaml

# Hoặc override trên config có sẵn
torchrun --nproc_per_node=2 train.py configs/train_vi_synthetic.yaml \
  model.text_tokenizer=vit5 model.text_tokenizer_path=/path/vit5-large/spiece.model \
  data.vietnamese_text_mode=diacritics model.text_head_init=decomposition
```
Log đầu run phải có dòng `[Text vocab] tokenizer=vit5 ... text_card=36001`. File `<run_dir>/config.json` ghi lại `text_tokenizer`, `text_tokenizer_path`, `text_tokenizer_sha256`, `text_head_init` và `text_vocab_learning_rate`.

Để A/B `text_linear`, chạy 2 run giống hệt nhau, chỉ khác `model.text_head_init=decomposition` và `model.text_head_init=random`. Sau đó so text loss và CER của transcript sinh ra.

**4. Inference.** Không cần flag gì thêm: tokenizer được chọn theo file `text_vocab.json` nằm cạnh checkpoint.
```bash
python -m tools.inference_smoke --config configs/infer.yaml \
  --adapter runs/<run>/checkpoints/<step> \
  model.text_tokenizer_path=/path/vit5-large/spiece.model   # chỉ cần khi file nằm chỗ khác lúc train
```
`inference_config.json` trong run cũng đã ghi sẵn tokenizer.

**Cơ chế và lưu ý**
- Lớp dịch ID giữ nguyên nghĩa các ID đặc biệt của PersonaPlex: PAD=3, EPAD=0, UNK=1, initial token=`text_card`. Các piece thường của ViT5 nằm ở 4..36000, nên `text_card=36001`.
- Ba module `text_emb`, `depformer_text_emb`, `text_linear` được resize. Mỗi hàng mới khởi tạo bằng trung bình embedding của các token cũ ghép thành piece đó, rồi được train full và lưu trong `lora.safetensors`. Transformer vẫn train bằng LoRA, Mimi frozen.
- Checkpoint luôn đi kèm `text_vocab.json` chứa sha256 của tokenizer. Resume, init-from và inference đều báo lỗi nếu tokenizer khác. Nếu copy checkpoint sang máy khác, phải copy cả `spiece.model`.
- Không dùng được với QLoRA. Cache lọc chunk tự tách theo tokenizer; Mimi cache dùng lại được.
- Model "base" trong eval khi chạy vit5 là model đã resize nhưng chưa train phần text, nên không phải PersonaPlex gốc. Muốn so với bản gốc, chạy inference riêng với `model.text_tokenizer=personaplex` và không truyền adapter.
- Không so text loss giữa hai tokenizer, vì số token khác nhau. Hãy so CER/WER của transcript sinh ra.

## Inference

```bash
python -m tools.inference_smoke --config configs/infer.yaml \
  --adapter runs/moshi-code-style/<checkpoint> \
  --window-seconds 30 --start 0 \
  --output-dir outputs/inference
```

`--sample-id` tìm trực tiếp trên các manifest train/validation/test đã cấu hình và không phụ thuộc `--split`; `--split` chỉ dùng khi chọn bằng `--index`. Với WAV/MP3 bên ngoài manifest, có thể truyền trực tiếp cả hai prompt qua CLI hoặc cấu hình sẵn trong `configs/infer.yaml` (`inference.voice_prompt`, `inference.text_prompt`, `inference.input_file`). Khi có đủ voice/text prompt riêng, chế độ standalone không nạp manifest. Khi chỉ có `input_file` và `sample_id`, file ngoài cung cấp user audio còn sample cung cấp voice/text conditioning:

```bash
python -m tools.inference_smoke --config configs/infer.yaml \
  --adapter /path/to/checkpoint \
  --input-file /path/to/user.wav \
  --voice-prompt /path/to/agent_voice.wav \
  --text-prompt "Bạn đang trò chuyện tự nhiên." \
  --window-seconds 30 --output-dir outputs/inference
```

Thêm `--force-filter` để xác thực manifest và lọc chunk lại thay vì dùng cache.

- `--adapter`: checkpoint LoRA cần nạp.
- `--window-seconds`, `--start`: độ dài và thời điểm bắt đầu đoạn audio.
- `--output-dir`: thư mục gốc; mỗi lần chạy tạo `infer_<YYYYMMDD_HHMMSS_microseconds>/` riêng, gồm WAV, transcript, `config.json`, `run.json` và `inference.log`. Config `inference.output_dir` cũng được hiểu là thư mục gốc.

Với `data.vietnamese_text_mode=telex`, model học và xuất text Telex; CER/WER được tính trên dạng Telex. Inference cũng ghi thêm `base_unicode.txt` và `finetuned_unicode.txt` để đọc transcript có dấu.

## Full fine-tuning

`configs/full-finetuning.yaml` chọn `train.method=full`, batch 1 mỗi GPU và gradient checkpointing. Trước khi chạy dài trên nhiều B200, chạy một bước với mẫu cố định:

```bash
python train.py --config configs/full-finetuning.yaml --smoke
```

Sau đó chạy DDP bằng `torchrun` với số GPU thực tế:

```bash
torchrun --nproc-per-node 4 train.py --config configs/full-finetuning.yaml
```

Đổi `4` thành số GPU được cấp. `train.method=full` cập nhật LM PersonaPlex, gồm embedding của cả hai kênh. Full checkpoint nằm trong `runs/full-finetuning/<run>/checkpoints/checkpoint_N/`, gồm `model.safetensors`, `training_state.pt` và `checkpoint.json`. Mỗi GPU DDP giữ một bản đầy đủ của model và AdamW; cần kiểm tra bộ nhớ GPU và dung lượng đĩa trước khi train dài. Dùng `--resume-from <checkpoint_dir>` để tiếp tục và `--checkpoint <checkpoint_dir>` với `tools.inference_smoke` để infer. `configs/config.yaml` và các lệnh LoRA cũ giữ nguyên mặc định.

### Trọng số loss

Mặc định huấn luyện là `user_loss: true`: user audio vừa là input context vừa được tính loss (cùng trọng số semantic/acoustic như agent). Để chỉ train stream agent, tắt bằng cờ `--no-user-loss` (CLI) hoặc override `user_loss=false` (Hydra). Layout forward luôn giữ nguyên 17 streams. Text token thật có weight `1.0`, PAD là `text_padding_weight` (`0.3`), EPAD là `epad_weight` (`1.0`); semantic codebook của agent là `1.0`, bảy acoustic codebook là `0.02` (khi bật `user_loss: true`, user codebook cũng nhận trọng số tương ứng). Log tách riêng `loss/text_real`, `loss/agent_semantic`, `loss/agent_acoustic`, `loss/user_semantic` và `loss/user_acoustic`.

## Log huấn luyện

```bash
tensorboard --logdir runs/moshi-code-style --port 6006
```

`runs/moshi-code-style` là thư mục log/checkpoint; thay bằng `train.output_dir` trong config nếu dùng đường dẫn khác.

Mỗi lần free-running validation lưu kết quả theo cấu trúc:

```text
<run_dir>/
├── checkpoints/
├── ranks/
├── free_running_metrics.jsonl
└── free-running/
    ├── free-running-report.json
    └── step_XXXXXX/
        └── <sample_id>/
            ├── dialogue_original.wav
            ├── dialogue_base.wav
            ├── dialogue_step.wav
            └── manifest.json
```

Ba WAV đều stereo 24 kHz: LEFT=agent, RIGHT=user. File `dialogue_original.wav`
là cửa sổ hội thoại gốc; hai file còn lại ghép agent do model sinh với user gốc.
Giữ nguyên độ dài và timeline user; phần agent được thêm silence/cắt để khớp
độ dài nguồn, kể cả các frame đầu chưa có output từ generator.

`step_000000` chạy base trước khi inject LoRA hoặc nạp checkpoint resume.
Các step sau tái sử dụng `dialogue_base.wav` từ step 0, không nạp thêm model 7B.
Tại step 0, `dialogue_step.wav` bằng kết quả base. Cửa sổ, seed, text mode và
generation settings phải khớp baseline để so sánh hợp lệ.

`manifest.json` chứa transcript thô của model (`transcript`/`hypothesis`),
`reference` chuẩn hóa theo `vietnamese_text_mode`, `raw_reference`, CER/WER,
cửa sổ, seed, generation settings và tên các WAV. Transcript là text tokens
do model sinh, không phải kết quả ASR của WAV. `free-running-report.json`
tổng hợp các lần eval hoàn tất, baseline CER và best step cải thiện baseline
(không chấp nhận transcript rỗng). `free_running_metrics.jsonl` ở gốc run
vẫn được duy trì cho công cụ tổng hợp cũ; các trường đường dẫn trỏ tới layout mới.
