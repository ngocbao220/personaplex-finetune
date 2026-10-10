"""Optional replacement text tokenizer (e.g. ViT5) behind a PersonaPlex token-ID layer.

The PersonaPlex text stream reserves frame-aligned special IDs that are *not*
SentencePiece specials of the new tokenizer:

    0 = EPAD (end of padding / word onset)   3 = PAD (agent silent)
    1, 2 = reserved (1 also carries the new tokenizer's <unk>)
    text_card = initial text token (last embedding row, as in Moshi)

Ordinary pieces of the new SentencePiece model map to 4..text_card-1. Its own
control pieces (<pad>, </s>, ...) are never emitted, so they cannot collide.
Train, chunk filtering, loss and generation all use this one class; checkpoints
record the tokenizer sha256 and inference refuses a different file.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

EPAD_ID = 0
UNK_ID = 1
PAD_ID = 3
FIRST_PIECE_ID = 4
TEXT_VOCAB_FILE = "text_vocab.json"
TEXT_VOCAB_MODULES = ("text_emb", "depformer_text_emb", "text_linear")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TranslatedTokenizer:
    padding_id = PAD_ID
    end_padding_id = EPAD_ID

    def __init__(self, path: Path) -> None:
        self._path = Path(path).expanduser().resolve()
        if not self._path.is_file():
            raise FileNotFoundError(f"text tokenizer model missing: {self._path}")
        sentencepiece = importlib.import_module("sentencepiece")
        self._processor = sentencepiece.SentencePieceProcessor(str(self._path))
        size = self._processor.vocab_size()
        self._special = {
            index for index in range(size)
            if self._processor.is_control(index) or self._processor.is_unknown(index)
            or self._processor.is_unused(index)
        }
        self._unk = self._processor.unk_id()
        pieces = [index for index in range(size) if index not in self._special]
        self._to_text = {source: FIRST_PIECE_ID + offset for offset, source in enumerate(pieces)}
        self._from_text = {target: source for source, target in self._to_text.items()}
        self.text_card = FIRST_PIECE_ID + len(pieces)
        self.sha256 = file_sha256(self._path)
        self.unknown_pieces = 0

    @property
    def path(self) -> Path:
        return self._path

    def __getstate__(self) -> dict[str, str]:
        return {"path": str(self._path)}

    def __setstate__(self, state: dict[str, str]) -> None:
        self.__init__(Path(state["path"]))

    def encode(self, text: str) -> list[int]:
        ids = []
        for source in self._processor.encode(text):
            if source in self._to_text:
                ids.append(self._to_text[source])
            else:
                self.unknown_pieces += 1
                ids.append(UNK_ID)
        return ids

    def source_ids(self, ids) -> list[int]:
        """PersonaPlex text IDs -> SentencePiece IDs, dropping PAD/EPAD/reserved/initial."""
        return [self._from_text[int(token)] for token in ids if int(token) in self._from_text]

    def decode(self, ids) -> str:
        return str(self._processor.decode(self.source_ids(ids)))

    def id_to_piece(self, token: int) -> str:
        token = int(token)
        if token in self._from_text:
            return self._processor.id_to_piece(self._from_text[token])
        return {EPAD_ID: "<epad>", PAD_ID: "<pad>", UNK_ID: "<unk>"}.get(token, f"<reserved:{token}>")

    def is_byte(self, token: int) -> bool:
        source = self._from_text.get(int(token))
        return source is not None and bool(self._processor.is_byte(source))

    def is_unk(self, token: int) -> bool:
        return int(token) == UNK_ID

    def metadata(self) -> dict:
        return {
            "kind": "translated_sentencepiece", "tokenizer_path": str(self._path),
            "tokenizer_sha256": self.sha256, "text_card": self.text_card,
            "epad_id": EPAD_ID, "pad_id": PAD_ID, "unk_id": UNK_ID, "first_piece_id": FIRST_PIECE_ID,
            "initial_token_id": self.text_card,
        }


def build_text_tokenizer(base_tokenizer_path: Path, text_tokenizer: str = "personaplex",
                         text_tokenizer_path: Path | None = None):
    """The tokenizer every train/filter/inference step must share."""
    if text_tokenizer == "personaplex":
        from .runtime import SentencePieceTokenizer
        return SentencePieceTokenizer(base_tokenizer_path)
    if text_tokenizer_path is None:
        raise ValueError(f"model.text_tokenizer={text_tokenizer} requires model.text_tokenizer_path")
    return TranslatedTokenizer(Path(text_tokenizer_path))


def tokenizer_fingerprint_path(base_tokenizer_path: Path, tokenizer) -> Path:
    """Filter caches must split by the tokenizer actually used."""
    return getattr(tokenizer, "path", None) or Path(base_tokenizer_path)


def _decomposed_rows(old_tokenizer, new_tokenizer) -> list[list[int]]:
    """Old PersonaPlex token IDs whose mean initializes each new text row."""
    rows: list[list[int]] = []
    for token in range(new_tokenizer.text_card):
        if token < FIRST_PIECE_ID:
            rows.append([token])  # EPAD, reserved/unk, PAD keep their old rows.
            continue
        piece = new_tokenizer.id_to_piece(token)
        text = piece.replace("▁", " ")
        # SentencePiece always adds the word-start marker; a word-internal piece
        # ("ng") must drop the bare "▁" the old model emits for it.
        old = [int(value) for value in old_tokenizer.encode(text.strip())] if text.strip() else []
        if old and not text.startswith(" ") and hasattr(old_tokenizer, "id_to_piece"):
            if old_tokenizer.id_to_piece(old[0]) == "▁" and len(old) > 1:
                old = old[1:]
        rows.append(old or [PAD_ID])
    return rows


def resize_text_vocab(model, old_tokenizer, new_tokenizer, head_init: str = "decomposition", seed: int = 0) -> dict:
    """Rebuild text_emb, depformer_text_emb and text_linear for the new vocabulary."""
    import torch

    if head_init not in {"decomposition", "random"}:
        raise ValueError("model.text_head_init must be decomposition or random")
    old_card = int(model.text_card)
    if getattr(model, "existing_text_padding_id", None) != PAD_ID:
        raise RuntimeError("PersonaPlex LM must use existing_text_padding_id=3 to keep PAD semantics")
    if model.text_linear.out_features != old_card:
        raise RuntimeError("unexpected text_linear shape; cannot resize text vocabulary")
    new_card = int(new_tokenizer.text_card)
    rows = _decomposed_rows(old_tokenizer, new_tokenizer)
    generator = torch.Generator().manual_seed(seed)

    def mean_rows(weight, initial_row: bool):
        source = weight.detach().float().cpu()
        out = torch.stack([source[ids].mean(dim=0) for ids in rows])
        if initial_row:
            out = torch.cat([out, source[old_card:old_card + 1]])  # initial token row
        return out.to(dtype=weight.dtype)

    for module in (model.text_emb, model.depformer_text_emb):
        # Resize in place: ScaledEmbedding keeps its norm and zero_idx behaviour.
        weight = module.weight
        if weight.shape[0] != old_card + 1:
            raise RuntimeError(f"unexpected text embedding rows {weight.shape[0]} (text_card={old_card})")
        module.weight = torch.nn.Parameter(mean_rows(weight, True).to(weight.device), requires_grad=False)
        module.num_embeddings = new_card + 1
    old_head = model.text_linear
    head = torch.nn.Linear(old_head.in_features, new_card, bias=old_head.bias is not None,
                           device=old_head.weight.device, dtype=old_head.weight.dtype)
    with torch.no_grad():
        if head_init == "decomposition":
            head.weight.copy_(mean_rows(old_head.weight, False))
            if old_head.bias is not None:
                bias = old_head.bias.detach().float().cpu()
                head.bias.copy_(torch.stack([bias[ids].mean() for ids in rows]).to(head.bias.dtype))
        else:
            std = float(old_head.weight.detach().float().std())
            head.weight.copy_((torch.randn(head.weight.shape, generator=generator) * std).to(head.weight.dtype))
            if head.bias is not None:
                head.bias.zero_()
            # Specials keep their pretrained output rows so PAD/EPAD start calibrated.
            for token in range(FIRST_PIECE_ID):
                head.weight[token].copy_(old_head.weight[token])
    head.requires_grad_(False)
    model.text_linear = head
    model.text_card = new_card
    model._personaplex_text_vocab = new_tokenizer.metadata()
    if not math.isfinite(float(head.weight.float().abs().sum())):
        raise RuntimeError("text vocabulary resize produced non-finite weights")
    return {"old_text_card": old_card, "new_text_card": new_card, "head_init": head_init}


_PARAMETRIZED = (".parametrizations.weight.original", ".parametrizations.bias.original")


def canonical_parameter_name(name: str) -> str:
    """Checkpoint key of a parameter, independent of the fp32 master-weight parametrization."""
    for marker, plain in zip(_PARAMETRIZED, (".weight", ".bias")):
        name = name.replace(marker, plain)
    return name


def keep_text_vocab_fp32(model) -> int:
    """Train the resized text modules on fp32 master weights.

    BF16 keeps ~3 significant digits, so AdamW updates of lr~1e-4 on BF16
    embeddings mostly round to zero. The forward pass still sees the original
    compute dtype through a cast parametrization; only storage/updates are fp32.
    Returns the number of tensors promoted.
    """
    import torch
    from torch.nn.utils import parametrize

    class CastTo(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.dtype = dtype

        def forward(self, value):
            return value.to(self.dtype)

    if getattr(model, "_personaplex_text_vocab", None) is None:
        return 0
    promoted = 0
    for module_name in TEXT_VOCAB_MODULES:
        module = getattr(model, module_name)
        for attribute in ("weight", "bias"):
            tensor = getattr(module, attribute, None)
            if tensor is None or parametrize.is_parametrized(module, attribute) or tensor.dtype == torch.float32:
                continue
            compute_dtype = tensor.dtype
            setattr(module, attribute, torch.nn.Parameter(tensor.detach().float(), requires_grad=tensor.requires_grad))
            parametrize.register_parametrization(module, attribute, CastTo(compute_dtype), unsafe=True)
            promoted += 1
    return promoted


def text_vocab_parameter_names(model) -> list[str]:
    if getattr(model, "_personaplex_text_vocab", None) is None:
        return []
    return [name for name, _ in model.named_parameters() if name.split(".", 1)[0] in TEXT_VOCAB_MODULES]


def write_text_vocab_metadata(directory: Path, model) -> None:
    metadata = getattr(model, "_personaplex_text_vocab", None)
    if metadata is not None:
        (Path(directory) / TEXT_VOCAB_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def read_text_vocab_metadata(checkpoint: Path) -> dict | None:
    checkpoint = Path(checkpoint)
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    path = directory / TEXT_VOCAB_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def verify_text_vocab(metadata: dict | None, tokenizer) -> None:
    """Fail when a checkpoint's text vocabulary differs from the configured tokenizer."""
    expected = None if metadata is None else metadata.get("tokenizer_sha256")
    actual = getattr(tokenizer, "sha256", None)
    if expected != actual:
        raise ValueError(
            "checkpoint text vocabulary does not match the configured tokenizer: "
            f"checkpoint={expected or 'personaplex'} configured={actual or 'personaplex'}; "
            "set model.text_tokenizer/model.text_tokenizer_path to the training values"
        )
