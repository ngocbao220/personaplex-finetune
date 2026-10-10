import io
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from personaplex_finetuning import text_vocab
from personaplex_finetuning.lora import adapter_state_dict, inject_lora, load_adapter
from personaplex_finetuning.text_vocab import (
    EPAD_ID, FIRST_PIECE_ID, PAD_ID, UNK_ID, TranslatedTokenizer, resize_text_vocab,
)

CORPUS = ["Tôi muốn chuyển khoản ngay", "Được rồi cảm ơn bạn nhé", "ừ để tôi kiểm tra"] * 40


def train_spm(directory: Path) -> Path:
    import sentencepiece as spm
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(CORPUS), model_writer=model, vocab_size=32,
        pad_id=0, eos_id=1, unk_id=2, bos_id=-1, character_coverage=1.0,
    )
    path = directory / "vit5-tiny.model"
    path.write_bytes(model.getvalue())
    return path


class OldTokenizer:
    """Stand-in for the PersonaPlex 32k model: one ID per character (>= 10)."""
    padding_id, end_padding_id = 3, 0

    def encode(self, text):
        return [10 + (ord(c) % 40) for c in text]

    def id_to_piece(self, token):
        return "▁" if token == 10 + (ord(" ") % 40) else "x"


class TinyLM(torch.nn.Module):
    def __init__(self, text_card=60, dim=8):
        super().__init__()
        self.text_card, self.existing_text_padding_id = text_card, 3
        self.text_emb = torch.nn.Embedding(text_card + 1, dim)
        self.depformer_text_emb = torch.nn.Embedding(text_card + 1, dim)
        self.text_linear = torch.nn.Linear(dim, text_card, bias=False)
        self.transformer = torch.nn.Sequential(torch.nn.Linear(dim, dim))


class TextVocabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = train_spm(Path(cls.tmp.name))
        cls.tok = TranslatedTokenizer(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_special_ids_are_reserved_and_round_trip(self):
        for text in CORPUS[:3]:
            ids = self.tok.encode(text)
            self.assertTrue(all(FIRST_PIECE_ID <= i < self.tok.text_card for i in ids))
            self.assertEqual(self.tok.decode(ids), text)
            # PAD/EPAD interleaved by the frame grid never reach the decoded text.
            self.assertEqual(self.tok.decode([PAD_ID, EPAD_ID, *ids, PAD_ID]), text)
        self.assertEqual((self.tok.padding_id, self.tok.end_padding_id), (3, 0))
        self.assertEqual(self.tok.encode("Q")[-1], UNK_ID)  # unseen char -> reserved unk

    def test_pickles_and_records_sha(self):
        clone = pickle.loads(pickle.dumps(self.tok))
        self.assertEqual(clone.encode(CORPUS[0]), self.tok.encode(CORPUS[0]))
        text_vocab.verify_text_vocab(self.tok.metadata(), clone)
        with self.assertRaises(ValueError):
            text_vocab.verify_text_vocab(None, clone)

    def test_resize_decomposition_keeps_specials_and_trains_text_modules(self):
        torch.manual_seed(0)
        model = TinyLM()
        old = {name: module.weight.detach().clone()
               for name, module in (("emb", model.text_emb), ("dep", model.depformer_text_emb),
                                    ("head", model.text_linear))}
        resize_text_vocab(model, OldTokenizer(), self.tok)
        card = self.tok.text_card
        self.assertEqual(model.text_card, card)
        self.assertEqual(model.text_emb.weight.shape[0], card + 1)
        self.assertEqual(model.depformer_text_emb.weight.shape[0], card + 1)
        self.assertEqual(model.text_linear.out_features, card)
        for token in (EPAD_ID, PAD_ID):
            self.assertTrue(torch.equal(model.text_emb.weight[token], old["emb"][token]))
            self.assertTrue(torch.equal(model.text_linear.weight[token], old["head"][token]))
        self.assertTrue(torch.equal(model.text_emb.weight[card], old["emb"][60]))  # initial token row
        token = FIRST_PIECE_ID + 5
        piece = self.tok.id_to_piece(token).replace("▁", " ")
        ids = OldTokenizer().encode(piece.strip())
        if not piece.startswith(" ") and len(ids) > 1 and OldTokenizer().id_to_piece(ids[0]) == "▁":
            ids = ids[1:]
        self.assertTrue(torch.allclose(model.depformer_text_emb.weight[token], old["dep"][ids].mean(0)))

        inject_lora(model, rank=2, alpha=4, prefixes=("transformer",))
        names = set(text_vocab.text_vocab_parameter_names(model))
        self.assertEqual({n.split(".")[0] for n in names}, set(text_vocab.TEXT_VOCAB_MODULES))
        state = adapter_state_dict(model)
        self.assertTrue(names <= set(state))
        with tempfile.TemporaryDirectory() as tmp:
            from safetensors.torch import save_file
            path = Path(tmp) / "lora.safetensors"
            save_file(state, str(path))
            load_adapter(model, path)
            plain = TinyLM()
            inject_lora(plain, rank=2, alpha=4, prefixes=("transformer",))
            with self.assertRaises(RuntimeError):
                load_adapter(plain, path)  # personaplex-vocab model cannot take a vit5 adapter

    def test_random_head_init_keeps_special_rows(self):
        model = TinyLM()
        before = model.text_linear.weight.detach().clone()
        resize_text_vocab(model, OldTokenizer(), self.tok, head_init="random")
        for token in range(FIRST_PIECE_ID):
            self.assertTrue(torch.equal(model.text_linear.weight[token], before[token]))


if __name__ == "__main__":
    unittest.main()


class TextVocabFp32MasterTest(unittest.TestCase):
    def test_bf16_text_modules_train_on_fp32_masters_and_adapter_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = TranslatedTokenizer(train_spm(Path(tmp)))
            model = TinyLM().to(torch.bfloat16)
            resize_text_vocab(model, OldTokenizer(), tok)
            inject_lora(model, rank=2, alpha=4)
            self.assertEqual(text_vocab.keep_text_vocab_fp32(model), 3)
            self.assertEqual(text_vocab.keep_text_vocab_fp32(model), 0)  # idempotent
            names = text_vocab.text_vocab_parameter_names(model)
            params = dict(model.named_parameters())
            self.assertEqual(len(names), 3)
            self.assertTrue(all(params[name].dtype == torch.float32 for name in names))
            self.assertEqual(model.text_emb.weight.dtype, torch.bfloat16)  # forward dtype unchanged

            # An update far below BF16 resolution survives in the fp32 master.
            master = params[[n for n in names if n.startswith("text_emb.")][0]]
            master.requires_grad_(True)
            before = master.detach().clone()
            optimizer = torch.optim.SGD([master], lr=1e-5)
            model.text_emb(torch.tensor([5])).float().sum().backward()
            optimizer.step()
            self.assertGreater(int((master.detach() != before).sum()), 0)

            state = adapter_state_dict(model)
            self.assertIn("text_emb.weight", state)
            self.assertFalse(any("parametrizations" in key for key in state))
            path = Path(tmp) / "lora.safetensors"
            from safetensors.torch import save_file
            save_file(state, str(path))

            # Load into a plain BF16 model (inference path, no fp32 masters).
            plain = TinyLM().to(torch.bfloat16)
            resize_text_vocab(plain, OldTokenizer(), tok)
            inject_lora(plain, rank=2, alpha=4)
            load_adapter(plain, path)
            torch.testing.assert_close(plain.text_emb.weight.float(), master.detach().to(torch.bfloat16).float())

            # And back into an fp32-master model (resume path).
            resumed = TinyLM().to(torch.bfloat16)
            resize_text_vocab(resumed, OldTokenizer(), tok)
            inject_lora(resumed, rank=2, alpha=4)
            text_vocab.keep_text_vocab_fp32(resumed)
            load_adapter(resumed, path)
            resumed_master = dict(resumed.named_parameters())[[n for n in names if n.startswith("text_emb.")][0]]
            torch.testing.assert_close(resumed_master, master.detach())
