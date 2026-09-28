import unicodedata
import unittest
from pathlib import Path

from personaplex_finetuning.runtime import SentencePieceTokenizer


TOKENIZER_PATH = Path(__file__).resolve().parents[2] / "models" / "tokenizer_spm_32k_3.model"
CORPUS = (
    "Tôi muốn chuyển khoản.",
    "Được rồi, cảm ơn bạn.",
    "Trường hợp này xử lý thế nào?",
    "Ừm, để tôi kiểm tra nhé.",
    "ă â ê ô ơ ư đ",
    "á à ả ã ạ",
    "ấ ầ ẩ ẫ ậ",
    "ớ ờ ở ỡ ợ",
)


class VietnameseTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not TOKENIZER_PATH.is_file():
            raise FileNotFoundError(f"local SentencePiece model missing: {TOKENIZER_PATH}")
        cls.tokenizer = SentencePieceTokenizer(TOKENIZER_PATH)

    def test_vietnamese_nfc_round_trip(self) -> None:
        for text in CORPUS:
            with self.subTest(text=text):
                normalized = unicodedata.normalize("NFC", text)
                ids = self.tokenizer.encode(normalized)
                decoded = self.tokenizer.decode(ids)

                self.assertEqual(unicodedata.normalize("NFC", decoded), normalized)
                self.assertNotIn("\ufffd", decoded, "replacement character indicates decoding damage")
                self.assertFalse(
                    any(marker in decoded for marker in ("Ã", "Â", "áº", "á»")),
                    f"possible UTF-8 mojibake: {decoded!r}",
                )
                self.assertNotIn(
                    self.tokenizer._processor.unk_id(), ids,
                    f"unexpected <unk> while encoding {normalized!r}",
                )


if __name__ == "__main__":
    unittest.main()
