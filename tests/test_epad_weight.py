import unittest
from types import SimpleNamespace

from personaplex_finetuning.train import tokenizer_text_padding_ids


class EpadPaddingTest(unittest.TestCase):
    tok = SimpleNamespace(padding_id=3, end_padding_id=0)

    def test_default_epad_is_real_target(self):
        self.assertEqual(tokenizer_text_padding_ids(self.tok), (3,))

    def test_legacy_flag(self):
        self.assertEqual(tokenizer_text_padding_ids(self.tok, True), (3, 0))

    def test_epad_full_weight(self):
        import torch
        from personaplex_finetuning.objective import stream_weights_torch
        codes = torch.full((17, 4), 5, dtype=torch.long)
        codes[0] = torch.tensor([3, 0, 42, 3])
        w = stream_weights_torch(codes, torch.ones(17, 4, dtype=torch.bool),
                                 tokenizer_text_padding_ids(self.tok), text_padding_weight=0.3)
        self.assertEqual(w[0].tolist(), [0.30000001192092896, 1.0, 1.0, 0.30000001192092896])

    def test_config_flag(self):
        from personaplex_finetuning.config import Config
        self.assertFalse(Config.__dataclass_fields__["epad_as_padding"].default)


if __name__ == "__main__":
    unittest.main()
