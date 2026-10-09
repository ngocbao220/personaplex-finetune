import unittest
from types import SimpleNamespace

from personaplex_finetuning.train import tokenizer_text_padding_ids


class EpadPaddingTest(unittest.TestCase):
    tok = SimpleNamespace(padding_id=3, end_padding_id=0)

    def test_default_epad_is_real_target(self):
        self.assertEqual(tokenizer_text_padding_ids(self.tok), (3,))

    def test_epad_full_weight(self):
        import torch
        from personaplex_finetuning.objective import stream_weights_torch
        codes = torch.full((17, 4), 5, dtype=torch.long)
        codes[0] = torch.tensor([3, 0, 42, 3])
        w = stream_weights_torch(codes, torch.ones(17, 4, dtype=torch.bool),
                                 tokenizer_text_padding_ids(self.tok), text_padding_weight=0.3)
        self.assertEqual(w[0].tolist(), [0.30000001192092896, 1.0, 1.0, 0.30000001192092896])

    def test_explicit_epad_weight(self):
        import torch
        from personaplex_finetuning.objective import stream_weights_torch
        codes = torch.full((17, 4), 5, dtype=torch.long)
        codes[0] = torch.tensor([3, 0, 42, 0])
        mask = torch.ones(17, 4, dtype=torch.bool)
        mask[0, 3] = False  # masked EPAD stays at zero weight
        w = stream_weights_torch(codes, mask, tokenizer_text_padding_ids(self.tok), text_padding_weight=0.3,
                                 epad_id=0, epad_weight=0.5)
        self.assertEqual(w[0].tolist(), [0.30000001192092896, 0.5, 1.0, 0.0])

    def test_config_reads_and_validates_epad_weight(self):
        import tempfile
        from pathlib import Path
        from personaplex_finetuning.config import load_config
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.json"
            base = '{"model":{"root":"/m","source":"/s"},"data":{"prepared_dir":"/d"}'
            config.write_text(base + "}")
            self.assertEqual(load_config(config).epad_weight, 1.0)
            config.write_text(base + ',"epad_weight":0.5}')
            self.assertEqual(load_config(config).epad_weight, 0.5)
            for bad in (',"epad_weight":1.5}', ',"epad_as_padding":true}'):
                config.write_text(base + bad)
                with self.assertRaises(ValueError):
                    load_config(config)

    def test_config_has_no_epad_as_padding(self):
        from personaplex_finetuning.config import Config
        self.assertNotIn("epad_as_padding", Config.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
