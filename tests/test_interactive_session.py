import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from personaplex_finetuning.session import (
    resolve_adapter_info,
    wrap_with_system_tags,
)

try:
    import numpy as np
    import torch
    HAS_TORCH_NUMPY = True
except ImportError:
    HAS_TORCH_NUMPY = False


class TestInteractiveSession(unittest.TestCase):
    def test_wrap_with_system_tags(self):
        self.assertEqual(wrap_with_system_tags("Hello"), "<system> Hello <system>")
        self.assertEqual(wrap_with_system_tags("  <system> Hello <system>  "), "<system> Hello <system>")
        self.assertEqual(wrap_with_system_tags("You are a helpful bot."), "<system> You are a helpful bot. <system>")

    def test_resolve_adapter_info_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            adapter_file = tmp_path / "lora.safetensors"
            adapter_file.write_bytes(b"dummy")
            meta_file = tmp_path / "adapter.json"
            meta_file.write_text(json.dumps({"rank": 8, "alpha": 16}), encoding="utf-8")

            resolved_file, rank, alpha = resolve_adapter_info(tmp_path)
            self.assertEqual(resolved_file, adapter_file.resolve())
            self.assertEqual(rank, 8)
            self.assertEqual(alpha, 16)

    def test_resolve_adapter_info_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            adapter_file = tmp_path / "lora.safetensors"
            adapter_file.write_bytes(b"dummy")

            resolved_file, rank, alpha = resolve_adapter_info(adapter_file)
            self.assertEqual(resolved_file, adapter_file.resolve())
            # Default fallback when adapter.json doesn't exist
            self.assertEqual(rank, 16)
            self.assertEqual(alpha, 32)

    def test_resolve_adapter_info_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                resolve_adapter_info(Path(tmpdir) / "non_existent.safetensors")

    @unittest.skipUnless(HAS_TORCH_NUMPY, "Requires torch and numpy")
    @patch("personaplex_finetuning.session.load_runtime")
    def test_session_step_audio_chunk_mocked(self, mock_load_runtime):
        from personaplex_finetuning.session import InteractiveSession

        # Mock runtime
        mock_runtime = MagicMock()
        mock_runtime.model.training = False
        mock_runtime.codec.sample_rate = 24000
        mock_runtime.codec.frame_rate = 12.5
        mock_runtime.codec.mimi.encode.return_value = torch.zeros(1, 8, 1, dtype=torch.long)
        mock_runtime.codec.mimi.decode.return_value = torch.zeros(1, 1, 1920, dtype=torch.float32)
        mock_runtime.tokenizer.padding_id = 3
        mock_runtime.tokenizer.end_padding_id = 0
        mock_runtime.tokenizer._processor.id_to_piece.return_value = "hello"
        mock_load_runtime.return_value = mock_runtime

        with patch("importlib.import_module") as mock_importlib:
            # Mock LMGen
            mock_lmgen_class = MagicMock()
            mock_generator = MagicMock()
            # tokens: (B=1, dep_q+1=17, T=1) -> tokens[0, 0, 0] = token_id 10
            mock_tokens = torch.zeros(1, 17, 1, dtype=torch.long)
            mock_tokens[0, 0, 0] = 10
            mock_generator.step.return_value = mock_tokens
            mock_lmgen_class.return_value = mock_generator

            mock_lm_module = MagicMock()
            mock_lm_module.LMGen = mock_lmgen_class
            mock_importlib.return_value = mock_lm_module

            with tempfile.TemporaryDirectory() as tmpdir:
                session = InteractiveSession(
                    model_root=tmpdir,
                    device="cpu",
                )
                dummy_pcm = np.zeros(1920, dtype=np.float32)
                agent_pcm, text_piece = session.step_audio_chunk(dummy_pcm)

                self.assertIsNotNone(agent_pcm)
                self.assertEqual(len(agent_pcm), 1920)
                self.assertEqual(text_piece, "hello")
                session.close()


if __name__ == "__main__":
    unittest.main()
