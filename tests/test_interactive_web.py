import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.interactive_web import find_preset_voice_prompts, create_gradio_ui


class TestInteractiveWeb(unittest.TestCase):
    def test_find_preset_voice_prompts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            conv1 = base / "conv_0001"
            conv1.mkdir()
            (conv1 / "voice_prompt_left.wav").write_bytes(b"dummy")

            conv2 = base / "conv_0002"
            conv2.mkdir()
            (conv2 / "voice_prompt_right.wav").write_bytes(b"dummy")

            # Non-prompt wav
            (conv2 / "stereo.wav").write_bytes(b"dummy")

            results = find_preset_voice_prompts(base)
            self.assertEqual(len(results), 2)
            labels = [r[0] for r in results]
            self.assertTrue(any("conv_0001" in l for l in labels))
            self.assertTrue(any("conv_0002" in l for l in labels))

    def test_find_preset_voice_prompts_empty_or_missing(self):
        self.assertEqual(find_preset_voice_prompts(Path("/non/existent/path")), [])

    def test_create_gradio_ui_structure(self):
        mock_app = MagicMock()
        mock_app.adapter_path = None
        mock_app.device = "cpu"
        mock_app.default_voice_prompt = "/fake/voice.wav"
        mock_app.default_text_prompt = "You are a helpful assistant."

        preset_prompts = [("Preset 1", "/fake/voice.wav")]
        demo = create_gradio_ui(mock_app, preset_prompts)

        # Gradio Blocks demo object should exist
        import gradio as gr
        self.assertIsInstance(demo, gr.Blocks)


if __name__ == "__main__":
    unittest.main()
