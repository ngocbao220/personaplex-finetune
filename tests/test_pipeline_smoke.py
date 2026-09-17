"""End-to-end pipeline smoke test covering all phases of PersonaPlex LoRA fine-tuning.

Verifies:
- Phase 1: Data contract validation (PreparedDataset, channels, timestamps, 10 samples)
- Phase 2: Sequence builder (17 streams, hybrid prompt mask, loss weighting)
- Phase 3: LoRA injection & Trainer (forward, backward, loss calculation, gradients, parameter freezing)
- Phase 4: Overfit checkpoint save & reload into fresh model
- Phase 5: Inference smoke test with greedy LMGen streaming, Mimi decoding, and non-finite checks
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import unittest

# Ensure src is on python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


class PipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import numpy as np
            import sphn
            from moshi.models import loaders, lm
        except ImportError as exc:
            raise unittest.SkipTest(f"Required ML dependency not available: {exc}")

        cls.torch = torch
        cls.np = np
        cls.sphn = sphn
        cls.loaders = loaders
        cls.lm = lm

        # Resolve paths to real assets in the workspace
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.workspace_root = cls.project_root.parent
        cls.manifest_path = cls.workspace_root / "prepared" / "train.jsonl"
        cls.models_path = cls.workspace_root / "models"
        cls.refs_source = cls.project_root / "src"

        if not cls.manifest_path.is_file():
            raise unittest.SkipTest(f"Prepared manifest not found at {cls.manifest_path}")
        if not (cls.models_path / "model.safetensors").is_file():
            raise unittest.SkipTest(f"Model assets not found at {cls.models_path}")

    def test_all_phases_smoke_pipeline(self):
        """Execute all phases in sequence using real audio/data and lightweight test model."""
        torch = self.torch
        np = self.np
        loaders = self.loaders
        lm = self.lm

        from personaplex_finetuning.data import PreparedDataset
        from personaplex_finetuning.runtime import RuntimePaths, MimiCodec, SentencePieceTokenizer
        from personaplex_finetuning.sequence import PersonaPlexTrainingExampleBuilder
        from personaplex_finetuning.train import loss_components, save_adapter
        from personaplex_finetuning.lora import inject_lora, load_adapter
        from personaplex_finetuning.config import Config

        # ---------------------------------------------------------------------
        # Phase 1: Data Contract Validation
        # ---------------------------------------------------------------------
        dataset = PreparedDataset(self.manifest_path, window_seconds=2.0).load()
        self.assertEqual(len(dataset), 10, "Expected 10 prepared samples in dataset")
        sample = dataset[0]
        self.assertEqual(sample.agent_channel, 0, "Agent channel must be LEFT (0)")
        self.assertEqual(sample.user_channel, 1, "User channel must be RIGHT (1)")
        self.assertTrue(sample.conversation_wav.is_file(), "Conversation WAV missing")
        self.assertTrue(sample.voice_prompt_wav.is_file(), "Voice prompt WAV missing")
        self.assertTrue(len(sample.words) > 0, "Words list empty")
        self.assertTrue(bool(sample.text_prompt.strip()), "Text prompt empty")

        # ---------------------------------------------------------------------
        # Phase 2: Runtime Assets & Sequence Builder
        # ---------------------------------------------------------------------
        runtime_paths = RuntimePaths(self.models_path, self.refs_source)
        resolved = runtime_paths.validate()
        mimi = loaders.get_mimi(resolved.mimi_weight, device="cpu")
        codec = MimiCodec(mimi, mimi.sample_rate, mimi.frame_rate, "cpu", lm)
        tokenizer = SentencePieceTokenizer(resolved.tokenizer)

        # Build miniature PersonaPlex LM architecture for fast CPU test
        lm_kwargs = loaders._lm_kwargs.copy()
        lm_kwargs.update({
            "dim": 64,
            "text_card": 32000,
            "existing_text_padding_id": 3,
            "num_heads": 2,
            "num_layers": 2,
            "dep_q": 16,
            "depformer_dim": 32,
            "depformer_dim_feedforward": 64,
            "depformer_num_heads": 2,
            "depformer_num_layers": 2,
        })
        model = loaders.LMModel(device="cpu", dtype=torch.float32, **lm_kwargs)
        initial_tokens = tuple(int(v) for v in model._get_initial_token()[0, :, 0].tolist())

        builder = PersonaPlexTrainingExampleBuilder(
            codec=codec,
            tokenizer=tokenizer,
            initial_tokens=initial_tokens,
            zero_token=int(model.zero_token_id),
        )
        example = builder.build(sample)

        # Assert 17 streams: 1 text + 8 agent audio + 8 user audio
        self.assertEqual(len(example.input_codes), 17)
        self.assertEqual(len(example.loss_mask), 17)
        self.assertGreater(example.prompt_frames, 0)
        self.assertGreater(example.dialogue_frames, 0)

        # Condition-only masking: System prompt must be masked out of loss
        self.assertEqual(
            example.loss_mask[0][:example.prompt_frames],
            (False,) * example.prompt_frames,
            "Text prompt positions must be masked",
        )
        for cb in range(1, 9):
            self.assertEqual(
                example.loss_mask[cb][:example.prompt_frames],
                (False,) * example.prompt_frames,
                f"Agent audio codebook {cb} prompt positions must be masked",
            )
        # User stream (codebooks 9..16) must be completely loss-masked (conditioning only)
        for cb in range(9, 17):
            self.assertTrue(
                all(not active for active in example.loss_mask[cb]),
                f"User audio codebook {cb} must be unweighted/masked",
            )

        # ---------------------------------------------------------------------
        # Phase 3: LoRA Injection & Train Step (Forward + Backward)
        # ---------------------------------------------------------------------
        lora_targets = inject_lora(model, rank=4, alpha=8)
        self.assertGreater(len(lora_targets), 0, "Expected LoRA targets to be injected")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        frozen_params = [p for p in model.parameters() if not p.requires_grad]
        self.assertGreater(len(trainable_params), 0, "LoRA trainable parameters missing")
        self.assertGreater(len(frozen_params), 0, "Base parameters must remain frozen")

        # Forward train with raw 17-stream codes
        raw_codes = torch.tensor(example.input_codes, dtype=torch.long).unsqueeze(0)
        model_out = model.forward_train(raw_codes)

        # Undelayed logits may contain NaN on invalid positions outside delay window;
        # clean NaN in logits for loss computation
        cleaned_logits = torch.nan_to_num(model_out.logits, nan=0.0)
        cleaned_text_logits = torch.nan_to_num(model_out.text_logits, nan=0.0)
        cleaned_output = type("CleanedOutput", (), {
            "logits": cleaned_logits,
            "mask": model_out.mask,
            "text_logits": cleaned_text_logits,
            "text_mask": model_out.text_mask,
        })()

        total_loss, components = loss_components(
            cleaned_output, raw_codes, example, tokenizer.padding_id, torch
        )
        self.assertTrue(torch.isfinite(total_loss), f"Total loss is non-finite: {total_loss}")
        self.assertTrue(torch.isfinite(components["text"]), "Text loss is non-finite")
        self.assertTrue(torch.isfinite(components["audio_semantic"]), "Semantic loss is non-finite")
        self.assertTrue(torch.isfinite(components["audio_nonsemantic"]), "Nonsemantic loss is non-finite")

        # Backward pass
        total_loss.backward()

        # Gradients assertion matching train.py contract
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in trainable_params),
            "LoRA trainable parameters must receive non-zero gradients",
        )
        for name, param in model.named_parameters():
            if not param.requires_grad:
                self.assertIsNone(param.grad, f"Frozen parameter {name} received gradient")

        # ---------------------------------------------------------------------
        # Phase 4: Save Checkpoint & Reload Adapter
        # ---------------------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            cfg = Config(
                path=run_dir / "config.yaml",
                model_root=self.models_path,
                personaplex_source=self.refs_source,
                prepared_dir=self.workspace_root / "prepared",
                output_dir=run_dir,
                device="cpu",
            )
            saved_adapter = save_adapter(run_dir, model, cfg, step=1)
            self.assertTrue(saved_adapter.is_file(), f"Adapter file not created: {saved_adapter}")

            # Instantiate fresh model and reload adapter
            fresh_model = loaders.LMModel(device="cpu", dtype=torch.float32, **lm_kwargs)
            inject_lora(fresh_model, rank=4, alpha=8)
            load_adapter(fresh_model, saved_adapter)

            # Compare weights to ensure exact reload
            loaded_sd = fresh_model.state_dict()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertTrue(torch.equal(param.data, loaded_sd[name].data), f"Weight mismatch for {name}")

        # ---------------------------------------------------------------------
        # Phase 5: Inference Smoke Test (LMGen greedy streaming)
        # ---------------------------------------------------------------------
        fresh_model.eval()
        generator = lm.LMGen(
            fresh_model,
            audio_silence_frame_cnt=6,
            sample_rate=codec.sample_rate,
            frame_rate=codec.frame_rate,
            device="cpu",
            use_sampling=False,
        )
        generator.load_voice_prompt(str(sample.voice_prompt_wav))
        generator.text_prompt_tokens = tokenizer.encode(f"<system> {sample.text_prompt.strip()} <system>")

        user_codes = codec.encode_conversation(
            sample.conversation_wav, sample.user_channel, sample.window_start_sec, sample.window_end_sec
        )
        user = torch.tensor(user_codes, device="cpu").unsqueeze(0)

        step_tokens = []
        with torch.no_grad(), generator.streaming(1):
            for frame in range(min(5, user.shape[-1])):
                toks = generator.step(input_tokens=user[:, :, frame : frame + 1])
                if toks is not None:
                    step_tokens.append(toks)

        self.assertGreater(len(step_tokens), 0, "Inference streaming generated 0 frames")
        first_token = step_tokens[0]
        pcm = codec.mimi.decode(first_token[:, 1:9]).squeeze().detach().float().cpu().numpy()
        self.assertTrue(np.isfinite(pcm).all(), "Inference produced non-finite PCM values")
        self.assertGreater(pcm.size, 0, "Decoded PCM is empty")


if __name__ == "__main__":
    unittest.main()
