import unittest
from types import SimpleNamespace

import torch

from personaplex_finetuning.train import (
    codebook_diagnostic_stats,
    fused_training_diagnostics,
    text_prediction_diagnostic_counts,
    text_supervision_counts,
    text_target_token_loss_stats,
)


class FusedDiagnosticsTest(unittest.TestCase):
    def test_matches_individual_diagnostics(self):
        torch.manual_seed(0)
        batch_size, frames, text_card, card = 2, 9, 12, 7
        labels = torch.randint(0, card, (batch_size, 17, frames))
        labels[:, 0] = torch.randint(0, text_card, (batch_size, frames))
        labels[:, 0, ::3] = 3  # PAD targets
        loss_mask = torch.rand(batch_size, 17, frames) > 0.3
        logits = torch.randn(batch_size, 16, frames, card, dtype=torch.bfloat16)
        text_logits = torch.randn(batch_size, 1, frames, text_card, dtype=torch.bfloat16)
        text_logits[0, 0, 1, 3] = 50  # force a PAD prediction
        output = SimpleNamespace(
            logits=logits, text_logits=text_logits,
            mask=torch.rand(batch_size, 16, frames) > 0.2,
            text_mask=torch.rand(batch_size, 1, frames) > 0.2,
        )
        batch = {"labels": labels, "loss_mask": loss_mask}
        padding = (3,)

        fused = fused_training_diagnostics(batch, output, padding)
        targets, padded = text_supervision_counts(batch, output, padding)
        ce_sum, ce_count = text_target_token_loss_stats(batch, output, padding)
        t_correct, t_count, t_loss, cb_correct, cb_count, cb_loss = codebook_diagnostic_stats(batch, output, padding)
        prediction_counts = text_prediction_diagnostic_counts(batch, output, padding)

        self.assertEqual(int(fused["text_target_tokens"]), int(targets))
        self.assertEqual(int(fused["text_target_tokens"]), int(ce_count))
        self.assertEqual(int(fused["text_padding_positions"]), int(padded))
        torch.testing.assert_close(fused["text_target_ce_sum"], ce_sum)
        self.assertEqual(int(fused["text_target_correct"]), int(t_correct))
        torch.testing.assert_close(fused["text_prediction_counts"], prediction_counts)
        torch.testing.assert_close(fused["audio_correct"], cb_correct)
        torch.testing.assert_close(fused["audio_count"], cb_count)
        torch.testing.assert_close(fused["audio_loss_sum"], cb_loss)


if __name__ == "__main__":
    unittest.main()
