import unittest

from personaplex_finetuning.objective import stream_weights, torch_weighted_cross_entropy


class ObjectiveTest(unittest.TestCase):
    def test_ignores_invalid_target_at_zero_weight_delay_position(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is unavailable")
        logits = torch.zeros(2, 3)
        targets = torch.tensor([1, -1])
        weights = torch.tensor([1.0, 0.0])

        loss = torch_weighted_cross_entropy(logits, targets, weights)

        self.assertAlmostEqual(float(loss), float(torch.log(torch.tensor(3.0))), places=6)

    def test_ignores_nan_logits_at_zero_weight_delay_position(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is unavailable")
        logits = torch.tensor([[0.0, 0.0, 0.0], [float("nan"), float("nan"), float("nan")]])
        targets = torch.tensor([1, -1])
        weights = torch.tensor([1.0, 0.0])

        loss = torch_weighted_cross_entropy(logits, targets, weights)

        self.assertTrue(torch.isfinite(loss))

    def test_both_dialogue_audio_streams_receive_semantic_and_nonsemantic_weights(self) -> None:
        mask = tuple(tuple(True for _ in range(3)) for _ in range(17))
        codes = (
            (3, 5, 3),
            *((10, 10, 10) for _ in range(8)),
            *((20, 20, 20) for _ in range(8)),
        )

        weights = stream_weights(codes, mask, text_padding_id=3)

        self.assertEqual(weights[0], (0.3, 1.0, 0.3))
        self.assertEqual(weights[1], (1.0, 1.0, 1.0))
        self.assertEqual(weights[2], (0.02, 0.02, 0.02))
        self.assertEqual(weights[9], (1.0, 1.0, 1.0))
        self.assertEqual(weights[10], (0.02, 0.02, 0.02))

    def test_disabling_user_loss_zeros_only_user_audio_stream_weights(self) -> None:
        codes = tuple((stream,) for stream in range(17))
        mask = tuple((True,) for _ in range(17))

        weights = stream_weights(codes, mask, text_padding_id=3, user_loss=False)

        self.assertEqual(weights[0], (1.0,))
        self.assertEqual(weights[1], (1.0,))
        self.assertEqual(weights[2], (0.02,))
        self.assertTrue(all(weights[index] == (0.0,) for index in range(9, 17)))

    def test_prompt_mask_disables_even_padding_weight(self) -> None:
        weights = stream_weights(
            ((3,),) + tuple(((1,),) for _ in range(16)),
            ((False,),) + tuple(((True,),) for _ in range(16)),
            text_padding_id=3,
        )
        self.assertEqual(weights[0], (0.0,))

    def test_first_codebook_multiplier_and_text_padding_are_configurable(self) -> None:
        weights = stream_weights(
            ((3, 4),) + tuple((8, 8) for _ in range(8)) + tuple((9, 9) for _ in range(8)),
            tuple((True, True) for _ in range(17)),
            text_padding_id=3, first_codebook_weight_multiplier=2.5, text_padding_weight=0.4,
        )
        self.assertEqual(weights[0], (0.4, 1.0))
        self.assertEqual(weights[1], (2.5, 2.5))
        self.assertEqual(weights[9], (2.5, 2.5))

    def test_torch_weights_match_configured_reference_weights(self) -> None:
        import torch
        from personaplex_finetuning.objective import stream_weights_torch

        codes = torch.full((1, 17, 2), 4, dtype=torch.long)
        codes[:, 0, 0] = 3
        mask = torch.ones_like(codes, dtype=torch.bool)
        weights = stream_weights_torch(codes, mask, 3, 0.02, 0.4, 2.5)
        self.assertTrue(torch.allclose(weights[0, 0], torch.tensor([0.4, 1.0])))
        self.assertTrue(torch.allclose(weights[0, 1], torch.tensor([2.5, 2.5])))
        self.assertTrue(torch.allclose(weights[0, 9], torch.tensor([2.5, 2.5])))

    def test_torch_weights_disable_user_audio_supervision(self) -> None:
        import torch
        from personaplex_finetuning.objective import stream_weights_torch

        codes = torch.full((1, 17, 2), 4, dtype=torch.long)
        mask = torch.ones_like(codes, dtype=torch.bool)

        weights = stream_weights_torch(codes, mask, 3, user_loss=False)

        self.assertTrue(torch.all(weights[:, 1:9] > 0))
        self.assertTrue(torch.equal(weights[:, 9:17], torch.zeros_like(weights[:, 9:17])))

    def test_both_text_padding_tokens_receive_padding_weight(self) -> None:
        import torch
        from personaplex_finetuning.objective import stream_weights_torch

        codes = torch.full((1, 17, 3), 4, dtype=torch.long)
        codes[:, 0] = torch.tensor([3, 0, 4])  # PAD, END_PAD, regular token.
        mask = torch.ones_like(codes, dtype=torch.bool)
        weights = stream_weights_torch(codes, mask, (3, 0), text_padding_weight=0.3)

        self.assertTrue(torch.allclose(weights[0, 0], torch.tensor([0.3, 0.3, 1.0])))
