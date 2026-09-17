import unittest

from personaplex_finetuning.objective import stream_weights


class ObjectiveTest(unittest.TestCase):
    def test_only_agent_dialogue_streams_receive_weight(self) -> None:
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
        self.assertEqual(weights[9], (0.0, 0.0, 0.0))

    def test_prompt_mask_disables_even_padding_weight(self) -> None:
        weights = stream_weights(
            ((3,),) + tuple(((1,),) for _ in range(16)),
            ((False,),) + tuple(((True,),) for _ in range(16)),
            text_padding_id=3,
        )
        self.assertEqual(weights[0], (0.0,))
