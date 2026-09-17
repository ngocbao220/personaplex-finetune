"""Tests for FSDP module utilities."""

import unittest
import torch
import torch.nn as nn
from personaplex_finetuning.fsdp import parse_gpu_ids, get_fsdp_policy, fsdp_adapter_state_dict, find_free_port


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(x)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = DummyLayer()
        self.layer2 = DummyLayer()

    def forward(self, x):
        return self.layer2(self.layer1(x))


class TestFSDP(unittest.TestCase):
    def test_parse_gpu_ids(self):
        self.assertEqual(parse_gpu_ids("0,1"), [0, 1])
        self.assertEqual(parse_gpu_ids(" 0 , 1 , 2 "), [0, 1, 2])
        with self.assertRaises(ValueError):
            parse_gpu_ids("0")
        with self.assertRaises(ValueError):
            parse_gpu_ids("")

    def test_find_free_port(self):
        port = find_free_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 1024)

    def test_get_fsdp_policy(self):
        policy = get_fsdp_policy(is_lora=True)
        self.assertTrue(callable(policy))

    def test_fsdp_adapter_state_dict_unwrapped(self):
        model = DummyModel()
        state = fsdp_adapter_state_dict(model)
        self.assertEqual(state, {})


if __name__ == "__main__":
    unittest.main()
