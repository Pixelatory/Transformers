import unittest

import torch

from tests.multi_head_attention_tests import BATCH_SIZE, D_MODEL, SEQ_LEN
from transformers.positional_encoding import SinusoidalEncoder


class PositionalEncodingTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.input = torch.rand((BATCH_SIZE, SEQ_LEN, D_MODEL))

    def test_sinusoidal_encoding_input(self):
        self.input = torch.rand((BATCH_SIZE, SEQ_LEN, D_MODEL))
        encoder = SinusoidalEncoder(SEQ_LEN, D_MODEL)
        result = encoder(self.input)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        self.assertFalse(torch.equal(self.input, result))

        encoder = SinusoidalEncoder(d_model=D_MODEL)
        result = encoder(self.input)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        self.assertFalse(torch.equal(self.input, result))

        encoder = SinusoidalEncoder()
        result = encoder(self.input)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        self.assertFalse(torch.equal(self.input, result))
