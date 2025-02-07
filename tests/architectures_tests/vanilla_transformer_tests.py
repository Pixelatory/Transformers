import unittest

import torch

from tests.common_tests import DIM_FEEDFORWARD
from tests.multi_head_attention_tests import BATCH_SIZE, D_MODEL, NHEADS, SEQ_LEN
from transformers.architectures.vanilla_transformer import VanillaTransformer

VOCAB_SIZE = 8


class ArchitectureTests(unittest.TestCase):
    def test_vanilla_transformer(self):
        input = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
        transformer = VanillaTransformer(
            VOCAB_SIZE, D_MODEL, NHEADS, DIM_FEEDFORWARD, 0.1, 4, 4
        )
        x, encoder_out, decoder_out = transformer(input)
        self.assertEqual(encoder_out.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        self.assertEqual(decoder_out.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        self.assertEqual(x.shape, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
