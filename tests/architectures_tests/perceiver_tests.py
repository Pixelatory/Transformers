import unittest

import torch

from tests.architectures_tests.vanilla_transformer_tests import VOCAB_SIZE
from tests.common_tests import DIM_FEEDFORWARD
from tests.multi_head_attention_tests import BATCH_SIZE, D_MODEL, NHEADS, SEQ_LEN
from transformers.architectures.perceiver import Perceiver


class PerceiverTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.input = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))

    def test_perceiver(self):
        perceiver = Perceiver(
            vocab_size=VOCAB_SIZE,
            emb_dim=D_MODEL,
            latent_dim=128,
            n_heads=NHEADS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=0.1,
            num_transformer_layers=4,
            num_perceiver_layers=4,
            num_latent_vecs=12,
            num_bands=6,
            sigma=10,
            tie_transformer_block_weights=False,
            tie_cross_attn_weights=False,
            tie_transformer_weights=False,
        )
        self.assertEqual(perceiver(self.input).shape, (BATCH_SIZE, 128))
