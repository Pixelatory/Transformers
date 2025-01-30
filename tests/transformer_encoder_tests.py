import unittest

import torch

from tests.common_tests import DIM_FEEDFORWARD
from tests.multi_head_attention_tests import BATCH_SIZE, D_MODEL, NHEADS, SEQ_LEN
from transformers.transformer import TransformerEncoder, TransformerEncoderLayer


class TransformerEncoderTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.input = torch.rand((BATCH_SIZE, SEQ_LEN, D_MODEL))

    def create_encoder(
        self, num_encoder_layers: int, mha_args: dict | None = None
    ) -> TransformerEncoder:
        encoder_layer = TransformerEncoderLayer(
            d_model=D_MODEL,
            n_heads=NHEADS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=0.1,
            mha_args=mha_args,
        )
        return TransformerEncoder(encoder_layer, num_encoder_layers)

    def test_standard_input(self):
        encoder = self.create_encoder(4)
        self.assertEqual(len(encoder.encoder_layers), 4)
        result = encoder(self.input)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))

    def test_mha_args(self):
        """MHA arguments are passed from encoder layer constructor to MHA."""
        mha_args = {
            "use_causal_mask": True,
            "scale": -1000,
            "value_dim": 2,
            "key_dim": 73,
        }
        encoder = self.create_encoder(4, mha_args)
        for encoder_layer in encoder.encoder_layers:
            mha_attn = encoder_layer.attention
            self.assertEqual(mha_attn.use_causal_mask, True)
            self.assertEqual(mha_attn.scale, -1000)
            self.assertEqual(mha_attn.value_dim, 2)
            self.assertEqual(mha_attn.key_dim, 73)
