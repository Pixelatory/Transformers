import unittest

import torch
import torch.nn as nn

from tests.common_tests import DIM_FEEDFORWARD
from tests.multi_head_attention_tests import BATCH_SIZE, D_MODEL, NHEADS, SEQ_LEN
from transformers.transformer import TransformerDecoder, TransformerDecoderLayer

try:
    from flash_attn.ops.fused_dense import FusedDense

    _fused_dense_found = True
except ImportError:
    _fused_dense_found = False


class TransformerDecoderTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.input = torch.rand((BATCH_SIZE, SEQ_LEN, D_MODEL))

    def assert_cuda_is_available(self):
        self.assertTrue(torch.cuda.is_available())

    def create_decoder(
        self,
        num_decoder_layers: int,
        mha_args: dict | None = None,
        fused_linear: bool = False,
    ) -> TransformerDecoder:
        decoder_layer = TransformerDecoderLayer(
            d_model=D_MODEL,
            n_heads=NHEADS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=0.1,
            fused_linear=fused_linear,
            mha_args=mha_args,
        )
        return TransformerDecoder(decoder_layer, num_decoder_layers)

    def test_standard_input(self):
        decoder = self.create_decoder(4)
        self.assertEqual(len(decoder.decoder_layers), 4)
        result = decoder(self.input, self.input)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))

    def test_mha_args(self):
        """MHA arguments are passed from decoder layer constructor to MHA."""
        mha_args = {
            "scale": -1000,
            "value_dim": 2,
            "key_dim": 73,
        }

        def _assert_mha(mha_attn):
            self.assertEqual(mha_attn.scale, -1000)
            self.assertEqual(mha_attn.value_dim, 2)
            self.assertEqual(mha_attn.key_dim, 73)

        encoder = self.create_decoder(4, mha_args)
        for encoder_layer in encoder.decoder_layers:
            _assert_mha(encoder_layer.attention)
            _assert_mha(encoder_layer.cross_attention)

    def test_fused_linear(self):
        self.assert_cuda_is_available()
        self.assertTrue(_fused_dense_found)
        decoder = self.create_decoder(4, fused_linear=True).to(
            device="cuda", dtype=torch.float16
        )
        has_standard_linear = False
        has_fused_linear = False
        for module in decoder.modules():
            if isinstance(module, FusedDense):
                has_fused_linear = True
            elif isinstance(module, nn.Linear):
                has_standard_linear = True
        self.assertTrue(has_fused_linear)
        self.assertFalse(has_standard_linear)

        input = self.input.to(device="cuda", dtype=torch.float16)
        result = decoder(input, input)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
