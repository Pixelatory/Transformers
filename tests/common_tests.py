import copy
import unittest

import torch
import torch.nn as nn

from tests.multi_head_attention_tests import D_MODEL, NHEADS
from transformers.common import reset_model_parameters
from transformers.transformer import TransformerEncoderLayer

DIM_FEEDFORWARD = 64


class TestCommon(unittest.TestCase):
    def test_reset_parameters(self):
        model = TransformerEncoderLayer(
            d_model=D_MODEL,
            n_heads=NHEADS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=0.1,
        )
        original_model = copy.deepcopy(model)

        # LayerNorm parameters are not randomly initialized,
        # and so for testing assign them random values to
        # check they are different.
        for module in original_model.modules():
            if isinstance(module, nn.LayerNorm):
                module.weight = nn.Parameter(torch.rand(size=module.weight.size()))
                module.bias = nn.Parameter(torch.rand(size=module.weight.size()))

        reset_model_parameters(model)

        original_parameters = dict(original_model.named_parameters())
        for name, parameter in model.named_parameters(recurse=True):
            self.assertIn(name, original_parameters)
            self.assertFalse(torch.equal(parameter, original_parameters[name]))
