import unittest

import torch

from tests.architecture_tests import VOCAB_SIZE
from tests.multi_head_attention_tests import BATCH_SIZE, D_MODEL, SEQ_LEN
from transformers.architectures.perceiver import Perceiver


class PerceiverTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.input = torch.rand((BATCH_SIZE, SEQ_LEN, D_MODEL))

    def test_perceiver(self):
        perceiver_layer = PerceiverLayer()
        perceiver = Perceiver(VOCAB_SIZE)