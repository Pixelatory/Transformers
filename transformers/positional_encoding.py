import math

import torch
import torch.nn as nn


class SinusoidalEncoder(nn.Module):
    """
    Module to apply sinusoidal encoding.

    Vaswani, A. "Attention is all you need."
    Advances in Neural Information Processing Systems (2017).
    """

    def __init__(self, max_seq_len: int | None = None, d_model: int | None = None):
        super().__init__()
        if max_seq_len is not None and d_model is not None:
            self.encoding = self._generate_encoding(max_seq_len, d_model)

    def _generate_encoding(self, seq_len: int, d_model: int):
        encoding = torch.zeros(size=(seq_len, d_model))
        pos = torch.arange(seq_len).unsqueeze(-1)
        div_term = torch.exp(
            torch.arange(start=0, end=d_model, step=2) * -math.log(10000) / d_model
        )
        encoding[:, 0::2] = torch.sin(pos * div_term)
        encoding[:, 1::2] = torch.cos(pos * div_term)
        return encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        if hasattr(self, "encoding"):
            encoding = self.encoding
        else:
            _, seq_len, d_model = x.size()
            encoding = self._generate_encoding(seq_len, d_model)
        print(x.shape, encoding.shape)
        return x + encoding
