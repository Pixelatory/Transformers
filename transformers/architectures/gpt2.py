import copy

import torch
import torch.nn as nn

from transformers.common import MLP, reset_model_parameters
from transformers.multi_head_attention import MultiHeadAttention


class GPT2DecoderLayer(nn.Module):
    """
    GPT-2 architecture decoder layer.

    Radford, Alec, et al.
    "Language models are unsupervised multitask learners."
    OpenAI blog 1.8 (2019): 9.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation_fn: nn.Module = nn.GELU,
        norm_fn: nn.Module = nn.LayerNorm,
        bias: bool = True,
        fused_linear: bool = True,
        mha_args: dict | None = None,
    ):
        super().__init__()
        if mha_args is None:
            mha_args = {}
            mha_args["causal_mask"] = True

        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout, bias=bias, fused_linear=fused_linear, **mha_args
        )
        self.norm1 = norm_fn(d_model)

        self.mlp = MLP(
            d_model,
            dim_feedforward,
            d_model,
            bias,
            dropout,
            activation_fn,
            fused=fused_linear,
        )
        self.norm2 = norm_fn(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Parameters:
        - mask: This is usually set to the causal mask. Masks out attention to values where mask is set to 1. \
            Should be shape (query_len x key_len/value_len), (batch_size x n_heads x query_len x key_len/value_len), \
            or (batch_size x key_len/value_len).
        """
        residual = x
        x = self.norm1(x)
        x = residual + self.attention(x, x, x, mask=mask)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.mlp(x))

        return x


class GPT2(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        activation_fn: nn.Module = nn.GELU,
        norm_fn: nn.Module = nn.LayerNorm,
        bias: bool = True,
        fused_linear: bool = True,
        mha_args: dict | None = None,
    ):
        super().__init__()
        decoder_layer = GPT2DecoderLayer(
            d_model,
            n_heads,
            dim_feedforward,
            dropout,
            activation_fn,
            norm_fn,
            bias,
            fused_linear,
            mha_args,
        )
        self.decoder_layers = nn.ModuleList()
        self.final_norm = norm_fn()

        for _ in range(num_layers):
            self.decoder_layers.append(copy.deepcopy(decoder_layer))
        reset_model_parameters(self)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Note: Does not apply a learned positional embedding or token embedding."""
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask=mask)
        return x
