import copy

import torch
import torch.nn as nn

from transformers.common import MLP, reset_model_parameters
from transformers.multi_head_attention import MultiHeadAttention


class GPTDecoderLayer(nn.Module):
    """
    GPT architecture decoder layer.

    Radford et al.
    "Improving language understanding by generative pre-training." (2018).
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
        mha_args["causal_mask"] = False

        self.attention = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            causal_mask=True,
            bias=bias,
            fused_linear=fused_linear,
            **mha_args
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
        x = residual + self.attention(x, x, x, mask=mask)
        x = self.norm1(x)

        residual = x
        x = residual + self.dropout(self.mlp(x))
        x = self.norm2(x)

        return x


class GPT(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        pre_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        norm_fn: nn.Module = nn.LayerNorm,
        bias: bool = True,
        fused_linear: bool = True,
        mha_args: dict | None = None,
    ):
        super().__init__()
        decoder_layer = GPTDecoderLayer(
            d_model,
            n_heads,
            dim_feedforward,
            dropout,
            pre_norm,
            activation_fn,
            norm_fn,
            bias,
            fused_linear,
            mha_args,
        )
        self.decoder_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.decoder_layers.append(copy.deepcopy(decoder_layer))
        reset_model_parameters(self)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Does not apply a learned positional embedding or token embedding."""
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(source, target, mask=mask)
        return x
