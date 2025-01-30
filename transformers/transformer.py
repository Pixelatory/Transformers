import copy

import torch
import torch.nn as nn

from transformers.common import MLP, reset_model_parameters
from transformers.multi_head_attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        pre_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        norm_fn: nn.Module = nn.LayerNorm,
        bias: bool = True,
        fused_linear: bool = False,
        mha_args: dict | None = None,
    ):
        super().__init__()
        if mha_args is None:
            mha_args = {}
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
        self.dropout = nn.Dropout(dropout)
        self.norm2 = norm_fn(d_model)
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Parameters:
        - mask: Masks out attention to values where mask is set to 1. Should be shape (query_len x key_len/value_len), \
            (batch_size x n_heads x query_len x key_len/value_len), or (batch_size x key_len/value_len)
        """
        residual = x
        x = self.norm1(x) if self.pre_norm else x
        x = residual + self.dropout(self.attention(x, x, x, mask))
        x = self.norm1(x) if not self.pre_norm else x

        residual = x
        x = self.norm2(x) if self.pre_norm else x
        x = residual + self.dropout(self.mlp(x))
        x = self.norm2(x) if not self.pre_norm else x
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.encoder_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.encoder_layers.append(copy.deepcopy(encoder_layer))
            reset_model_parameters(self.encoder_layers[-1])

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        pre_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        norm_fn: nn.Module = nn.LayerNorm,
        bias: bool = True,
        **kwargs
    ) -> None:
        super().__init__()

        self.cross_attention = MultiHeadAttention(
            d_model, n_heads, dropout, bias=bias, **kwargs
        )
        self.norm3 = norm_fn(d_model)
        self.mlp2 = MLP(d_model, dim_feedforward, d_model, bias, dropout, activation_fn)
        self.norm4 = norm_fn(d_model)

        self.attention = MultiHeadAttention(d_model, n_heads, dropout, bias=bias)
        self.norm1 = norm_fn(d_model)
        self.mlp1 = MLP(d_model, dim_feedforward, d_model, bias, dropout, activation_fn)
        self.norm2 = norm_fn(d_model)

        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(
        self,
        encoder_out: torch.Tensor,
        tgt: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None,
    ):
        """
        Parameters:
        - self_attn_mask: This is usually set to the causal mask. Masks out attention to values where mask is set to 1. \
            Should be shape (query_len x key_len/value_len), (batch_size x n_heads x query_len x key_len/value_len), \
            or (batch_size x key_len/value_len)
        - cross_attn_mask: Masks out attention to values where mask is set to 1. \
            Should be shape (query_len x key_len/value_len), (batch_size x n_heads x query_len x key_len/value_len), \
            or (batch_size x key_len/value_len)
        """
        # self-attention followed by feedforward on tgt tensor
        residual = tgt
        tgt = self.norm1(tgt) if self.pre_norm else tgt
        tgt = residual + self.dropout(
            self.attention(tgt, tgt, tgt, mask=self_attn_mask)
        )
        tgt = self.norm1(tgt) if not self.pre_norm else tgt

        residual = tgt
        tgt = self.norm2(tgt) if self.pre_norm else tgt
        tgt = residual + self.dropout(self.mlp1(tgt))
        tgt = self.norm2(tgt) if not self.pre_norm else tgt

        # cross-attention followed by feedforward with encoder_out and tgt tensors
        residual = tgt
        tgt = self.norm3(tgt) if self.pre_norm else tgt
        tgt = residual + self.dropout(
            self.cross_attention(tgt, encoder_out, encoder_out, mask=cross_attn_mask)
        )
        tgt = self.norm3(tgt) if not self.pre_norm else tgt

        residual = tgt
        tgt = self.norm4(tgt) if self.pre_norm else tgt
        tgt = residual + self.dropout(self.mlp2(tgt))
        tgt = self.norm4(tgt) if not self.pre_norm else tgt

        return tgt
