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
        x = residual + self.attention(x, x, x, mask)
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
        fused_linear: bool = True,
        mha_args: dict | None = None,
    ):
        super().__init__()
        if mha_args is None:
            mha_args = {}

        self.cross_attention = MultiHeadAttention(
            d_model, n_heads, dropout, bias=bias, fused_linear=fused_linear, **mha_args
        )
        self.norm1 = norm_fn(d_model)

        self.attention = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            causal_mask=True,
            bias=bias,
            fused_linear=fused_linear,
            **mha_args
        )
        self.norm2 = norm_fn(d_model)

        self.mlp = MLP(
            d_model,
            dim_feedforward,
            d_model,
            bias,
            dropout,
            activation_fn,
            fused=fused_linear,
        )
        self.norm3 = norm_fn(d_model)

        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
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
        # Self-attention followed by feedforward on target tensor.
        residual = target
        target = self.norm1(target) if self.pre_norm else target
        target = residual + self.attention(target, target, target, mask=self_attn_mask)
        target = self.norm1(target) if not self.pre_norm else target

        # Cross-attention followed by feedforward with source and target tensors.
        residual = target
        target = self.norm2(target) if self.pre_norm else target
        target = residual + self.cross_attention(
            target, source, source, mask=cross_attn_mask
        )
        target = self.norm2(target) if not self.pre_norm else target

        residual = target
        target = self.norm3(target) if self.pre_norm else target
        target = residual + self.dropout(self.mlp(target))
        target = self.norm3(target) if not self.pre_norm else target

        return target


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.decoder_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.decoder_layers.append(copy.deepcopy(decoder_layer))
            reset_model_parameters(self.decoder_layers[-1])

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(source, target, self_attn_mask, cross_attn_mask)
        return x
