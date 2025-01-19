import math
import torch
import torch.nn.functional as F
import torch.nn as nn

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
        **kwargs
    ) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout, bias=bias, **kwargs
        )
        self.norm1 = norm_fn(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=bias),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=bias),
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
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=bias),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=bias),
        )
        self.norm4 = norm_fn(d_model)

        self.attention = MultiHeadAttention(d_model, n_heads, dropout, bias=bias)
        self.norm1 = norm_fn(d_model)
        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=bias),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=bias),
        )
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
