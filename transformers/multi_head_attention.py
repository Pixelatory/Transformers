import torch
import torch.nn.functional as F
import torch.nn as nn

"""
Referenced: https://github.com/lightmatmul/Transformer-from-scratch
"""

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        key_dim: int | None = None,
        value_dim: int | None = None,
        scale: float | None = None,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        value_dim = value_dim if value_dim is not None else d_model
        self.key_dim = key_dim if key_dim is not None else d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.scale = scale if scale is not None else self.head_dim**0.5

        assert (
            self.head_dim * self.n_heads == self.d_model
        ), "d_model must be divisible by num_heads"

        if self.key_dim == value_dim and not use_flash_attn:
            self.W_qkv = nn.Linear(d_model * 3, d_model * 3, bias=bias)
        elif not use_flash_attn:
            self.W_q = nn.Linear(d_model, d_model, bias=bias)
            self.W_k = nn.Linear(self.key_dim, d_model, bias=bias)
            self.W_v = nn.Linear(value_dim, d_model, bias=bias)

        self.use_flash_attn = use_flash_attn
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        Calculate the attention weights and return the weighted sum of values.

        Attention scores: (query * key^T) / sqrt(head_dim).
        Attention probabilities: softmax(mask(Attention Scores)).

        During training, Dropout is performed after attention probabilities
        are calculated.

        :param query: Projected many-head query tensor of shape
            [batch_size, n_heads, seq_len, d_model].
        :param key: Projected many-head key tensor of shape
            [batch_size, n_heads, seq_len, d_model].
        :param value: Projected many-head value tensor of shape
            [batch_size, n_heads, seq_len, d_model].
        :output: Tensor of shape [batch_size, n_heads, seq_len, d_model].
        """
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, self.dropout, training=self.training)
        output = torch.matmul(attn_probs, value)
        output = F.dropout(output, self.dropout, training=self.training)
        return output

    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Split the projected tensors' last dimensions into [num_heads, d_model].
        Transpose the result such that the shape is [batch_size, n_heads, seq_len, head_dim].

        :param x: Input tensor of shape [batch_size, seq_len, d_model]
            which is split into many-head form.
        """
        x = x.view(batch_size, -1, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverses the operation performed by `split_heads`.

        :param x: Input tensor of shape [batch_size, n_heads, seq_len, head_dim].

        :outputs: Tensor of shape [batch_size, seq_len, d_model].
        """
        x = x.transpose(1, 2).contiguous()
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters:
        - mask: Masks out attention to values where mask is set to 1. Should be shape (query_len x key_len/value_len), \
            (batch_size x n_heads x query_len x key_len/value_len), or (batch_size x key_len/value_len)
        """
        batch_size = query.size(0)

        if self.use_flash_attn:
            pass
        elif hasattr(self, "W_qkv"):
            qkv = torch.concat((query, key, value), dim=-1)  # [batch_size, seq_len, d_model*3]
            qkv = self.W_qkv(qkv)  # [batch_size, seq_len, d_model*3]
            query, key, value = torch.chunk(qkv, chunks=3, dim=-1)  # Each has shape [batch_size, seq_len, d_model]

            query = self.split_heads(query, batch_size)  # [batch_size, n_heads, seq_len, head_dim]
            key = self.split_heads(key, batch_size)  # [batch_size, n_heads, seq_len, head_dim]
            value = self.split_heads(value, batch_size)  # [batch_size, n_heads, seq_len, head_dim]

            attn_output = self.scaled_dot_product_attention(query, key, value)  # [batch_size, n_heads, seq_len, head_dim]
            attn_output = self.combine_heads(attn_output)  # [batch_size, n_heads, seq_len, head_dim]
        else:
            query = self.W_q(query)  # [batch_size, seq_len, d_model]
            key = self.W_k(key)  # [batch_size, seq_len, d_model]
            value = self.W_v(value)  # [batch_size, seq_len, d_model]

            query = self.split_heads(query, batch_size)  # [batch_size, n_heads, seq_len, head_dim]
            key = self.split_heads(key, batch_size)  # [batch_size, n_heads, seq_len, head_dim]
            value = self.split_heads(value, batch_size)  # [batch_size, n_heads, seq_len, head_dim]

            attn_output = self.scaled_dot_product_attention(query, key, value)  # [batch_size, n_heads, seq_len, head_dim]
            attn_output = self.combine_heads(attn_output)  # [batch_size, n_heads, seq_len, head_dim]
        output = self.W_o(attn_output)
        return F.dropout(output, p=self.dropout, training=self.training)
        


"""
if mask is not None:
            print("mask", mask)
            if list(mask.shape) == [energy.shape[0], energy.shape[-1]]:
                mask = mask.unsqueeze(1).unsqueeze(
                    1
                )  # source sequence padding mask: batch_size x 1 x 1 x (key_len/value_len)
            energy = energy.masked_fill_(mask == 1, float("-inf"))
"""

if __name__ == "__main__":
    mha = MultiHeadAttention(32, 4, 0.1, bias=True)
    input = torch.rand((12, 10, 32))
    mha(input, input, input)