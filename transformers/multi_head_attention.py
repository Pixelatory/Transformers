import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.common import Linear

try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

    _flash_attn_found = True
except ImportError:
    _flash_attn_found = False


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        key_dim: int | None = None,
        value_dim: int | None = None,
        scale: float | None = None,
        causal_mask: bool = False,
        flash_attn: bool = False,
        fused_linear: bool = False,
    ):
        super().__init__()

        if (d_model // n_heads) * n_heads != d_model:
            raise ValueError("d_model must be evenly divisible by num_heads")

        if flash_attn and not _flash_attn_found:
            raise ValueError(
                "flash_attn is True, but Flash Attention is not installed. "
                "View https://github.com/Dao-AILab/flash-attention for installation procedure."
            )

        if causal_mask and not flash_attn:
            logging.warning(
                "causal_mask set to True is only used when flash_attn is True."
            )

        self.value_dim = value_dim if value_dim is not None else d_model
        self.key_dim = key_dim if key_dim is not None else d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.scale = scale if scale is not None else self.head_dim**0.5
        self.causal_mask = causal_mask

        if self.is_qkv_packed():
            self.W_qkv = Linear(d_model * 3, d_model * 3, bias=bias, fused=fused_linear)
        else:
            self.W_q = Linear(d_model, d_model, bias=bias, fused=fused_linear)
            self.W_k = Linear(self.key_dim, d_model, bias=bias, fused=fused_linear)
            self.W_v = Linear(value_dim, d_model, bias=bias, fused=fused_linear)

        self.use_flash_attn = flash_attn
        self.W_o = Linear(d_model, d_model, bias=bias, fused=fused_linear)
        self.dropout = dropout

    def is_qkv_packed(self) -> bool:
        return self.d_model == self.value_dim == self.key_dim

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate the attention weights and return the weighted sum of values.

        Attention scores: (query * key^T) / sqrt(head_dim).
        Attention probabilities: softmax(mask(Attention scores)).

        During training, Dropout is performed after attention probabilities
        are calculated and after the final attention output.

        :param query: Projected many-head query tensor of shape
            [batch_size, n_heads, seq_len, d_model].
        :param key: Projected many-head key tensor of shape
            [batch_size, n_heads, seq_len, d_model].
        :param value: Projected many-head value tensor of shape
            [batch_size, n_heads, seq_len, d_model].
        :param mask: Mask values should already be filled out before calling.
        :output: Tensor of shape [batch_size, n_heads, seq_len, d_model].
        """
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_scores += mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, self.dropout, training=self.training)
        output = torch.matmul(attn_probs, value)
        output = F.dropout(output, self.dropout, training=self.training)
        return output

    def split_heads(
        self,
        xs: tuple[torch.Tensor] | torch.Tensor,
        batch_size: int,
        transpose: bool = True,
    ) -> tuple[torch.Tensor] | torch.Tensor:
        """
        Split the projected tensors' last dimensions into [num_heads, d_model].
        Transpose the result such that the shape is [batch_size, n_heads, seq_len, head_dim].

        :param x: Input tensor of shape [batch_size, seq_len, d_model]
            which is split into many-head form.
        :param transpose: Whether to transpose from shape [batch_size, seq_len, n_heads, head_dim]
            to [batch_size, n_heads, seq_len, head_dim].
        """
        single_tensor = False
        if isinstance(xs, torch.Tensor):
            single_tensor = True
            xs = (xs,)
        new_xs = []
        for x in xs:
            x = x.view(batch_size, -1, self.n_heads, self.head_dim)
            new_xs.append(x.transpose(1, 2) if transpose else x)

        if single_tensor:
            return new_xs.pop()
        else:
            return (*new_xs,)

    def combine_heads(self, x: torch.Tensor, transpose: bool = True) -> torch.Tensor:
        """
        Reverses the operation performed by `split_heads`.

        :param x: Input tensor of shape [batch_size, n_heads, seq_len, head_dim]
            or [batch_size, seq_len, n_heads, head_dim].
        :param transpose: Whether to transpose from shape [batch_size, n_heads, seq_len, head_dim]
            to [batch_size, seq_len, n_heads, head_dim]. You will likely want to set this to the
            same value as when `split_heads` was used.

        :outputs: Tensor of shape [batch_size, seq_len, d_model].
        """
        if transpose:
            x = x.transpose(1, 2)
        x = x.contiguous()
        batch_size, seq_length, _, _ = x.size()
        return x.view(batch_size, seq_length, self.d_model)

    def _prepare_packed_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepares a packed qkv tensor.

        Concatenates query, key, and value into a single
        tensor and applies a linear projection.

        :returns: A tensor of shape [batch_size, seq_len, d_model*3].
        """
        # [batch_size, seq_len, d_model*3].
        qkv = torch.concat((query, key, value), dim=-1)

        # [batch_size, seq_len, d_model*3]
        qkv = self.W_qkv(qkv)

        return qkv

    def _prepare_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepares separate q, k, and v tensors.

        Applies a linear projection to each of query,
        key, and value.
        """
        query = self.W_q(query)  # [batch_size, seq_len, d_model]
        key = self.W_k(key)  # [batch_size, seq_len, d_model]
        value = self.W_v(value)  # [batch_size, seq_len, d_model]

        return query, key, value

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        :param mask: A mask of shape [batch_size, 1, seq_len, 1] or [1, 1, seq_len, seq_len] (no flash attention)
            or [batch_size, seq_len, 1] (using flash attention). Mask values should already
            be filled out to -inf or 0 before calling. If using flash attention and only
            causal mask, do not use the mask argument and instead set causal_mask to True.
        :param cu_seqlens: Cumulative sequence lengths. Not currently used, but will be soon.
        """
        if query is None:
            raise ValueError("Query should not be None.")
        elif key is None:
            raise ValueError("Key should not be None.")
        elif value is None:
            raise ValueError("Value should not be None.")

        batch_size = query.size(0)

        if self.use_flash_attn:
            if self.is_qkv_packed():
                qkv = self._prepare_packed_qkv(query, key, value)

                if mask is not None:
                    qkv += mask

                qkv = qkv.view(batch_size, -1, 3, self.n_heads, self.head_dim)

                attn_output = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.dropout if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=self.causal_mask,
                )  # [batch_size, seq_len, n_heads, head_dim]

                attn_output = self.combine_heads(
                    attn_output, transpose=False
                )  # [batch_size, seq_len, d_model]
            else:
                # Each of shape [batch_size, seq_len, d_model].
                query, key, value = self._prepare_qkv(
                    query,
                    key,
                    value,
                )

                if mask is not None:
                    query += mask
                    key += mask
                    value += mask

                # Each has shape [batch_size, seq_len, n_heads, head_dim].
                query, key, value = self.split_heads(
                    (query, key, value), batch_size, transpose=False
                )

                # [batch_size, seq_len, n_heads, head_dim].
                attn_output = flash_attn_func(
                    q=query,
                    k=key,
                    v=value,
                    dropout_p=self.dropout if self.training else 0.0,
                    softmax_scale=self.scale,
                    causal=self.causal_mask,
                )

                # [batch_size, seq_len, d_model].
                attn_output = self.combine_heads(attn_output, transpose=False)

        elif hasattr(self, "W_qkv"):
            qkv = self._prepare_packed_qkv(query, key, value)

            # Each has shape [batch_size, seq_len, d_model].
            query, key, value = torch.chunk(qkv, chunks=3, dim=-1)

            query, key, value = self.split_heads((query, key, value), batch_size)

            attn_output = self.scaled_dot_product_attention(
                query, key, value, mask=mask
            )  # [batch_size, n_heads, seq_len, head_dim]
            attn_output = self.combine_heads(
                attn_output
            )  # [batch_size, n_heads, seq_len, head_dim]
        else:
            query, key, value = self._prepare_qkv(query, key, value)

            query, key, value = self.split_heads(
                (query, key, value), batch_size, transpose=True
            )  # Each has shape [batch_size, seq_len, n_heads, head_dim]

            attn_output = self.scaled_dot_product_attention(
                query, key, value, mask=mask
            )  # [batch_size, n_heads, seq_len, head_dim]
            attn_output = self.combine_heads(
                attn_output
            )  # [batch_size, n_heads, seq_len, head_dim]
        output = self.W_o(attn_output)
        return F.dropout(output, p=self.dropout, training=self.training)
