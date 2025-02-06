import copy

import torch
import torch.nn as nn

from transformers.architectures.gpt2 import GPT2DecoderLayer
from transformers.common import MLP, reset_model_parameters
from transformers.multi_head_attention import MultiHeadAttention
from transformers.positional_encoding import GaussianFourierEncoder


# TODO: fourier features positional encoding.
class PerceiverLayer(nn.Module):
    """
    Perceiver architecture layer.

    Jaegle, Andrew, et al.
    "Perceiver: General perception with iterative attention."
    International conference on machine learning. PMLR, 2021.
    """

    def __init__(
        self,
        latent_dim: int,
        kv_dim: int,
        cross_attn_heads: int,
        latent_attn_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_transformer_layers: int,
        tie_transformer_block_weights: bool,
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

        self.kv_dim = kv_dim
        self.latent_dim = latent_dim
        self.cross_attention = MultiHeadAttention(
            latent_dim,
            cross_attn_heads,
            dropout,
            bias=bias,
            key_dim=kv_dim,
            value_dim=kv_dim,
            fused_linear=fused_linear,
            **mha_args
        )
        self.latent_norm = norm_fn(latent_dim)
        self.kv_norm = norm_fn(kv_dim)

        self.mlp = MLP(
            latent_dim,
            dim_feedforward,
            latent_dim,
            bias,
            dropout,
            activation_fn,
            fused=fused_linear,
        )
        self.mlp_norm = norm_fn(latent_dim)

        latent_transformer_layer = GPT2DecoderLayer(
            latent_dim,
            latent_attn_heads,
            dim_feedforward,
            dropout,
            activation_fn,
            norm_fn,
            bias,
            fused_linear,
            mha_args,
        )

        self.latent_transformer_layers = nn.ModuleList()

        for _ in range(num_transformer_layers):
            if tie_transformer_block_weights:
                self.latent_transformer_layers.append(latent_transformer_layer)
            else:
                self.latent_transformer_layers.append(
                    copy.deepcopy(latent_transformer_layer)
                )
                reset_model_parameters(self.latent_transformer_layers[-1])

    def forward(self, latent_vec: torch.Tensor, kv_vec: torch.Tensor):
        residual = latent_vec
        latent_vec = self.latent_norm(latent_vec)
        kv_vec = self.kv_norm(kv_vec)
        latent_vec = residual + self.cross_attention(latent_vec, kv_vec, kv_vec)

        residual = latent_vec
        latent_vec = residual + self.mlp(self.mlp_norm(latent_vec))

        for layer in self.latent_transformer_layers:
            latent_vec = layer(latent_vec)
        return latent_vec


class Perceiver(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        latent_dim: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_transformer_layers: int,
        num_perceiver_layers: int,
        num_latent_vecs: int,
        num_bands: int,
        sigma: float,
        tie_transformer_block_weights: bool,
        tie_cross_attn_weights: bool,
        tie_transformer_weights: bool,
        activation_fn: nn.Module = nn.ReLU,
        norm_fn: nn.Module = nn.LayerNorm,
        bias: bool = True,
        low: float = -1,
        high: float = 1,
        fused_linear: bool = True,
        padding_idx: int = 0,
        mha_args: dict | None = None,
    ):
        """
        :param tie_cross_attn_weights: Tie cross attention weights between perceiver
            layers. Note that if True, all cross attention weights except the first
            will be tied, as the paper does not recommend tying the first.
        :param tie_transformer_block_weights: Tie weights of each transformer block
            within a perceiver layer.
        :param tie_transformer_weights: Tie transformer weights between perceiver layers.
        :param num_bands: Number of frequency bands for fourier positional encoding.
        :param sigma: Standard deviation used during normal distribution initialization of 
            the frequencies for fourier encoding.
        :param low: Low value when generating an equally spaced position vector within a range.
        :param high: High value when generating an equally spaced position vector within a range.
        """
        super().__init__()
        if mha_args is None:
            mha_args = {}
            mha_args["causal_mask"] = False

        self.low = low
        self.high = high
        self.fourier_encoder = GaussianFourierEncoder(
            input_dim=1, num_bands=num_bands, sigma=sigma, concat_original=True
        )
        self.embedding = nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx
        )

        perceiver_layer = PerceiverLayer(
            latent_dim=latent_dim,
            cross_attn_heads=n_heads,
            latent_attn_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_transformer_layers=num_transformer_layers,
            tie_transformer_block_weights=tie_transformer_block_weights,
            activation_fn=activation_fn,
            norm_fn=norm_fn,
            bias=bias,
            fused_linear=fused_linear,
            mha_args=mha_args,
        )

        self.layers = nn.ModuleList()
        cross_attn = perceiver_layer.cross_attention
        latent_transformer_layers = perceiver_layer.latent_transformer_layers

        latent_vec = torch.empty(size=(num_latent_vecs, perceiver_layer.latent_dim))
        latent_vec = torch.nn.init.trunc_normal_(
            latent_vec, mean=0, std=0.02, a=-2, b=-2
        )
        self.latent_vec = nn.Parameter(latent_vec)

        for index in range(num_perceiver_layers):
            self.layers.append(copy.deepcopy(perceiver_layer))
            reset_model_parameters(self.layers[-1])
            if tie_cross_attn_weights and index > 1:
                self.layers[-1].cross_attention = cross_attn
            if tie_transformer_weights:
                self.layers[-1].latent_transformer_layers = latent_transformer_layers

    def forward(self, kv_vec: torch.Tensor):
        batch_size, seq_len = kv_vec.shape

        # Create fourier positional encodings.
        pos_vec = torch.linspace(self.low, self.high, seq_len, device=kv_vec.device).unsqueeze(-1)
        enc_vec = self.fourier_encoder(pos_vec).expand(size=(batch_size, -1, -1))

        kv_vec = self.embedding(kv_vec)

        # Concatenate positional encodings with token embeddings.
        kv_vec = torch.concat((kv_vec, enc_vec), dim=-1)

        # TODO: the kv_dim likely does not match the perceiver layer at this moment.

        # Expand latent_vec to have same batch size.
        latent_vec = self.latent_vec.expand(size=(batch_size, -1, -1))
        for layer in self.layers:
            latent_vec = layer(latent_vec, kv_vec)
        return latent_vec
