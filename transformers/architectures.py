import torch
import torch.nn as nn

from transformers.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        num_encoder_layers: int,
        num_decoder_layers: int,
        pre_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        norm_fn: nn.Module = nn.LayerNorm,
        bias: bool = True,
        fused_linear: bool = True,
        mha_args: dict | None = None,
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
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
        decoder_layer = TransformerDecoderLayer(
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

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(
        self,
        x: torch.Tensor,
        encoder_mask: torch.Tensor = None,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None,
    ):
        encoder_out = self.encoder(x, mask=encoder_mask)
        decoder_out = self.decoder(
            x,
            encoder_out,
            self_attn_mask=self_attn_mask,
            cross_attn_mask=cross_attn_mask,
        )

        # TODO: positional encoding and embedding.
        return encoder_out, decoder_out
