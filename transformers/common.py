import torch
import torch.nn as nn

try:
    from flash_attn.ops.fused_dense import FusedDense

    _flash_attn_found = True
except ImportError:
    _flash_attn_found = False

try:
    from flash_attn.ops.fused_dense import FusedDense

    _fused_dense_found = True
except ImportError:
    _fused_dense_found = False


class Linear(nn.Module):
    """A standard PyTorch Linear, or FlashAttention's FusedDense."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool, fused: bool = False
    ):
        super().__init__()
        if fused and not _fused_dense_found:
            raise ValueError(
                "fused is True, but FusedDense is not installed. "
                "View https://github.com/Dao-AILab/flash-attention/tree/main/csrc/fused_dense_lib for installation procedure."
            )
        linear_module = FusedDense if fused and _flash_attn_found else nn.Linear
        self.module = linear_module(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x):
        return self.module(x)


class MLP(nn.Module):
    """Multilayer Perceptron."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bias: bool,
        dropout: float,
        activation_fn: nn.Module,
        fused: bool = False,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            Linear(in_dim, hidden_dim, bias=bias, fused=fused),
            activation_fn(),
            nn.Dropout(dropout),
            Linear(hidden_dim, out_dim, bias=bias, fused=fused),
        )

    def forward(self, x):
        return self.layer(x)


def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Generate a causal mask.

    :returns: Tensor of shape [seq_len, seq_len] with upper
        triangle values set to 1 (not including diagonal),
        and others 0.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask


def fill_mask_values(mask: torch.Tensor):
    """
    Fills a mask with -inf or 0 values.

    All values higher than 0 are set to -inf (masked out),
    and values less than or equal to 0 are set to 0.0.
    """
    return mask.masked_fill(mask > 0, float("-inf")).masked_fill(mask <= 0, float(0.0))


def reset_model_parameters(model: nn.Module):
    """
    Re-initialize a model's parameters.

    Calls the reset_parameters() method on each module
    within the given `model`, if the method exists.
    """
    for layer in model.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
