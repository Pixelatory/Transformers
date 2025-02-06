import math

import torch
import torch.nn as nn


class SinusoidalEncoder(nn.Module):
    """
    Module to apply sinusoidal encoding.

    Vaswani et al. "Attention is all you need."
    Advances in Neural Information Processing Systems (2017).
    """

    def __init__(self, max_seq_len: int | None = None, d_model: int | None = None):
        super().__init__()
        if max_seq_len is not None and d_model is not None:
            self.encoding = self._generate_encoding(max_seq_len, d_model)

    def _generate_encoding(self, seq_len: int, d_model: int):
        encoding = torch.zeros(size=(seq_len, d_model))
        pos = torch.arange(seq_len).unsqueeze(-1)
        div_term = torch.exp(
            torch.arange(start=0, end=d_model, step=2) * -math.log(10000) / d_model
        )
        encoding[:, 0::2] = torch.sin(pos * div_term)
        encoding[:, 1::2] = torch.cos(pos * div_term)
        return encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        if hasattr(self, "encoding"):
            encoding = self.encoding
        else:
            _, seq_len, d_model = x.size()
            encoding = self._generate_encoding(seq_len, d_model)
        print(x.shape, encoding.shape)
        return x + encoding


class GaussianFourierEncoder(nn.Module):
    """
    Tancik, Matthew, et al.
    "Fourier features let networks learn high frequency functions in low dimensional domains."
    Advances in neural information processing systems 33 (2020): 7537-7547.
    """

    def __init__(
        self,
        input_dim: int,
        num_bands: int,
        sigma: float,
        concat_original: bool = True,
    ):
        """
        :param input_dim: The dimensionality of the input position.
            E.g. 1 for word sequence, 2 for (x, y) pixels on image, etc.
        :param num_bands: Number of frequency bands.
        :param concat_original: Whether to concatenate the original position
            tensor with the fourier encoding tensor.
        """
        super().__init__()
        self.input_dim = input_dim
        self.sigma = sigma
        self.concat_original = concat_original

        # TODO: look into regularly spaced frequency matrix,
        # like in NERF and Perceiver paper.
        frequencies = torch.randn(num_bands, input_dim) * sigma
        self.register_buffer("frequencies", frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape [batch_size, ..., input_dim].
        :returns: Tensor of shape [batch_size, ..., input_dim + 2 * num_bands].
        """
        self.frequencies = self.frequencies.to(x.device)
        batch_shape = x.shape[:-1]

        # Flatten input to 2D shape [batch_size, input_dim]
        x = x.view(-1, self.input_dim)

        # [batch_size, num_bands]
        angles = 2 * torch.pi * (x @ self.frequencies.T)

        # [batch_size, num_bands * 2]
        fourier_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if self.concat_original:
            fourier_features = torch.cat([x, fourier_features], dim=-1)

        return fourier_features.view(*batch_shape, -1)


if __name__ == "__main__":
    batch_size = 3
    seq_len = 5
    input_dim = 1
    encoder = GaussianFourierEncoder(input_dim=input_dim, num_bands=4, sigma=10.0)
    x = torch.linspace(start=-1, end=1, steps=seq_len).unsqueeze(-1)
    print(x)
    encoded_x = encoder(x)
    print(encoded_x)
    print(encoded_x.expand(size=(batch_size, -1, -1)))
    print(encoded_x.expand(size=(batch_size, -1, -1)).shape)
    exit(1)
    batched_tensor = encoded_x.expand((batch_size, -1, -1))
    print(batched_tensor.shape)
    print(batched_tensor)
