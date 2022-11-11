"""Module containing Transformer model for time series."""
import torch
import torch.nn as nn

from .transformer_enc import TransformerEncoder
from ..embeddings import AbsolutePositionalEmbedding


class TSTransformer(nn.Module):
    """`TSTransformer class.`

    """
    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        ff_int_dim: int,
        num_heads: int,
        num_hid_layers: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(input_dim, hid_dim)
        self.encoder = TransformerEncoder(
            hid_dim, ff_int_dim, num_heads, num_hid_layers, max_len, dropout
        )
        self.embedding = AbsolutePositionalEmbedding(hid_dim, dropout, max_len)
        self.linear_1 = nn.Linear(hid_dim * max_len, 64)
        self.dropout = nn.Dropout()
        self.linear_2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of `TSTransformer`.

        Args:
            x (torch.Tensor): TS sequence data to predict.

        Returns:
            torch.Tensor: Prediction for next time step.
        """
        x = self.input_layer(x)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        return self.linear_2(self.dropout(x))
