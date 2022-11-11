"""This module contains implementation of different types of Embeddings."""
import math
import torch
import torch.nn as nn


class AbsolutePositionalEmbedding(nn.Module):
    """Class for Absolute Positional Embeddings.
    """
    def __init__(
        self,
        hid_dim: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first: bool = True
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hid_dim, 2) * (-math.log(10000.0) / hid_dim)
        )

        pe = torch.zeros(max_len, 1, hid_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.transpose(1, 0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds input.

        Args:
            x (torch.Tensor): Input tokens.

        Returns:
            torch.Tensor: Embedded input tokens.
        """
        seq_len_dim = 1 if self.batch_first else 0
        x = x + self.pe[:x.size(seq_len_dim)]
        return self.dropout(x)
