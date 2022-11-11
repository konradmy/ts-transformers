"""This module contains implementation of Transformers' Attention heads."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multihead attention network.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ) -> None:
        super().__init__()

        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"

        head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )

        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward pass of multihead attention network.

        Args:
            hidden_state (torch.Tensor): Input hidden state.

        Returns:
            torch.Tensor: Output of hidden state passed through
                multihead attention.
        """
        x = torch.cat([head(hidden_state) for head in self.heads], dim=-1)

        return self.output_linear(x)


class AttentionHead(nn.Module):
    """Singular Attention Head.
    """
    def __init__(self, embed_dim: int, head_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward pass of a singular attention network.

        Args:
            hidden_state (torch.Tensor): Input hidden state.

        Returns:
            torch.Tensor: Output of hidden state passed through
                a signular attention.
        """
        out = _scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
        )

        return out


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor
) -> torch.Tensor:
    """Calculates attention weights for query, key and values vectors.

    Args:
        query (torch.Tensor): Query vectors (batched).
        key (torch.Tensor): Key vectors (batched).
        value (torch.Tensor): Value vectors (batched).

    Returns:
        torch.Tensor: _description_
    """
    dim_k = query.size(-1)

    att_scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim_k)
    att_weights = F.softmax(att_scores)

    return torch.bmm(att_weights, value)
