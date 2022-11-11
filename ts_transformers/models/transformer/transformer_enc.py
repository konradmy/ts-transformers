"""Module containing Transformer Encoder classes."""
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from ..embeddings import AbsolutePositionalEmbedding

from .transf_feed_forward import FeedForward


class TransformerEncoder(nn.Module):
    """Transformers Encoder module.
    """
    def __init__(
        self,
        hid_dim: int,
        ff_int_dim: int,
        num_heads: int,
        num_hid_layers: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        emb_type: str = None
    ) -> None:
        super().__init__()

        kwargs = {
            "hid_dim": hid_dim,
            "ff_int_dim": ff_int_dim,
            "num_heads": num_heads,
            "num_hid_layers": num_hid_layers,
            "max_len": max_len,
            "dropout": dropout
        }

        self.embeddings = _get_embedding_type(emb_type, **kwargs)

        self.layers = nn.ModuleList([
            TransformersEncoderLayer(hid_dim, ff_int_dim, num_heads, dropout)
            for _ in range(num_hid_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of The whole Transformer Encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Fully encoded input tensor.
        """
        if self.embeddings is not None:
            x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)

        return x


class TransformersEncoderLayer(nn.Module):
    """Class for Transformers Encoding Layer.
    """
    def __init__(
        self,
        hid_dim: int,
        ff_int_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hid_dim)
        self.layer_norm_2 = nn.LayerNorm(hid_dim)
        self.attention = MultiHeadAttention(hid_dim, num_heads)
        self.ff = FeedForward(hid_dim, ff_int_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of `TransformersEncoderLayer`.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded input tensor.
        """
        hid_state = self.layer_norm_1(x)
        x = x + self.attention(hid_state)

        x = x + self.ff(self.layer_norm_2(x))

        return x


def _get_embedding_type(emb_type: str, **kwargs) -> nn.Module:
    """Instantiates embeddings of a specific type.

    Args:
        emb_type (str): Type of embeddings to instantiate. Available:
            - `abs_positional`

    Returns:
        nn.Module: Embedding module.
    """
    if isinstance(emb_type, str):
        if emb_type == "abs_positional":
            args_names = ["hid_dim", "dropout", "max_len", "batch_first"]
            kwargs = {
                arg_name: kwargs.get(arg_name) for arg_name in args_names
                if arg_name is not None
            }
            return AbsolutePositionalEmbedding(**kwargs)
    elif emb_type is None:
        return None
