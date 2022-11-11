"""This module contains Transformers Feed Forward network implementation."""
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Feed Forward network used in Transformers.
    """
    def __init__(
        self,
        hid_dim: int,
        int_dim: int,
        activation: str = 'gelu',
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(hid_dim, int_dim)
        self.linear_2 = nn.Linear(int_dim, hid_dim)

        self.act = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a Feed Forward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Input tensor passed through Feed Forward network.
        """
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


def _get_activation_fn(
    activation: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Retrieves activation function for a given name.

    Args:
        activation (str): Name of the activation function

    Raises:
        RuntimeError: Raised when not supported activation function provided.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: Activation function.
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation)
    )
