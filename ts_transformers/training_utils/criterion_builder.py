"""This module contains utility functions for building a loss functions.
"""

from typing import Callable
import torch


def build_loss(name: str) -> Callable:
    """Builds loss function for provided name.

    Args:
        name (str): NAme of a loss function.

    Returns:
        Callable: Loss function.
    """
    if name == "mse":
        return torch.nn.MSELoss()
