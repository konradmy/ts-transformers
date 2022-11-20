"""This module contains a helper function for computing loss.

The helper function calculates loss based on different models.
"""
from typing import Callable, List, Union
import torch
import torch.nn as nn

from .criterion_builder import build_loss
from ..models import DirTSTransformer


def calculate_loss(
    model: nn.Module,
    targets: Union[torch.Tensor, List[torch.Tensor]],
    preds: Union[torch.Tensor, List[torch.Tensor]],
    input: torch.Tensor,
    criterion: Callable
):

    if isinstance(model, DirTSTransformer):
        preds_close, preds_dir = preds
        targ_close, _, _, targ_dir = targets

        close_loss = criterion(preds_close, targ_close)

        crit_ce = build_loss("cross_entropy")
        weights = _get_weights(targ_dir)
        dir_loss = crit_ce(
            preds_dir.reshape(-1), targ_dir, weight=weights[targ_dir.long()]
        )

        # loss = 0.5 * close_loss + 0.5 * dir_loss
        loss = 1/170 * close_loss + dir_loss
        # loss = dir_loss

    else:
        targ_close, _, _, targ_dir = targets

        # # Criterion for penalising predictions going into wrong direction
        # last_close = input[:, -1]
        # direction = preds > last_close

        # mse_loss = criterion(preds, targ_close)

        # crit_ce = build_loss("cross_entropy")
        # weights = _get_weights(targ_dir)
        # dir_loss = crit_ce(
        #     direction.reshape(-1).float(),
        #     targ_dir,
        #     weight=weights[targ_dir.long()]
        # )

        # loss = 1/170 * mse_loss + dir_loss

        loss = criterion(preds, targ_close)

    return loss


def _get_weights(labels: torch.Tensor) -> torch.Tensor:
    """Calculates classification weights based on labels distribution.

    Args:
        labels (torch.Tensor): Labels for a specific batch.

    Returns:
        torch.Tensor: Tensor with a weight for each class. For example
        if input labels contain two classes [0, 1] then returned tensor will
        have a weight for class 0 at index 0 and a weight for class 1
        at index 1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (labels.shape[0] == 0):
        return None
    labels = labels.clone()
    labels = labels.long().squeeze()
    labels_ratio = torch.bincount(labels) / labels.shape[0]
    reversed_ratio = (
        torch.tensor([1]).to(device) / labels_ratio
    ).nan_to_num(posinf=0)
    return reversed_ratio / reversed_ratio.sum()
