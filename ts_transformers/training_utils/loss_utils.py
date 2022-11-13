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
    criterion: Callable
):

    if isinstance(model, DirTSTransformer):
        preds_close, preds_dir = preds
        targ_close, _, _, targ_dir = targets

        close_loss = criterion(preds_close, targ_close)

        crit_ce = build_loss("cross_entropy")
        dir_loss = crit_ce(preds_dir, targ_dir.reshape(-1, 1))

        # loss = 0.5 * close_loss + 0.5 * dir_loss
        loss = 1/170 * close_loss + dir_loss

    else:
        loss = criterion(preds, targets)

    return loss
