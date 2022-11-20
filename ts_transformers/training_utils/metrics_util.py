"""This module contains a helper function for computing metrics.

The helper function calculates metrics based on different models.
"""
from typing import Callable, List, Union
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn

from .criterion_builder import build_loss
from .loss_utils import _get_weights
from ..models import DirTSTransformer


def calculate_metrics(
    model: nn.Module,
    targets: Union[torch.Tensor, List[torch.Tensor]],
    preds: Union[torch.Tensor, List[torch.Tensor]],
    input: torch.Tensor,
    criterion: Callable,
):

    if isinstance(model, DirTSTransformer):
        preds_close, preds_dir = preds
        targ_close, _, _, targ_dir = targets

        close_loss = criterion(preds_close, targ_close)

        crit_ce = build_loss("cross_entropy")
        dir_loss = crit_ce(preds_dir, targ_dir.reshape(-1, 1))

        # loss = 0.5 * close_loss + 0.5 * dir_loss
        loss = 1/170 * close_loss + dir_loss

        weights = _get_weights(targ_dir.reshape(-1, 1))
        weights = weights[targ_dir.long()]
        weights = weights.cpu()

        roc_auc = roc_auc_score(
            targ_dir.reshape(-1, 1).cpu(), preds_dir, sample_weight=weights
        )
        avg_prec = average_precision_score(
            targ_dir.reshape(-1, 1).cpu(), preds_dir, sample_weight=weights
        )

        metrics = {
            "loss": loss.item(),
            "mse": close_loss.item(),
            "cross_entropy": dir_loss.item(),
            "roc_auc": roc_auc,
            "avgerage_precision": avg_prec
        }

    else:
        targ_close, _, _, targ_dir = targets
        last_close = input[:, -1]
        direction = preds > last_close

        mse_loss = criterion(preds, targ_close)

        crit_ce = build_loss("cross_entropy")
        weights = _get_weights(targ_dir)
        dir_loss = crit_ce(
            direction.reshape(-1).float(),
            targ_dir,
            weight=weights[targ_dir.long()]
        )

        # loss = 1/170 * mse_loss + dir_loss

        weights = _get_weights(targ_dir.reshape(-1, 1))
        weights = weights[targ_dir.long()]
        weights = weights.cpu()

        roc_auc = roc_auc_score(
            targ_dir.reshape(-1, 1).cpu(), direction, sample_weight=weights
        )
        avg_prec = average_precision_score(
            targ_dir.reshape(-1, 1).cpu(), direction, sample_weight=weights
        )

        metrics = {
            "mse": mse_loss.item(),
            "cross_entropy": dir_loss.item(),
            "loss": mse_loss.item(),
            "roc_auc": roc_auc,
            "avgerage_precision": avg_prec
        }

    return metrics
