"""This module contains a helper function for buidling optimiser."""
from typing import Iterator, Tuple
import torch
from torch.nn import Parameter
from .model_args import ModelArgs


def build_optimizer(
    args: ModelArgs, params: Iterator[Parameter]
) -> Tuple[torch.optim.lr_scheduler._LRScheduler, torch.optim.Optimizer]:
    """Returns pytorch optimiser for given configurations.

    Args:
        args (ModelArgs): ModelArgs object with model's configurations.
        params (Iterator[Parameter]): Iterator with model's parameters, which
            are returned by '.parameters()' function from pytorch model.

    Returns:
        torch.optim.Optimizer: Optimiser for given configurations
            and parameters.
    """
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad & (not p.grad_fn), params)
    if args.optim == "adam":
        optimizer = torch.optim.Adam(
            filter_fn, lr=args.lr, weight_decay=weight_decay
        )
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(
            filter_fn,
            lr=args.lr,
            momentum=0.95,
            weight_decay=weight_decay,
        )
    elif args.optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            filter_fn, lr=args.lr, weight_decay=weight_decay
        )
    elif args.optim == "adagrad":
        optimizer = torch.optim.SGD(
            filter_fn, lr=args.lr, weight_decay=weight_decay
        )
    if args.opt_scheduler == "none":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate
        )
    elif args.opt_scheduler == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart
        )
    return scheduler, optimizer
