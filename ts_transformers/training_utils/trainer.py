"""This module contains a Trainer class responsible for training models.
"""

from typing import Callable, List
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from .loss_utils import calculate_loss
from .metrics_util import calculate_metrics
from ..utils import create_directory


class Trainer:
    """Class managing training loop.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        optimiser: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        n_epochs: int,
        checkpoint_freq: int = 10,
        stats_freq: int = 10,
        checkpoint_dir: int = None,
        writer: SummaryWriter = None,
        execution_id: str = None,
        val_metrics: bool = True,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.optimiser = optimiser
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.n_epochs = n_epochs
        self.checkpoint_freq = checkpoint_freq
        self.stats_freq = stats_freq
        self.checkpoint_dir = checkpoint_dir
        self.writer = writer
        self.execution_id = execution_id
        self.val_metrics = val_metrics

        if execution_id is None:
            self.execution_id = datetime.now().strftime("%d%m%Y%H%M%S")
        else:
            self.execution_id = execution_id

    def _train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimiser: torch.optim.Optimizer,
        criterion: Callable,
    ) -> float:
        """Performs forward and back propagation on a full batch.

        Args:
            model (torch.nn.Module): NN model.
            train_loader (DataLoader): PyTorch `DataLoader` with training data.
            optimiser (torch.optim.Optimizer): Optimiser used for training.
            criterion (Callable): Loss function used for training.

        Returns:
            float: Mean training loss.
        """

        model.train()
        train_loss = []

        for batch_x, batch_y in train_loader:
            optimiser.zero_grad()

            prediction = model(batch_x)

            loss = calculate_loss(
                model, batch_y, prediction, batch_x, criterion
            )

            loss.backward()

            train_loss.append(loss.item())
            optimiser.step()

        return np.mean(train_loss)

    @torch.no_grad()
    def _test(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        criterion: Callable,
    ) -> dict:
        """Performs prediction and outputs metrics.

        Args:
            model (torch.nn.Module): NN model.
            test_loader (DataLoader): PyTorch `DataLoader` with
                evaluation data.
            criterion (Callable): Loss function.

        Returns:
            dict: Mean metrics for a given dataset.
        """
        model.eval()
        test_metrics = []

        for batch_x, batch_y in test_loader:

            prediction = model(batch_x)

            temp_metrs = calculate_metrics(
                model, batch_y, prediction, batch_x, criterion
            )

            test_metrics.append(temp_metrs)

        # Calculate mean for each metric
        metrics = _aggr_dicts(test_metrics)
        metrics = {key: np.mean(value) for key, value in metrics.items()}

        return metrics

    def training_loop(self):
        """Runs a training loop based on data provided in the init."""
        for epoch in range(1, self.n_epochs + 1):
            train_loss = self._train(
                model=self.model,
                train_loader=self.train_loader,
                optimiser=self.optimiser,
                criterion=self.criterion
            )

            train_stats = {
                "loss": train_loss
            }
            _log_stats(
                stats=train_stats,
                epoch=epoch,
                writer=self.writer,
                test_type="train"
            )

            if epoch % self.checkpoint_freq == 0:
                if self.checkpoint_dir:
                    self._save_model_checkpoint(
                        self.model, self.optimiser, self.checkpoint_dir, epoch
                    )

            if epoch % self.stats_freq == 0 and self.val_metrics:
                val_stats = self._test(
                    model=self.model,
                    test_loader=self.test_loader,
                    criterion=self.criterion
                )
                _log_stats(
                    stats=val_stats,
                    epoch=epoch,
                    writer=self.writer,
                    test_type="val",
                )

    def _save_model_checkpoint(
        self,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        checkpoint_dir: str,
        epoch: int
    ):
        """Saves model checkpoint.

        Args:
            model (torch.nn.Module): NN model.
            optimiser (torch.optim.Optimizer): Optimiser used for training.
            checkpoint_dir (str): Directory to store model checkpoint in.
            epoch (int): Current epoch.
        """

        create_directory(checkpoint_dir)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_{epoch}.pth")


def _log_stats(
    stats: dict,
    epoch: int,
    writer: SummaryWriter = None,
    test_type: str = "val"
):
    """Logs metrics in a TensorboardX `SummaryWriter`.

    Args:
        stats (dict): Metrics to log.
        epoch (int): Epoch for which stats are logged.
        writer (SummaryWriter, optional): TensorboardX SummaryWriter.
            Defaults to None.
        test_type (str, optional): Test split type. Defaults to "val".
    """
    for stat_name, stat_value in stats.items():
        if stat_value is None:
            continue
        if writer:
            writer.add_scalar(f"{test_type}_{stat_name}", stat_value, epoch)


def _aggr_dicts(dicts_list: List[dict]) -> dict:
    """For a list of dict with the same structure aggregates values for key.

    For example for two given dictionaries:
        dict_1 = {"a": 1, "b": 2}
        dict_2 = {"a": 10, "b": 20}
    it will output the following aggregated dictionary:
        {
            "a": [1, 10],
            "b": [2, 20]
        }

    Args:
        dicts_list (List[dict]): List of dictionaries with the same keys.

    Returns:
        dict: Aggregates dictionary.
    """
    dict_keys = dicts_list[0].keys()
    return {
        key: [dict_entity.get(key) for dict_entity in dicts_list]
        for key in dict_keys
    }
