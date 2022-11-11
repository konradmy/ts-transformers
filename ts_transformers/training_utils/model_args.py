"""This module contains ModelArgs class which manages model parameters.
"""
from datetime import datetime

from ts_transformers.utils import create_directory, to_json


class ModelArgs:
    """Class for managing properties for model's parameters.
    """
    def __init__(
        self,
        model_name: str = "",
        hid_dim: int = 256,
        ff: int = 1024,
        num_heads: int = 8,
        num_tr_lays: int = 3,
        wind_size: int = 60,
        dropout: int = 0,
        input_dim: int = 1,
        epochs: int = 5,
        train_size: float = 0.6,
        val_size: float = 0.2,
        dataset: str = "yfinance",
        ticker: str = "AAPL",
        batch_size: int = 60,
        execution_id: str = None,
        optim: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        opt_scheduler: str = "none",
        opt_restart: bool = None,
        opt_decay_step: float = None,
        opt_decay_rate: float = None,
        criterion: str = "mse",
    ) -> None:
        self.model_name = model_name
        self.hid_dim = hid_dim
        self.ff = ff
        self.num_heads = num_heads
        self.num_tr_lays = num_tr_lays
        self.wind_size = wind_size
        self.dropout = dropout
        self.input_dim = input_dim
        self.epochs = epochs
        self.train_size = train_size
        self.val_size = val_size
        self.dataset = dataset
        self.ticker = ticker
        self.batch_size = batch_size
        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion
        self.opt_scheduler = opt_scheduler
        self.opt_restart = opt_restart
        self.opt_decay_step = opt_decay_step
        self.opt_decay_rate = opt_decay_rate

        if execution_id is None:
            self.execution_id = datetime.now().strftime("%d%m%Y%H%M%S")
        else:
            self.execution_id = execution_id

    def get_checkpoint_dir(self) -> str:
        """Creates path to store model checkpoints based on model's attribute.

        Returns:
            str: Path to save model checkpoints for a model in a given run.
        """
        return f"models/checkpoints/{self.execution_id}"

    def get_model_args_dir(self) -> str:
        """Creates path to store paramaters used to instantiate GNN model.

        Returns:
            str: Path to pickled model parameters
        """
        create_directory("models/model_args/")
        return f"models/model_args/{self.execution_id}.pkl"

    def save_run_metadata(self):
        """Save JSON file with run metadata."""
        metadata = {
            "execution_id": self.execution_id,
            "model_name": self.model_name,
            "hid_dim": self.hid_dim,
            "ff": self.ff,
            "num_heads": self.num_heads,
            "num_tr_lays": self.num_tr_lays,
            "wind_size": self.wind_size,
            "dropout": self.dropout,
            "input_dim": self.input_dim,
            "epochs": self.epochs,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "dataset": self.dataset,
            "ticker": self.ticker,
            "batch_size": self.batch_size,
            "optim": self.optim,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "criterion": self.criterion,
            "opt_scheduler": self.opt_scheduler,
            "opt_restart": self.opt_restart,
            "opt_decay_step": self.opt_decay_step,
            "opt_decay_rate": self.opt_decay_rate,
        }
        create_directory("models/run_metadata/")
        to_json(metadata, f"models/run_metadata/{self.execution_id}.json")
