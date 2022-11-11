"""Script for running training loop from CLI."""
import click
from datetime import datetime
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader

from .datasets import dataset_mapper
from .models import model_mapper
from .training_utils import (
    build_loss, build_optimizer, ModelArgs, train_test_split, Trainer
)
from .utils import to_pickle


@click.command()
@click.option("--model-name", default="ts_simple_transformer")
@click.option("--hid-dim", default=256)
@click.option("--ff", default=1024)
@click.option("--num-heads", default=8)
@click.option("--num-tr-lays", default=3)
@click.option("--wind-size", default=60)
@click.option("--dropout", default=0)
@click.option("--input-dim", default=1)
@click.option("--epochs", default=5)
@click.option("--train-size", default=0.6)
@click.option("--val-size", default=0.2)
@click.option("--dataset", default="yfinance")
@click.option("--ticker", default="AAPL")
@click.option("--batch-size", default=60)
@click.option("--optim", default="adam")
@click.option("--lr", default=1e-3)
@click.option("--weight-decay", default=0)
@click.option("--opt-decay-rate", default=1)
@click.option("--opt-decay-step", default=1000)
@click.option("--criterion", default="mse")
@click.option("--check-freq", default=10)
@click.option("--stats-freq", default=10)
@click.option("--val-metrics", default=True)
def _main(
    model_name: str,
    hid_dim: int,
    ff: int,
    num_heads: int,
    num_tr_lays: int,
    wind_size: int,
    dropout: int,
    input_dim: int,
    epochs: int,
    train_size: float,
    val_size: float,
    dataset: str,
    ticker: str,
    batch_size: int,
    optim: str,
    lr: float,
    weight_decay: float,
    opt_decay_step: float,
    opt_decay_rate: float,
    criterion: str,
    check_freq: int,
    stats_freq: int,
    val_metrics: bool,
):
    """Runs training loop.

    Args:
        model_name (str): Model name.
        hid_dim (int): Hidden dimension.
        ff (int): Dimension of Transformers' feedforward network
        num_heads (int): Number of heads in Multihead attention.
        num_tr_lays (int): Number of transformers layers.
        wind_size (int): Window size of a sequence in dataset training input.
        dropout (int): Dropout rate.
        input_dim (int): Input dimension.
        epochs (int): Number of epochs.
        train_size (float): Training ratio size.
        val_size (float): Validation ratio size.
        dataset (str): Dataset name.
        ticker (str): Ticker for a stock for prediction
        batch_size (int): Batch size.
        optim (str): Optimiser type.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        criterion (str): Loss function to optimise.
    """
    execution_id = datetime.now().strftime("%d%m%Y%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_mapper.get(model_name)
    assert model is not None, "Provided model name doesn't exist"

    data = dataset_mapper.get(dataset)
    data = data(ticker, wind_size)
    assert data is not None, "Provided dataset name doesn't exist"

    # Get model params, save them and instantiate model
    model_args = ModelArgs(
        model_name=model_name,
        hid_dim=hid_dim,
        ff=ff,
        num_heads=num_heads,
        num_tr_lays=num_tr_lays,
        wind_size=wind_size,
        dropout=dropout,
        input_dim=input_dim,
        epochs=epochs,
        train_size=train_size,
        val_size=val_size,
        dataset=dataset,
        ticker=ticker,
        batch_size=batch_size,
        execution_id=execution_id,
        optim=optim,
        lr=lr,
        weight_decay=weight_decay,
        opt_decay_step=opt_decay_step,
        opt_decay_rate=opt_decay_rate,
        criterion=criterion
    )

    model_params = {
        "input_dim": input_dim,
        "hid_dim": hid_dim,
        "ff_int_dim": ff,
        "num_heads": num_heads,
        "num_hid_layers": num_tr_lays,
        "max_len": wind_size,
        "dropout": dropout
    }

    model_args_dir = model_args.get_model_args_dir()
    to_pickle(model_params, model_args_dir)

    model_args.save_run_metadata()

    model = model(**model_params)
    model = model.to(device)

    # Dataset

    data_splits = train_test_split(data, train_size, val_size, True)
    train_loader, val_loader, test_loader = [
        DataLoader(split, batch_size=batch_size, drop_last=True)
        for split in data_splits
    ]

    # Loss and optimiser
    criterion = build_loss(criterion)
    _, optimiser = build_optimizer(model_args, model.parameters())

    # TensorboardX SummaryWriter
    writer = SummaryWriter(f"models_logs/{execution_id}")
    checkpoint_dir = model_args.get_checkpoint_dir()

    trainer = Trainer(
        model=model,
        model_name=model_name,
        optimiser=optimiser,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        n_epochs=epochs,
        checkpoint_freq=check_freq,
        stats_freq=stats_freq,
        checkpoint_dir=checkpoint_dir,
        writer=writer,
        execution_id=execution_id,
        val_metrics=val_metrics,
    )

    trainer.training_loop()


if __name__ == '__main__':
    _main()
