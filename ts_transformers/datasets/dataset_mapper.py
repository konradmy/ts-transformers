"""This module contains mapper from dataset name to a dataset class."""
from .y_fin_dataset import YFinanceDataset


dataset_mapper = {
    "yfinance": YFinanceDataset
}
