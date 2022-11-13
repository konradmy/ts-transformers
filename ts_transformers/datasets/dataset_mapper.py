"""This module contains mapper from dataset name to a dataset class."""
from .y_fin_dataset import YFinanceDataset
from .rr_vol_cl_dataset import YFinanceRRVolDataset


dataset_mapper = {
    "yfinance": YFinanceDataset,
    "rr_vol_cl_dataset": YFinanceRRVolDataset,
}
