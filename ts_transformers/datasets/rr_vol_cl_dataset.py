"""This module contains a `YFinanceRRVolDataset` dataset

This dataset contains the following targets:
    - stock close
    - reutrn rates
    - volatility.
"""
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch

from . import dataset_utils as utils


class YFinanceRRVolDataset(Dataset):
    """Dataset with financial close prices, daily returns and volatility.
    """
    def __init__(
        self,
        ticker: str,
        window_size: int,
        start_date: str = '2017-01-01',
        end_date: str = '2021-10-01',
        transform=None,
        target_transform=None,
        scale_target: bool = False
    ):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.ticker = ticker
        self.transform = transform
        self.target_transform = target_transform

        stock_df = yf.download(ticker, start=start_date, end=end_date)
        stock_close = stock_df["Close"].values

        scaled_data = self.scaler.fit_transform(stock_close.reshape(-1, 1))

        daily_rr = utils.get_daily_returns(stock_close)
        vol = utils.get_volatility(stock_close, window_size)
        dir_change = utils.get_pos_neg_change(stock_close)

        y_aux = np.concatenate(
            [daily_rr[None], vol[None], dir_change[None]]
        ).transpose()

        x_data, y_data, y_aux = utils.split_dataset_aux(
            scaled_data, window_size, y_aux
        )

        x_data, y_data = np.array(x_data), np.array(y_data)
        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
        # y_data = np.reshape(y_data, (y_data.shape[0], 1))
        if not scale_target:
            y_data = self.scaler.inverse_transform(y_data)

        self.x = torch.tensor(x_data).float()
        self.y = torch.tensor(y_data).float()
        self.y_aux = torch.tensor(y_aux).float()

    def __len__(self) -> int:
        """Returns a length of a dataset.

        Returns:
            int: Length of a dataset
        """
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple:
        """Gets individual item from a dataset.

        Args:
            idx (int): Index of data to fetch.

        Returns:
            tuple: Tuple with the following elements:
                - input data
                - label data
                - auxilary label data
                    (daily returns, volatility, up/down change)
        """
        x_data = self.x[idx]
        y_data = self.y[idx]
        y_aux = self.y_aux[idx]

        if self.transform:
            x_data = self.transform(x_data)
        if self.target_transform:
            y_data = self.target_transform(y_data)

        return x_data, [y_data, y_aux[0], y_aux[1], y_aux[2]]
