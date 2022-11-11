"""This module contains a simple dataset for stock close prices."""
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch


class YFinanceDataset(Dataset):
    """Simple dataset with financial stock close prices for a given ticker.
    """
    def __init__(
        self,
        ticker: str,
        window_size: int,
        start_date: str = '2017-01-01',
        end_date: str = '2021-10-01',
        transform=None,
        target_transform=None
    ):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.ticker = ticker
        self.transform = transform
        self.target_transform = target_transform

        stock_df = yf.download(ticker, start=start_date, end=end_date)
        stock_close = stock_df["Close"].values

        scaled_data = self.scaler.fit_transform(stock_close.reshape(-1, 1))

        x_data, y_data = split_dataset(scaled_data, window_size)

        x_data, y_data = np.array(x_data), np.array(y_data)
        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
        y_data = np.reshape(y_data, (y_data.shape[0], 1))

        self.x = torch.tensor(x_data).float()
        self.y = torch.tensor(y_data).float()

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
            tuple: Tuple with input data and label data.
        """
        x_data = self.x[idx]
        y_data = self.y[idx]

        if self.transform:
            x_data = self.transform(x_data)
        if self.target_transform:
            y_data = self.target_transform(y_data)

        return x_data, y_data


def split_dataset(data: np.ndarray, split_size: int) -> tuple:
    """Splits time series into lists of n endogenous and 1 exogenous variables.

    Args:
        data (np.ndarray): Time series data to be split.
        split_size (int): Size of an endogenous data used for forecasting.

    Returns:
        tuple: Tuple with:
            - endogenous data used for forecasting
            - ground truth prediction value
    """
    x_data = []
    y_data = []

    for i in range(split_size, len(data)):
        x_data.append(data[i-split_size: i, 0])
        y_data.append(data[i, 0])

    return x_data, y_data
