"""Module containing utils for dataset."""
import numpy as np
import pandas as pd


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


def split_dataset_aux(
    data: np.ndarray, split_size: int, aux_targets: np.ndarray = None
) -> tuple:
    """Splits time series into lists of n endogenous and 1 exogenous variables.

    Args:
        data (np.ndarray): Time series data to be split.
        split_size (int): Size of an endogenous data used for forecasting.
        aux_targets (np.ndarray): Array with auxilary targets where
            0 dimension is of the same lentgh as `data` and 1 dimension
            contains targets for a fiven timestep.

    Returns:
        tuple: Tuple with:
            - endogenous data used for forecasting
            - ground truth prediction value
            - auxilary targets
    """
    x_data = []
    y_data = []
    y_aux = []

    if aux_targets is not None:
        aux_targets = aux_targets.tolist()

    for i in range(split_size, len(data)):
        x_data.append(data[i-split_size: i, 0])
        y_data.append([data[i, 0]])
        if aux_targets:
            y_aux.append(aux_targets[i])

    return x_data, y_data, y_aux


def get_daily_returns(data: np.ndarray) -> np.ndarray:
    """Calculates daily returns.

    Args:
        data (np.ndarray): Time series data.

    Returns:
        np.ndarray: Daily returns.
    """
    daily_return = data[1:] / data[:-1] - 1
    daily_return = np.insert(daily_return, 0, np.nan, axis=0)
    return daily_return


def get_pos_neg_change(data: np.ndarray) -> np.ndarray:
    """Calculates up/down daily change.

    Args:
        data (np.ndarray): Time series data.

    Returns:
        np.ndarray: Up/down daily change.
    """
    change_direction = (data[1:] > data[:-1]).astype(float)
    change_direction = np.insert(change_direction, 0, np.nan, axis=0)
    return change_direction


def get_volatility(data: np.ndarray, window_size: int) -> np.ndarray:
    """Calculates moving standard deviation for a given window size.

    Args:
        data (np.ndarray): Time series data.
        window_size (int): Size of the window for rolling std.

    Returns:
        np.ndarray: Moving standard deviation for a given window size.
    """
    return pd.Series(data).rolling(window_size, closed="left").std().to_numpy()
