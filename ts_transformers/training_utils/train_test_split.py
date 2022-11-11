"""Module containing a function for TS dataset train-test-val split"""
import numpy as np
from torch import randperm
from torch.utils.data import Dataset, Subset


def train_test_split(
    dataset: Dataset, train_rat: float, val_rat: float, shuffle: bool = False
) -> tuple:
    """Splits given dataset into train test val splits based on given ratios.

    Args:
        dataset (Dataset): PyTorch dataset to split.
        train_rat (float): Training ratio.
        val_rat (float): Validation ratio.
        shuffle (bool): If True, shuffle splits. Defaults to False.

    Raises:
        ValueError: Raised when given ratios are lower than 0.
        ValueError: Raised when sum of given ratios is greater than 1.

    Returns:
        tuple: Tuple with split datasets.
    """
    ds_len = len(dataset)
    if train_rat < 0 or val_rat < 0:
        raise ValueError("Split ratio cannot be smaller than 0.")

    if train_rat + val_rat > 1:
        raise ValueError("Split ratios cannot be greater than 1")

    train_size = round(ds_len * train_rat)
    val_size = round(ds_len * val_rat)
    test_size = ds_len - train_size - val_size

    train_indices = _get_split_idx(id_end=train_size, shuffle=shuffle)
    train_ds = Subset(dataset, train_indices)

    if val_size > 0:
        val_indices = _get_split_idx(
            train_size + val_size, train_size, shuffle
        )
        val_ds = Subset(dataset, val_indices)
    else:
        val_ds = None

    if test_size > 0:
        test_indices = _get_split_idx(ds_len, train_size + val_size, shuffle)
        test_ds = Subset(dataset, test_indices)
    else:
        test_ds = None

    return train_ds, val_ds, test_ds


def _get_split_idx(
    id_end: int, id_start: int = 0, shuffle: bool = False
) -> list:
    """Generates a list with indices with optional permutation.

    Args:
        id_end (int): End index.
        id_start (int): Start index. Defaults to 0.
        shuffle (bool): If True indices will be permuted. Defaults to False.

    Returns:
        list: List with indices with optional permutation.
    """
    idx = np.array(range(id_start, id_end))
    if shuffle:
        idx = idx[randperm(idx.shape[0])]
    return idx
