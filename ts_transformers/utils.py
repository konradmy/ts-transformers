"""This module contains general utility functions.
"""
import os
import json
import pickle
from typing import Any


def load_json(file_path: str) -> dict:
    """Loads json to a dictionary from a given path.

    Args:
        file_path (str): File path to a json file.

    Returns:
        dict: Loaded json gile.
    """
    with open(file_path, "r", encoding="utf-8") as in_file:
        json_content = json.load(in_file)
    return json_content


def to_json(dict_obj: dict, file_path: str):
    """Save given dictionary to a JSON file.

    Args:
        dict_obj (dict): Dictionary to store as a JSON.
        file_path (str): File path to give dictionary..
    """
    with open(file_path, "w", encoding="utf-8") as in_file:
        json.dump(dict_obj, in_file)


def load_pickle(file_path: str) -> Any:
    """Loads pickle object from a given file path.

    Args:
        file_path (str): File path to saved pickled object.

    Returns:
        Any: Loaded pickled object.
    """
    with open(file_path, "rb") as f_in:
        pickled_obj = pickle.load(f_in)
    return pickled_obj


def to_pickle(obj: Any, file_path: str):
    """Pickles given object under given file path.

    Args:
        obj (Any): Object to pickle.
        file_path (str): File path to save pickled object.
    """
    with open(file_path, "wb") as f_out:
        pickle.dump(obj, f_out)


def create_directory(directory: str):
    """Creates a directory if it doesn't exist.

    Args:
        directory (str): Directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
