"""This module contains mapper from model name to a model class."""
from .transformer import TSTransformer


model_mapper = {
    "ts_simple_transformer": TSTransformer
}
