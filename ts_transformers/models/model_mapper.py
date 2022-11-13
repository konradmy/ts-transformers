"""This module contains mapper from model name to a model class."""
from .transformer import DirTSTransformer, TSTransformer


model_mapper = {
    "ts_simple_transformer": TSTransformer,
    "direction_transformer": DirTSTransformer
}
