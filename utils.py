from collections.abc import Mapping
import numpy as np
import pandas as pd

NULL = frozenset([np.nan, pd.NA, None])


def numeric_cols(data: pd.DataFrame) -> list:
    """Returns a list of all numeric column names.

    Args:
        data (pd.DataFrame): DataFrame to get column names from.

    Returns:
        list: All numeric column names.
    """
    numeric = data.dtypes.map(pd.api.types.is_numeric_dtype)
    return data.columns[numeric].to_list()

def noncat_cols(data: pd.DataFrame) -> list:
    categorical = data.dtypes.map(pd.api.types.is_categorical_dtype)
    return data.columns[~categorical].to_list()

def map_list_likes(data: pd.Series, mapper: dict):
    """Apply `mapper` to elements of elements of `data`.

    Args:
        data (pd.Series): Series containing only list-like elements.
        mapper (dict): Dict-like or callable to apply to elements of elements of `data`.
    """

    def transform(list_):
        if isinstance(mapper, Mapping):
            return [mapper[x] if x not in NULL else x for x in list_]
        else:
            return [mapper(x) if x not in NULL else x for x in list_]

    return data.map(transform, na_action="ignore")
