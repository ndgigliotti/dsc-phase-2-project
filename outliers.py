from functools import singledispatch
import pandas as pd
import numpy as np
from scipy.stats import mstats
import utils


def iqr_fences(data: pd.Series, log=False) -> pd.Series:
    if log:
        data = np.log(data)
    q1 = data.quantile(0.25, interpolation="midpoint")
    q3 = data.quantile(0.75, interpolation="midpoint")
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    if log:
        lower_fence = np.e ** lower_fence
        upper_fence = np.e ** upper_fence
    return lower_fence, upper_fence


@singledispatch
def iqr_outliers(data: pd.Series, log=False) -> pd.Series:
    lower, upper = iqr_fences(data, log=log)
    return (data < lower) | (data > upper)


@iqr_outliers.register
def _(data: pd.DataFrame, log=False) -> pd.DataFrame:
    data = data.loc[:, utils.numeric_cols(data)]
    return data.apply(iqr_outliers, log=log)


@singledispatch
def iqr_clip(data: pd.Series, log=False):
    lower, upper = iqr_fences(data, log=log)
    return data.clip(lower=lower, upper=upper)


@iqr_clip.register
def _(data: pd.DataFrame, log=False):
    return data.apply(iqr_clip, log=log)


@singledispatch
def winsorize(data: pd.DataFrame, limits=(0.05, 0.05), **kwargs):
    return data.apply(mstats.winsorize, raw=True, limits=limits, **kwargs)


@winsorize.register
def _(data: pd.Series, **kwargs):
    data = winsorize(data.to_frame(), **kwargs)
    return data.squeeze()