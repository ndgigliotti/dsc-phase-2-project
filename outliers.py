from functools import singledispatch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import mstats
import utils


def iqr_fences(data: pd.Series, log=False) -> pd.Series:
    if log:
        data = np.log10(data)
    q1 = data.quantile(0.25, interpolation="midpoint")
    q3 = data.quantile(0.75, interpolation="midpoint")
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    if log:
        lower_fence = 10 ** lower_fence
        upper_fence = 10 ** upper_fence
    return lower_fence, upper_fence


@singledispatch
def trim(data: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    report = outliers.sum()
    n_dropped = outliers.any(axis=1).sum()
    report["Overlapping"] = report.sum() - n_dropped
    report["Total Dropped"] = n_dropped
    print("Trim Results\n")
    print(report.to_frame("Observations"))
    return data.loc[~outliers.any(axis=1)].copy()


@trim.register
def _(data: pd.Series, outliers: pd.Series) -> pd.Series:
    print(f"Trim Results: {outliers.sum()} dropped")
    return data.loc[~outliers].copy()


@singledispatch
def iqr_outliers(data: pd.Series, log=False) -> pd.Series:
    lower, upper = iqr_fences(data, log=log)
    return (data < lower) | (data > upper)


@iqr_outliers.register
def _(data: pd.DataFrame, log=False) -> pd.DataFrame:
    # data = data.loc[:, utils.numeric_cols(data)]
    return data.apply(iqr_outliers, log=log)


@singledispatch
def iqr_clip(data: pd.Series, log=False):
    lower, upper = iqr_fences(data, log=log)
    return data.clip(lower=lower, upper=upper)


@iqr_clip.register
def _(data: pd.DataFrame, log=False) -> pd.DataFrame:
    return data.apply(iqr_clip, log=log)


def iqr_trim(data: pd.DataFrame, log=False) -> pd.DataFrame:
    outliers = iqr_outliers(data, log=log)
    return trim(data, outliers)


@singledispatch
def winsorize(data: pd.DataFrame, limits=(0.05, 0.05), **kwargs):
    return data.apply(mstats.winsorize, raw=True, limits=limits, **kwargs)


@winsorize.register
def _(data: pd.Series, **kwargs):
    data = winsorize(data.to_frame(), **kwargs)
    return data.squeeze()


@singledispatch
def z_outliers(data: pd.DataFrame, thresh=3, log=False) -> pd.DataFrame:
    if log:
        data = np.log10(data)
    ss = StandardScaler()
    # data = data.loc[:, utils.numeric_cols(data)]
    z_data = ss.fit_transform(data)
    z_data = pd.DataFrame(z_data, index=data.index, columns=data.columns)
    return z_data.abs() > thresh


@z_outliers.register
def _(data: pd.Series, thresh=3, log=False) -> pd.Series:
    return z_outliers(data.to_frame(), thresh=thresh, log=log).squeeze()


def z_clip(data: pd.DataFrame, thresh=3, log=False) -> pd.DataFrame:
    if log:
        data = np.log10(data)
    ss = StandardScaler()
    # data = data.loc[:, utils.numeric_cols(data)]
    z_data = ss.fit_transform(data)
    clipped = np.clip(z_data, -1 * thresh, thresh)
    clipped = ss.inverse_transform(clipped)
    if log:
        clipped = 10 ** clipped
    clipped = pd.DataFrame(clipped, index=data.index, columns=data.columns)
    return clipped


def z_trim(data: pd.DataFrame, thresh=3, log=False) -> pd.DataFrame:
    outliers = z_outliers(data, thresh=thresh, log=log)
    return trim(data, outliers)
