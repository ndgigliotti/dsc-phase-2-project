from functools import singledispatch

import numpy as np
import pandas as pd
from pandas.api.types import is_integer, is_integer_dtype, is_float_dtype
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler

import utils

_rng = np.random.default_rng(42)

def get_iqr(data: pd.Series):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    return q3 - q1


def iqr_fences(data: pd.Series) -> tuple:
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    return lower_fence, upper_fence


def _display_report(outliers, verb):
    if isinstance(outliers, pd.Series):
        outliers = outliers.to_frame()
    report = outliers.sum()
    n_modified = outliers.any(axis=1).sum()
    report["total_obs"] = n_modified
    report = report.to_frame(f"n_{verb}")
    report[f"pct_{verb}"] = (report.squeeze() / outliers.shape[0]) * 100
    display(report)


@singledispatch
def iqr_outliers(data: pd.Series) -> pd.Series:
    lower, upper = iqr_fences(data)
    return (data < lower) | (data > upper)


@iqr_outliers.register
def _(data: pd.DataFrame) -> pd.DataFrame:
    return data.apply(iqr_outliers)


def _jitter(shape, dist, dtype=np.float64, positive=True):
    if is_float_dtype(dtype):
        if positive:
            jitter = _rng.uniform(0, dist, shape).astype(dtype)
        else:
            jitter = _rng.uniform(dist * -1, dist, shape).astype(dtype)
    elif is_integer_dtype(dtype):
        dist = round(dist)
        if positive:
            jitter = _rng.integers(dist, size=shape, dtype=dtype, endpoint=True)
        else:
            jitter = _rng.integers(
                dist * -1, high=dist, size=shape, dtype=dtype, endpoint=True
            )
    else:
        raise ValueError(f"`dtype` must be either int or float dtype, got {dtype}")
    return jitter


def _jitter_like(data: pd.Series, dist: float, positive=True):
    return _jitter(data.shape, dist, dtype=data.dtype, positive=positive)


def _jitter_clipped(
    clipped: pd.Series, lower_outs: pd.Series, upper_outs: pd.Series, dist: float
):
    lower_jitter = _jitter_like(clipped[lower_outs], dist)
    upper_jitter = _jitter_like(clipped[upper_outs], dist)
    jittered = clipped.copy()
    jittered[lower_outs] += lower_jitter
    jittered[upper_outs] -= upper_jitter
    return jittered


@singledispatch
def iqr_clip(data: pd.Series, jitter=0, silent=False):
    lower, upper = iqr_fences(data)
    clipped = data.clip(lower=lower, upper=upper)
    if is_integer_dtype(data):
        clipped = clipped.round().astype(data.dtype)
    if not silent:
        outliers = (data < lower) | (data > upper)
        _display_report(outliers, "clipped")
    if jitter:
        if is_integer_dtype(data):
            jitter = round(jitter)
        clipped = _jitter_clipped(clipped, data < lower, data > upper, jitter)
    return clipped


@iqr_clip.register
def _(data: pd.DataFrame, jitter=0, silent=False) -> pd.DataFrame:
    clipped = data.apply(iqr_clip, jitter=jitter, silent=True)
    if not silent:
        _display_report(iqr_outliers(data), "clipped")
    return clipped


@singledispatch
def iqr_tuck(data: pd.Series, silent=False):
    lower, upper = iqr_fences(data)
    lower_outs = data < lower
    upper_outs = data > upper
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    if is_float_dtype(data):
        lower_vals = _rng.uniform(low=lower, high=q1, size=lower_outs.sum())
        res = np.finfo(np.float64).resolution
        upper_vals = _rng.uniform(low=q3 + res, high=upper + res, size=upper_outs.sum())
    elif is_integer_dtype(data):
        lower_vals = _rng.integers(lower, high=q1, size=lower_outs.sum())
        upper_vals = _rng.integers(q3 + 1, high=upper + 1, size=upper_outs.sum())
    else:
        raise TypeError("`data` must have either float or integer dtype")
    tucked = data.copy()
    tucked[lower_outs] = lower_vals
    tucked[upper_outs] = upper_vals
    if not silent:
        _display_report(lower_outs | upper_outs, "tucked")
    return tucked


@iqr_tuck.register
def _(data: pd.DataFrame, silent=False):
    tucked = data.apply(iqr_tuck, silent=True)
    if not silent:
        _display_report(iqr_outliers(data), "tucked")
    return tucked


def iqr_tuck_demo(data: pd.Series):
    outliers = iqr_outliers(data)
    tucked = iqr_tuck(data)
    clipped = iqr_clip(data, jitter=False, silent=True)
    dropped = iqr_drop(data, silent=True)
    outliers.name = outliers.name.title()
    return outliers, (data, tucked, clipped, dropped)


@singledispatch
def iqr_drop(data: pd.DataFrame, silent=False) -> pd.DataFrame:
    outliers = iqr_outliers(data)
    if not silent:
        _display_report(outliers, "dropped")
    return data.loc[~outliers.any(axis=1)].copy()


@iqr_drop.register
def _(data: pd.Series, silent=False) -> pd.Series:
    outliers = iqr_outliers(data)
    if not silent:
        _display_report(outliers, "dropped")
    return data.loc[~outliers].copy()


@singledispatch
def winsorize(
    data: pd.DataFrame, limits=(0.05, 0.05), silent=False, **kwargs
) -> pd.DataFrame:
    clipped = data.apply(mstats.winsorize, raw=True, limits=limits, **kwargs)
    return clipped


@winsorize.register
def _(data: pd.Series, limits=(0.05, 0.05), silent=False, **kwargs) -> pd.Series:
    clipped = winsorize(data.to_frame(), silent=True, **kwargs).squeeze()
    return clipped


@singledispatch
def z_outliers(data: pd.DataFrame, thresh=3) -> pd.DataFrame:
    ss = StandardScaler()
    z_data = ss.fit_transform(data)
    z_data = pd.DataFrame(z_data, index=data.index, columns=data.columns)
    return z_data.abs() > thresh


@z_outliers.register
def _(data: pd.Series, thresh=3) -> pd.Series:
    return z_outliers(data.to_frame(), thresh=thresh).squeeze()


@singledispatch
def z_clip(data: pd.DataFrame, thresh=3, silent=False) -> pd.DataFrame:
    ss = StandardScaler()
    z_data = ss.fit_transform(data)
    clipped = np.clip(z_data, -1 * thresh, thresh)
    clipped = ss.inverse_transform(clipped)
    clipped = pd.DataFrame(clipped, index=data.index, columns=data.columns)
    if not silent:
        _display_report(z_outliers(data), "clipped")
    return clipped


@z_clip.register
def _(data: pd.Series, thresh=3, silent=False) -> pd.Series:
    clipped = z_clip(data.to_frame(), thresh=thresh, silent=True).squeeze()
    if not silent:
        _display_report(z_outliers(data), "clipped")
    return clipped


@singledispatch
def z_drop(data: pd.DataFrame, thresh=3, silent=False) -> pd.DataFrame:
    outliers = z_outliers(data, thresh=thresh)
    if not silent:
        _display_report(outliers, "dropped")
    return data.loc[~outliers.any(axis=1)].copy()


@z_drop.register
def _(data: pd.Series, thresh=3, silent=False) -> pd.Series:
    outliers = z_outliers(data, thresh=thresh)
    if not silent:
        _display_report(outliers, "dropped")
    return data.loc[~outliers].copy()
