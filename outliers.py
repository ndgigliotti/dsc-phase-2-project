from functools import singledispatch

import numpy as np
import pandas as pd
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler

import utils


def get_iqr(data: pd.Series, log=False):
    if log:
        data = np.log10(data)
    q1 = data.quantile(0.25, interpolation="midpoint")
    q3 = data.quantile(0.75, interpolation="midpoint")
    return q3 - q1


def iqr_fences(data: pd.Series, log=False) -> tuple:
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
    if pd.api.types.is_integer_dtype(data.dtype):
        lower_fence, upper_fence = np.round([lower_fence, upper_fence]).astype(
            data.dtype
        )
    return lower_fence, upper_fence


@singledispatch
def _print_drop_report(outliers: pd.DataFrame) -> None:
    report = outliers.sum()
    n_dropped = outliers.any(axis=1).sum()
    report["total"] = n_dropped
    report = report.to_frame("n_dropped")
    report["pct_dropped"] = (report.squeeze() / outliers.shape[0]) * 100
    print("Drop Results\n")
    print(report)


@_print_drop_report.register
def _(outliers: pd.Series) -> None:
    report = pd.Series(data=[outliers.sum()], index=["n_dropped"])
    report["pct_dropped"] = (report["n_dropped"] / outliers.size) * 100
    print("Drop Results\n")
    print(report.to_frame("observations"))


@singledispatch
def _print_clip_report(before: pd.DataFrame, after: pd.DataFrame) -> None:
    changes = before.compare(after)
    report = changes.count().unstack()["self"]
    report["total"] = changes.shape[0]
    report = report.to_frame("n_clipped")
    report["pct_clipped"] = (report.squeeze() / before.shape[0]) * 100
    print("Clip Results\n")
    print(report)


@_print_clip_report.register
def _(before: pd.Series, after: pd.Series) -> None:
    changes = before.compare(after)
    report = pd.Series(data=[changes.shape[0]], index=["n_clipped"])
    report["pct_clipped"] = (report["n_clipped"] / before.shape[0]) * 100
    print("Clip Results\n")
    print(report.to_frame("observations"))


@singledispatch
def iqr_outliers(data: pd.Series, log=False) -> pd.Series:
    lower, upper = iqr_fences(data, log=log)
    return (data < lower) | (data > upper)


@iqr_outliers.register
def _(data: pd.DataFrame, log=False) -> pd.DataFrame:
    return data.apply(iqr_outliers, log=log)


def _jitter(shape, dist, dtype=np.float64, positive=True):
    rng = np.random.default_rng()
    if pd.api.types.is_float_dtype(dtype):
        if positive:
            jitter = rng.uniform(0, dist, shape).astype(dtype)
        else:
            jitter = rng.uniform(dist * -1, dist, shape).astype(dtype)
    elif pd.api.types.is_integer_dtype(dtype):
        if not pd.api.types.is_integer(dist):
            raise TypeError(f"`dist` must be integer if `dtype` is integer")
        if positive:
            jitter = rng.integers(dist, size=shape, dtype=dtype, endpoint=True)
        else:
            jitter = rng.integers(
                dist * -1, high=dist, size=shape, dtype=dtype, endpoint=True
            )
    else:
        raise ValueError("`dtype` must be either int or float dtype")
    return jitter


def _jitter_like(data: pd.Series, dist: float, positive=True):
    return _jitter(data.shape, dist, dtype=data.dtype, positive=positive)


@singledispatch
def _derive_outliers(unclipped: pd.Series, clipped: pd.Series):
    outliers = unclipped.compare(clipped, keep_shape=True).notnull()
    outliers = outliers.drop(columns="self").squeeze()
    outliers.name = "outliers"
    return outliers


@_derive_outliers.register
def _(unclipped: pd.DataFrame, clipped: pd.DataFrame):
    outliers = unclipped.compare(clipped, keep_shape=True).notnull()
    outliers = outliers.T.reset_index(level=1, drop=True).T
    return outliers


def _get_former_outliers(unclipped: pd.Series, clipped: pd.Series):
    outliers = _derive_outliers(unclipped, clipped)
    limits = clipped[outliers].unique()
    upper = unclipped.max()
    lower = unclipped.min()
    if limits.size == 0:
        pass
    elif limits.size == 1:
        if (clipped < limits[0]).sum() > 0:
            upper = limits[0]
        else:
            lower = limits[0]
    elif limits.size == 2:
        lower = limits.min()
        upper = limits.max()
    else:
        raise ValueError("former outliers in `clipped` have more than 2 unique values")
    lower_end = clipped[unclipped < lower]
    upper_end = clipped[unclipped > upper]
    return lower_end, upper_end


def _jitter_clipped_outliers(unclipped: pd.Series, clipped: pd.Series, dist: float):
    lower_end, upper_end = _get_former_outliers(unclipped, clipped)
    lower_jitter = _jitter_like(lower_end, dist)
    upper_jitter = _jitter_like(upper_end, dist)
    jittered = clipped.copy()
    jittered[lower_end.index] += lower_jitter
    jittered[upper_end.index] -= upper_jitter
    return jittered


@singledispatch
def iqr_clip(data: pd.Series, jitter=True, dist=None, log=False, silent=False):
    lower, upper = iqr_fences(data, log=log)
    clipped = data.clip(lower=lower, upper=upper)
    if not silent:
        _print_clip_report(data, clipped)
    if jitter:
        dist = dist or get_iqr(data, log=log)
        if pd.api.types.is_integer_dtype(data.dtype):
            dist = round(dist)
        clipped = _jitter_clipped_outliers(data, clipped, dist)
    return clipped


@iqr_clip.register
def _(data: pd.DataFrame, jitter=True, log=False, silent=False) -> pd.DataFrame:
    clipped = data.apply(iqr_clip, jitter=True, log=log, silent=True)
    if not silent:
        _print_clip_report(data, clipped)
    return clipped


def iqr_clip_demo(data: pd.Series, dist=None, log=False):
    clipped_jit = iqr_clip(data, jitter=True, dist=dist, silent=True, log=log)
    clipped_hard = iqr_clip(data, jitter=False, silent=True, log=log)
    dropped = iqr_drop(data, silent=True, log=log)
    outliers = _derive_outliers(data, clipped_jit)
    outliers.name = outliers.name.title()
    return outliers, (data, clipped_jit, clipped_hard, dropped)


@singledispatch
def iqr_drop(data: pd.DataFrame, log=False, silent=False) -> pd.DataFrame:
    outliers = iqr_outliers(data, log=log)
    if not silent:
        _print_drop_report(outliers)
    return data.loc[~outliers.any(axis=1)].copy()


@iqr_drop.register
def _(data: pd.Series, log=False, silent=False) -> pd.Series:
    outliers = iqr_outliers(data, log=log)
    if not silent:
        _print_drop_report(outliers)
    return data.loc[~outliers].copy()


@singledispatch
def winsorize(
    data: pd.DataFrame, limits=(0.05, 0.05), silent=False, **kwargs
) -> pd.DataFrame:
    clipped = data.apply(mstats.winsorize, raw=True, limits=limits, **kwargs)
    if not silent:
        _print_clip_report(data, clipped)
    return clipped


@winsorize.register
def _(data: pd.Series, limits=(0.05, 0.05), silent=False, **kwargs) -> pd.Series:
    clipped = winsorize(data.to_frame(), silent=True, **kwargs).squeeze()
    if not silent:
        _print_clip_report(data, clipped)
    return clipped


@singledispatch
def z_outliers(data: pd.DataFrame, thresh=3, log=False) -> pd.DataFrame:
    if log:
        data = np.log10(data)
    ss = StandardScaler()
    z_data = ss.fit_transform(data)
    z_data = pd.DataFrame(z_data, index=data.index, columns=data.columns)
    return z_data.abs() > thresh


@z_outliers.register
def _(data: pd.Series, thresh=3, log=False) -> pd.Series:
    return z_outliers(data.to_frame(), thresh=thresh, log=log).squeeze()


@singledispatch
def z_clip(data: pd.DataFrame, thresh=3, log=False, silent=False) -> pd.DataFrame:
    if log:
        data = np.log10(data)
    ss = StandardScaler()
    z_data = ss.fit_transform(data)
    clipped = np.clip(z_data, -1 * thresh, thresh)
    clipped = ss.inverse_transform(clipped)
    if log:
        clipped = 10 ** clipped
    clipped = pd.DataFrame(clipped, index=data.index, columns=data.columns)
    if not silent:
        _print_clip_report(data, clipped)
    return clipped


@z_clip.register
def _(data: pd.Series, thresh=3, log=False, silent=False) -> pd.Series:
    clipped = z_clip(data.to_frame(), thresh=thresh, log=log, silent=True).squeeze()
    if not silent:
        _print_clip_report(data, clipped)
    return clipped


@singledispatch
def z_drop(data: pd.DataFrame, thresh=3, log=False, silent=False) -> pd.DataFrame:
    outliers = z_outliers(data, thresh=thresh, log=log)
    if not silent:
        _print_drop_report(outliers)
    return data.loc[~outliers.any(axis=1)].copy()


@z_drop.register
def _(data: pd.Series, thresh=3, log=False, silent=False) -> pd.Series:
    outliers = z_outliers(data, thresh=thresh, log=log)
    if not silent:
        _print_drop_report(outliers)
    return data.loc[~outliers].copy()
