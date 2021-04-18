import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib import ticker

import utils


def _format_big_number(num, dec):
    abb = ""
    if num != 0:
        mag = np.log10(np.abs(num))
        if mag >= 12:
            num = num / 10 ** 12
            abb = "T"
        elif mag >= 9:
            num = num / 10 ** 9
            abb = "B"
        elif mag >= 6:
            num = num / 10 ** 6
            abb = "M"
        elif mag >= 3:
            num = num / 10 ** 3
            abb = "K"
        num = round(num, dec)
    return f"{num:,.{dec}f}{abb}"


def big_number_formatter(dec=0):
    @ticker.FuncFormatter
    def formatter(num, pos):
        return _format_big_number(num, dec)

    return formatter


def big_money_formatter(dec=0):
    @ticker.FuncFormatter
    def formatter(num, pos):
        return f"${_format_big_number(num, dec)}"

    return formatter


def topn_ranking(
    data: pd.DataFrame,
    rankby: str,
    names: str = None,
    topn: int = 15,
    figsize: tuple = (5, 8),
    **kwargs,
) -> Axes:
    """Plot the top observations sorted by the specified column.

    Args:
        data (pd.DataFrame): Data to plot.
        names (str): Column containing names, titles, or identifiers.
        rankby (str): Column to sort by.
        topn (int, optional): Number of observations to show. Defaults to 15.
        figsize (tuple, optional): Figure size. Defaults to (5, 8).

    Returns:
        Axes: Axes for the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    rank_df = data.sort_values(rankby, ascending=False).head(topn)
    if not names:
        names = rank_df.index.to_numpy()
    ax = sns.barplot(data=rank_df, x=rankby, y=names, ec="gray", ax=ax, **kwargs)
    return ax


def pair_corr_heatmap(
    data, ignore=None, annot=True, thresh=None, figsize=(10, 10), **kwargs
):
    if not ignore:
        ignore = []
    corr_df = data.corr().drop(columns=ignore, index=ignore)
    center = 0
    cbar = True
    if thresh is not None:
        if annot:
            annot = corr_df.values
        corr_df = corr_df > thresh
        cbar = False
        center = None
    mask = np.triu(np.ones_like(corr_df, dtype="int64"), k=0)
    fig, ax = plt.subplots(figsize=figsize)
    return sns.heatmap(
        data=corr_df,
        cmap="vlag",
        mask=mask,
        square=True,
        center=center,
        annot=annot,
        fmt=".2f",
        ax=ax,
        cbar=cbar,
        linewidths=0.1,
        linecolor="k",
        **kwargs,
    )

def multi_dist(
    data: pd.DataFrame, ncols=3, sp_height=5, **kwargs
) -> np.ndarray:
    data = data.loc[:, utils.numeric_cols(data)]
    nrows = round(data.columns.size / ncols)
    figsize = (ncols * sp_height, nrows * sp_height)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ax, column in zip(axs.flat, data.columns):
        ax = sns.histplot(data=data, x=column, ax=ax, **kwargs)
        ax.set_title(f"Distribution of `{column}`")
    return axs

def linearity_scatters(
    data: pd.DataFrame, target: str, ncols=3, sp_height=5, yformatter=None, **kwargs
) -> np.ndarray:
    data = data.loc[:, utils.numeric_cols(data)]
    nrows = round(data.columns.size / ncols)
    figsize = (ncols * sp_height, nrows * sp_height)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=figsize)
    for ax, column in zip(axs.flat, data.columns):
        ax = sns.scatterplot(data=data, x=column, y=target, ax=ax, **kwargs)
        if yformatter:
            ax.yaxis.set_major_formatter(yformatter)
        ax.set_title(f"{column} vs. {target}")
    return axs


def linearity_hists(
    data: pd.DataFrame,
    target: str,
    ncols=3,
    sp_height=5,
    cmap="dark:salmon_r",
    yformatter=None,
) -> np.ndarray:
    data = data.loc[:, utils.numeric_cols(data)]
    nrows = round(data.columns.size / ncols)
    figsize = (ncols * sp_height, nrows * sp_height)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=figsize)
    for ax, column in zip(axs.flat, data.columns):
        ax = sns.histplot(data=data, x=column, y=target, ax=ax, cmap=cmap)
        if yformatter:
            ax.yaxis.set_major_formatter(yformatter)
        ax.set_title(f"{column} vs. {target}")
    return axs


def multi_joint(data: pd.DataFrame, target: str, **kwargs) -> np.ndarray:
    data = data.loc[:, utils.numeric_cols(data)]
    grids = []
    for i, column in enumerate(data.columns):
        g = sns.jointplot(data=data, x=column, y=target, **kwargs)
        g.fig.suptitle(f"{column} vs. {target}")
        g.fig.subplots_adjust(top=0.9)
        grids.append(g)
    return np.array(grids)


def heated_barplot(
    data: pd.Series, desat: float = 0.6, ax: Axes = None, figsize: tuple = (8, 10)
) -> Axes:
    """Plot a sharply divided ranking of positive and negative values.

    Args:
        data (pd.Series): Data to plot.
        desat (float, optional): Saturation of bar colors. Defaults to 0.6.
        ax (Axes, optional): Axes to plot on. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (8, 10).

    Returns:
        Axes: Axes for the plot.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    data.index = data.index.astype(str)
    data.sort_values(ascending=False, inplace=True)
    blues = sns.color_palette("Blues", (data <= 0).sum(), desat=desat)
    reds = sns.color_palette("Reds_r", (data > 0).sum(), desat=desat)
    palette = reds + blues
    ax = sns.barplot(
        x=data.values, y=data.index, palette=palette, orient="h", ec="gray", ax=ax
    )
    ax.axvline(0.0, color="gray", lw=1, ls="-")
    return ax


def cat_correlation(crosstab: pd.DataFrame, other: pd.Series, **kwargs) -> Axes:
    """Make a heated bar plot of the correlation between a crosstab and `other`.

    Args:
        crosstab (pd.DataFrame): Crosstab frequency table for categorical variables.
        other (pd.Series): Data for correlation. Must share index with `crosstab`.

    Returns:
        Axes: Axes for the plot.
    """
    corr = crosstab.corrwith(other).dropna().sort_values(ascending=False)
    ax = heated_barplot(corr, **kwargs)
    ax.set_xlabel("Correlation", labelpad=15)
    return ax


def cat_corr_by_bins(
    corr: pd.DataFrame,
    bin1: str,
    bin2: str,
    interval1: pd.Interval,
    interval2: pd.Interval,
    suptitle: str,
    **kwargs,
) -> np.array:
    """Plot correlation data for two bins side-by-side.

    Args:
        corr (pd.DataFrame): Table of correlations indexed by bin.
        bin1 (str): Row index for left plot.
        bin2 (str): Row index for right plot.
        interval1 (pd.Interval): Interval for left plot.
        interval2 (pd.Interval): Interval for right plot.
        suptitle (str): Title of figure.

    Returns:
        np.array: Array of Axes.
    """
    fig, axs = plt.subplots(ncols=2, sharex=True, figsize=(15, 10))
    bins = [bin1, bin2]
    intervals = [interval1, interval2]
    for bin_, interval, ax in zip(bins, intervals, axs.flat):
        data = corr.loc[bin_].dropna().sort_values(ascending=False)
        ax = heated_barplot(data, ax=ax, **kwargs)
        left = round(interval.left)
        right = round(interval.right)
        ax.set_title(f"{bin_}\n\${left:,.0f} to \${right:,.0f}")
        ax.set_xlabel("Correlation", labelpad=15)
        ax.set_ylabel(None)

    fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()
    return axs


def boolean_violinplots(
    crosstab: pd.DataFrame,
    y_series: pd.Series,
    suptitle: str,
    xlabels: list = None,
    ylabel: str = None,
    include: list = None,
    figsize: tuple = (12, 8),
    **kwargs,
) -> np.array:
    """Create multiple violin plots showing distributions for True and False.

    Args:
        crosstab (pd.DataFrame): Crosstab frequency table for categorical variables.
        y_series (pd.Series): Data for y-axis.
        suptitle (str): Figure title.
        xlabels (list, optional): Labels for x-axes. Defaults to None.
        ylabel (str, optional): Label for y-axis. Defaults to None.
        include (list, optional): Columns of `crosstab` to plot. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (12, 8).

    Returns:
        np.array: Array of Axes.
    """
    ncols = 2
    nrows = int(np.ceil(crosstab.shape[1] / 2))
    if include:
        crosstab = crosstab.loc[:, include]
        nrows = int(np.ceil(len(include) / 2))
    corr = crosstab.corrwith(y_series)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=figsize)
    for i, ax in enumerate(axs.flat):
        ax = sns.violinplot(x=crosstab.iloc[:, i], y=y_series, ax=ax, **kwargs)
        ax.set_ylabel(None)
        if xlabels:
            ax.set_xlabel(xlabel[i])
        cat_corr = np.round(corr.iloc[i], 2)
        text = f"Corr: {cat_corr}"
        ax.text(
            0.975,
            1.025,
            text,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
        )
    if ylabel:
        for ax in axs[:, 0]:
            ax.set_ylabel(ylabel, labelpad=10)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return axs
