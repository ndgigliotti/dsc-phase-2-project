import re
from functools import singledispatch
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import ticker
from matplotlib.axes import Axes

import utils
import outliers

HEATMAP_STYLE = MappingProxyType(
    {
        "square": True,
        "annot": True,
        "fmt": ".2f",
        "cbar": False,
        "center": 0,
        "cmap": "vlag",
        "linewidths": 0.1,
        "linecolor": "k",
    }
)


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


def add_tukey_marks(
    data, ax, iqr_color="r", fence_color="k", fence_style="--", show_quarts=False
):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    ax.axvspan(q1, q3, color=iqr_color, alpha=0.2)
    iqr_mp = q1 + ((q3 - q1) / 2)
    lower, upper = outliers.iqr_fences(data)
    ax.axvline(lower, c=fence_color, ls=fence_style)
    ax.axvline(upper, c=fence_color, ls=fence_style)
    text_yval = ax.get_ylim()[1]
    text_yval *= 1.01
    ax.text(iqr_mp, text_yval, "IQR", ha="center")
    if show_quarts:
        ax.text(q1, text_yval, "Q1", ha="center")
        ax.text(q3, text_yval, "Q3", ha="center")
    ax.text(upper, text_yval, "Fence", ha="center")
    ax.text(lower, text_yval, "Fence", ha="center")
    return ax


@singledispatch
def rotate_ticks(ax: Axes, deg: float, axis: str = "x"):
    get_labels = getattr(ax, f"get_{axis}ticklabels")
    for label in get_labels():
        label.set_rotation(deg)


@rotate_ticks.register
def _(ax: np.ndarray, deg: float, axis: str = "x"):
    axs = ax
    for ax in axs:
        rotate_ticks(ax, deg=deg, axis=axis)


def topn_ranking(
    data: pd.DataFrame,
    rankby: str,
    names: str = None,
    topn: int = 15,
    orient: str = "h",
    figsize: tuple = (5, 8),
    reverse: bool = False,
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
    data.index = data.index.astype(str)
    rank_df = data.sort_values(rankby, ascending=reverse).head(topn)
    if not names:
        names = rank_df.index.to_numpy()
    if orient.lower() == "h":
        x, y = rankby, names
    elif orient.lower() == "v":
        y, x = rankby, names
        if figsize == (5, 8):
            figsize = (8, 5)
    else:
        raise ValueError("orient must be 'h' or 'v'")
    ax = sns.barplot(data=rank_df, x=x, y=y, ec="gray", ax=ax, **kwargs)
    if orient.lower() == "v":
        for label in ax.get_xticklabels():
            label.set_rotation(90)
    return ax


def pair_corr_heatmap(
    data, ignore=None, annot=True, thresh=None, figsize=(10, 10), **kwargs
):
    if not ignore:
        ignore = []
    corr_df = data.corr().drop(columns=ignore, index=ignore)
    center = 0
    if thresh is not None:
        if annot:
            annot = corr_df.values
        corr_df = corr_df > thresh
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
        cbar=False,
        linewidths=0.1,
        linecolor="k",
        **kwargs,
    )


def _calc_figsize(nplots, ncols, sp_height):
    nrows = round(nplots / ncols)
    figsize = (ncols * sp_height, nrows * sp_height)
    return nrows, figsize


def multi_dist(data: pd.DataFrame, ncols=3, sp_height=5, **kwargs) -> np.ndarray:
    data = data.loc[:, utils.numeric_cols(data)]
    nrows, figsize = _calc_figsize(data.columns.size, ncols, sp_height)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ax, column in zip(axs.flat, data.columns):
        ax = sns.histplot(data=data, x=column, ax=ax, **kwargs)
        ax.set_title(f"Distribution of `{column}`")
    if axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.set_ylabel(None)
    elif axs.size > 1:
        for ax in axs[1:]:
            ax.set_ylabel(None)
    return axs


def multi_scatter(
    data: pd.DataFrame,
    target: str,
    ncols=3,
    sp_height=5,
    reflexive=False,
    yformatter=None,
    **kwargs,
) -> np.ndarray:
    data = data.select_dtypes(include="number")
    target_data = data.loc[:, target]
    if not reflexive:
        data.drop(columns=target, inplace=True)
    nrows, figsize = _calc_figsize(data.columns.size, ncols, sp_height)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=figsize)
    for ax, column in zip(axs.flat, data.columns):
        ax = sns.scatterplot(x=data[column], y=target_data, ax=ax, **kwargs)
        ax.set_ylabel(target, labelpad=10)
        if yformatter:
            ax.yaxis.set_major_formatter(yformatter)
        ax.set_title(f"{column} vs. {target}")
    return axs


def linearity_scatters(
    data: pd.DataFrame, target: str, ncols=3, sp_height=5, yformatter=None, **kwargs
) -> np.ndarray:
    data = data.loc[:, utils.numeric_cols(data)]
    corr_df = data.corrwith(data[target]).round(2)
    nrows, figsize = _calc_figsize(data.columns.size, ncols, sp_height)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=figsize)
    for ax, column in zip(axs.flat, data.columns):
        ax = sns.scatterplot(data=data, x=column, y=target, ax=ax, **kwargs)
        text = f"r={corr_df[column]:.2f}"
        ax.text(
            0.975,
            1.025,
            text,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        if yformatter:
            ax.yaxis.set_major_formatter(yformatter)
        ax.set_title(f"{column} vs. {target}")
    return axs


def multi_joint(
    data: pd.DataFrame, target: str, reflexive=False, **kwargs
) -> np.ndarray:
    data = data.select_dtypes(include="number")
    grids = []
    columns = data.columns if reflexive else data.columns.drop(target)
    for column in columns:
        g = sns.jointplot(data=data, x=column, y=target, **kwargs)
        g.fig.suptitle(f"{column} vs. {target}")
        g.fig.subplots_adjust(top=0.9)
        grids.append(g)
    return np.array(grids)


def _annot_vbars(ax, color="k"):
    raise NotImplementedError()
    # max_bar = np.abs([b.get_width() for b in ax.patches]).max()
    # dist = 0.15 * max_bar
    # for bar in ax.patches:
    #     x = bar.get_x() + bar.get_width() / 2
    #     y = bar.get_height() - dist
    #     val = round(bar.get_height(), 2)
    #     text = f"{val:,.2f}"
    #     ax.annotate(text, (x, y), ha="center", va="center", c=color, fontsize=14)
    # return ax


def _annot_hbars(ax, color="k"):
    raise NotImplementedError()
    # max_bar = np.abs([b.get_width() for b in ax.patches]).max()
    # dist = 0.15 * max_bar
    # for bar in ax.patches:
    #     x = bar.get_width()
    #     x = x + dist if x < 0 else x - dist
    #     y = bar.get_y() + bar.get_height() / 2
    #     val = round(bar.get_width(), 2)
    #     text = f"{val:,.2f}"
    #     ax.annotate(text, (x, y), ha="center", va="center", c=color, fontsize=14)
    # return ax


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


def diagnostics(
    model,
    height=5,
    xformatter=big_number_formatter(2),
    yformatter=big_number_formatter(2),
):
    fig, (qq, hs) = plt.subplots(ncols=2, figsize=(height * 2, height))
    sm.graphics.qqplot(model.resid, fit=True, line="45", ax=qq)
    qq.set_title("Normality of Residuals")
    hs = sns.scatterplot(x=model.predict(), y=model.resid, s=5)
    hs.set_ylabel("Residuals", labelpad=10)
    hs.set_xlabel("Predicted Values", labelpad=10)
    hs.yaxis.set_major_formatter(yformatter)
    hs.xaxis.set_major_formatter(xformatter)
    for label in hs.get_xticklabels():
        label.set_rotation(45)
    hs.set_title("Homoscedasticity Check")
    fig.tight_layout()
    return np.array([qq, hs])


def cat_palette(name: str, keys: list, offset=0):
    pal = sns.color_palette(name, n_colors=len(keys) + offset)[offset:]
    return dict(zip(keys, pal))


def derive_coeff_labels(coeff_df):
    re_cat = r"C\(\w+\)\[T\.([\w\s]+)\]"
    label = coeff_df.index.to_series(name="label")
    cat_label = label.filter(regex=re_cat, axis=0)
    label.update(cat_label.str.extract(re_cat).squeeze())
    return coeff_df.assign(label=label)


def simple_barplot(data, x, y, estimator=np.mean, figsize=(5, 5), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(
        data=data,
        x=x,
        y=y,
        estimator=estimator,
        ax=ax,
        **kwargs
    )
    est_name = estimator.__name__.title()
    ax.set_title(f"{est_name} {y.title()} by {x.title()}", pad=10)
    ax.set_xlabel(x.title(), labelpad=10)
    ax.set_ylabel(y.title(), labelpad=15)
    return ax


def coeffs_endog_barplot(main_df, coeff_df, exog, endog, estimator=np.median):
    if "label" not in coeff_df.columns:
        coeff_df = derive_coeff_labels(coeff_df)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    uniq_exog = main_df[exog].sort_values().unique()
    # pal = pd.Series(sns.color_palette("deep", uniq_exog.size), index=uniq_exog)
    pal = cat_palette("deep", uniq_exog)
    coeff_df = coeff_df.filter(like=exog, axis=0)
    coeff_df = coeff_df.assign(label=coeff_df.label.astype(uniq_exog.dtype))
    coeff_df.sort_values("label", inplace=True)
    ax1 = sns.barplot(
        data=coeff_df,
        x="label",
        y="coeff",
        palette=pal,
        ax=ax1,
    )
    ax2 = sns.barplot(
        data=main_df,
        x=exog,
        y=endog,
        estimator=estimator,
        palette=pal,
        ax=ax2,
    )
    ax1.set_ylabel(f"Effect on {endog.title()}", labelpad=10)
    ax2.set_ylabel(endog.title(), labelpad=10)
    ax1.set_title(f"Projected Effects of {exog.title()} on {endog.title()}", pad=10)
    est_name = estimator.__name__.title()
    ax2.set_title(f"{est_name} {endog.title()} by {exog.title()}", pad=10)
    for ax in (ax1, ax2):
        ax.set_xlabel(exog.title())
    fig.tight_layout()
    return fig