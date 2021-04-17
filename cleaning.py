import re
import json
import unidecode
from string import punctuation
import pandas as pd
import numpy as np
import utils

RE_PUNCT = re.compile(f"[{re.escape(punctuation)}]")
RE_WHITESPACE = re.compile(r"\s+")


def nan_info(data: pd.DataFrame):
    df = data.isna().sum().to_frame("Total")
    df["Percent"] = (df["Total"] / data.shape[0]) * 100
    return df.sort_values("Total", ascending=False)


def dup_info(data: pd.DataFrame):
    df = data.duplicated().sum().to_frame("Total")
    df["Percent"] = (df["Total"] / data.shape[0]) * 100
    return df.sort_values("Total", ascending=False)


def nan_rows(data: pd.DataFrame):
    return data[data.isna().any(axis=1)]


def dup_rows(data: pd.DataFrame, **kwargs):
    return data[data.duplicated(**kwargs)]


def who_is_nan(data: pd.DataFrame, col: str, name_col: str):
    return nan_rows(data)[data[col].isna()][name_col]


def process_strings(strings: pd.Series) -> pd.Series:
    df = strings.str.lower()
    df = df.str.replace(RE_PUNCT, "").str.replace(RE_WHITESPACE, " ")
    df = df.map(unidecode.unidecode, na_action="ignore")
    return df


def detect_json_list(x):
    return isinstance(x, str) and bool(re.fullmatch(r"\[.*\]", x))


def coerce_list_likes(data):
    if not isinstance(data, pd.Series):
        raise TypeError("`data` must be pd.Series")
    json_strs = data.map(detect_json_list, na_action="ignore")
    clean = data.copy()
    clean[json_strs] = clean.loc[json_strs].map(json.loads)
    list_like = clean.map(pd.api.types.is_list_like)
    clean[~list_like] = clean.loc[~list_like].map(lambda x: [x], na_action="ignore")
    clean = clean.map(list, na_action="ignore")
    return clean


def info(data: pd.DataFrame, round_pct: int = 2) -> pd.DataFrame:
    n_rows = data.shape[0]
    nan = data.isna().sum().to_frame("nan")
    dup = data.apply(lambda x: x.duplicated()).sum().to_frame("dup")
    uniq = data.nunique().to_frame("uniq")
    info = pd.concat([nan, dup, uniq], axis=1)
    pcts = (info / n_rows) * 100
    pcts.columns = pcts.columns.map(lambda x: f"{x}_%")
    pcts = pcts.round(round_pct)
    info = pd.concat([info, pcts], axis=1)
    return info.sort_index(axis=1).sort_values("nan", ascending=False)


def show_uniques(data: pd.DataFrame, include: list = None, cut: int = None):
    if include:
        data = data.loc[:, include]
    if cut:
        data = data.loc[:, data.nunique(dropna=False) <= cut]
    for name, column in data.iteritems():
        df = pd.DataFrame(data=column.unique(), columns=[name])
        df = df.sort_values(name).reset_index(drop=True)
        display(df)