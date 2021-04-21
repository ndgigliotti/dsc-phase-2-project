import datetime
import glob
import pickle
import itertools
import os
from functools import partial
from multiprocessing.pool import ThreadPool
from operator import itemgetter

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols

import plotting
import utils

TEST_DIR = "test_models"
MV_SWEEP_DIR = os.path.join(TEST_DIR, "mv_sweep")
PERM_SWEEP_DIR = os.path.join(TEST_DIR, "perm_sweep")


def reg_model(data, formula):
    model = ols(formula=formula, data=data).fit()
    display(model.summary())
    plots = plotting.diagnostics(model)
    return model, plots


def _build_and_record(data, formula, path):
    model = ols(formula=formula, data=data).fit()
    with open(path, "w") as f:
        f.write(model.summary().as_html())


def _format_overall_summary_table(df):
    srs1 = df[[0, 1]].set_index(0).squeeze()
    srs2 = df[[2, 3]].set_index(2).squeeze()
    df = pd.concat([srs1, srs2], axis=0).dropna().to_frame().T.convert_dtypes()
    df.columns = df.columns.to_series().str.replace(":", "")
    return df.T.squeeze()


def _format_params_summary_table(df):
    df.columns = df.loc[0].fillna("param")
    df = df.drop(0).set_index("param").astype("float64")
    return df


def _format_stats_summary_table(df):
    srs1 = df[[0, 1]].set_index(0).squeeze()
    srs2 = df[[2, 3]].set_index(2).squeeze()
    return pd.concat([srs1, srs2], axis=0).to_frame("Stats").astype("float64")


def _format_summary_doc(dfs):
    df1 = _format_overall_summary_table(dfs[0])
    df2 = _format_params_summary_table(dfs[1])
    df3 = _format_stats_summary_table(dfs[2])
    return (df1, df2, df3)


def get_rsquared(sweep_results, drop_trivial=True):
    formulae, dfs = zip(*sweep_results.items())
    dfs = np.asarray(dfs, dtype="object")[:, 0]
    r2 = map(itemgetter("R-squared"), dfs)
    adj_r2 = map(itemgetter("Adj. R-squared"), dfs)
    r2 = pd.Series(r2, index=formulae, name="rsquared")
    adj_r2 = pd.Series(adj_r2, index=formulae, name="adj_rsquared")
    df = pd.concat([r2, adj_r2], axis=1)
    if drop_trivial:
        dependents = {x[: x.find("~")] for x in formulae}
        re_filt = "|".join([f"^{x}~.*{x}.*$" for x in dependents])
        df.drop(df.filter(regex=re_filt, axis=0).index, inplace=True)
    return df


def load_sweep_results(glob_path):
    paths = glob.glob(glob_path)
    with ThreadPool() as pool:
        docs = pool.map(pd.read_html, paths)
    docs = map(_format_summary_doc, docs)
    fnames = map(os.path.basename, paths)
    formulae, _ = zip(*map(os.path.splitext, fnames))
    docs = {x: y for x, y in zip(formulae, docs)}
    return docs

def pickle_sweep_results(sweep_results, dst):
    with open(dst, "wb") as f:
        f.write(pickle.dumps(sweep_results, pickle.HIGHEST_PROTOCOL))


def rfe_feature_ranking(data, target, dummify_cats=False, n_features=None, ignore=None):
    if ignore:
        data = data.drop(columns=ignore)
    predictors = data.drop(columns=target)
    if dummify_cats:
        dummies = pd.get_dummies(predictors.select_dtypes(include="category"))
        predictors = predictors.select_dtypes(include="number")
        predictors = pd.concat([predictors, dummies], axis=1)
    else:
        predictors = predictors.select_dtypes(include="number")
    reg = LinearRegression()
    selector = RFE(reg, n_features_to_select=n_features)
    selector = selector.fit(predictors, data[target])
    selected = selector.get_support(indices=True)
    results = pd.Series(selector.ranking_, index=predictors.columns, name="RFE Ranking")
    return results.sort_values()


def multivariate_sweep(data, target, n_vars=2, ignore=None, dst=MV_SWEEP_DIR):
    if ignore:
        data = data.drop(columns=ignore)
    var_names = utils.noncat_cols(data)
    var_names += [f"C({x})" for x in utils.cat_cols(data)]
    var_names.remove(target)
    combos = list(itertools.combinations(var_names, n_vars))
    combo_strs = ["+".join(x) for x in combos]
    formulae = [f"{target}~{x}" for x in combo_strs]
    dst = os.path.join(dst, f"{target}~{n_vars}")
    os.makedirs(dst, exist_ok=True)
    paths = [os.path.join(dst, f"{x}.html") for x in formulae]
    with ThreadPool() as pool:
        build = partial(_build_and_record, data)
        pool.starmap(build, zip(formulae, paths))


def permutation_sweep(data, formula, dst=PERM_SWEEP_DIR):
    target, predictors = formula.split("~")
    predictors = predictors.split("+")
    perms = itertools.permutations(predictors, len(predictors))
    perm_strs = ["+".join(x) for x in perms]
    formulae = [f"{target}~{x}" for x in perm_strs]
    dst = os.path.join(dst, formula)
    os.makedirs(dst, exist_ok=True)
    paths = [os.path.join(dst, f"{x}.html") for x in formulae]
    with ThreadPool() as pool:
        build = partial(_build_and_record, data)
        pool.starmap(build, zip(formulae, paths))

def corr(frame: pd.DataFrame, other: pd.DataFrame):
    return other.apply(lambda x: frame.corrwith(x))