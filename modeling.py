import datetime
import glob
import itertools
import os
import pickle
import re
import shutil
from functools import partial, singledispatch
from multiprocessing.pool import ThreadPool
from operator import itemgetter
from time import perf_counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.regression.linear_model import RegressionResultsWrapper
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols

import plotting
import utils

TEST_DIR = "test_models"
OLS_SWEEP_DIR = os.path.join(TEST_DIR, "ols_sweep")


def ols_model(data, formula):
    model = ols(formula=formula, data=data).fit()
    display(model.summary())
    fig = plotting.diagnostics(model)
    bp = breusch_pagan(model).to_frame("breusch_pagan")
    bad_p = bad_pvalues(model).to_frame("bad_pvalue")
    display(bp)
    display(bad_p)
    return model


def _build_and_record(data, formula, path):
    model = ols(formula=formula, data=data).fit()
    summary = summarize(model)
    summary.to_json(path)


def summarize(model):
    model.summary()
    results = pd.Series(dtype=np.float64)
    results["rsquared"] = model.rsquared
    results["rsquared_adj"] = model.rsquared_adj
    results["fval"] = model.fvalue
    results["f_pval"] = model.f_pvalue
    results["nobs"] = model.nobs
    results = results.append(pd.Series(model.diagn))
    results = results.append(model.pvalues.add_prefix("pval_"))
    results = results.append(model.params.add_prefix("coef_"))
    results = results.append(breusch_pagan(model).add_prefix("bp_"))
    results["bp_hetero"] = results["bp_lm_pval"] < 0.05
    results["high_corr_exog"] = check_multicol(model).sum().sum()
    results["bad_pvals"] = bad_pvalues(model).size
    return results


def load_results(glob_path, jobs=os.cpu_count()):
    paths = glob.glob(glob_path)
    with ThreadPool(jobs) as pool:
        read_json = partial(pd.read_json, typ="Series")
        docs = pool.map(read_json, paths)
    fnames = map(os.path.basename, paths)
    formulae, _ = zip(*map(os.path.splitext, fnames))
    return list(zip(formulae, docs))


def load_results_as_frame(glob_path):
    results = load_results(glob_path)
    df = pd.DataFrame(dict(results))
    df.index.name = "formula"
    return df.loc[df.notnull().all(axis=1)].T


def consolidate_results(glob_path):
    start = perf_counter()
    df = load_results_as_frame(glob_path)
    first_path = glob.glob(glob_path)[0]
    dir_ = os.path.dirname(first_path)
    path = f"{dir_}_summary.json"
    df.to_json(path)
    shutil.make_archive(dir_, "zip", dir_)
    shutil.rmtree(dir_)
    print(utils.elapsed(start))


@singledispatch
def _strip_patsy_cat(feature: str):
    match = re.fullmatch(r"C\((\w+)\)", feature.strip())
    return match.group(1) if match else feature


@_strip_patsy_cat.register
def _(feature: list):
    return list(map(_strip_patsy_cat, feature))


def feature_summary(sweep_results, agg=np.mean, filter_mc=True):
    summ = sweep_results.copy()
    if filter_mc:
        summ = summ.query("high_corr_exog < 1")
    feature = summ.index.to_series(name="feature")
    feature = feature.str.split("~").map(itemgetter(1)).str.split("+")
    feature = feature.map(_strip_patsy_cat)
    summ = summ.join(feature)
    return summ.explode("feature").groupby("feature").agg(agg)


def rfe_feature_ranking(data, target, dummify_cats=False, n_features=None, ignore=None):
    if ignore:
        data = data.drop(columns=ignore)
    predictors = data.drop(columns=target)
    if dummify_cats:
        dummies = pd.get_dummies(
            predictors.select_dtypes(include="category"), drop_first=True
        )
        predictors = predictors.select_dtypes(include="number")
        predictors = pd.concat([predictors, dummies], axis=1)
    else:
        predictors = predictors.select_dtypes(include="number")
    selector = RFE(estimator=LinearRegression(), n_features_to_select=n_features)
    selector = selector.fit(predictors, data[target])
    selected = selector.get_support(indices=True)
    results = pd.Series(selector.ranking_, index=predictors.columns, name="RFE Ranking")
    return results.sort_values()


def seq_feature_selection(data, target, n_features=None):
    predictors = data.drop(columns=target).select_dtypes(np.number)
    selector = SequentialFeatureSelector(
        estimator=LinearRegression(), n_features_to_select=n_features
    )
    selector = selector.fit(predictors, data[target])
    selected = selector.get_support(indices=True)
    return predictors.iloc[:, selected].columns.to_list()


def ols_sweep(
    data, target, n_vars=2, ignore=None, dst=OLS_SWEEP_DIR, jobs=os.cpu_count()
):
    start = perf_counter()
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
    paths = [os.path.join(dst, f"{x}.json") for x in formulae]
    with ThreadPool(jobs) as pool:
        build = partial(_build_and_record, data)
        pool.starmap(build, zip(formulae, paths))
    print(utils.elapsed(start))


def goldfeld_quandt(
    model: RegressionResultsWrapper,
    split: float = 0.45,
    drop: float = 0.1,
    jobs: int = os.cpu_count(),
) -> pd.DataFrame:
    """Run a battery of GQ tests, sorting by each exog variable in `model`.

    Args:
        model (RegressionResultsWrapper): Statsmodels regression results.
        split (float, optional): Fraction of observations for split point. Defaults to 0.45.
        drop (float, optional): Fraction of observations to drop. Defaults to 0.1.
        jobs (int, optional): Number of threads to create. Defaults to os.cpu_count().

    Returns:
        [pd.DataFrame]: DataFrame of results for each exog variable.
    """
    resid = model.resid
    exog = model.model.data.orig_exog
    resid, exog = resid.align(exog, axis=0)
    sort_cols = np.arange(exog.shape[1])

    if jobs > 1:
        gq = partial(
            sms.het_goldfeldquandt,
            resid.to_numpy(),
            exog.to_numpy(),
            alternative="two-sided",
            split=split,
            drop=drop,
        )
        with ThreadPool(jobs) as pool:
            all_results = pool.map(lambda x: gq(idx=x), sort_cols)
    else:
        all_results = []
        for idx in sort_cols:
            results = sms.het_goldfeldquandt(
                resid.to_numpy(),
                exog.to_numpy(),
                idx=idx,
                alternative="two-sided",
                split=split,
                drop=drop,
            )
            all_results.append(results)
    all_results = pd.DataFrame(
        all_results, columns=["f_val", "p_val", "hypothesis"], index=sort_cols
    )
    all_results.index = all_results.index.map(lambda x: exog.columns.values[x])
    all_results.index.name = "sort_by"
    return all_results.sort_values("p_val")


def gq_summary(model, split=0.45, drop=0.1):
    results = goldfeld_quandt(model, split=split, drop=drop)
    n_hetero = (results["p_val"] < 0.05).sum()
    n_total = results.shape[0]
    n_pass = n_total - n_hetero
    ratio = n_pass / n_total
    max_f = results.query("p_val < .05")["f_val"].max()
    summ = pd.Series([ratio, max_f], index=["pass_ratio", "max_f"])
    summ.name = "goldfeld_quandt"
    return summ


def white(model):
    results = sms.het_white(model.resid, model.model.exog)
    return pd.Series(results, index=["lm", "lm_pval", "f_val", "f_pval"])


def breusch_pagan(model, **kwargs):
    results = sms.het_breuschpagan(model.resid, model.model.exog, **kwargs)
    return pd.Series(results, index=["lm", "lm_pval", "f_val", "f_pval"])


def jarque_bera(model):
    results = sms.jarque_bera(model.resid)
    return pd.Series(results, index=["jb", "p_val", "skew", "kurt"])


def check_multicol(model, high_corr=0.7):
    regs = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    regs.drop(columns="Intercept", inplace=True)
    corr_df = regs.corr() >= high_corr
    mask = np.triu(np.ones_like(corr_df), 0)
    masked_df = pd.DataFrame(np.ma.MaskedArray(corr_df.values, mask=mask))
    masked_df.index = corr_df.index
    masked_df.columns = corr_df.columns
    return masked_df.loc[:, masked_df.notnull().any()]


def bad_pvalues(model, alpha=0.05):
    bad = model.pvalues >= alpha
    return model.pvalues.loc[bad].copy()


def corr(frame: pd.DataFrame, other: pd.DataFrame):
    return other.apply(lambda x: frame.corrwith(x))


def get_high_corrs(data, high_corr=0.7):
    corr_df = data.corr()
    mask = np.tril(np.ones_like(corr_df, dtype=np.bool_))
    corr_df = corr_df.mask(mask).stack()
    high_mask = corr_df >= high_corr
    return corr_df[high_mask].index.to_list()