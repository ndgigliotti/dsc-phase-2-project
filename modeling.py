import re
import datetime
import glob
import pickle
import itertools
import os
import shutil
from functools import partial
from multiprocessing.pool import ThreadPool
from operator import itemgetter

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols


import plotting
import utils

TEST_DIR = "test_models"
OLS_SWEEP_DIR = os.path.join(TEST_DIR, "ols_sweep")


def reg_model(data, formula):
    model = ols(formula=formula, data=data).fit()
    display(model.summary())
    axs = plotting.diagnostics(model)
    gq = goldfeld_quandt(model)
    gq.index.name = "goldfeld_quandt"
    bad_p = bad_pvalues(model).to_frame("bad_pvalue")
    display(gq)
    display(bad_p)
    display(axs)
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
    gq = goldfeld_quandt(model)
    results = results.append(gq["p_val"].add_prefix("gq_"))
    results["high_corr_exog"] = check_multicol(model).sum().sum()
    results["bad_pvals"] = bad_pvalues(model).size
    return results


def load_results(glob_path):
    paths = glob.glob(glob_path)
    with ThreadPool() as pool:
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
    df = load_results_as_frame(glob_path)
    first_path = glob.glob(glob_path)[0]
    dir_ = os.path.dirname(first_path)
    path = f"{dir_}_summary.json"
    df.to_json(path)
    shutil.make_archive(dir_, "zip", dir_)
    shutil.rmtree(dir_)


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


def ols_sweep(data, target, n_vars=2, ignore=None, dst=OLS_SWEEP_DIR):
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
    with ThreadPool() as pool:
        build = partial(_build_and_record, data)
        pool.starmap(build, zip(formulae, paths))


def goldfeld_quandt(model, split=0.45, drop=0.1):
    all_alts = []
    for alt in ["increasing", "decreasing", "two-sided"]:
        results = sms.het_goldfeldquandt(
            model.resid, model.model.exog, alternative=alt, split=split, drop=drop
        )
        all_alts.append(results)
    all_alts = pd.DataFrame(all_alts, columns=["f_val", "p_val", "hypothesis"])
    return all_alts.set_index("hypothesis")


def breusch_pagan(model, robust=True):
    results = sms.het_breuschpagan(model.resid, model.model.exog, robust=robust)
    index = ["lagrange_mult", "p_val", "f_val", "f_pval"]
    return pd.Series(results, index=index)


def jarque_bera(model):
    results = sms.jarque_bera(model.resid)
    return pd.Series(results, index=["jb", "p_val", "skew", "kurt"])


def check_multicol(model, high_corr=0.75):
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