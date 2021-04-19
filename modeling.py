import os
import glob
import itertools
import datetime
from functools import partial
from multiprocessing.pool import ThreadPool
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import utils


def reg_model(data, formula):
    model = ols(formula=formula, data=data).fit()
    display(model.summary())
    return model


def _build_and_record(data, formula, path):
    model = ols(formula=formula, data=data).fit()
    index = ["rsquared", "rsquared_adj", "mse_resid"]
    summary = pd.Series(data=[getattr(model, x) for x in index], index=index)
    summary = summary.to_frame(formula)
    summary.to_csv(path)
    # model.save(path, remove_data=True)
    return model

def load_sweep_results(glob_path):
    paths = glob.glob(glob_path)
    read_csv = partial(pd.read_csv, index_col=0, squeeze=True)
    with ThreadPool() as pool:
        columns = pool.map(read_csv, paths)
    df = pd.concat(columns, axis=1).T
    return df



def multivariate_sweep(data, target, n_vars=2, ignore=None, dst="test_model_results"):
    if ignore:
        data = data.drop(columns=ignore)
    var_names = utils.noncat_cols(data)
    var_names += [f"C({x})" for x in utils.cat_cols(data)]
    combos = list(itertools.combinations(var_names, n_vars))
    combo_strs = ["+".join(x) for x in combos]
    formulae = [f"{target}~{x}" for x in combo_strs]
    dst = os.path.join(dst, utils.now_name())
    os.makedirs(dst, exist_ok=True)
    paths = [os.path.join(dst, f"{x}.csv") for x in formulae]
    with ThreadPool() as pool:
        build = partial(_build_and_record, data)
        models = pool.starmap(build, zip(formulae, paths))
    index = formulae
    columns = ["rsquared", "mse_resid"]
    summary = pd.DataFrame(columns=columns, index=index)
    for column in summary.columns:
        summary[column] = [getattr(x, column) for x in models]
    return summary.sort_values("rsquared", ascending=False)


def permutation_sweep(data, target, *vars, dst="test_model_results"):
    perms = itertools.permutations(vars, len(vars))
    perm_strs = ["+".join(x) for x in perms]
    formulae = [f"{target}~{x}" for x in perm_strs]
    dst = os.path.join(dst, utils.now_name())
    os.makedirs(dst, exist_ok=True)
    paths = [os.path.join(dst, f"{x}.csv") for x in formulae]
    with ThreadPool() as pool:
        build = partial(_build_and_record, data)
        models = pool.starmap(build, zip(formulae, paths))
    index = formulae
    columns = ["rsquared", "mse_resid"]
    summary = pd.DataFrame(columns=columns, index=index)
    for column in summary.columns:
        summary[column] = [getattr(x, column) for x in models]
    return summary.sort_values("rsquared", ascending=False)