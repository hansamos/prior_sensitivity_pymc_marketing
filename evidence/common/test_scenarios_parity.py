"""Parity: the NumPy log ratio of a scenario equals PyMC's own joint log-density difference.

For a scenario, rebuild the changed model with `scenarios.fit_kwargs`, evaluate prior + likelihood
log densities of both models on the same reference draws with `pm.compute_log_prior` /
`pm.compute_log_likelihood`, and compare (new total - reference total) with `scenarios.log_ratio`.
This ties the SciPy densities to the PyMC `Prior` parametrization (e.g. Gamma rate, StudentT scale).

    python -I evidence/common/test_scenarios_parity.py [--n_draws 200]
Run from the repository root; needs runs/pmm/ref_business and runs/pmm/ref_lift.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pmm_model  # noqa: E402
import scenarios  # noqa: E402

warnings.filterwarnings("ignore")
p = argparse.ArgumentParser()
p.add_argument("--n_draws", type=int, default=200)
p.add_argument("--tol", type=float, default=1e-6)
args = p.parse_args()

TABLES = {"business": HERE.parent / "scenarios" / "priors.csv", "lift": HERE.parent / "scenarios" / "likelihood.csv"}


def total_logp(model, post_ds):
    """Sum over all prior and likelihood terms of `model` at the draws: (S,)."""
    dt = xr.DataTree.from_dict({"posterior": post_ds})
    with model:
        pm.compute_log_likelihood(dt)
        pm.stats.compute_log_prior(dt)
    tot = 0.0
    for group in ("log_prior", "log_likelihood"):
        ds = dt[group].to_dataset()
        for v in ds.data_vars:
            a = ds[v].stack(s=("chain", "draw")).transpose("s", ...).values
            tot = tot + a.reshape(a.shape[0], -1).sum(1)
    return tot


def pick(table):
    """First scenario of each family, two joints, and the special cases."""
    ids = list(table.groupby("family").head(1).id) + list(table[table.family == "joint"].id.iloc[[3, 7]])
    ids += [i for i in table.id if i.endswith(("_none", "_drop_both"))]
    return table.set_index("id").loc[list(dict.fromkeys(ids))]


worst = 0.0
for model, csv in TABLES.items():
    path = f"runs/pmm/ref_{model}/idata.nc"
    d = scenarios.load_draws(path)
    dt = az.from_netcdf(path)
    post = dt["posterior"].to_dataset()
    free = [v.name for v in pmm_model.build_model(model)[0].model.free_RVs]
    sub = post[free].isel(draw=slice(0, args.n_draws // post.sizes["chain"] + 1))
    n = sub.sizes["chain"] * sub.sizes["draw"]
    ref_total = total_logp(pmm_model.build_model(model)[0].model, sub)
    # the draws of `sub` in the order of the stacked (s) axis -> indices into the flattened draws `d`
    idx = (np.arange(post.sizes["chain"])[:, None] * post.sizes["draw"] + np.arange(sub.sizes["draw"])[None, :]).ravel()
    for sid, row in pick(pd.read_csv(csv)).iterrows():
        spec = json.loads(row.spec)
        new_total = total_logp(pmm_model.build_model(model, **scenarios.fit_kwargs(spec))[0].model, sub)
        mine = scenarios.log_ratio(spec, d)[idx]
        diff = float(np.max(np.abs((new_total - ref_total) - mine)))
        worst = max(worst, diff)
        print(f"{model:8s} {sid:20s} n={n}  max|PyMC - NumPy| = {diff:.2e}  {'ok' if diff < args.tol else 'FAIL'}", flush=True)
print("worst:", f"{worst:.2e}")
sys.exit(0 if worst < args.tol else 1)
