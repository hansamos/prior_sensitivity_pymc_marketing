"""Scenarios: one JSON spec, read by two functions that must agree.

    fit_kwargs(spec)       the changed model, for a full refit (`pmm_model.build_model`)
    log_ratio(spec, d)     log importance ratio on the reference draws `d` (`load_draws`), from NumPy/SciPy

Spec keys (all optional; a missing key means "as in the reference"):
    business    {"roas": [x1, x2], "sigma": [x1, x2]}  or "none"      reference: roas [100, 150], sigma [20, 50]
    sat_beta    s          HalfNormal(s) on saturation_beta           reference: 1.0
    adstock     [a, b]     Beta(a, b) on adstock_alpha                reference: [2, 3]
    sat_lam     [a, b]     Gamma(shape a, rate b) on saturation_lam   reference: [2, 2]
    y_sigma     s          HalfNormal(s) on the sales noise scale     reference: 0.5
    sales_t_nu  nu         StudentT(nu) sales noise (fixed nu)        reference: Normal
    lift_sigma  s          standard error of the lift tests           reference: 3.0   (lift model only)
    lift_keep   [rows]     lift tests kept                            reference: [0, 1]
    lift_t_nu   nu         StudentT(nu) lift noise (fixed nu)         reference: Normal
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pymc_extras.prior import Prior
from scipy import stats

REF = dict(business={"roas": [100.0, 150.0], "sigma": [20.0, 50.0]}, sat_beta=1.0, adstock=[2.0, 3.0],
           sat_lam=[2.0, 2.0], y_sigma=0.5, lift_sigma=3.0, lift_keep=[0, 1])

PRIOR_KEYS = ("sat_beta", "adstock", "sat_lam", "y_sigma")
LIKELIHOOD_KEYS = ("sales_t_nu", "lift_sigma", "lift_keep", "lift_t_nu")


# --------------------------------------------------------------------------- densities
def _dist(key: str, value):
    if key in ("sat_beta", "y_sigma"):
        return stats.halfnorm(scale=value)
    if key == "adstock":
        return stats.beta(*value)
    if key == "sat_lam":
        return stats.gamma(value[0], scale=1.0 / value[1])
    raise KeyError(key)


def _lift_logpdf(obs, est, sigma, nu=None):
    return stats.norm.logpdf(obs, est, sigma) if nu is None else stats.t.logpdf(obs, nu, est, sigma)


# --------------------------------------------------------------------------- refit side
def fit_kwargs(spec: dict) -> dict:
    """Keyword arguments of `pmm_model.build_model` for the changed model (without `model`)."""
    kw: dict = {}
    over = {}
    if "sat_beta" in spec:
        over["saturation_beta"] = Prior("HalfNormal", sigma=spec["sat_beta"], dims="channel")
    if "adstock" in spec:
        over["adstock_alpha"] = Prior("Beta", alpha=spec["adstock"][0], beta=spec["adstock"][1], dims="channel")
    if "sat_lam" in spec:
        over["saturation_lam"] = Prior("Gamma", alpha=spec["sat_lam"][0], beta=spec["sat_lam"][1], dims="channel")
    if over:
        kw["prior_overrides"] = over
    if "y_sigma" in spec or "sales_t_nu" in spec:
        scale = Prior("HalfNormal", sigma=spec.get("y_sigma", REF["y_sigma"]))
        kw["sales_likelihood"] = (Prior("StudentT", nu=spec["sales_t_nu"], sigma=scale)
                                  if "sales_t_nu" in spec else Prior("Normal", sigma=scale))
    if "business" in spec:
        kw["business_prior"] = spec["business"] if spec["business"] == "none" else {
            "channel": ["x1", "x2"], "roas": list(spec["business"]["roas"]), "sigma": list(spec["business"]["sigma"])}
    if "lift_sigma" in spec:
        kw["lift_sigma"] = spec["lift_sigma"]
    if "lift_keep" in spec:
        kw["lift_rows"] = list(spec["lift_keep"])
    if "lift_t_nu" in spec:
        kw["lift_likelihood"] = Prior("StudentT", nu=spec["lift_t_nu"])
    return kw


# --------------------------------------------------------------------------- reweighting side
def load_draws(idata_path: str) -> dict:
    """Flattened (S, ...) reference draws and data needed by `log_ratio`."""
    import arviz as az
    dt = az.from_netcdf(idata_path)
    post = dt["posterior"].to_dataset().stack(s=("chain", "draw")).transpose("s", ...)
    d = {k: post[k].values for k in ("ROAS", "adstock_alpha", "saturation_lam", "saturation_beta", "y_sigma", "mu")}
    if "lift_measurements_model_estimated_lift" in post:
        d["lift_est"] = post["lift_measurements_model_estimated_lift"].values
        d["lift_obs"] = dt["observed_data"].to_dataset()["lift_measurements"].values
    d["y"] = dt["observed_data"].to_dataset()["y"].values
    d["target_scale"] = float(dt["constant_data"].to_dataset()["target_scale"])
    d["n_chains"] = dt["posterior"].sizes["chain"]
    return d


def log_ratio_terms(spec: dict, d: dict) -> dict[str, np.ndarray]:
    """Per-term (S,) log ratios; their sum is the log importance ratio."""
    terms: dict[str, np.ndarray] = {}
    if "business" in spec:
        ref = REF["business"]
        lp0 = stats.norm.logpdf(ref["roas"], d["ROAS"], ref["sigma"]).sum(1)
        new = spec["business"]
        lp1 = 0.0 if new == "none" else stats.norm.logpdf(new["roas"], d["ROAS"], new["sigma"]).sum(1)
        terms["business"] = lp1 - lp0
    for key, var in (("sat_beta", "saturation_beta"), ("adstock", "adstock_alpha"), ("sat_lam", "saturation_lam")):
        if key in spec:
            x = d[var]
            terms[key] = (_dist(key, spec[key]).logpdf(x) - _dist(key, REF[key]).logpdf(x)).sum(1)
    if "y_sigma" in spec:
        x = d["y_sigma"]
        terms["y_sigma"] = _dist("y_sigma", spec["y_sigma"]).logpdf(x) - _dist("y_sigma", REF["y_sigma"]).logpdf(x)
    if "sales_t_nu" in spec:  # evaluated at the reference noise scale; a changed y_sigma prior is a separate term
        y, mu, sg = d["y"], d["mu"], d["y_sigma"][:, None]
        terms["sales_t_nu"] = (stats.t.logpdf(y, spec["sales_t_nu"], mu, sg) - stats.norm.logpdf(y, mu, sg)).sum(1)
    if any(k in spec for k in ("lift_sigma", "lift_keep", "lift_t_nu")):
        ts, obs, est = d["target_scale"], d["lift_obs"], d["lift_est"]
        old = _lift_logpdf(obs, est, REF["lift_sigma"] / ts)  # (S, 2)
        keep = list(spec.get("lift_keep", REF["lift_keep"]))
        new = _lift_logpdf(obs[keep], est[:, keep], spec.get("lift_sigma", REF["lift_sigma"]) / ts,
                           spec.get("lift_t_nu"))
        terms["lift"] = new.sum(1) - old.sum(1)
    return terms


def log_ratio(spec: dict, d: dict) -> np.ndarray:
    return np.sum(list(log_ratio_terms(spec, d).values()), axis=0)


# --------------------------------------------------------------------------- scenario tables
def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _round(x, n=3):
    return float(np.round(x, n))


def sample_family(key: str, rng) -> dict:
    """One random spec fragment for a family, spread from mild to extreme changes."""
    if key == "business":
        roas = [_round(r * rng.uniform(0.4, 1.6), 1) for r in REF["business"]["roas"]]
        sig = [_round(s * _log_uniform(rng, 0.25, 4), 1) for s in REF["business"]["sigma"]]
        return {"business": {"roas": roas, "sigma": sig}}
    if key == "sat_beta":
        return {"sat_beta": _round(_log_uniform(rng, 0.2, 4))}
    if key == "adstock":
        mean, conc = rng.uniform(0.1, 0.9), 5.0 * _log_uniform(rng, 0.4, 4)
        return {"adstock": [_round(max(mean * conc, 1.0)), _round(max((1 - mean) * conc, 1.0))]}  # shapes >= 1
    if key == "sat_lam":
        shape, mean = 2.0 * _log_uniform(rng, 0.5, 4), _log_uniform(rng, 0.25, 4)
        return {"sat_lam": [_round(shape), _round(shape / mean)]}
    if key == "y_sigma":
        return {"y_sigma": _round(_log_uniform(rng, 0.05, 2.0))}
    if key == "sales_t_nu":
        return {"sales_t_nu": float(rng.choice([1, 2, 3, 5, 8, 15, 30, 60]))}
    if key == "lift_sigma":
        return {"lift_sigma": _round(REF["lift_sigma"] * _log_uniform(rng, 0.25, 8))}
    if key == "lift_t_nu":
        return {"lift_t_nu": float(rng.choice([1, 2, 3, 5, 8, 15, 30, 60]))}
    raise KeyError(key)


def shrink(spec: dict, m: float) -> dict:
    """Move a full-strength spec fragment toward the reference by a factor m in (0, 1] (log scale for scales)."""
    def geo(ref, v):
        return _round(ref * (v / ref) ** m, 4)

    out = dict(spec)
    if isinstance(spec.get("business"), dict):
        b, r = spec["business"], REF["business"]
        out["business"] = {"roas": [_round(r0 + m * (v - r0), 1) for r0, v in zip(r["roas"], b["roas"])],
                           "sigma": [_round(geo(s0, v), 1) for s0, v in zip(r["sigma"], b["sigma"])]}
    for k in ("sat_beta", "y_sigma", "lift_sigma"):
        if k in spec:
            out[k] = geo(REF[k], spec[k])
    if "adstock" in spec:
        a, b = spec["adstock"]
        mean0, conc0 = REF["adstock"][0] / sum(REF["adstock"]), sum(REF["adstock"])
        mean, conc = mean0 + m * (a / (a + b) - mean0), conc0 * ((a + b) / conc0) ** m
        out["adstock"] = [_round(max(mean * conc, 1.0)), _round(max((1 - mean) * conc, 1.0))]
    if "sat_lam" in spec:
        shape, rate = spec["sat_lam"]
        shape_ref = REF["sat_lam"][0]
        new_shape, new_mean = shape_ref * (shape / shape_ref) ** m, (shape / rate) ** m  # reference mean is 1
        out["sat_lam"] = [_round(new_shape), _round(new_shape / new_mean)]
    for k in ("sales_t_nu", "lift_t_nu"):
        if k in spec:
            out[k] = float(min(round(spec[k] / m), 1000))  # nu -> infinity is the Normal reference
    return out


POOL_WEIGHTS = {
    "priors": {"business": 0.25, "sat_beta": 0.10, "adstock": 0.10, "sat_lam": 0.10, "y_sigma": 0.05, "joint": 0.40},
    "likelihood": {"lift_sigma": 0.12, "lift_t_nu": 0.08, "sales_t_nu": 0.12, "business": 0.08, "joint": 0.60},
}
JOINT_FAMILIES = {"priors": ["business", "sat_beta", "adstock", "sat_lam", "y_sigma"],
                  "likelihood": ["lift_sigma", "lift_t_nu", "sales_t_nu", "business", "sat_beta", "adstock", "sat_lam", "y_sigma"]}
SPECIAL = {"priors": [("p_business_none", "business", {"business": "none"})],
           "likelihood": [("l_lift_drop_0", "lift_drop", {"lift_keep": [1]}),
                          ("l_lift_drop_1", "lift_drop", {"lift_keep": [0]}),
                          ("l_lift_drop_both", "lift_drop", {"lift_keep": []})]}


def candidate_pool(table: str, n: int, seed: int) -> list[tuple[str, dict]]:
    """n candidate (family, spec): single-family changes and joint changes of 2-3 families, each of random
    strength from nearly null to extreme."""
    rng = np.random.default_rng(seed)
    fams = list(POOL_WEIGHTS[table])
    p = np.array(list(POOL_WEIGHTS[table].values()))
    pool = []
    for _ in range(n):
        fam = str(rng.choice(fams, p=p / p.sum()))
        parts = (rng.choice(JOINT_FAMILIES[table], size=int(rng.integers(2, 4)), replace=False)
                 if fam == "joint" else [fam])
        spec = {}
        for f in parts:
            spec.update(shrink(sample_family(str(f), rng), _log_uniform(rng, 0.05, 1.0)))
        pool.append((fam, spec))
    return pool


def select_stratified(pool, k, edges, per_bin, weights, seed):
    """Pick `per_bin` candidates in each k-hat bin (bins from `edges`), balanced over families
    in proportion to `weights`. `k` is the k-hat of each pool member on the reference draws."""
    rng = np.random.default_rng(seed)
    bins = np.digitize(k, edges)
    chosen = []
    for b in range(len(edges) + 1):
        members = [i for i in rng.permutation(len(pool)) if bins[i] == b]
        counts = {f: 0 for f in weights}
        picked = []
        while members and len(picked) < per_bin:
            avail = sorted({pool[i][0] for i in members})  # sorted: ties must not depend on set/hash order
            fam = min(avail, key=lambda f: counts[f] / weights[f])  # family furthest below its share
            i = next(i for i in members if pool[i][0] == fam)
            members.remove(i)
            counts[fam] += 1
            picked.append(i)
        chosen += picked
    return chosen, bins


def build_table(rows) -> pd.DataFrame:
    return pd.DataFrame([{"id": i, "family": f, "spec": json.dumps(s, sort_keys=True)} for i, f, s in rows])
