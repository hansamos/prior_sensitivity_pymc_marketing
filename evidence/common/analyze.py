"""Reweighted reference posterior vs full refit, per scenario and posterior quantity.

    python -I evidence/common/analyze.py --table priors \
        --ref runs/pmm/ref_business/idata.nc --refits_a runs/pmm/refits_priors_A [--refits_b runs/pmm/refits_priors_B] \
        [--reruns runs/pmm/rerun_priors_C runs/pmm/rerun_priors_D ...] --out evidence/results

A refit A that did not converge is replaced by the first converged refit of the same scenario in `--reruns`
(same model, other seeds); `refit_source` records which one was used.

Quantities: ROAS, adstock_alpha, saturation_lam, saturation_beta per channel; y_sigma.
Per scenario x quantity (reweighted = reference draws with PSIS weights; refit = refit A):
  err_mean_sd  |mean_reweighted - mean_refit| / sd_refit
  sd_ratio     sd_reweighted / sd_refit
  w1           Wasserstein-1 distance (units of the quantity)
  ks           Kolmogorov-Smirnov distance
  kl           KL(refit || reweighted) on smoothed 200-bin histograms, probabilities floored at 1e-8
  err_q05_sd, err_q95_sd   |quantile_reweighted - quantile_refit| / sd_refit at the 5% and 95% quantiles
Baselines with the same metrics:
  *_none       the unweighted reference posterior as the answer ("do nothing"); err_mean_sd_none is the effect size
  *_split      refit A chains {0,1,2} vs {3,4,5} (Monte Carlo noise of one fit); *_ab refit A vs refit B (needs B)
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

import reweight as rw  # noqa: E402
import scenarios  # noqa: E402

QUANTITIES = [("ROAS", ["x1", "x2"]), ("adstock_alpha", ["x1", "x2"]), ("saturation_lam", ["x1", "x2"]),
              ("saturation_beta", ["x1", "x2"]), ("y_sigma", [None])]


def wquantile(x, w, q):
    """Quantiles of samples x with normalized weights w (midpoint rule)."""
    o = np.argsort(x)
    c = np.cumsum(w[o]) - w[o] / 2
    return np.interp(q, c, x[o])


def converged(meta):
    return meta["max_r_hat"] <= 1.01 and meta["min_ess_bulk"] >= 400


def metrics(x, wx, y, wy=None):
    """x with normalized weights wx against samples y (weights wy, default uniform)."""
    wy = np.full(len(y), 1.0 / len(y)) if wy is None else wy
    mx, my = np.sum(wx * x), np.sum(wy * y)
    sx, sy = np.sqrt(np.sum(wx * (x - mx) ** 2)), np.sqrt(np.sum(wy * (y - my) ** 2))
    grid = np.sort(np.concatenate([x, y]))
    ox, oy = np.argsort(x), np.argsort(y)
    Fx = np.concatenate([[0], np.cumsum(wx[ox])])[np.searchsorted(x[ox], grid, side="right")]
    Fy = np.concatenate([[0], np.cumsum(wy[oy])])[np.searchsorted(y[oy], grid, side="right")]
    edges = np.linspace(grid[0], grid[-1], 201)
    px = gaussian_filter1d(np.histogram(x, edges, weights=wx)[0], 2)
    py = gaussian_filter1d(np.histogram(y, edges, weights=wy)[0], 2)
    px, py = np.maximum(px / px.sum(), 1e-8), np.maximum(py / py.sum(), 1e-8)
    qx, qy = wquantile(x, wx, [0.05, 0.95]), wquantile(y, wy, [0.05, 0.95])
    return dict(err_mean_sd=abs(mx - my) / sy, sd_ratio=sx / sy,
                err_q05_sd=abs(qx[0] - qy[0]) / sy, err_q95_sd=abs(qx[1] - qy[1]) / sy,
                w1=float(np.sum(np.abs(Fx - Fy)[:-1] * np.diff(grid))), ks=float(np.max(np.abs(Fx - Fy))),
                kl=float(np.sum(py * np.log(py / px))))


def flat(ds, var, k):
    a = ds[var].values  # (chain, draw[, channel])
    return a if a.ndim == 2 else a[..., k]


def relative_efficiency(ref_path):
    """Mean over the reported quantities of ESS(mean)/S of the reference fit, capped at 1 (as `az.loo`)."""
    ds = az.from_netcdf(ref_path)["posterior"].to_dataset()
    ess = []
    for v, chs in QUANTITIES:
        for k in range(len(chs)):
            x = ds[v].isel(channel=k) if chs[k] is not None else ds[v]
            ess.append(float(az.ess(x.to_dataset(name=v), method="mean")[v]))
    return float(min(np.mean(ess) / (ds.sizes["chain"] * ds.sizes["draw"]), 1.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", choices=["priors", "likelihood"], required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--refits_a", required=True)
    p.add_argument("--refits_b")
    p.add_argument("--reruns", nargs="*", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()

    d = scenarios.load_draws(a.ref)
    reff = relative_efficiency(a.ref)
    print(f"relative MCMC efficiency of the reference: {reff:.2f}")
    table = pd.read_csv(HERE.parent / "scenarios" / f"{a.table}.csv")
    rows, srows = [], []
    for _, r in table.iterrows():
        src = Path(a.refits_a) / r.id
        if not (src / "refit.nc").exists():
            continue
        if not converged(json.loads((src / "meta.json").read_text())):
            for d_ in a.reruns:
                c_ = Path(d_) / r.id / "meta.json"
                if c_.exists() and converged(json.loads(c_.read_text())):
                    src = c_.parent
                    break
        pa = src / "refit.nc"
        spec = json.loads(r.spec)
        res = rw.reweight(scenarios.log_ratio(spec, d), r_eff=reff)
        A = xr.open_dataset(pa)
        pb = Path(a.refits_b) / r.id / "refit.nc" if a.refits_b else None
        B = (xr.open_dataset(pb) if pb is not None and pb.exists()
             and converged(json.loads((pb.parent / "meta.json").read_text())) else None)  # B: noise baseline only
        meta = json.loads((src / "meta.json").read_text())
        if meta["spec"] != spec:  # a stale run directory from an older table
            raise ValueError(f"{r.id}: refit was run with a different spec than the table's")
        errs = []
        for var, chs in QUANTITIES:
            for k, ch in enumerate(chs):
                x = d[var][:, k] if d[var].ndim == 2 else d[var]
                ya = flat(A, var, k)
                row = dict(id=r.id, family=r.family, quantity=var if ch is None else f"{var}[{ch}]")
                m = metrics(x, res.weights, ya.ravel())
                row.update(m)
                row.update({f"{k_}_none": v for k_, v in metrics(x, np.full(len(x), 1.0 / len(x)), ya.ravel()).items()})
                h = ya.shape[0] // 2
                row.update({f"{k_}_split": v for k_, v in metrics(ya[:h].ravel(), np.full(ya[:h].size, 1.0 / ya[:h].size),
                                                                  ya[h:].ravel()).items()})
                if B is not None:
                    yb = flat(B, var, k).ravel()
                    row.update({f"{k_}_ab": v for k_, v in metrics(ya.ravel(), np.full(ya.size, 1.0 / ya.size), yb).items()})
                rows.append(row)
                errs.append(m["err_mean_sd"])
        srows.append(dict(id=r.id, family=r.family, spec=r.spec, pareto_k=res.pareto_k, ess_fraction=res.ess_fraction,
                          status=res.status, n_zero_weight=res.n_zero_weight, k_threshold=res.k_threshold,
                          err_mean_sd_max=max(errs), err_mean_sd_median=float(np.median(errs)),
                          refit_divergences=meta["divergences"], refit_max_r_hat=meta["max_r_hat"],
                          refit_min_ess_bulk=meta["min_ess_bulk"], refit_seconds=meta["seconds"],
                          refit_source=src.parent.name, refit_seed=meta["seed"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{a.table}_quantity_level.csv", index=False)
    pd.DataFrame(srows).to_csv(out / f"{a.table}_scenario_level.csv", index=False)
    print(len(srows), "scenarios,", len(rows), "scenario x quantity rows ->", out)


if __name__ == "__main__":
    main()
