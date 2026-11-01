"""Summary tables from the analysis CSVs: error of the reweighted posterior by Pareto k-hat bin.

    python -I evidence/common/summarize.py [--results evidence/results]

Per scenario, over the nine posterior quantities (ROAS, adstock_alpha, saturation_lam, saturation_beta per
channel; y_sigma):
  err_median / err_max   median / worst |mean_reweighted - mean_refit| / sd_refit
  do_nothing_max         the same error if the unweighted reference posterior is used (effect size of the change)
  closer_worst           share of scenarios whose worst-quantity error is below the do-nothing one
Per scenario x quantity, split by whether the change moves the quantity (do-nothing error above MOVED SDs):
  reweighting helps on the quantities that move, and costs accuracy on the ones that do not once k-hat is high.
Refits that did not converge (max r-hat > 1.01 or min bulk ESS < 400) are no ground truth: they are left
out of the error statistics and listed. Divergences are reported, not used to exclude.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

K_EDGES, K_LABELS = [0.3, 0.5, 0.7, 1.0], ["<0.3", "0.3-0.5", "0.5-0.7", "0.7-1", ">=1"]
ESS_EDGES, ESS_LABELS = [0.05, 0.2], ["<5%", "5-20%", ">20%"]
MOVED = 0.1  # a quantity counts as moved by the change if the do-nothing error exceeds this many refit SDs


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    df = df.reset_index() if index else df
    rows = [[str(c) for c in df.columns]] + [[f"{v:g}" if isinstance(v, float) else str(v) for v in r] for r in df.itertuples(index=False)]
    line = lambda r: "| " + " | ".join(r) + " |"  # noqa: E731
    return "\n".join([line(rows[0]), line(["---"] * len(rows[0]))] + [line(r) for r in rows[1:]])


def load(results: Path, table: str) -> pd.DataFrame:
    s = pd.read_csv(results / f"{table}_scenario_level.csv")
    q = pd.read_csv(results / f"{table}_quantity_level.csv")
    g = q.groupby("id").agg(err_median=("err_mean_sd", "median"), err_max=("err_mean_sd", "max"),
                            do_nothing_median=("err_mean_sd_none", "median"), do_nothing_max=("err_mean_sd_none", "max"),
                            split_noise=("err_mean_sd_split", "median"), ab_noise=("err_mean_sd_ab", "median"),
                            sd_ratio=("sd_ratio", "median"))
    s = s.merge(g, left_on="id", right_index=True)
    s["refit_ok"] = (s.refit_max_r_hat <= 1.01) & (s.refit_min_ess_bulk >= 400)
    s["k_bin"] = pd.cut(s.pareto_k, [-np.inf] + K_EDGES + [np.inf], labels=K_LABELS, right=False)
    s["ess_bin"] = pd.cut(s.ess_fraction, [0] + ESS_EDGES + [1.01], labels=ESS_LABELS, right=False)
    return s


def by_bin(s: pd.DataFrame, key: str) -> pd.DataFrame:
    p90 = lambda x: x.quantile(0.9)  # noqa: E731
    s = s.assign(closer=s.err_max < s.do_nothing_max)
    out = s.groupby(key, observed=True).agg(
        n=("id", "size"), ess_median=("ess_fraction", "median"),
        err_median=("err_median", "median"), err_max_median=("err_max", "median"), err_max_p90=("err_max", p90),
        do_nothing_max_median=("do_nothing_max", "median"), closer_worst=("closer", "mean"))
    return out.round(3)


def load_quantities(results: Path, table: str, s: pd.DataFrame) -> pd.DataFrame:
    """Quantity-level rows of converged scenarios, with the k-hat bin and whether the change moves the quantity."""
    q = pd.read_csv(results / f"{table}_quantity_level.csv")
    q = q.merge(s[s.refit_ok][["id", "k_bin"]], on="id")
    q["moved"] = q.err_mean_sd_none > MOVED
    q["closer"] = q.err_mean_sd < q.err_mean_sd_none
    return q


def by_bin_quantities(q: pd.DataFrame) -> pd.DataFrame:
    """Per k-hat bin: errors on the quantities the change moves, and on those it does not."""
    rows = []
    for (b, moved), d in q.groupby(["k_bin", "moved"], observed=True):
        rows.append(dict(k_bin=b, quantities="moved" if moved else "not moved", n=len(d),
                         err_reweighted=d.err_mean_sd.median(), err_do_nothing=d.err_mean_sd_none.median(),
                         share_closer=d.closer.mean()))
    return pd.DataFrame(rows).round(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(Path(__file__).resolve().parent.parent / "results"))
    results = Path(p.parse_args().results)
    md = ["# Reweighted posterior vs full refit, by Pareto k-hat\n"]
    for table in ("priors", "likelihood"):
        s = load(results, table)
        ok = s[s.refit_ok]
        md.append(f"\n## {table}: {len(s)} scenarios, {len(ok)} with a converged refit\n")
        md.append(f"Refit noise (median error between two refits of one scenario, refit SDs): "
                  f"{s.ab_noise.median():.3f} ({int(s.ab_noise.notna().sum())} scenarios); "
                  f"split-chain noise {s.split_noise.median():.3f}. Refits with divergences: "
                  f"{int((s.refit_divergences > 0).sum())} (total {int(s.refit_divergences.sum())} of "
                  f"{int(s.refit_divergences.size * 12000)} draws).\n")
        md.append("\nBy k-hat (error in refit posterior SDs; `do_nothing` = unweighted reference posterior):\n")
        t = by_bin(ok, "k_bin")
        t.to_csv(results / f"{table}_by_khat.csv")
        md.append(md_table(t) + "\n")
        tq = by_bin_quantities(load_quantities(results, table, s))
        tq.to_csv(results / f"{table}_by_khat_quantities.csv", index=False)
        md.append(f"\nBy k-hat, per posterior quantity (median error in refit SDs). `moved`: the change shifts the "
                  f"quantity by more than {MOVED} SD (do-nothing error); `share_closer`: share of quantities where "
                  f"the reweighted mean is closer to the refit than the unweighted reference:\n")
        md.append(md_table(tq, index=False) + "\n")
        low = ok[ok.pareto_k < 0.7]
        md.append("\nScenarios with k-hat < 0.7, by ESS fraction:\n")
        t2 = by_bin(low, "ess_bin")
        t2.to_csv(results / f"{table}_by_ess_khat_below_0.7.csv")
        md.append(md_table(t2) + "\n")
        bad = s[~s.refit_ok]
        if len(bad):
            md.append("\nRefits that did not converge (excluded):\n")
            md.append(md_table(bad[["id", "pareto_k", "ess_fraction", "refit_max_r_hat", "refit_min_ess_bulk"]].round(3),
                               index=False) + "\n")
    (results / "SUMMARY.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
