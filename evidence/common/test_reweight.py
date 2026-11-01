"""Unit tests for reweight.py (no MCMC).

    python -I evidence/common/test_reweight.py            # analytic checks
    python -I evidence/common/test_reweight.py --dump F   # also write the cases to F (npz)
    <meridian env> python -I evidence/common/test_reweight.py --compare F
        # same cases through ArviZ 0.19 `az.psislw` (the Meridian core): k-hat and weights must agree
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

p = argparse.ArgumentParser()
p.add_argument("--dump")
p.add_argument("--compare")
args = p.parse_args()


def cases():
    rng = np.random.default_rng(0)
    S = 8000
    x = rng.normal(size=S)
    out = {}
    out["shift"] = stats.norm(0.5, 1).logpdf(x) - stats.norm(0, 1).logpdf(x)  # mild
    out["narrow"] = stats.norm(0, 0.7).logpdf(x) - stats.norm(0, 1).logpdf(x)  # bounded weights
    out["wide"] = stats.norm(0, 1.4).logpdf(x) - stats.norm(0, 1).logpdf(x)  # heavy tail
    out["far"] = stats.norm(2.0, 1).logpdf(x) - stats.norm(0, 1).logpdf(x)  # very heavy tail
    return x, out


if args.compare:  # ArviZ 0.19 side
    import arviz as az
    d = np.load(args.compare)
    for name in [k for k in d.files if k.startswith("lr_")]:
        lr = d[name]
        lw, k = az.psislw(lr.copy())
        w = np.exp(lw - np.max(lw)); w /= w.sum()
        w1, k1 = d["w_" + name[3:]], float(d["k_" + name[3:]])
        ess0, ess1 = 1 / np.sum(w**2), 1 / np.sum(w1**2)
        print(f"{name[3:]:7s} k-hat 0.19: {k:6.3f}  1.x: {k1:6.3f}   ESS 0.19: {ess0:6.0f}  1.x: {ess1:6.0f}   "
              f"max |dw|/mean w: {np.max(np.abs(w - w1)) * len(w):.3f}")
        # the two ArviZ versions implement PSIS differently (tail size, GPD fit): same sign, similar size
        assert abs(k - k1) < 0.15 and abs(ess0 - ess1) < 0.05 * len(w), name
    print("ArviZ 0.19 and 1.x agree within tolerance")
    sys.exit()

import reweight as rw  # noqa: E402

x, cs = cases()
dump = {}
for name, lr in cs.items():
    r = rw.reweight(lr)
    dump[f"lr_{name}"], dump[f"k_{name}"], dump[f"w_{name}"] = lr, r.pareto_k, r.weights
    print(f"{name:7s} k-hat {r.pareto_k:5.2f}  ESS {r.ess_fraction:5.1%}  {r.status}")

# mild shift: weighted mean must move to the target mean and the sign convention must be right
r = rw.reweight(cs["shift"])
m = rw.weighted_mean(x, r.weights)
assert abs(m - 0.5) < 0.05, m
assert r.status == "good" and r.pareto_k < 0.5
# far shift: heavy tail must be flagged
assert rw.reweight(cs["far"]).pareto_k > rw.reweight(cs["shift"]).pareto_k
# a wrong sign would put the mean on the wrong side
assert rw.weighted_mean(x, rw.reweight(-cs["shift"]).weights) < 0
# identical priors, truncation, zero-density draws
assert rw.reweight(np.zeros(100)).status == "identical"
lr = np.where(x > 0, 0.0, -np.inf)
rt = rw.reweight(lr)
assert rt.status == "bounded" and rt.n_zero_weight == (x <= 0).sum() and np.all(rt.weights[x <= 0] == 0)
lr = cs["shift"].copy(); lr[:50] = -np.inf
rz = rw.reweight(lr)
assert np.all(rz.weights[:50] == 0) and abs(rz.weights.sum() - 1) < 1e-12
for bad in (np.nan, np.inf):
    try:
        rw.reweight(np.r_[cs["shift"][:10], bad]); raise SystemExit("accepted " + str(bad))
    except ValueError:
        pass
print("all checks passed")
if args.dump:
    np.savez(args.dump, **dump)
