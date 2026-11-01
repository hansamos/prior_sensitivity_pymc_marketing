"""Write the two scenario tables (notebook 1: priors, notebook 2: likelihood and combined).

A large pool of candidate changes, from nearly null to extreme, is reweighted on the reference
draws; PER_BIN scenarios per Pareto k-hat bin are selected, balanced over families, and the special
cases (business prior dropped, lift tests dropped) are added. Selecting on k-hat gives the evidence coverage of the range the diagnostic is meant
to judge. It does not bias error-given-k-hat results, but it means the share of scenarios in a bin
says nothing about how often a typical change is safe.

    python -I evidence/common/make_tables.py        # needs runs/pmm/ref_{business,lift}
"""
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import reweight as rw  # noqa: E402
import scenarios  # noqa: E402
from analyze import relative_efficiency  # noqa: E402

warnings.filterwarnings("ignore")
K_EDGES = [0.3, 0.5, 0.7, 1.0]  # bins: <0.3, 0.3-0.5, 0.5-0.7, 0.7-1, >=1
N_POOL = 3000
PER_BIN = {"priors": 20, "likelihood": 8}  # scenarios per k-hat bin, plus the special cases of scenarios.SPECIAL

out = HERE.parent / "scenarios"
out.mkdir(exist_ok=True)
for table, model, seed in (("priors", "business", 20261101), ("likelihood", "lift", 20261102)):
    path = f"runs/pmm/ref_{model}/idata.nc"
    d, reff = scenarios.load_draws(path), relative_efficiency(path)
    pool = scenarios.candidate_pool(table, N_POOL, seed)
    k = np.array([rw.reweight(scenarios.log_ratio(spec, d), r_eff=reff).pareto_k for _, spec in pool])
    assert not np.isnan(k).any(), "a candidate has a constant ratio (no k-hat)"
    special = scenarios.SPECIAL[table]
    chosen, bins = scenarios.select_stratified(pool, k, K_EDGES, PER_BIN[table], scenarios.POOL_WEIGHTS[table], seed)
    rows = list(special)
    for j, i in enumerate(chosen):
        fam, spec = pool[i]
        rows.append((f"{table[0]}_{fam}_{j + 1:03d}", fam, spec))
    t = scenarios.build_table(rows)
    t.to_csv(out / f"{table}.csv", index=False)
    print(f"{table}: pool k-hat bins {np.bincount(bins, minlength=len(K_EDGES) + 1).tolist()}; selected {len(t)}")
    print("  families:", t.family.value_counts().to_dict())
