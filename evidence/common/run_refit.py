"""Full refit of one scenario; saves the draws the analysis needs (not the full InferenceData).

    python -I evidence/common/run_refit.py --table priors --index 3 --which A --out_root runs/pmm/refits_priors_A

--table priors -> reference model "business" (notebook 1); --table likelihood -> "lift" (notebook 2).
`--index` is 1-based (SLURM_ARRAY_TASK_ID). The seed is seed_base + index, so refits A and B differ by seed_base only.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arviz as az  # noqa: E402
import pandas as pd  # noqa: E402

import pmm_model  # noqa: E402
import scenarios  # noqa: E402

warnings.filterwarnings("ignore")
MODEL = {"priors": "business", "likelihood": "lift"}
KEEP = ["ROAS", "adstock_alpha", "saturation_lam", "saturation_beta", "y_sigma"]
SEED_BASE = {"A": 10_000, "B": 20_000}

p = argparse.ArgumentParser()
p.add_argument("--table", choices=MODEL, required=True)
p.add_argument("--index", type=int, required=True)
p.add_argument("--which", choices=SEED_BASE, default="A")
p.add_argument("--out_root", required=True)
args = p.parse_args()

row = pd.read_csv(HERE.parent / "scenarios" / f"{args.table}.csv").iloc[args.index - 1]
spec = json.loads(row.spec)
seed = SEED_BASE[args.which] + args.index
out = Path(args.out_root) / row.id
out.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
mmm = pmm_model.build_and_fit(seed, MODEL[args.table], **scenarios.fit_kwargs(spec))
secs = time.perf_counter() - t0
dt = mmm.idata
free = [v.name for v in mmm.model.free_RVs]
summ = az.summary(dt, var_names=free, kind="diagnostics")
meta = dict(id=row.id, family=row.family, spec=spec, seed=seed, seconds=round(secs),
            divergences=int(dt["sample_stats"]["diverging"].sum()),
            max_r_hat=float(summ["r_hat"].astype(float).max()),
            min_ess_bulk=float(summ["ess_bulk"].astype(float).min()))
dt["posterior"].to_dataset()[KEEP].to_netcdf(out / "refit.nc")
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta))
