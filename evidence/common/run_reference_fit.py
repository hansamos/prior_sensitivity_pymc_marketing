"""Fit the reference model and save the InferenceData (posterior, log_prior, log_likelihood, ROAS).

    python -I evidence/common/run_reference_fit.py --model business --seed 9001 --out runs/pmm/ref_business
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pmm_model  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["business", "lift"], required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--out", required=True, help="output directory")
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
mmm = pmm_model.build_and_fit(args.seed, args.model)
fit_s = time.perf_counter() - t0
mmm.idata.to_netcdf(out / "idata.nc")
(out / "meta.txt").write_text(f"model={args.model}\nseed={args.seed}\nfit_and_derive_seconds={fit_s:.0f}\n")
print(f"done in {fit_s:.0f} s ->", out)
