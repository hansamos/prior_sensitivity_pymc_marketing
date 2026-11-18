# Evidence

Scripts, scenario tables, results and figures behind the numbers in the notebooks `mmm_prior_reweighting` and `mmm_likelihood_reweighting` of [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing). Every experiment compares a posterior reweighted from **one reference fit** with **independent full refits** of the changed model. The model and data are those of PyMC-Marketing's prior and likelihood sensitivity notebook (synthetic weekly data, two channels, 131 weeks).

## Layout

| Folder | Content |
|---|---|
| `common/` | The reweighting core (`reweight.py`), the model (`pmm_model.py`), scenario definitions (`scenarios.py`), fit and refit scripts, analysis and tests. |
| `scenarios/` | `priors.csv` (101 prior changes, observational model) and `likelihood.csv` (46 likelihood and combined changes, lift-calibrated model). One JSON spec per row. |
| `results/` | Per-scenario and per-quantity results, tables by Pareto k̂ bin, and `SUMMARY.md`. |
| `figures/` | Error against k̂ and ESS, and reweighting against doing nothing. |

## Setup

Python 3.12 or later.

```
pip install -r evidence/requirements.txt
```

The data are read from the PyMC-Marketing repository on GitHub. Set `PMM_DATA_CSV` to read a local copy.

## Reproducing the experiments

Run everything from the repository root. Fits are written to `runs/pmm/` (untracked, about 1 GB). A fit takes 20 to 45 seconds on 6 cores. The `*.sbatch` files are for Slurm: adapt the partition, and set `PYTHON` if `python` is not your environment's interpreter.

```
# reference fits: observational model (priors) and lift-calibrated model (likelihood)
python -I evidence/common/run_reference_fit.py --model business --seed 9001 --out runs/pmm/ref_business
python -I evidence/common/run_reference_fit.py --model lift --seed 9002 --out runs/pmm/ref_lift

# one refit per scenario; seed A for all rows, seed B for a few (refit noise)
sbatch --array=1-101 evidence/common/run_refit.sbatch --table priors --which A --out_root runs/pmm/refits_priors_A
sbatch --array=1-46 evidence/common/run_refit.sbatch --table likelihood --which A --out_root runs/pmm/refits_likelihood_A
sbatch --array=1,18,35,52 evidence/common/run_refit.sbatch --table priors --which B --out_root runs/pmm/refits_priors_B
sbatch --array=1,8,15,22,44-46 evidence/common/run_refit.sbatch --table likelihood --which B --out_root runs/pmm/refits_likelihood_B

# refit A of rows 1, 65 and 85 had stuck chains: rerun with other seeds (the analysis takes the first converged one)
for w in C D E; do sbatch --array=1,65,85 evidence/common/run_refit.sbatch --table priors --which $w --out_root runs/pmm/rerun_priors_$w; done

# reweighted vs refit, tables and figures
python -I evidence/common/analyze.py --table priors --ref runs/pmm/ref_business/idata.nc \
    --refits_a runs/pmm/refits_priors_A --refits_b runs/pmm/refits_priors_B \
    --reruns runs/pmm/rerun_priors_C runs/pmm/rerun_priors_D runs/pmm/rerun_priors_E --out evidence/results
python -I evidence/common/analyze.py --table likelihood --ref runs/pmm/ref_lift/idata.nc \
    --refits_a runs/pmm/refits_likelihood_A --refits_b runs/pmm/refits_likelihood_B --out evidence/results
python -I evidence/common/summarize.py
python -I evidence/common/make_figures.py
```

`make_tables.py` regenerates the scenario tables from the reference fits (the tables are in `scenarios/`, so this is only needed to change the selection).

**Tests** (no MCMC for the first):
```
python -I evidence/common/test_reweight.py            # analytic checks of the core
sbatch evidence/common/run_parity.sbatch              # NumPy log ratios equal PyMC's log-density differences
```

## How the scenarios were chosen

A pool of 3,000 candidate changes per table, from nearly null to extreme (single changes and combinations of two or three), was reweighted on the reference draws. 20 scenarios per Pareto k̂ bin (priors) or 8 (likelihood) were selected, balanced over families, plus special cases (business prior dropped, lift tests dropped). The selection covers the k̂ range the diagnostic is meant to judge. Three rows were added by hand to test the ESS rule: lift sigma 1.5, 1 and 0.75 (`l_lift_tight_*`). The selection does not bias error-given-k̂ results, but the number of scenarios in a bin says nothing about how often a typical change is safe.

## Notes

- Errors are in refit posterior SDs, for ROAS and the media parameters of each channel and the noise scale: of the mean, the SD and the 5% and 95% quantiles. Quantiles are noisier: two refits differ by about 0.07 at the worst quantile. "Do nothing" uses the unweighted reference posterior.
- A refit counts as converged at max r̂ ≤ 1.01 and bulk ESS ≥ 400. Three prior refits (business prior removed or weakened) had chains stuck in a region the other chains left; they were rerun with other seeds (`p_business_none` needed three tries) and the first converged refit is used. `results/SUMMARY.md` lists them. Refit B is used as a noise baseline only when it converged. Divergences are reported, not used to exclude.
- One reference fit per model. Over eight reference fits of the notebook models, the same change moved k̂ by up to 0.34 (priors) and 0.4 (likelihood).
- The likelihood bins are small (8 to 11 scenarios). Read that table as a pattern, not as precise rates.
- Synthetic data, one dataset, two channels.
