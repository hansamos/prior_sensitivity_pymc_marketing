# Evidence

Scripts, scenario tables, results and figures behind the numbers in the notebooks `mmm_prior_reweighting` and `mmm_likelihood_reweighting` of [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing). Every experiment compares a posterior reweighted from **one reference fit** with **independent full refits** of the changed model. The model and data are those of PyMC-Marketing's prior and likelihood sensitivity notebook (synthetic weekly data, two channels, 131 weeks).

## Layout

| Folder | Content |
|---|---|
| `common/` | The reweighting core (`reweight.py`), the model (`pmm_model.py`), scenario definitions (`scenarios.py`), fit and refit scripts, analysis and tests. |
| `scenarios/` | `priors.csv` (101 prior changes, observational model) and `likelihood.csv` (43 likelihood and combined changes, lift-calibrated model). One JSON spec per row. |
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
sbatch --array=1-43 evidence/common/run_refit.sbatch --table likelihood --which A --out_root runs/pmm/refits_likelihood_A
sbatch --array=1,18,35,52 evidence/common/run_refit.sbatch --table priors --which B --out_root runs/pmm/refits_priors_B
sbatch --array=1,8,15,22 evidence/common/run_refit.sbatch --table likelihood --which B --out_root runs/pmm/refits_likelihood_B

# reweighted vs refit, tables and figures
python -I evidence/common/analyze.py --table priors --ref runs/pmm/ref_business/idata.nc \
    --refits_a runs/pmm/refits_priors_A --refits_b runs/pmm/refits_priors_B --out evidence/results
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

A pool of 3,000 candidate changes per table, from nearly null to extreme (single changes and combinations of two or three), was reweighted on the reference draws. 20 scenarios per Pareto k̂ bin (priors) or 8 (likelihood) were selected, balanced over families, plus special cases (business prior dropped, lift tests dropped). The selection covers the k̂ range the diagnostic is meant to judge. It does not bias error-given-k̂ results, but the number of scenarios in a bin says nothing about how often a typical change is safe.

## Notes

- Errors are in refit posterior SDs, for ROAS and the media parameters of each channel and the noise scale. "Do nothing" uses the unweighted reference posterior.
- Refits that did not converge (max r̂ above 1.01 or bulk ESS below 400) are no ground truth and are left out of the error statistics: three prior scenarios, all with the business prior removed or weakened, where the posterior of x1 ROAS has two modes. `results/SUMMARY.md` lists them. Divergences are reported, not used to exclude.
- One reference fit per model. For the same change, k̂ differed by up to about 0.25 (priors) and 0.4 (likelihood) between this reference fit and the one in the notebooks.
- The likelihood bins are small (8 to 10 scenarios). Read that table as a pattern, not as precise rates.
- Synthetic data, one dataset, two channels.
