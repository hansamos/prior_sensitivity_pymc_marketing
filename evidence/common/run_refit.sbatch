#!/bin/bash
# Usage (from the repository root), one array task per scenario row:
#   sbatch --array=1-100%50 --job-name=rf_pri_A evidence/common/run_refit.sbatch --table priors --which A --out_root runs/pmm/refits_priors_A
#   sbatch --array=1-100%50 --job-name=rf_lik_A evidence/common/run_refit.sbatch --table likelihood --which A --out_root runs/pmm/refits_likelihood_A
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}" || exit 1
unset PYTHONPATH
${PYTHON:-python} -I evidence/common/run_refit.py --index "$SLURM_ARRAY_TASK_ID" "$@"
