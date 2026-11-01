#!/bin/bash
# Usage (from the repository root):
#   sbatch --job-name=ref_business evidence/common/run_reference_fit.sbatch --model business --seed 9001 --out runs/pmm/ref_business
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}" || exit 1
mkdir -p logs runs/pmm
unset PYTHONPATH
${PYTHON:-python} -I evidence/common/run_reference_fit.py "$@"
