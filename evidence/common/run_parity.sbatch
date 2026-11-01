#!/bin/bash
# Usage (from the repository root): sbatch evidence/common/run_parity.sbatch
#SBATCH --job-name=pmm_parity
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}" || exit 1
unset PYTHONPATH
${PYTHON:-python} -I evidence/common/test_scenarios_parity.py "$@"
