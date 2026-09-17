#!/bin/bash
#SBATCH --job-name=econ282e
#SBATCH --account=bepc-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=tools/logs/%x-%j.out
#SBATCH --error=tools/logs/%x-%j.err
#
# Run course code on a compute node.  Notebooks, simulations and homework
# reference code must NEVER run on the Delta login node (CLAUDE.md -> Build).
#
# Why a GPU partition for CPU work: the only account available here
# (bepc-delta-gpu) is a GPU account, and Delta rejects jobs that request no GPU
# resources under such an account.  One GPU is the smallest allocation that is
# accepted.  The S7-S8 deep-learning labs will use it for real.
#
# Usage, from the course root:
#     sbatch tools/run_code.slurm                       # regenerate the S3 ledger
#     sbatch tools/run_code.slurm python3 tools/figures/s03_generate.py
#     sbatch tools/run_code.slurm jupyter nbconvert --execute --inplace labs/L03_Python_and_VFI.ipynb

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p tools/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "host      : $(hostname)"
echo "started   : $(date -Is)"
echo "python    : $(command -v python3)"
python3 -c "import numpy, scipy; print('numpy', numpy.__version__, '| scipy', scipy.__version__)"
echo "command   : ${*:-python3 tools/figures/s03_generate.py}"
echo "---"

if [ "$#" -eq 0 ]; then
    python3 tools/figures/s03_generate.py
else
    "$@"
fi

echo "---"
echo "finished  : $(date -Is)"
