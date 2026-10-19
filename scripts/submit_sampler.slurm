#!/bin/bash
#SBATCH --job-name=fastwindemu-sampler
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=23          # cluster policy: >= 16 CPUs per GPU on capacity, >= 32 on performance
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sampler-%j.out
#SBATCH --error=logs/sampler-%j.out

# Usage, from the repository folder:
#   sbatch scripts/submit_sampler.slurm --mock --nwalkers 512 --nsteps 20000 --outdir mock_512x20000
# Every argument is forwarded straight to run_sampler_gpu.py.
#
# The #SBATCH lines above are set for one cluster (partition names, GPU type,
# CPU count); change them to match yours.
#
# Partition choice: 'capacity' has 8x NVIDIA L4 (23 GB) per node and is the
# default; 'performance' has 8x RTX 6000 Ada, request it with
#   --partition=performance --gres=gpu:rtx_6000_ada:1 --cpus-per-task=32
# (performance requires 32 CPUs per GPU, more than the 23 requested above)
# This workload is small enough that one GPU is plenty - it is latency bound,
# not throughput bound, so more GPUs would not help a single run.

set -eo pipefail

# SLURM runs a copy of this script, so the repository is found as the folder
# the job was submitted from.
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f "$ROOT/run_sampler_gpu.py" ]]; then
    echo "submit from the repository folder (no run_sampler_gpu.py in $ROOT)" >&2
    exit 1
fi
cd "$ROOT"
mkdir -p logs
source "$ROOT/env.sh"

echo "SLURM_CPUS_PER_TASK : ${SLURM_CPUS_PER_TASK:-unset}"
echo "SLURM_TRES_PER_TASK : ${SLURM_TRES_PER_TASK:-unset}"
echo "SLURM_CPUS_ON_NODE  : ${SLURM_CPUS_ON_NODE:-unset}"

echo "job      : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "gpu      : ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "python   : $(command -v python3)"
echo "args     : $*"
echo "---------------------------------------------------------------"

srun python3 "$ROOT/run_sampler_gpu.py" "$@"
