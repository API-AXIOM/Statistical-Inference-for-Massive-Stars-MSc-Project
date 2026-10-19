#!/bin/bash
# Grab an interactive GPU shell with the environment already loaded.
#   ./scripts/interactive_gpu.sh [hours]
# [hours] is the session length in whole hours (default 2); fractions such as
# 0.5 are not accepted.
#
# The partition, GPU type and CPU count below are set for one cluster; change
# them to match yours, as in submit_sampler.slurm.
HOURS="${1:-2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the repository folder
exec srun --partition=capacity --gres=gpu:l4:1 --cpus-per-task=16 \
    --mem=64G --time="${HOURS}:00:00" --pty \
    bash --init-file "$ROOT/env.sh"
