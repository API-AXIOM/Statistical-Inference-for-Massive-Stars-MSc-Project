#!/bin/bash
# ---------------------------------------------------------------------------
# Shared environment for the fitting programs.
# Source from the login node and from inside SLURM jobs alike:
#     source <repository folder>/env.sh
#
# The module block below loads Python on the cluster this code was developed
# on (EESSI software stack) and is skipped where EESSI is not available. On
# another machine, provide Python 3 with the packages in requirements.txt
# in your own way, or adapt that block.
# ---------------------------------------------------------------------------

# The repository folder: the one holding this file.
export MCMC_GPU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Directory holding the emulator packages: line_emulators_13par/ and fluxcont/.
export EMULATOR_ROOT="${EMULATOR_ROOT:-$MCMC_GPU_ROOT/emulators}"
export MCMC_GPU_VENV="${MCMC_GPU_VENV:-$MCMC_GPU_ROOT/venv}"

if [ -d /cvmfs/software.eessi.io/init/modules ]; then
    # Lmod's init trips over `set -u`, so make sure it is off while sourcing.
    set +u
    if ! type module >/dev/null 2>&1; then
        [ -f /opt/ohpc/admin/lmod/lmod/init/bash ] && source /opt/ohpc/admin/lmod/lmod/init/bash
    fi

    # Pin the EESSI software subdirectory so the venv built on the login node
    # resolves to the exact same interpreter inside the job.
    export EESSI_VERSION=2025.06
    export EESSI_NVIDIA_OVERRIDE_DEFAULT=/opt/eessi/nvidia
    export EESSI_ACCELERATOR_TARGET_OVERRIDE=accel/nvidia/cc80
    export EESSI_SOFTWARE_SUBDIR_OVERRIDE=x86_64/amd/zen4
    module use /cvmfs/software.eessi.io/init/modules
    module --ignore_cache load EESSI/2025.06
    module load Python/3.12.3-GCCcore-13.3.0

    # EESSI runs under its own compat-layer loader, which does not search
    # /usr/lib64 -- so the host NVIDIA driver (libcuda.so.1) is invisible to
    # PyTorch unless the site's host-injection directory is on the search path.
    if [ -d /opt/eessi/nvidia ]; then
        export LD_LIBRARY_PATH="/opt/eessi/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

if [ -f "$MCMC_GPU_VENV/bin/activate" ]; then
    source "$MCMC_GPU_VENV/bin/activate"
fi

# One process drives one GPU here; extra BLAS threads only add contention.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$MCMC_GPU_ROOT/src:$EMULATOR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
