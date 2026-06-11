#!/usr/bin/env bash
# run_tdse_lrz.sh -- LRZ SLURM batch script for the recipe full-TDSE
# (Route A) at production size.  Submits one job at the chosen knob
# combination; convergence sweeps are driven by sweep_lrz.sh.
#
#top in cm4: srun -M cm4 --jobid=168943 --overlap --pty top
#salloc in cm4:salloc -M cm4 --account=lxcusers --partition=cm4_tiny --nodes=1 --ntasks-per-node=1 --cpus-per-task=112 --time=24:00:00
# Usage on LRZ:
#     sbatch scripts/run_tdse_lrz.sh \
#         --L 500000 --dr 0.5 \
#         --E_cut_open_kHz 300 --E_cut_m5_kHz 20000000 \
#         --tau_us 30 --T_us 90 --dt_us 0.01 \
#         --out_prefix $WORK/tdse/run_L5e5_E5m20G
#
# Adjust SBATCH partition/account to match your LRZ allocation.
#
#SBATCH --job-name=mc_tdse
#SBATCH --partition=cm2_std            # CooLMUC2 standard; adjust as needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28              # one full CooLMUC2 node
#SBATCH --time=24:00:00
#SBATCH --mem=120G                      # request all memory on the node
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# --------------------------------------------------------------------
# Environment.  Adjust 'module load' lines to match your LRZ stack.
# --------------------------------------------------------------------
module purge
module load slurm_setup
module load gcc
module load cmake
# Eigen3 is usually system; if not, set EIGEN3_INC env var or pass
# -DEigen3_DIR= during cmake configure.

# OpenMP threads = the cpus-per-task above
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-28}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# --------------------------------------------------------------------
# Paths.  $REPO is the multichannel_tdse/full_tdse directory.  Set
# $WORK to your scratch / work area where outputs land.
# --------------------------------------------------------------------
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
BUILD="${BUILD:-$REPO/build_lrz}"
RUN_TDSE="$BUILD/run_tdse"

# --------------------------------------------------------------------
# Build (once per environment; safe to re-run, CMake will skip if up to date).
# --------------------------------------------------------------------
if [[ ! -x "$RUN_TDSE" ]]; then
    echo "[run_tdse_lrz] configuring + building in $BUILD"
    cmake -S "$REPO" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD" -j "$OMP_NUM_THREADS" --target run_tdse
fi

echo "[run_tdse_lrz] launching:"
echo "  $RUN_TDSE $*"
echo "[run_tdse_lrz] OMP_NUM_THREADS=$OMP_NUM_THREADS"
date

srun --cpu-bind=cores "$RUN_TDSE" "$@"

date
echo "[run_tdse_lrz] done."
