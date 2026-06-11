#!/usr/bin/env bash
# submit_ablation.sh -- launch the 3-block (no M_F=-5) ablation run.
#
# Test (i) of validation_tests_spec.txt: rerun the anchor parameters
# with the M_F=-5 ("virtual") block removed from the TDSE basis, to
# measure how much it shifts the M_F=-4 ZEPE peak position and yield.
#
# Usage:
#   ./scripts/submit_ablation.sh
#
# Drops the result into $GRID_DIR/ablation_3block/ alongside the
# existing s1_om8 anchor run.  After it finishes, compare with:
#   python3 scripts/ablation_compare.py $GRID_DIR

set -euo pipefail

CODE_DIR=/dss/dsshome1/08/di35ker/static_exchangeHF/multichannel_tdse/full_tdse
WORK=/hppfs/work/pr28fa/di35ker/multichannel_tdse
EXE=$CODE_DIR/build/run_tdse
# Separate basis-cache subdir: the 3-block basis has a different cache
# key than the full 4-block one (the M_F=-5 block is absent), so we
# don't want to pollute the production cache.
BASIS_CACHE_BASE=$WORK/run_L5e5_E5m20G/.basis_cache
BASIS_CACHE_3B=$WORK/.basis_cache_3block

GRID_DIR=${GRID_DIR:-$(ls -td "$WORK"/production_grid_* 2>/dev/null | head -1)}
if [[ -z "${GRID_DIR:-}" || ! -d "$GRID_DIR" ]]; then
    echo "no production_grid_* directory found in $WORK" >&2
    exit 1
fi
TAG=ablation_3block
OUTDIR=$GRID_DIR/$TAG
mkdir -p "$OUTDIR" "$GRID_DIR/logs" "$BASIS_CACHE_3B"

# Anchor parameters (matches s1_om8).
OMEGA=80.896
TAU=30
T_US=90
OMR=1.0

JID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=mc_${TAG}
#SBATCH --account=pr28fa
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=95
#SBATCH --time=42:00:00
#SBATCH --output=$GRID_DIR/logs/${TAG}_%j.out
#SBATCH --error=$GRID_DIR/logs/${TAG}_%j.err

echo "========================================"
echo "JOB STARTED at \$(date)"
echo "host   = \$(hostname)"
echo "pwd    = \$(pwd)"
echo "jobid  = \$SLURM_JOB_ID"
echo "cpus   = \$SLURM_CPUS_PER_TASK"
echo "tag    = $TAG  (Test i: closed-channel ablation, no M_F=-5)"
echo "========================================"

export outdir="$OUTDIR"

source /dss/dsshome1/08/di35ker/load_modules_spion.sh
echo "module script returned: \$?"
echo
echo "Loaded modules:"
module list

set -uo pipefail

cd "$CODE_DIR"
echo "Now in: \$(pwd)"

mkdir -p "\$outdir"

cmd=(
    "$EXE"
    --omp_threads 95
    --L 400000
    --dr 0.5
    --E_cut_open_kHz 10000
    --E_cut_m5_kHz 1000000
    --omega_kHz "$OMEGA"
    --tau_us "$TAU"
    --T_us "$T_US"
    --dt_us 0.01
    --delta_E_kHz 5.0
    --Omega_R_kHz "$OMR"
    --blocks "-4,-3,-2"
    --basis_cache_dir "$BASIS_CACHE_3B"
    --out_prefix "\$outdir/$TAG"
)

echo "Command:"
printf '  %q' "\${cmd[@]}"
echo
echo

echo "Starting run_tdse at \$(date)"
time "\${cmd[@]}"
rc=\$?

echo
echo "run_tdse exit code: \$rc"
echo "FINISHED at \$(date)"
exit "\$rc"
EOF
)

echo "Submitted ablation 3-block run: jobid=$JID"
echo "Output dir: $OUTDIR"
echo "Watch with:  squeue --me"
echo
echo "Once finished, compare against the s1_om8 anchor:"
echo "  python3 $CODE_DIR/scripts/ablation_compare.py $GRID_DIR"
