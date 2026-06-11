#!/usr/bin/env bash
# submit_range.sh -- submit ONE SLURM job per parameter row, for a
# range LO..HI of runs.tsv.
#
# Usage:
#   ./scripts/submit_range.sh 1 10
#   ./scripts/submit_range.sh 11 20
#
# Override grid:
#   GRID_DIR=/path/to/production_grid_xxx ./scripts/submit_range.sh 1 10

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 LO HI" >&2
    exit 2
fi

LO=$1
HI=$2

# ---- paths ------------------------------------------------------------
WORK=/hppfs/work/pr28fa/di35ker/multichannel_tdse
CODE_DIR=/dss/dsshome1/08/di35ker/static_exchangeHF/multichannel_tdse/full_tdse
EXE=$CODE_DIR/build/run_tdse
BASIS_CACHE=$WORK/run_L5e5_E5m20G/.basis_cache

GRID_DIR=${GRID_DIR:-$(ls -td "$WORK"/production_grid_* 2>/dev/null | head -1)}

if [[ -z "${GRID_DIR:-}" || ! -d "$GRID_DIR" ]]; then
    echo "no production_grid_* directory found in $WORK" >&2
    echo "run submit_production_grid.sh --dry-run first to create the table." >&2
    exit 1
fi

TABLE=$GRID_DIR/runs.tsv

if [[ ! -f "$TABLE" ]]; then
    echo "param table not found: $TABLE" >&2
    exit 1
fi

if [[ ! -x "$EXE" ]]; then
    echo "executable not found or not executable: $EXE" >&2
    exit 1
fi

mkdir -p "$GRID_DIR/logs"

echo "Grid       : $GRID_DIR"
echo "Code dir   : $CODE_DIR"
echo "Executable : $EXE"
echo "Table      : $TABLE"
echo "Submitting : rows $LO..$HI"
echo

for ROW_IDX in $(seq "$LO" "$HI"); do
    # Read row IDX from runs.tsv.
    # NR==(ROW_IDX+1) skips the header.
    # tr -d '\r' removes possible Windows CRLF characters.
    ROW=$(awk -v n="$ROW_IDX" 'NR==(n+1)' "$TABLE" | tr -d '\r')

    if [[ -z "$ROW" ]]; then
        echo "  row $ROW_IDX empty, stop"
        break
    fi

    IFS=$'\t' read -r TAG OMEGA TAU T_US OMR EXTRA <<< "$ROW"

    if [[ -z "${TAG:-}" || -z "${OMEGA:-}" || -z "${TAU:-}" || -z "${T_US:-}" || -z "${OMR:-}" ]]; then
        echo "ERROR: bad row $ROW_IDX:" >&2
        printf '[%s]\n' "$ROW" >&2
        exit 1
    fi

    OUTDIR=$GRID_DIR/$TAG
    mkdir -p "$OUTDIR"

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
echo "tag    = $TAG"
echo "========================================"

export outdir="$OUTDIR"

echo
echo "Loading modules..."
source /dss/dsshome1/08/di35ker/load_modules_spion.sh
echo "module script returned: \$?"

echo
echo "Loaded modules:"
module list

set -uo pipefail

echo
echo "Changing to code directory..."
cd "$CODE_DIR"

echo "Now in: \$(pwd)"
echo "Executable:"
ls -lh "$EXE"

echo
echo "Output directory:"
mkdir -p "\$outdir"
ls -ld "\$outdir"

echo
echo "Parameters:"
echo "  TAG         = $TAG"
echo "  omega_kHz   = $OMEGA"
echo "  tau_us      = $TAU"
echo "  T_us        = $T_US"
echo "  Omega_R_kHz = $OMR"
echo "  outdir      = \$outdir"
echo

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
    --basis_cache_dir "$BASIS_CACHE"
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

    echo "  row $ROW_IDX  $TAG  jobid=$JID"
done

echo
echo "Submitted rows $LO..$HI. Watch with:"
echo "  squeue --me"
echo
echo "Next batch:"
echo "  ./scripts/submit_range.sh $((HI+1)) $((HI+10))"
