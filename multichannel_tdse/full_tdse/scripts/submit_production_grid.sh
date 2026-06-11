#!/usr/bin/env bash
# submit_production_grid.sh -- generate the production parameter grid
# and submit it as N CHUNKED SLURM JOBS on LRZ.
#
# LRZ user limits: max 2 jobs running, max 10 jobs submitted (queue +
# running).  So we can NOT use a 81-task array job (each task counts as
# one queued job, exceeds the 10-submit cap).  Instead we split the
# parameter table into a small number of CHUNK jobs, each of which runs
# its slice of param sets SEQUENTIALLY within its 42-h wall.  With max-
# 2-running, two chunks execute in parallel and the rest queue.
#
# Usage:
#   ./submit_production_grid.sh                 # 21 runs in 1 chunk
#   ./submit_production_grid.sh --with-2d       # 81 runs in ~5 chunks
#   ./submit_production_grid.sh --chunks N      # force N chunks
#   ./submit_production_grid.sh --dry-run       # print table + chunk plan, no submit
#
# Run-time budget per chunk: ≤ 42 h.  At τ=30 μs, ~50 min/run.  At τ=100
# μs the run is ~3× longer (T scales with τ).  We default to 18 runs per
# chunk, which leaves headroom for the longest τ + basis-rebuild buffer.

set -euo pipefail

# ---- LRZ paths (edit to match your environment) -----------------------
WORK=/hppfs/work/pr28fa/di35ker/multichannel_tdse
EXE=$WORK/build/run_tdse
BASIS_CACHE=$WORK/run_L5e5_E5m20G/.basis_cache
SBATCH_TPL="$(cd "$(dirname "$0")" && pwd)/run_one_grid_task.sbatch"

# ---- Physics constant -------------------------------------------------
E_B_KHZ=10.112      # halo binding at B = 155.04 G (recipe field)

# ---- Fixed (anchor) settings ------------------------------------------
TAU0_US=30
OMR0_KHZ=1.0
OMEGA0_KHZ=$(python3 -c "print(8 * ${E_B_KHZ})")     # = 80.896

# ---- 1-D scans --------------------------------------------------------
OMEGA_RATIOS=(3 4 5 6 8 10 12)
TAUS_US=(15 20 25 30 40 50 75 100)
OMRS_KHZ=(0.2 0.5 1.0 1.5 2.0 3.0)

# ---- 2-D grid (optional) ----------------------------------------------
TWO_D_TAUS=(20 30 50 75)
TWO_D_OMEGAS=(4 6 8 10 12)
TWO_D_OMRS=(1.0 1.5 2.0)

# ---- Chunk default ----------------------------------------------------
DEFAULT_RUNS_PER_CHUNK=18

# ---- Parse flags ------------------------------------------------------
WITH_2D=0
DRY_RUN=0
CHUNKS_OVERRIDE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-2d)  WITH_2D=1; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --chunks)   CHUNKS_OVERRIDE="$2"; shift 2 ;;
        -h|--help)  sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1"; exit 2 ;;
    esac
done

# ---- Build the parameter table ---------------------------------------
GRID_DIR="$WORK/production_grid_$(date +%Y%m%d_%H%M)"
mkdir -p "$GRID_DIR"
TABLE="$GRID_DIR/runs.tsv"
printf "tag\tomega_kHz\ttau_us\tT_us\tOmega_R_kHz\n" > "$TABLE"

T_for_tau() { python3 -c "print(max(90, 3 * $1))"; }

# Scan 1: ω/E_b ratio scan.
for r in "${OMEGA_RATIOS[@]}"; do
    omega=$(python3 -c "print(${r} * ${E_B_KHZ})")
    printf "s1_om%g\t%s\t%s\t%s\t%s\n" \
        "$r" "$omega" "$TAU0_US" "$(T_for_tau $TAU0_US)" "$OMR0_KHZ" >> "$TABLE"
done
# Scan 2: τ scan.
for tau in "${TAUS_US[@]}"; do
    printf "s2_tau%g\t%s\t%s\t%s\t%s\n" \
        "$tau" "$OMEGA0_KHZ" "$tau" "$(T_for_tau $tau)" "$OMR0_KHZ" >> "$TABLE"
done
# Scan 3: Ω_R scan.
for omr in "${OMRS_KHZ[@]}"; do
    printf "s3_om_r%g\t%s\t%s\t%s\t%s\n" \
        "$omr" "$OMEGA0_KHZ" "$TAU0_US" "$(T_for_tau $TAU0_US)" "$omr" >> "$TABLE"
done
# 2-D grid (optional).
if [[ "$WITH_2D" -eq 1 ]]; then
    for tau in "${TWO_D_TAUS[@]}"; do
        for r in "${TWO_D_OMEGAS[@]}"; do
            omega=$(python3 -c "print(${r} * ${E_B_KHZ})")
            for omr in "${TWO_D_OMRS[@]}"; do
                printf "g2_t%g_om%g_r%g\t%s\t%s\t%s\t%s\n" \
                    "$tau" "$r" "$omr" "$omega" "$tau" "$(T_for_tau $tau)" "$omr" >> "$TABLE"
            done
        done
    done
fi

N=$(($(wc -l < "$TABLE") - 1))
echo "Grid root  : $GRID_DIR"
echo "Param table: $TABLE   ($N runs)"

# ---- Compute chunk plan ----------------------------------------------
if [[ "$CHUNKS_OVERRIDE" -gt 0 ]]; then
    NCHUNKS=$CHUNKS_OVERRIDE
    CHUNK_SIZE=$(( (N + NCHUNKS - 1) / NCHUNKS ))
else
    CHUNK_SIZE=$DEFAULT_RUNS_PER_CHUNK
    NCHUNKS=$(( (N + CHUNK_SIZE - 1) / CHUNK_SIZE ))
fi

echo "Chunk plan : $NCHUNKS chunk(s) of up to $CHUNK_SIZE runs each"
echo
column -ts $'\t' "$TABLE" | head -8
echo "..."

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "Chunk slices (LO..HI of $TABLE, 1-indexed, header excluded):"
    for c in $(seq 0 $((NCHUNKS - 1))); do
        LO=$((c * CHUNK_SIZE + 1))
        HI=$(( (c + 1) * CHUNK_SIZE ))
        [[ $HI -gt $N ]] && HI=$N
        echo "  chunk $((c+1))/$NCHUNKS: rows $LO..$HI"
    done
    echo
    echo "Dry-run mode: not submitting.  Inspect $TABLE and rerun without --dry-run."
    exit 0
fi

# ---- Submit one SBATCH per chunk -------------------------------------
mkdir -p "$GRID_DIR/logs"
JOBIDS=()
for c in $(seq 0 $((NCHUNKS - 1))); do
    LO=$((c * CHUNK_SIZE + 1))
    HI=$(( (c + 1) * CHUNK_SIZE ))
    [[ $HI -gt $N ]] && HI=$N
    JOBNAME="mcgrid_$(basename $GRID_DIR)_c$((c+1))of${NCHUNKS}"
    OUT_LOG="$GRID_DIR/logs/chunk_$((c+1))_%j.out"
    ERR_LOG="$GRID_DIR/logs/chunk_$((c+1))_%j.err"
    JID=$(sbatch --parsable \
        --job-name="$JOBNAME" \
        --output="$OUT_LOG" \
        --error="$ERR_LOG" \
        --export=ALL,GRID_DIR="$GRID_DIR",TABLE="$TABLE",EXE="$EXE",BASIS_CACHE="$BASIS_CACHE",ROW_LO=$LO,ROW_HI=$HI \
        "$SBATCH_TPL")
    JOBIDS+=("$JID")
    echo "  submitted chunk $((c+1))/$NCHUNKS  rows $LO..$HI  jobid $JID"
done

echo
echo "All $NCHUNKS chunk jobs submitted."
echo "Job IDs    : ${JOBIDS[*]}"
echo "Per-chunk logs in $GRID_DIR/logs/"
echo
echo "LRZ will run ≤ 2 chunks at once; the rest sit in the queue.  Watch with:"
echo "  squeue -u \$USER"
echo
echo "Once all chunks finish, post-process each subdirectory with:"
echo "  for d in $GRID_DIR/*/; do scripts/run_plots.sh \$d/\$(basename \$d); done"
