#!/bin/bash
# run_one_grid_task.sbatch -- ONE chunk of the production grid.  Iterates
# rows ROW_LO..ROW_HI of $TABLE, running each parameter set sequentially.
#
# Driver: submit_production_grid.sh sets GRID_DIR, TABLE, EXE,
# BASIS_CACHE, ROW_LO, ROW_HI via --export=ALL.  Do not run directly.
#
# Each chunk fits in 42 h: at τ=30 μs (~50 min/run), 18 sequential runs
# = ~15 h.  τ scaling extends the longest runs; chunk plan in the
# driver picks 18 to leave headroom.

#SBATCH --account=pr28fa
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=95
#SBATCH --time=42:00:00

set -euo pipefail

source /dss/dsshome1/08/di35ker/load_modules_spion.sh > /dev/null 2>&1
module list 2>/dev/null | head -20 || true

# ---- Required env (set by submit_production_grid.sh --export=ALL) -----
: "${GRID_DIR:?GRID_DIR not set}"
: "${TABLE:?TABLE not set}"
: "${EXE:?EXE not set}"
: "${BASIS_CACHE:?BASIS_CACHE not set}"
: "${ROW_LO:?ROW_LO not set}"
: "${ROW_HI:?ROW_HI not set}"

echo "================================================================"
echo "Chunk job $SLURM_JOB_ID: rows $ROW_LO..$ROW_HI of $TABLE"
echo "================================================================"

CHUNK_T0=$SECONDS
N_DONE=0
N_FAIL=0

for ROW_IDX in $(seq "$ROW_LO" "$ROW_HI"); do
    # +1 for header.
    ROW=$(awk -v n="$ROW_IDX" 'NR==(n+1)' "$TABLE")
    if [[ -z "$ROW" ]]; then
        echo "WARN: empty row at index $ROW_IDX, skipping"
        continue
    fi
    TAG=$(echo "$ROW"   | cut -f1)
    OMEGA=$(echo "$ROW" | cut -f2)
    TAU=$(echo "$ROW"   | cut -f3)
    T_US=$(echo "$ROW"  | cut -f4)
    OMR=$(echo "$ROW"   | cut -f5)

    OUTDIR="$GRID_DIR/$TAG"
    mkdir -p "$OUTDIR"

    echo
    echo "----------------------------------------------------------------"
    echo "[chunk $SLURM_JOB_ID  row $ROW_IDX/$ROW_HI]   $TAG"
    echo "  ω=$OMEGA kHz   τ=$TAU μs (T=$T_US)   Ω_R=$OMR kHz"
    echo "  outdir=$OUTDIR"
    echo "----------------------------------------------------------------"
    RUN_T0=$SECONDS

    if time "$EXE" \
        --omp_threads 95 \
        --L 400000 --dr 0.5 \
        --E_cut_open_kHz 10000 \
        --E_cut_m5_kHz 1000000 \
        --omega_kHz "$OMEGA" \
        --tau_us "$TAU" \
        --T_us "$T_US" \
        --dt_us 0.01 \
        --delta_E_kHz 5.0 \
        --Omega_R_kHz "$OMR" \
        --basis_cache_dir "$BASIS_CACHE" \
        --out_prefix "$OUTDIR/$TAG"
    then
        N_DONE=$((N_DONE + 1))
        echo "[chunk $SLURM_JOB_ID]  $TAG OK  ($((SECONDS - RUN_T0)) s)"
    else
        N_FAIL=$((N_FAIL + 1))
        echo "[chunk $SLURM_JOB_ID]  $TAG FAIL after $((SECONDS - RUN_T0)) s"
    fi
done

echo
echo "================================================================"
echo "Chunk $SLURM_JOB_ID summary:"
echo "  rows $ROW_LO..$ROW_HI"
echo "  done : $N_DONE"
echo "  fail : $N_FAIL"
echo "  wall : $((SECONDS - CHUNK_T0)) s"
echo "================================================================"
