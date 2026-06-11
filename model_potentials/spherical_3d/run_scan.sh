#!/usr/bin/env bash
# run_scan.sh -- spherical_3d ik scan driver.
#
# Runs the spherical_3d executable over a range of ik values.  Energy-
# independent state (bound-state E + eigenfunctions) is cached in
# $WORK/bound_<molhash>.h5 so multiple runs at different continuum
# energies don't recompute the bisection.  Per-ik dipole files
# ($WORK/dipole_<molhash>_<scan_id>/ikNNNN.h5) are atomic-written and
# the binary skips ik's that already exist -- so the script is safe to
# re-run after a crash / preemption.
#
# Optionally runs the parent project's gather_dipoles.py +
# cross_section_delay.py post-processing at the end.  Output is
# byte-compatible with those scripts.
#
# ---------------- SLURM (delete or comment these lines for local runs) ----
#SBATCH --job-name=sph3
#SBATCH --account=pr28fa
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=48:00:00
#SBATCH --output=c8f8_scatter_%j.out
#SBATCH --error=c8f8_scatter_%j.err

source /dss/dsshome1/08/di35ker/load_modules_spion.sh > /dev/null 2>&1

module list


# --------------------------------------------------------------------------

set -euo pipefail

# ---- Default parameters (override on CLI or via env) ---------------------
# Potential
: "${KIND:=cubic}"          # cubic | spherical | gaussian | anis_gauss |
                            # harmonic | soft_coul | h2plus_johnson | free
: "${V0:=0.75}"             # well depth (a.u.)
: "${L100:=294}"            # box size in centi-bohr  (-> L_box = L100 * 0.01)

# Grid + angular cutoff
: "${LMAX:=15}"
: "${NGRID:=10001}"
: "${DR:=0.01}"

# Energy scan (k_n = ik*dk;  E_kin = k_n^2 / 2)
: "${IK_MIN:=10}"
: "${IK_MAX:=300}"
: "${DK:=0.01}"

# Output
: "${SCAN_ID:=}"            # empty -> auto:  ik<min>-<max>_dk<dk>
: "${MOL_NAME:=spherical_3d_model}"
: "${WORK:=/hppfs/work/pr28fa/di35ker/static_exHF/c8f8_data}"
: "${SCRATCH:=/hppfs/work/pr28fa/di35ker/static_exHF/c8f8_data/scratch}"


export SCRATCH=/hppfs/work/pr28fa/di35ker/static_exHF/c8f8_data/scratch

# Threading
: "${OMP_NUM_THREADS:=${SLURM_CPUS_PER_TASK:-112}}"

# Post-processing (set to 0 to skip)
: "${RUN_GATHER:=1}"
: "${RUN_XSEC:=1}"

# ---- Locate binaries -----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/build/spherical_3d}"
PARENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
GATHER="${GATHER:-$PARENT_DIR/postprocessing/gather_dipoles.py}"
XSEC="${XSEC:-$PARENT_DIR/postprocessing/cross_section_delay.py}"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: $BIN not found or not executable." >&2
    echo "  Build with:  cmake -S $SCRIPT_DIR -B $SCRIPT_DIR/build -DCMAKE_BUILD_TYPE=Release && \\" >&2
    echo "               cmake --build $SCRIPT_DIR/build -j" >&2
    exit 1
fi

mkdir -p "$WORK" "$SCRATCH"
export OMP_NUM_THREADS

# ---- CLI override ---------------------------------------------------------
# Accept three positional args: ik_min ik_max dk  (matches parent's pattern).
if [[ $# -ge 3 ]]; then IK_MIN="$1"; IK_MAX="$2"; DK="$3"; fi

# ---- Banner --------------------------------------------------------------
echo "============================================================"
echo " spherical_3d ik scan"
echo "============================================================"
echo "  kind        = $KIND"
echo "  V0          = $V0      L100 = $L100   (-> L_box = $(awk "BEGIN{print $L100*0.01}"))"
echo "  lmax        = $LMAX     N_grid = $NGRID    dr = $DR"
echo "  ik          = [$IK_MIN .. $IK_MAX]    dk = $DK"
echo "  WORK        = $WORK"
echo "  SCRATCH     = $SCRATCH"
echo "  threads     = $OMP_NUM_THREADS"
echo "  bin         = $BIN"
[[ -n "$SCAN_ID" ]] && echo "  scan_id     = $SCAN_ID"
echo "------------------------------------------------------------"

t_start=$SECONDS

# ---- 1. ik scan ----------------------------------------------------------
EXTRA_ARGS=()
[[ -n "$SCAN_ID" ]] && EXTRA_ARGS+=( -scan-id "$SCAN_ID" )

WORK="$WORK" SCRATCH="$SCRATCH" "$BIN" \
    -k "$KIND" \
    -V0 "$V0" \
    -L "$L100" \
    -lmax "$LMAX" \
    -N "$NGRID" \
    -dr "$DR" \
    -ik-min "$IK_MIN" \
    -ik-max "$IK_MAX" \
    -dk "$DK" \
    -name "$MOL_NAME" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# ---- Locate the scan directory the binary just wrote ---------------------
# It uses molhash(potential params) + scan_id; safest to glob since we
# don't replicate the C++ hash here.
SCAN_DIR="$(ls -dt "$WORK"/dipole_*_* 2>/dev/null | head -n 1 || true)"
if [[ -z "$SCAN_DIR" ]]; then
    echo "ERROR: no dipole_<hash>_<scan> directory found under $WORK" >&2
    exit 1
fi
echo
echo "[run_scan] scan dir = $SCAN_DIR"

# ---- 2. Gather (per-channel .dat files) ----------------------------------
if [[ "$RUN_GATHER" == "1" ]]; then
    echo
    echo "============================================================"
    echo " gather_dipoles.py"
    echo "============================================================"
    if [[ -f "$GATHER" ]]; then
        python3 "$GATHER" "$SCAN_DIR" \
            --output-dir "$WORK/gathered_$(basename "$SCAN_DIR")"
    else
        echo "  SKIP: $GATHER not found"
    fi
fi

# ---- 3. Cross-section + delay + beta -------------------------------------
if [[ "$RUN_XSEC" == "1" ]]; then
    echo
    echo "============================================================"
    echo " cross_section_delay.py"
    echo "============================================================"
    GATHERED="$WORK/gathered_$(basename "$SCAN_DIR")"
    XSEC_OUT="$WORK/xsec_$(basename "$SCAN_DIR")"
    if [[ -d "$GATHERED" && -f "$XSEC" ]]; then
        python3 "$XSEC" "$GATHERED" --output-dir "$XSEC_OUT"
    else
        echo "  SKIP: $GATHERED or $XSEC missing"
    fi
fi

t_end=$SECONDS
echo
echo "============================================================"
echo " DONE.  total wall = $((t_end - t_start)) s"
echo "============================================================"
echo "  scan dir   : $SCAN_DIR"
[[ "$RUN_GATHER" == "1" ]] && echo "  gathered   : $WORK/gathered_$(basename "$SCAN_DIR")"
[[ "$RUN_XSEC"   == "1" ]] && echo "  xsec       : $WORK/xsec_$(basename "$SCAN_DIR")"
