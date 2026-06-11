#!/usr/bin/env bash
# run_plots.sh -- generate per-block, observable-overlay, ZEPE close-up,
# and population-bar plots for one TDSE run.
#
# Usage:
#   ./run_plots.sh PREFIX [--fixed DELTA_kHz] [--no-adaptive]
#
# PREFIX is the path stem of the run, e.g.
#   ../../tdse/run_L4e5_E5m20G/run_L4e5_E5m20G
#
# By default we produce BOTH:
#   * adaptive smoothing  (σ_α = local level spacing)         ← *_dEadap.pdf
#   * fixed Gaussian      (σ = 5 kHz unless --fixed override) ← *_dE5.0.pdf
#
# Pass --no-adaptive to skip the adaptive pass; pass --fixed N to
# override the fixed δE.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 PREFIX [--fixed DELTA_kHz] [--no-adaptive]" >&2
    exit 2
fi

PREFIX="$1"; shift

DELTA_FIXED=5.0
DO_ADAPTIVE=1
DO_FIXED=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fixed)        DELTA_FIXED="$2"; shift 2 ;;
        --no-adaptive)  DO_ADAPTIVE=0; shift ;;
        --no-fixed)     DO_FIXED=0; shift ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLOT="$SCRIPT_DIR/plot_run.py"

if [[ ! -f "$PLOT" ]]; then
    echo "plot_run.py not found at $PLOT" >&2
    exit 1
fi

if [[ "$DO_FIXED" == 1 ]]; then
    echo "==> fixed δE = ${DELTA_FIXED} kHz"
    python3 "$PLOT" "$PREFIX" --delta_E_kHz "$DELTA_FIXED"
fi

if [[ "$DO_ADAPTIVE" == 1 ]]; then
    echo "==> adaptive σ_α (local spacing)"
    python3 "$PLOT" "$PREFIX" --adaptive
fi

echo "done."
