#!/usr/bin/env bash
# sweep_lrz.sh -- recipe convergence sweep for full-TDSE Route A.
#
# Submits a series of sbatch jobs varying one knob at a time, holding
# the others at the recipe reference point.  After all jobs finish,
# postprocess the dP/dE CSVs and check 1% convergence of:
#
#     P_ZEPE^(-4), P_aa^(-2), E_peak^(-4), P_ea/P_ae, Δ_coh
#
# (recipe item 6 of the convergence checklist).  The recipe says to
# run E_cut^(-5) in {5, 10, 20, 50} GHz; that's the FIRST sweep here.
# The second sweep doubles L; the third doubles 1/dr (refines grid).
#
# Usage on LRZ:
#     bash scripts/sweep_lrz.sh
# (run on the login node; this just submits jobs to the queue.)
#
# Output layout under $OUT_BASE:
#     $OUT_BASE/conv_E5_<X>GHz/   one per E_cut^(-5) value
#     $OUT_BASE/conv_L_<X>/        one per box-length value
#     $OUT_BASE/conv_dr_<X>/       one per dr value

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="${OUT_BASE:-${WORK:-$HOME}/mc_tdse_conv}"
mkdir -p "$OUT_BASE"

# Recipe reference point.  Change carefully.
L_REF=500000              # a_0 (recipe says ~5e5 a_0)
DR_REF=0.5                # a_0
E_CUT_OPEN_REF=300        # kHz, recipe = 30·E_b ≈ 300 kHz
E_CUT_M5_REF_kHz=20000000 # 20 GHz, recipe practical starting point
TAU_REF=30                # μs
T_REF=90                  # μs
DT_REF=0.01               # μs
DELTA_E_REF=1.0           # kHz, dP/dE smoothing width

# Submit one run.  Args: tag, L, dr, E_open_kHz, E_m5_kHz
submit_run() {
    local tag="$1"
    local L="$2"
    local dr="$3"
    local Eo="$4"
    local Em5="$5"

    local outdir="$OUT_BASE/$tag"
    mkdir -p "$outdir"
    local prefix="$outdir/run"

    echo "[sweep] submitting $tag  L=$L dr=$dr E_open=$Eo E_m5=$Em5"
    sbatch \
        --job-name="mc_tdse_${tag}" \
        "$REPO/scripts/run_tdse_lrz.sh" \
            --L "$L" --dr "$dr" \
            --E_cut_open_kHz "$Eo" --E_cut_m5_kHz "$Em5" \
            --tau_us "$TAU_REF" --T_us "$T_REF" --dt_us "$DT_REF" \
            --delta_E_kHz "$DELTA_E_REF" \
            --out_prefix "$prefix"
}

# ---- Sweep 1: E_cut^(-5) ramp (recipe item 6) ----
for Em5_GHz in 5 10 20 50; do
    Em5_kHz=$((Em5_GHz * 1000000))
    submit_run "conv_E5_${Em5_GHz}GHz" "$L_REF" "$DR_REF" "$E_CUT_OPEN_REF" "$Em5_kHz"
done

# ---- Sweep 2: box-length L ramp (recipe item 7) ----
for L in 250000 500000 1000000; do
    submit_run "conv_L_${L}" "$L" "$DR_REF" "$E_CUT_OPEN_REF" "$E_CUT_M5_REF_kHz"
done

# ---- Sweep 3: open-block cutoff ramp (recipe item 5) ----
for Eo in 200 300 400; do
    submit_run "conv_Eo_${Eo}kHz" "$L_REF" "$DR_REF" "$Eo" "$E_CUT_M5_REF_kHz"
done

# ---- Sweep 4: time-step ramp (recipe item 4) ----
# (Different driver flag, so use a custom wrapper.  For dt sweep just
# call run_tdse directly with --dt_us values.)
for dt in 0.02 0.01 0.005; do
    tag="conv_dt_${dt}"
    outdir="$OUT_BASE/$tag"
    mkdir -p "$outdir"
    sbatch \
        --job-name="mc_tdse_${tag}" \
        "$REPO/scripts/run_tdse_lrz.sh" \
            --L "$L_REF" --dr "$DR_REF" \
            --E_cut_open_kHz "$E_CUT_OPEN_REF" \
            --E_cut_m5_kHz "$E_CUT_M5_REF_kHz" \
            --tau_us "$TAU_REF" --T_us "$T_REF" --dt_us "$dt" \
            --delta_E_kHz "$DELTA_E_REF" \
            --out_prefix "$outdir/run"
done

echo "[sweep] queued.  Watch with 'squeue -u $USER'."
echo "[sweep] outputs land in $OUT_BASE."
