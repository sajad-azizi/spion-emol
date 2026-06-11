#!/usr/bin/env python3
"""ablation_compare.py -- Test (i) results in the spec's output format.

Compares the full 4-block anchor run (s1_om8) against the 3-block
ablation run (ablation_3block).  Prints the spec's TEST (i) block plus
the anchor reproduction block.

Usage:
    python3 ablation_compare.py [grid_dir]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

from plot_run import load_states, adaptive_sigma, smooth_states


def parse_summary(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        try: out[k.strip()] = float(v.strip())
        except ValueError: out[k.strip()] = v.strip()
    return out


def zepe_peak(grid_root, tag):
    """Return (E_peak_kHz, dPdE_max) for the M_F=-4 ZEPE channel using
    adaptive Gaussian smoothing on the per-state CSV."""
    sub = grid_root / tag
    sp  = sub / f"{tag}_block_M-4_states.csv"
    if not sp.exists():
        return float("nan"), float("nan")
    E_lab, E_block, prob, _ = load_states(sp)
    sigma = adaptive_sigma(E_block)
    E_grid = np.linspace(0.0, 25.0, 4000)
    y = smooth_states(E_block, prob, E_grid, sigma, only_continuum=True)
    if y.max() <= 0:
        return float("nan"), float("nan")
    return float(E_grid[np.argmax(y)]), float(y.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_dir", nargs="?", default=None,
                    help="production_grid_* directory")
    args = ap.parse_args()
    if args.grid_dir is None:
        candidates = sorted(Path("..").rglob("production_grid_*"))
        if not candidates:
            sys.exit("no production_grid_* found")
        args.grid_dir = str(candidates[-1])
    grid = Path(args.grid_dir).resolve()

    # ---- Anchor reproduction (Run A) --------------------------------
    A_tag = "s1_om8"
    A_summ = grid / A_tag / f"{A_tag}_summary.txt"
    if not A_summ.exists():
        sys.exit(f"missing anchor run: {A_summ}")
    A = parse_summary(A_summ)
    Ep_A, _ = zepe_peak(grid, A_tag)

    # ---- Ablation (Run B) -------------------------------------------
    B_tag = "ablation_3block"
    B_summ = grid / B_tag / f"{B_tag}_summary.txt"
    have_B = B_summ.exists()
    if have_B:
        B = parse_summary(B_summ)
        Ep_B, _ = zepe_peak(grid, B_tag)
    else:
        B = None; Ep_B = float("nan")

    bin_kHz = 5.0  # delta_E_kHz the runs were saved with

    # ---- Print exactly the spec's output block ----------------------
    print("ANCHOR REPRODUCTION CHECK")
    print(f"    halo_retention            = {A['P_halo']:.4f}")
    print(f"    E_peak^ZEPE [kHz]         = {Ep_A:.3f}")
    print(f"    P_ZEPE                    = {A['P_M-4_continuum']:.3e}")
    print(f"    P_1g                      = {A['P_M-3_total']:.3e}")
    print(f"    P_2g_aa                   = {A['P_M-2_total']:.3e}")
    print(f"    P_-5_final                = {A.get('P_M-5_total', float('nan')):.3e}")
    print()
    print("TEST (i) -- CLOSED-CHANNEL ABLATION")
    print(f"    Run A (4-block, full):")
    print(f"        E_peak^ZEPE_A [kHz]   = {Ep_A:.3f}")
    print(f"        P_ZEPE_A              = {A['P_M-4_continuum']:.3e}")
    if have_B:
        print(f"    Run B (3-block, no M_F=-5):")
        print(f"        E_peak^ZEPE_B [kHz]   = {Ep_B:.3f}")
        print(f"        P_ZEPE_B              = {B['P_M-4_continuum']:.3e}")
        dE   = Ep_A - Ep_B
        dP_p = (A['P_M-4_continuum'] - B['P_M-4_continuum']) \
               / A['P_M-4_continuum'] * 100.0
        print(f"    Differences:")
        print(f"        Delta E_peak [kHz]    = {dE:+.3f}")
        print(f"        Delta P_ZEPE / P_ZEPE [percent] = {dP_p:+.2f}")
        print()
        # Verdict per spec acceptance threshold.
        ok_E = abs(dE) < bin_kHz
        ok_P = abs(dP_p) < 5.0
        print("Acceptance check (spec):")
        print(f"    |Delta E_peak| < {bin_kHz:.1f} kHz (one bin):  "
              f"|{dE:+.3f}| {'<' if ok_E else '>='} {bin_kHz:.1f}  "
              f"-> {'PASS' if ok_E else 'FAIL'}")
        print(f"    |Delta P_ZEPE/P_ZEPE| < 5%:               "
              f"|{dP_p:+.2f}| {'<' if ok_P else '>='} 5.0      "
              f"-> {'PASS' if ok_P else 'FAIL'}")
        print()
        if ok_E and ok_P:
            print("Suggested End-Matter sentence:")
            print(f'  "Removing the M_F=-5 block changes the ZEPE-band '
                  f'yield by {abs(dP_p):.2f}% and shifts the peak by '
                  f'{abs(dE):.2f} kHz, less than one energy bin."')
        else:
            print("M_F=-5 contribution is NON-NEGLIGIBLE; the discussion")
            print("paragraph should report the percent-level effect.")
    else:
        print(f"    Run B (3-block, no M_F=-5):  NOT YET RUN")
        print(f"        Submit with:  scripts/submit_ablation.sh")
        print(f"        Expected output dir:  {grid / B_tag}")


if __name__ == "__main__":
    main()
