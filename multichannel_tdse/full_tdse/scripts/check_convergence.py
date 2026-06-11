#!/usr/bin/env python3
"""Post-process a convergence sweep produced by sweep_lrz.sh.

Reads every {tag}/run_summary.txt + {tag}/run_block_M*.csv pair under
$OUT_BASE and reports the recipe convergence observables:

    P_ZEPE^(-4), P_aa^(-2), E_peak^(-4), P_ea/P_ae, P_M-3 total

It prints a table (one row per knob value) and computes the relative
change between consecutive runs.  Recipe item 6: stop ramping when
the relative change is below 1%.

Run on LRZ (or anywhere with python3 + pandas/numpy) AFTER all sweep
jobs have finished:

    python3 scripts/check_convergence.py $OUT_BASE
"""
import sys
import csv
import re
from pathlib import Path
from collections import defaultdict

import numpy as np


def parse_summary(path: Path) -> dict:
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                out[k.strip()] = v.strip()
    return out


def load_dPdE(path: Path):
    E_kHz = []
    y     = []
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            E_kHz.append(float(row[0]))
            y.append(float(row[1]))
    return np.array(E_kHz), np.array(y)


def integrate_peak(E_kHz, y, E_lo_kHz, E_hi_kHz):
    """Integrate dP/dE over [E_lo, E_hi] (trapezoid)."""
    mask = (E_kHz >= E_lo_kHz) & (E_kHz <= E_hi_kHz)
    if mask.sum() < 2:
        return 0.0
    return np.trapz(y[mask], E_kHz[mask])


def find_peak_E(E_kHz, y, E_lo_kHz, E_hi_kHz):
    mask = (E_kHz >= E_lo_kHz) & (E_kHz <= E_hi_kHz)
    if mask.sum() == 0:
        return float("nan")
    sub_E = E_kHz[mask]
    sub_y = y[mask]
    return sub_E[np.argmax(sub_y)]


def analyze_run(run_dir: Path) -> dict:
    summary = parse_summary(run_dir / "run_summary.txt")
    out = dict(summary)

    # E_h is the halo binding (lowest M_F=-4 state).  Read from M-4 dPdE.
    paths = {
        -5: run_dir / "run_block_M-5_dPdE.csv",
        -4: run_dir / "run_block_M-4_dPdE.csv",
        -3: run_dir / "run_block_M-3_dPdE.csv",
        -2: run_dir / "run_block_M-2_dPdE.csv",
    }
    for mf, p in paths.items():
        if not p.exists():
            continue
        E_kHz, y = load_dPdE(p)
        # Recipe peaks: ZEPE near halo (-10 kHz); 1γ around +ω; 2γ around +(2ω - E_b);
        # use ±50 kHz windows around the expected peaks.
        out[f"area_M{mf}_full"] = np.trapz(y, E_kHz)
        if mf == -4:
            out["E_peak_M-4"]   = find_peak_E(E_kHz, y, -50.0, +50.0)
            out["P_ZEPE"]       = integrate_peak(E_kHz, y, -50.0, +200.0) - integrate_peak(E_kHz, y, -50.0, -5.0)
        elif mf == -3:
            out["P_1gamma"]     = integrate_peak(E_kHz, y, +20.0, +120.0)
        elif mf == -2:
            out["P_aa"]         = integrate_peak(E_kHz, y, +100.0, +200.0)
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    base = Path(sys.argv[1])
    if not base.is_dir():
        print(f"ERROR: {base} is not a directory")
        sys.exit(1)

    by_sweep = defaultdict(list)
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "run_summary.txt").exists():
            continue
        m = re.match(r"^conv_([A-Za-z0-9]+)_(.+)$", run_dir.name)
        if not m:
            continue
        sweep_name = m.group(1)
        knob_value = m.group(2)
        try:
            res = analyze_run(run_dir)
        except Exception as e:
            print(f"WARN: failed to read {run_dir.name}: {e}")
            continue
        res["__knob_value__"] = knob_value
        by_sweep[sweep_name].append(res)

    for sweep_name, runs in by_sweep.items():
        print(f"\n=== sweep: {sweep_name}  ({len(runs)} runs) ===")
        if not runs:
            continue
        cols = ["__knob_value__", "P_ZEPE", "P_1gamma", "P_aa",
                "E_peak_M-4", "P_M-5", "err_unitary"]
        # Print header
        print("  " + "  ".join(f"{c:>14}" for c in cols))
        for r in sorted(runs, key=lambda x: x["__knob_value__"]):
            cells = []
            for c in cols:
                v = r.get(c, float("nan"))
                if isinstance(v, str):
                    cells.append(f"{v:>14}")
                else:
                    cells.append(f"{v:>14.4e}")
            print("  " + "  ".join(cells))
        # Convergence: relative change between consecutive runs (sorted).
        ordered = sorted(runs, key=lambda x: x["__knob_value__"])
        if len(ordered) >= 2:
            print("  relative changes:")
            for c in ["P_ZEPE", "P_1gamma", "P_aa"]:
                rel = []
                for i in range(1, len(ordered)):
                    a = ordered[i - 1].get(c, float("nan"))
                    b = ordered[i].get(c, float("nan"))
                    if a == 0 or np.isnan(a) or np.isnan(b):
                        rel.append("--")
                    else:
                        rel.append(f"{abs(b - a) / abs(a) * 100:.2f}%")
                print(f"    Δ{c:8} {' '.join(f'{r:>8}' for r in rel)}")


if __name__ == "__main__":
    main()
