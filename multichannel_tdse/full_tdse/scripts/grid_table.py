#!/usr/bin/env python3
"""grid_table.py -- summarise every run in a production_grid_* directory.

Reads each <tag>/<tag>_summary.txt, prints a table with parameters,
populations, pulse-area, and PT-regime classification.

Usage:
    python3 grid_table.py [grid_dir]

Default grid_dir is the newest production_grid_* under the cwd's
sibling tdse/ tree (or pass it explicitly).

PT-regime rule of thumb (pulse area A = √(2π) · Ω_R · τ in radians;
in our knob units A = 2π·Ω_R[kHz]·√(2π)·τ[μs]·1e-3 since 1 kHz · 1 μs
= 2π·1e-3 rad of phase):

    A < 0.30 rad   →  deep PT          (1γ is leading, Ω² scaling exact)
    0.30–0.70 rad  →  PT               (clean Ω²/Ω⁴; ~0.5–5% depletion)
    0.70–1.50 rad  →  marginal         (saturation onset visible)
    > 1.50 rad     →  saturated        (NOT PT — stop trusting closed-form)

The pulse-area number is the cleanest PT-regime indicator because every
relevant probability scales as A^n at fixed pulse shape.

Halo depletion (1 − P_halo) is the empirical cross-check.  In PT they
agree.
"""
import argparse
import math
import re
from pathlib import Path


def parse_summary(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            out[k.strip()] = v.strip()
    return out


# Halo binding (used for ω/E_b ratio).
E_B_KHZ = 10.112


def pulse_area_rad(Omega_R_kHz: float, tau_us: float) -> float:
    """A = √(2π) · Ω_R · τ in radians.  With Ω_R in 2π·kHz units (the
    code's convention: Omega_R_au = 2π · kHz_to_au(Omega_R_kHz))."""
    return 2.0 * math.pi * Omega_R_kHz * 1e-3 * math.sqrt(2.0 * math.pi) * tau_us


def regime_of(A: float) -> str:
    if A < 0.30:   return "deep PT"
    if A < 0.70:   return "PT"
    if A < 1.50:   return "marginal"
    return "SATURATED"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid_dir", nargs="?", default=None)
    ap.add_argument("--csv", action="store_true",
                    help="emit comma-separated for spreadsheet import")
    args = ap.parse_args()

    if args.grid_dir is None:
        candidates = sorted(Path("..").rglob("production_grid_*"))
        if not candidates:
            raise SystemExit("no production_grid_* found; pass an explicit path")
        args.grid_dir = str(candidates[-1])
    grid = Path(args.grid_dir).resolve()

    rows = []
    for sub in sorted(grid.iterdir()):
        if not sub.is_dir():
            continue
        # Find the summary file for this run.  Prefer <tag>_summary.txt
        # in the subdir; otherwise skip (logs/ etc.).
        tag = sub.name
        summ = sub / f"{tag}_summary.txt"
        if not summ.exists():
            cands = list(sub.glob("*_summary.txt"))
            if not cands:
                continue
            summ = cands[0]
        s = parse_summary(summ)
        if "Omega_R_kHz" not in s or "tau_us" not in s or "P_halo" not in s:
            continue

        omega = s["omega_kHz"]
        tau   = s["tau_us"]
        OmR   = s["Omega_R_kHz"]
        A     = pulse_area_rad(OmR, tau)
        rows.append({
            "tag":        tag,
            "ω/E_b":      omega / E_B_KHZ,
            "ω_kHz":      omega,
            "τ_μs":       tau,
            "Ω_R_kHz":    OmR,
            "A_rad":      A,
            "P_halo":     s["P_halo"],
            "P_1γ":       s.get("P_M-3_total", 0.0),
            "P_ZEPE":     s.get("P_M-4_continuum", 0.0),
            "P_2γ":       s.get("P_M-2_total", 0.0),
            "P_M-5":      s.get("P_M-5_total", 0.0),
            "deplete":    1.0 - s["P_halo"],
            "regime":     regime_of(A),
            "‖c‖−1":      s.get("err_unitary", 0.0),
        })

    if not rows:
        raise SystemExit(f"no completed runs found under {grid}")

    # ---- Sort: by scan group (s1 / s2 / s3 / g2) then natural ----
    def sortkey(r):
        m = re.match(r"(s[123]|g[2-9])_", r["tag"])
        group = m.group(1) if m else "z"
        return (group, r["ω/E_b"], r["τ_μs"], r["Ω_R_kHz"])
    rows.sort(key=sortkey)

    # ---- Output ----
    cols = ["tag", "ω/E_b", "τ_μs", "Ω_R_kHz", "A_rad",
            "P_halo", "deplete", "P_1γ", "P_ZEPE", "P_2γ", "P_M-5",
            "regime", "‖c‖−1"]

    if args.csv:
        print(",".join(cols))
        for r in rows:
            print(",".join(f"{r[c]:.4e}" if isinstance(r[c], float) else str(r[c])
                           for c in cols))
        return

    fmt = {
        "tag":     ("{:<14}",  "{:<14}"),
        "ω/E_b":   ("{:>6}",   "{:>6.2f}"),
        "τ_μs":    ("{:>5}",   "{:>5.0f}"),
        "Ω_R_kHz": ("{:>7}",   "{:>7.2f}"),
        "A_rad":   ("{:>6}",   "{:>6.3f}"),
        "P_halo":  ("{:>9}",   "{:>9.6f}"),
        "deplete": ("{:>9}",   "{:>9.4f}"),
        "P_1γ":    ("{:>10}",  "{:>10.3e}"),
        "P_ZEPE":  ("{:>10}",  "{:>10.3e}"),
        "P_2γ":    ("{:>10}",  "{:>10.3e}"),
        "P_M-5":   ("{:>10}",  "{:>10.3e}"),
        "regime":  ("{:<10}",  "{:<10}"),
        "‖c‖−1":   ("{:>8}",   "{:>8.1e}"),
    }
    headers = "  ".join(fmt[c][0].format(c) for c in cols)
    print(headers)
    print("-" * len(headers))

    last_group = None
    for r in rows:
        g = sortkey(r)[0]
        if last_group is not None and g != last_group:
            print("")            # blank line between scan groups
        last_group = g
        print("  ".join(fmt[c][1].format(r[c]) for c in cols))

    print()
    print(f"{len(rows)} runs in {grid}")
    print()
    print("Regime legend: deep PT (A<0.30), PT (<0.70), marginal (<1.50), SATURATED.")
    print("A = √(2π)·Ω_R·τ  in radians  (Gaussian-pulse area).")


if __name__ == "__main__":
    main()
