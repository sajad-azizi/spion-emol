#!/usr/bin/env python3
"""Quick spectral / population analysis of a multichannel-TDSE run.

Reports peak positions, FWHM, integrated areas of the dP/dE features
in BLOCK-RELATIVE energy coordinates (i.e. above each block's own
threshold).  Reads the per-state CSVs so smoothing width δE is freely
adjustable without re-running the TDSE.

Usage:
    python3 analyze_run.py PREFIX [--delta_E_kHz 5.0]

PREFIX = path stem like  tdse/run_L5e5_E5m20G/run_L5e5_E5m20G
"""
import sys
import csv
import argparse
from pathlib import Path
import numpy as np


def load_states(p):
    n, El, Eb, prob = [], [], [], []
    with open(p) as f:
        r = csv.reader(f); next(r)
        for row in r:
            n.append(int(row[0]))
            El.append(float(row[1]))
            Eb.append(float(row[2]))
            prob.append(float(row[3]))
    return np.array(n), np.array(El), np.array(Eb), np.array(prob)


def smooth(E_a, prob, E_grid, sigma):
    norm = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    out = np.zeros_like(E_grid)
    for E, p in zip(E_a, prob):
        out += p * norm * np.exp(-(E_grid - E) ** 2 * inv2s2)
    return out


def parse_summary(p):
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                out[k.strip()] = v.strip()
    return out


def analyze_block(prefix, mf, label, exp_peak_kHz, sigma_kHz):
    states_path = prefix.parent / f"{prefix.name}_block_M{mf:+d}_states.csv"
    if not states_path.exists():
        return None
    n, E_lab, E_block, prob = load_states(states_path)

    E_lo = E_block.min() - 5 * sigma_kHz
    E_hi = E_block.max() + 5 * sigma_kHz
    E_grid = np.linspace(E_lo, E_hi, 8000)
    y = smooth(E_block, prob, E_grid, sigma_kHz)
    P_total = prob.sum()

    if exp_peak_kHz is not None:
        m = (E_grid >= exp_peak_kHz - 50) & (E_grid <= exp_peak_kHz + 50)
        if m.any():
            E_peak = E_grid[m][np.argmax(y[m])]
            y_peak = y[m].max()
        else:
            E_peak = E_grid[np.argmax(y)]; y_peak = y.max()
    else:
        E_peak = E_grid[np.argmax(y)]
        y_peak = y.max()

    half = y_peak / 2
    idx = np.where(y >= half)[0]
    FWHM = E_grid[idx[-1]] - E_grid[idx[0]] if len(idx) >= 2 else 0.0

    if exp_peak_kHz is not None and FWHM > 0:
        E_lo_int = E_peak - 2 * FWHM
        E_hi_int = E_peak + 2 * FWHM
    else:
        E_lo_int = E_peak - 5 * sigma_kHz
        E_hi_int = E_peak + 5 * sigma_kHz
    m_int = (E_grid >= E_lo_int) & (E_grid <= E_hi_int)
    peak_area = np.trapz(y[m_int], E_grid[m_int])
    n_state_in_window = int(((E_block >= E_lo_int) & (E_block <= E_hi_int)).sum())

    return dict(label=label, n=n, E_lab=E_lab, E_block=E_block, prob=prob,
                P_total=P_total, E_peak=E_peak, y_peak=y_peak, FWHM=FWHM,
                peak_area=peak_area, E_lo_int=E_lo_int, E_hi_int=E_hi_int,
                n_state_in_window=n_state_in_window, exp_peak_kHz=exp_peak_kHz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", type=str)
    ap.add_argument("--delta_E_kHz", type=float, default=5.0)
    args = ap.parse_args()

    prefix = Path(args.prefix).resolve()
    summary = parse_summary(prefix.with_suffix("").as_posix() + "_summary.txt")
    omega = summary["omega_kHz"]
    E_b = abs(omega / 8.0)
    sigma_kHz = args.delta_E_kHz

    print(f"Run: {prefix.name}")
    print(f"  ω = {omega:.3f} kHz  Ω_R/2π = {summary['Omega_R_kHz']:.0f} kHz  "
          f"τ = {summary['tau_us']:.0f} μs")
    print(f"  L = {int(summary['L_a0'])} a_0  E_cut^open = {int(summary['E_cut_open_kHz'])} kHz  "
          f"E_cut^(-5) = {summary['E_cut_m5_kHz']/1e3:.0f} MHz")
    print(f"  ‖c‖−1 = {summary['err_unitary']:.2e}    K_avg = {int(summary['K_avg'])}")
    print()

    # ---- ZEPE perturbative prediction (Azizi/Saalmann/Rost 2024, Eq. 5) ----
    # Convert our pulse to the paper's convention.
    # Our chi(t) = exp(-t²/(2τ²)); paper's g(t) = exp(-t²/T_paper²).
    # ⇒ T_paper = τ √2.  Spectral width ΔE = 2/T_paper [angular] →
    # in cyclic kHz: ΔE_kHz = 1000/(π · T_paper_μs).
    tau_us       = summary["tau_us"]
    T_paper_us   = tau_us * np.sqrt(2.0)
    DeltaE_kHz   = 1000.0 / (np.pi * T_paper_us)
    E_b_kHz      = E_b
    beta         = E_b_kHz / DeltaE_kHz
    print("ZEPE PT prediction (Azizi/Saalmann/Rost PRL 2024, Eq. 5):")
    print(f"  pulse FWHM (Gaussian)         = {tau_us*np.sqrt(2*np.log(2))*2:.2f} μs  "
          f"(τ={tau_us:.0f} μs)")
    print(f"  T_paper (= τ √2)              = {T_paper_us:.2f} μs")
    print(f"  ΔE (= 2/T_paper, cyclic kHz)  = {DeltaE_kHz:.3f} kHz")
    print(f"  E_b (halo binding)            = {E_b_kHz:.3f} kHz")
    print(f"  β = E_b / ΔE                  = {beta:.3f}")
    print("  Eq. (5):  E_max = (E_b/2) · [√(1 + (2ℓ+1)/β²) − 1]")
    # Our σ⁺ photon raises M_F (spin) but does NOT change orbital ℓ.
    # The halo is s-wave (ℓ=0), and so are all final states reachable
    # by absorb-emit/emit-absorb σ⁺ photons.  Only ℓ=0 is physical here;
    # the paper's ℓ=2 d-wave channel is absent (would require an
    # orbital-changing dipole).
    E_max_kHz = 0.5 * E_b_kHz * (np.sqrt(1.0 + 1.0 / beta ** 2) - 1.0)
    print(f"      ℓ=0 (s-wave): E_max^ZEPE = {E_max_kHz:+.3f} kHz")
    # Strong/weak-field flag.  Pulse area is the canonical weak-field
    # check: A_pulse = Ω_R · ∫χ(t) dt = Ω_R · √(2π) · τ.  Rad units.
    A_pulse_rad = (2.0 * np.pi * summary["Omega_R_kHz"] * 1e3) \
                  * (np.sqrt(2 * np.pi) * tau_us * 1e-6)   # 2π·Ω_R[Hz] · √(2π) · τ[s]
    print(f"  pulse area Ω_R · √(2π) · τ   = {A_pulse_rad:.2f} rad")
    if A_pulse_rad > 1.0:
        print(f"  WARNING: pulse area ≫ 1 → strong-driving regime.")
        print(f"           Eq. (5) is perturbative; expect peak shift / broadening.")
    print()
    P_sum = (summary['P_halo'] + summary['P_M-4_continuum']
             + summary['P_M-3_total'] + summary['P_M-2_total']
             + summary['P_M-5_total'])
    print("Block totals:")
    print(f"  P_halo  (M_F=-4 bound)        = {summary['P_halo']:.4e}")
    print(f"  P_ZEPE  (M_F=-4 continuum)    = {summary['P_M-4_continuum']:.4e}")
    print(f"  P_M-3   (1γ branch)           = {summary['P_M-3_total']:.4e}")
    print(f"  P_M-2   (2γ branch)           = {summary['P_M-2_total']:.4e}")
    print(f"  P_M-5   (virtual)             = {summary['P_M-5_total']:.4e}")
    print(f"  Σ                             = {P_sum:.10f}  (must = 1.0)")
    print()

    print("─── Spectral peaks (block-relative E) ─────────────────────────────")
    print(f"  smoothing δE = {sigma_kHz} kHz")
    print()
    blocks = [
        (-4, "ZEPE  (M_F=-4 cont)", None),
        (-3, "1γ    (M_F=-3)",      omega),
        (-2, "2γ    (M_F=-2)",      2 * omega - E_b),
        (-5, "virt. (M_F=-5)",      None),
    ]
    for mf, label, exp_peak in blocks:
        res = analyze_block(prefix, mf, label, exp_peak, sigma_kHz)
        if res is None:
            continue
        print(f"  {label}")
        print(f"    N_states                  : {len(res['n']):>6}     "
              f"E_block range  [{res['E_block'].min():+.2f}, {res['E_block'].max():+.2f}] kHz")
        print(f"    Σ |b|² in block           : {res['P_total']:.4e}")
        if exp_peak is not None:
            print(f"    Expected peak             : {exp_peak:+.2f} kHz")
        print(f"    Peak position             : {res['E_peak']:+.2f} kHz   "
              f"(dP/dE_max = {res['y_peak']:.4e} per kHz)")
        print(f"    FWHM                      : {res['FWHM']:.2f} kHz")
        print(f"    Peak area  ([{res['E_lo_int']:+.1f}, {res['E_hi_int']:+.1f}] kHz)"
              f" = {res['peak_area']:.4e}")
        print(f"    States within peak window : {res['n_state_in_window']}")
        print()

    print("─── Top-5 most populated states per block ───────────────────────")
    for mf, label, _ in blocks:
        res = analyze_block(prefix, mf, label, None, sigma_kHz)
        if res is None:
            continue
        idx = np.argsort(res['prob'])[::-1][:5]
        print(f"\n  {label}:")
        print(f"    {'n':>4} {'E_block (kHz)':>14} {'E_lab (kHz)':>14} {'|b|²':>14}")
        for i in idx:
            print(f"    {int(res['n'][i]):>4} {res['E_block'][i]:>14.4f} "
                  f"{res['E_lab'][i]:>14.4f} {res['prob'][i]:>14.4e}")


if __name__ == "__main__":
    main()
