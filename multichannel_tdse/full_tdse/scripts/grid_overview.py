#!/usr/bin/env python3
"""grid_overview.py -- grid-level summary plots for a production_grid_*
directory.  Reads each run's summary.txt and (for ZEPE close-up) its
M_F=-4 states.csv, and produces:

  {grid}_scan_trends.pdf      4 panels: P_halo, P_1γ, P_ZEPE, P_2γ vs
                              swept parameter, one curve per scan group
  {grid}_zepe_close_up.pdf    M_F=-4 ZEPE-channel dP/dE overlay across
                              every run, with adaptive σ smoothing

Usage:
    python3 grid_overview.py [grid_dir]
"""
import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the helpers we already have.
from plot_run import load_states, adaptive_sigma, smooth_states


E_B_KHZ = 10.112


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


def regime_of(A):
    if A < 0.30:   return "deep PT"
    if A < 0.70:   return "PT"
    if A < 1.50:   return "marginal"
    return "SAT"


def collect(grid: Path):
    """Return list of dicts, one per run; sorted by scan group."""
    rows = []
    for sub in sorted(grid.iterdir()):
        if not sub.is_dir(): continue
        tag = sub.name
        summ = sub / f"{tag}_summary.txt"
        if not summ.exists(): continue
        s = parse_summary(summ)
        if "Omega_R_kHz" not in s or "P_halo" not in s: continue
        OmR = s["Omega_R_kHz"]
        tau = s["tau_us"]
        omega = s["omega_kHz"]
        A = 2 * math.pi * OmR * 1e-3 * math.sqrt(2 * math.pi) * tau
        m = re.match(r"(s[123]|g[2-9]|zmax)_", tag)
        group = m.group(1) if m else "z"
        rows.append({
            "tag": tag, "group": group, "subdir": sub,
            "omega_ratio": omega / E_B_KHZ, "tau": tau, "OmR": OmR,
            "A": A, "regime": regime_of(A),
            "P_halo": s["P_halo"],
            "P_1g":   s.get("P_M-3_total", 0.0),
            "P_zepe": s.get("P_M-4_continuum", 0.0),
            "P_2g":   s.get("P_M-2_total", 0.0),
        })
    return rows


def panel(ax, xs, P_halo, P_1g, P_zepe, P_2g, xlabel, title):
    ax.semilogy(xs, P_halo, "o-", color="tab:blue",   label=r"$P_{\mathrm{halo}}$")
    ax.semilogy(xs, P_1g,   "s-", color="tab:green",  label=r"$P_{1\gamma}$")
    ax.semilogy(xs, P_zepe, "^-", color="tab:red",    label=r"$P_{\mathrm{ZEPE}}$")
    ax.semilogy(xs, P_2g,   "v-", color="tab:orange", label=r"$P_{2\gamma}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("population")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best", framealpha=0.85)


def trends_figure(rows, outpath: Path):
    """3-panel figure: one panel per scan group, each showing the four
    populations vs the scan variable."""
    s1 = sorted([r for r in rows if r["group"] == "s1"], key=lambda r: r["omega_ratio"])
    s2 = sorted([r for r in rows if r["group"] == "s2"], key=lambda r: r["tau"])
    s3 = sorted([r for r in rows if r["group"] == "s3"], key=lambda r: r["OmR"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    if s1:
        panel(axes[0],
              [r["omega_ratio"] for r in s1],
              [r["P_halo"] for r in s1], [r["P_1g"] for r in s1],
              [r["P_zepe"] for r in s1], [r["P_2g"] for r in s1],
              r"$\hbar\omega / E_b$",
              r"Scan 1: $\omega$ (τ=30 μs, $\Omega_R$=1 kHz)")
    if s2:
        panel(axes[1],
              [r["tau"] for r in s2],
              [r["P_halo"] for r in s2], [r["P_1g"] for r in s2],
              [r["P_zepe"] for r in s2], [r["P_2g"] for r in s2],
              r"$\tau$ [μs]",
              r"Scan 2: $\tau$ ($\omega/E_b$=8, $\Omega_R$=1 kHz)")
        # Add A=0.7 and 1.5 regime markers (vertical lines)
        # A = 2π·Ω·1e-3·√(2π)·τ, with Ω_R=1 → τ_threshold = A/(2π·1e-3·√(2π))
        for A_thr, lbl, c in [(0.7, "PT→marginal", "0.5"), (1.5, "marginal→SAT", "k")]:
            tau_thr = A_thr / (2 * math.pi * 1e-3 * math.sqrt(2 * math.pi))
            axes[1].axvline(tau_thr, color=c, ls="--", lw=0.7, alpha=0.6)
            axes[1].text(tau_thr, axes[1].get_ylim()[0] * 1.5, "  " + lbl,
                         fontsize=7, color=c, alpha=0.7, va="bottom")
    if s3:
        panel(axes[2],
              [r["OmR"] for r in s3],
              [r["P_halo"] for r in s3], [r["P_1g"] for r in s3],
              [r["P_zepe"] for r in s3], [r["P_2g"] for r in s3],
              r"$\Omega_R/2\pi$ [kHz]",
              r"Scan 3: $\Omega_R$ ($\omega/E_b$=8, τ=30 μs)")
        # Add slope-2 (Ω²) and slope-4 (Ω⁴) reference lines pinned at the
        # smallest-Ω anchor point.
        anchor = s3[0]
        x0, p_1g_0, p_zepe_0 = anchor["OmR"], anchor["P_1g"], anchor["P_zepe"]
        xs = np.array([r["OmR"] for r in s3])
        axes[2].plot(xs, p_1g_0   * (xs/x0)**2, ":", color="tab:green",  lw=1, alpha=0.7,
                     label=r"$\propto\Omega_R^2$")
        axes[2].plot(xs, p_zepe_0 * (xs/x0)**4, ":", color="tab:red",    lw=1, alpha=0.7,
                     label=r"$\propto\Omega_R^4$")
        axes[2].legend(fontsize=8, loc="best", framealpha=0.85)

    fig.suptitle("Production-grid scan trends", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath)
    plt.close(fig)
    print(f"wrote {outpath}")


def zepe_overlay_figure(rows, outpath: Path):
    """Overlay every run's ZEPE-channel dP/dE on a single semilog plot
    (E_block 0..50 kHz), labelled by run tag.  Three subplots for the
    three scan groups."""
    s1 = sorted([r for r in rows if r["group"] == "s1"], key=lambda r: r["omega_ratio"])
    s2 = sorted([r for r in rows if r["group"] == "s2"], key=lambda r: r["tau"])
    s3 = sorted([r for r in rows if r["group"] == "s3"], key=lambda r: r["OmR"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    E_grid = np.linspace(0.01, 60.0, 1500)

    def plot_one(ax, group, title, label_of):
        for i, r in enumerate(group):
            sp = r["subdir"] / f"{r['tag']}_block_M-4_states.csv"
            if not sp.exists(): continue
            res = load_states(sp)
            E_lab, E_block, prob, _ = res
            sigma = adaptive_sigma(E_block)
            y = smooth_states(E_block, prob, E_grid, sigma, only_continuum=True)
            cmap = matplotlib.cm.viridis(i / max(1, len(group) - 1))
            ax.semilogy(E_grid, y, color=cmap, lw=1.4, label=label_of(r))
        ax.set_xlabel(r"E above $M_F$=−4 threshold  [kHz]")
        ax.set_ylabel(r"$dP/dE$  [1/kHz]")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85, ncol=1)
        ax.set_xlim(0, 60)

    if s1:
        plot_one(axes[0], s1, r"Scan 1: ZEPE peak vs $\omega/E_b$",
                 lambda r: rf"$\omega/E_b$={r['omega_ratio']:.0f}")
    if s2:
        plot_one(axes[1], s2, r"Scan 2: ZEPE peak vs $\tau$",
                 lambda r: rf"$\tau$={int(r['tau'])}μs ({r['regime']})")
    if s3:
        plot_one(axes[2], s3, r"Scan 3: ZEPE peak vs $\Omega_R$",
                 lambda r: rf"$\Omega_R$={r['OmR']:g} kHz ({r['regime']})")

    fig.suptitle("ZEPE channel ($M_F$=−4 continuum) — adaptive smoothing", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath)
    plt.close(fig)
    print(f"wrote {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid_dir", nargs="?", default=None)
    args = ap.parse_args()

    if args.grid_dir is None:
        candidates = sorted(Path("..").rglob("production_grid_*"))
        if not candidates:
            raise SystemExit("no production_grid_* found; pass a path")
        args.grid_dir = str(candidates[-1])

    grid = Path(args.grid_dir).resolve()
    rows = collect(grid)
    if not rows:
        raise SystemExit(f"no completed runs in {grid}")

    out_trends = grid / f"{grid.name}_scan_trends.pdf"
    out_zepe   = grid / f"{grid.name}_zepe_overlay.pdf"
    trends_figure(rows, out_trends)
    zepe_overlay_figure(rows, out_zepe)


if __name__ == "__main__":
    main()
