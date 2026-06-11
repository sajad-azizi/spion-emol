#!/usr/bin/env python3
"""paper_figs.py -- generate the 4 main-text figures for the
multichannel-ZEPE PRL from a production-grid directory.

The grid must contain (at minimum) these runs:
    s1_om8                            (anchor: ω/E_b=8, τ=30, Ω_R=1)
    s1_om6, s1_om8, s1_om10           (ω-scan for Fig 3b)
    s3_om_r0.2, 0.5, 1.0, 1.5, 2.0    (Ω_R-scan for Fig 4a)
    s2_tau{15,20,25,30,40,50,75,100}  (τ-scan for Fig 4b)

Outputs (in <grid>/figs/):
    fig2_anchor_spectra.pdf        3-panel TDSE spectrum, M_F=-2,-3,-4
    fig3a_zepe_closeup.pdf         M_F=-4 ZEPE close-up: TDSE vs 2nd-order PT
    fig3b_omega_invariance.pdf     ZEPE peak invariance under ω
    fig4a_omega_R_scaling.pdf      Ω_R-yield scaling w/ Ω² and Ω⁴ guides
    fig4b_tau_scan.pdf             ZEPE yield + peak position vs τ

Usage:
    python3 paper_figs.py [grid_dir]
"""
import argparse
import math
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_run import load_states, adaptive_sigma, smooth_states


E_B_KHZ = 10.112


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
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


def load_run(grid: Path, tag: str):
    """Return (summary_dict, dict_of_states_per_block).  The states
    dict has keys -2, -3, -4, -5 with each value (E_block, prob, prob_pt)."""
    sub = grid / tag
    s = parse_summary(sub / f"{tag}_summary.txt")
    states = {}
    for mf in (-5, -4, -3, -2):
        sp = sub / f"{tag}_block_M{mf:+d}_states.csv"
        if not sp.exists():
            continue
        E_lab, E_block, prob, prob_pt = load_states(sp)
        states[mf] = (E_block, prob, prob_pt)
    return s, states


def adaptive_smooth_curve(E_block, prob, E_grid, only_continuum=True):
    """Smooth using per-state σ_α = local spacing."""
    sigma = adaptive_sigma(E_block)
    return smooth_states(E_block, prob, E_grid, sigma,
                         only_continuum=only_continuum)


# ------------------------------------------------------------------ #
# Fig 2 — 3-panel channel-resolved spectrum at anchor (s1_om8)
# ------------------------------------------------------------------ #
def fig2(grid: Path, outdir: Path):
    s, states = load_run(grid, "s1_om8")
    omega = s["omega_kHz"]
    E_b   = E_B_KHZ
    E_1g  = omega - E_b
    E_2g  = 2 * omega - E_b

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cfg = [
        (-4, axes[0], "tab:blue",  r"$M_F=-4$ (ZEPE)",
         (0.0, 50.0),
         [(8.0, "ZEPE peak\nnear threshold")]),
        (-3, axes[1], "tab:green", r"$M_F=-3$ ($1\gamma$)",
         (0.0, omega * 1.5),
         [(E_1g, rf"$E_{{1\gamma}}\!=\!\hbar\omega\!-\!E_b\!=\!{E_1g:.1f}$ kHz")]),
        (-2, axes[2], "tab:red",   r"$M_F=-2$ ($2\gamma_{\rm aa}$)",
         (0.0, 2 * omega * 1.1),
         [(E_2g, rf"$E_{{2\gamma_{{\rm aa}}}}\!=\!2\hbar\omega\!-\!E_b"
                 rf"\!=\!{E_2g:.1f}$ kHz")]),
    ]
    for mf, ax, color, title, (Elo, Ehi), markers in cfg:
        if mf not in states:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        E_block, prob, _ = states[mf]
        E_grid = np.linspace(Elo, Ehi, 4000)
        y = adaptive_smooth_curve(E_block, prob, E_grid, only_continuum=True)
        m = y > 0
        ax.semilogy(E_grid[m], y[m], color=color, lw=1.6)
        for E_ref, label in markers:
            if Elo <= E_ref <= Ehi:
                # mark with vertical line only when the E_ref is meaningful
                # (skip for pure annotation labels offset from E=0)
                if mf != -4:
                    ax.axvline(E_ref, color="k", ls=":", lw=0.8, alpha=0.7)
                ax.text(E_ref, ax.get_ylim()[1] * 0.6, "  " + label,
                        fontsize=8, ha="left", va="top")
        ax.set_xlim(Elo, Ehi)
        ax.set_xlabel("relative kinetic energy $E$ above $M_F$ threshold  [kHz]")
        ax.set_ylabel(r"$dP/dE$  [1/kHz]")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")

    pulse_area = 2 * math.pi * s["Omega_R_kHz"] * 1e-3 \
                 * math.sqrt(2 * math.pi) * s["tau_us"]
    fig.suptitle(rf"Channel-resolved TDSE spectra at the anchor point   "
                 rf"$\hbar\omega/E_b={omega/E_b:.0f}$, "
                 rf"$\tau={s['tau_us']:.0f}~\mu$s, "
                 rf"$\Omega_R/2\pi={s['Omega_R_kHz']:.1f}$ kHz   "
                 rf"($A={pulse_area:.2f}$ rad)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = outdir / "fig2_anchor_spectra.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------ #
# Fig 3a — ZEPE close-up at anchor: TDSE vs 2nd-order PT
# ------------------------------------------------------------------ #
def analytic_zepe(E_kHz, A, E_star_kHz, beta, ell=0, E_b_kHz=E_B_KHZ):
    """Paper Eq. (4): P_ZEPE^{ℓ,β}(E) = P_*(E) · s_{ℓ,β}(E/E_EA)
        P_*(E)        = (1 + E/E_*)^{-1}                       (Eq. 4a)
        s_{ℓ,β}(x)    = β^4 · x^{ℓ+1/2} · exp(-β² (x+1)²)      (Eq. 4b)
    A is an overall amplitude (the paper normalises so that the integral
    matches; we keep it as a free fit parameter).
    """
    x = E_kHz / E_b_kHz
    P_star = 1.0 / (1.0 + E_kHz / E_star_kHz)
    s_lb   = (beta**4) * np.power(np.maximum(x, 0.0), ell + 0.5) \
             * np.exp(-(beta**2) * (x + 1.0)**2)
    return A * P_star * s_lb


def fit_E_star(E_data, y_data, beta, ell=0, E_b_kHz=E_B_KHZ):
    """Fit the analytic Eq.(4) shape to (E, y) data.  Two free params:
    overall amplitude A and E_* (slow-background scale).  Robust to
    outliers via log-residual (the peak is sharply localised)."""
    from scipy.optimize import curve_fit
    # Restrict to the part of the spectrum where the peak lives;
    # outside, the shape is dominated by FFT-tail / Floquet sidebands.
    m = (E_data > 0) & (y_data > 0) & (E_data < 25.0)
    if m.sum() < 5:
        return None, None, None
    E_fit, y_fit = E_data[m], y_data[m]
    A0   = y_fit.max() / max(beta**4, 1e-30)
    E_st0 = E_b_kHz * 5.0       # initial guess: a few × E_b
    try:
        popt, pcov = curve_fit(
            lambda E, A, E_st: analytic_zepe(E, A, E_st, beta, ell, E_b_kHz),
            E_fit, y_fit,
            p0=[A0, E_st0],
            bounds=([1e-30, 1e-3], [1e3, 1e6]),
            maxfev=8000)
        return popt[0], popt[1], pcov
    except Exception as exc:
        print(f"fit_E_star: curve_fit failed ({exc})")
        return None, None, None


def fig3a(grid: Path, outdir: Path):
    s, states = load_run(grid, "s1_om8")
    if -4 not in states:
        print("fig3a: missing M_F=-4 data"); return

    E_block, prob, prob_pt = states[-4]
    E_grid = np.linspace(0.0, 60.0, 4000)
    y_tdse = adaptive_smooth_curve(E_block, prob,    E_grid)
    y_pt   = adaptive_smooth_curve(E_block, prob_pt, E_grid)

    # Paper Eq.5 PT prediction for ℓ=0:
    tau    = s["tau_us"]
    T_p    = tau * math.sqrt(2.0)
    DE     = 1000.0 / (math.pi * T_p)
    beta   = E_B_KHZ / DE
    E_max  = 0.5 * E_B_KHZ * (math.sqrt(1.0 + 1.0/beta**2) - 1.0)

    # Fit the paper's analytic shape Eq.(4) to the TDSE peak.
    # E_* is the slowly-varying background scale, NOT a peak position.
    # We keep it as a one-parameter shape fit; the resulting value is
    # written to the figure caption (printed by paper_figs at run time).
    A_fit, E_star_fit, _ = fit_E_star(E_grid, y_tdse, beta=beta, ell=0)
    if E_star_fit is not None:
        y_anal = analytic_zepe(E_grid, A_fit, E_star_fit, beta, ell=0)
        anal_label = "analytic Eq. (4)"
        print(f"[fig3a] analytic Eq.(4) shape fit: A={A_fit:.3e},  "
              f"E_*={E_star_fit:.3f} kHz "
              f"(= {E_star_fit/E_B_KHZ:.2f} E_b)   "
              f"[background scale, not a peak]")
    else:
        y_anal = None
        anal_label = None

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_lin.plot(E_grid, y_tdse, color="tab:blue", lw=1.7,
                label=r"full TDSE")
    ax_lin.plot(E_grid, y_pt,   color="tab:purple", lw=1.4, ls="--",
                label=r"2nd-order PT (Eq. 7)")
    if y_anal is not None:
        ax_lin.plot(E_grid, y_anal, color="tab:green", lw=1.4, ls="-.",
                    label=anal_label)
    ax_lin.axvline(E_max, color="tab:orange", lw=1.0, ls=":",
                   label=rf"PT peak Eq. (5):  $E_{{\rm peak}}^{{\rm PT}}={E_max:.2f}$ kHz")
    ax_lin.set_xlim(0.0, 25.0)
    ax_lin.set_xlabel("relative kinetic energy $E$  [kHz]")
    ax_lin.set_ylabel(r"$dP/dE$  [1/kHz]")
    ax_lin.set_title("(a) ZEPE peak — linear scale")
    ax_lin.legend(fontsize=8, framealpha=0.85, loc="upper right")
    ax_lin.grid(alpha=0.3)

    E_log = np.logspace(-2, 2, 4000)
    y_tdse_lg = adaptive_smooth_curve(E_block, prob,    E_log)
    y_pt_lg   = adaptive_smooth_curve(E_block, prob_pt, E_log)
    ax_log.loglog(E_log[y_tdse_lg > 0], y_tdse_lg[y_tdse_lg > 0],
                  color="tab:blue", lw=1.7, label="full TDSE")
    ax_log.loglog(E_log[y_pt_lg > 0], y_pt_lg[y_pt_lg > 0],
                  color="tab:purple", lw=1.4, ls="--", label="2nd-order PT")
    if y_anal is not None:
        y_anal_lg = analytic_zepe(E_log, A_fit, E_star_fit, beta, ell=0)
        ax_log.loglog(E_log[y_anal_lg > 0], y_anal_lg[y_anal_lg > 0],
                      color="tab:green", lw=1.4, ls="-.",
                      label=r"analytic Eq. (4)")
    # Wigner E^{1/2} guide.
    if y_pt_lg.max() > 0:
        i_ref = np.argmin(np.abs(E_log - 0.5))
        if y_pt_lg[i_ref] > 0:
            yguide = y_pt_lg[i_ref] * (E_log / 0.5) ** 0.5
            mg = (yguide > 1e-30) & (E_log < 5.0)
            ax_log.loglog(E_log[mg], yguide[mg], color="gray", ls=":", lw=0.8,
                          alpha=0.7, label=r"Wigner $\propto E^{1/2}$")
    ax_log.axvline(E_max, color="tab:orange", lw=1.0, ls=":")
    ax_log.set_xlim(0.05, 50)
    peak_y = max(y_tdse_lg.max() if y_tdse_lg.size else 0,
                 y_pt_lg.max()   if y_pt_lg.size   else 0)
    if peak_y > 0:
        ax_log.set_ylim(1e-14, peak_y * 3)
    ax_log.set_xlabel("relative kinetic energy $E$  [kHz]")
    ax_log.set_ylabel(r"$dP/dE$  [1/kHz]")
    ax_log.set_title("(b) ZEPE peak — log-log (threshold law + tail)")
    ax_log.legend(fontsize=8, framealpha=0.85, loc="lower center")
    ax_log.grid(alpha=0.3, which="both")

    suptitle = (rf"$M_F=-4$ ZEPE close-up at anchor "
                rf"($\hbar\omega/E_b=8$, $\tau_0=30$ μs, $\Omega_R/2\pi=1$ kHz);"
                rf"  $\beta=E_b/\Delta E={beta:.2f}$")
    if E_star_fit is not None:
        suptitle += (rf"   |   Eq. (4) shape fit: "
                     rf"$E_*={E_star_fit:.2f}$ kHz "
                     rf"$={E_star_fit/E_B_KHZ:.2f}\,E_b$ (background scale)")
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = outdir / "fig3a_zepe_closeup.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------ #
# Fig 3b — ω-independence of the ZEPE peak position
# ------------------------------------------------------------------ #
def fig3b(grid: Path, outdir: Path,
          tags=("s1_om6", "s1_om8", "s1_om10")):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = matplotlib.cm.viridis
    E_grid = np.linspace(0.0, 25.0, 4000)
    extracted = []      # (omega_ratio, E_peak_kHz, peak_height)
    for i, tag in enumerate(tags):
        s, states = load_run(grid, tag)
        if -4 not in states:
            continue
        E_block, prob, _ = states[-4]
        y = adaptive_smooth_curve(E_block, prob, E_grid)
        peak = y.max()
        ynorm = y / peak if peak > 0 else y
        E_pk = E_grid[np.argmax(y)] if peak > 0 else float("nan")
        c = cmap(i / max(1, len(tags) - 1))
        ax.plot(E_grid, ynorm, color=c, lw=1.6,
                label=rf"$\hbar\omega/E_b={s['omega_kHz']/E_B_KHZ:.0f}$")
        extracted.append((s['omega_kHz']/E_B_KHZ, E_pk, peak))
    ax.set_xlim(0.0, 25.0)
    ax.set_xlabel("relative kinetic energy $E$ above $M_F=-4$ threshold  [kHz]")
    ax.set_ylabel("normalized $dP/dE$")
    ax.set_title(r"Carrier-frequency-independent ZEPE peak "
                 r"($\tau_0=30$ μs, $\Omega_R/2\pi=1$ kHz)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.85)

    # Extracted-peak inset table.
    if extracted:
        rows = "\n".join(rf"$\omega/E_b={r[0]:.0f}$:  $E_{{\rm peak}}={r[1]:.2f}$ kHz,  "
                         rf"$(dP/dE)_{{\max}}={r[2]:.2e}$/kHz"
                         for r in extracted)
        ax.text(0.97, 0.50, rows, transform=ax.transAxes, fontsize=9,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec="lightgray", alpha=0.92))

    out = outdir / "fig3b_omega_invariance.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------ #
# Fig 4a — Ω_R scaling
# ------------------------------------------------------------------ #
def fig4a(grid: Path, outdir: Path,
          tags=("s3_om_r0.2", "s3_om_r0.5", "s3_om_r1",
                "s3_om_r1.5", "s3_om_r2")):
    rows = []
    for tag in tags:
        s, _ = load_run(grid, tag)
        rows.append((s["Omega_R_kHz"],
                     s["P_M-3_total"], s["P_M-4_continuum"],
                     s["P_M-2_total"]))
    rows.sort()
    OmR  = np.array([r[0] for r in rows])
    P_1g = np.array([r[1] for r in rows])
    P_zp = np.array([r[2] for r in rows])
    P_2g = np.array([r[3] for r in rows])

    # Linear least-squares slope fits in log-log space, restricted to the
    # PT/marginal regime (Ω_R ≤ 2 kHz) where higher-order corrections are
    # < 5 %.  The "anchor" Ω_R=0.2 kHz point is included if present.
    def slope_fit(OmR_arr, P_arr):
        m = (P_arr > 0) & (OmR_arr <= 2.01)
        if m.sum() < 2: return float("nan")
        return np.polyfit(np.log(OmR_arr[m]), np.log(P_arr[m]), 1)[0]
    s_1g = slope_fit(OmR, P_1g)
    s_zp = slope_fit(OmR, P_zp)
    s_2g = slope_fit(OmR, P_2g)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(OmR, P_1g, "s-", color="tab:green",  lw=1.5, ms=7,
              label=rf"$P_{{1\gamma}}$, slope ${s_1g:.2f}$")
    ax.loglog(OmR, P_zp, "^-", color="tab:red",    lw=1.5, ms=7,
              label=rf"$P_{{\rm ZEPE}}$, slope ${s_zp:.2f}$")
    ax.loglog(OmR, P_2g, "v-", color="tab:orange", lw=1.5, ms=7,
              label=rf"$P_{{2\gamma_{{\rm aa}}}}$, slope ${s_2g:.2f}$")
    # Power-law guides anchored at the smallest-Ω data point.
    O0 = OmR[0]
    ax.loglog(OmR, P_1g[0] * (OmR / O0) ** 2, ":", color="tab:green",
              lw=1.0, alpha=0.7, label=r"$\Omega_R^{\,2}$")
    ax.loglog(OmR, P_zp[0] * (OmR / O0) ** 4, ":", color="tab:red",
              lw=1.0, alpha=0.7, label=r"$\Omega_R^{\,4}$")
    ax.set_xlabel(r"$\Omega_R / 2\pi$  [kHz]")
    ax.set_ylabel("integrated yield")
    ax.set_title(r"Yield scaling vs $\Omega_R$ "
                 r"($\hbar\omega/E_b=8$, $\tau=30$ μs)")
    ax.grid(alpha=0.3, which="both")
    # Legend outside the axes on the right so it doesn't cover data.
    ax.legend(fontsize=9, framealpha=0.85,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    out = outdir / "fig4a_omega_R_scaling.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------ #
# Fig 4b — τ scan
# ------------------------------------------------------------------ #
def fig4b(grid: Path, outdir: Path,
          tags=("s2_tau15","s2_tau20","s2_tau25","s2_tau30",
                "s2_tau40","s2_tau50","s2_tau75","s2_tau100")):
    rows = []
    for tag in tags:
        s, states = load_run(grid, tag)
        E_peak = math.nan
        if -4 in states:
            E_block, prob, _ = states[-4]
            E_grid = np.linspace(0.0, 25.0, 4000)
            y = adaptive_smooth_curve(E_block, prob, E_grid)
            if y.max() > 0:
                E_peak = E_grid[np.argmax(y)]
        rows.append((s["tau_us"], s["P_M-4_continuum"], E_peak))
    rows.sort()
    tau   = np.array([r[0] for r in rows])
    P_zp  = np.array([r[1] for r in rows])
    E_pk  = np.array([r[2] for r in rows])
    # Eq.(5) PT prediction for E_peak (ℓ=0).
    T_p   = tau * math.sqrt(2.0)
    DE    = 1000.0 / (math.pi * T_p)
    beta  = E_B_KHZ / DE
    E_pt  = 0.5 * E_B_KHZ * (np.sqrt(1.0 + 1.0/beta**2) - 1.0)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color1 = "tab:red"
    color2 = "tab:blue"

    # Shade the τ ≥ 75 μs region: yield is ≲ 1e-9, hits numerical-noise
    # floor of the TDSE basis (peak position there is unreliable).
    NOISE_TAU = 75
    if (tau >= NOISE_TAU).any():
        ax1.axvspan(NOISE_TAU, tau.max() * 1.05, color="lightgray",
                    alpha=0.35, zorder=0,
                    label=(rf"$\tau\!\geq\!{NOISE_TAU}\,\mu$s:  yield approaches"
                           "\n"
                           r"the smoothing-floor; peak position unreliable"))

    ax1.semilogy(tau, P_zp, "^-", color=color1, lw=1.6, ms=7,
                 label=r"$P_{\rm ZEPE}$ (TDSE)")
    ax1.set_xlabel(r"$\tau_0$  [μs]")
    ax1.set_ylabel(r"$P_{\rm ZEPE}$ (integrated yield)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(alpha=0.3, which="both")

    ax2 = ax1.twinx()
    ax2.plot(tau, E_pk, "o-", color=color2, lw=1.5, ms=6,
             label=r"$E_{\rm peak}$ (TDSE)")
    ax2.plot(tau, E_pt, ":", color="tab:gray", lw=1.2,
             label=r"$E_{\max}^{\ell=0}$  PT (Eq. 5)")
    ax2.set_ylabel(r"$E_{\rm peak}^{\rm ZEPE}$  [kHz]", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, loc="center right", framealpha=0.85)

    ax1.set_title(r"$\tau\!\downarrow\,\Rightarrow\,P_{\rm ZEPE}\!\uparrow,"
                  r"\,E_{\rm peak}^{\rm ZEPE}\!\uparrow$"
                  r"   ($\hbar\omega/E_b\!=\!8$, $\Omega_R/2\pi\!=\!1$ kHz)")
    out = outdir / "fig4b_tau_scan.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid_dir", nargs="?", default=None)
    args = ap.parse_args()

    if args.grid_dir is None:
        candidates = sorted(Path("..").rglob("production_grid_*"))
        if not candidates:
            raise SystemExit("no production_grid_* found")
        args.grid_dir = str(candidates[-1])

    grid   = Path(args.grid_dir).resolve()
    outdir = grid / "figs"
    outdir.mkdir(exist_ok=True)

    fig2(grid,  outdir)
    fig3a(grid, outdir)
    fig3b(grid, outdir)
    fig4a(grid, outdir)
    fig4b(grid, outdir)


if __name__ == "__main__":
    main()
