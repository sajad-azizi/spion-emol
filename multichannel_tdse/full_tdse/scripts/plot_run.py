#!/usr/bin/env python3
"""Plot dP/dE per block from a run_tdse run, in BLOCK-RELATIVE energy
coordinates (E above each block's own threshold) and with a chosen
Gaussian smoothing width.

Usage:
    python3 plot_run.py PREFIX [--delta_E_kHz 5.0] [--outdir DIR]

Inputs read from PREFIX path stem:
    {stem}_summary.txt
    {stem}_block_M{MF}_dPdE.csv      (smoothed at write time)
    {stem}_block_M{MF}_states.csv    (per-state populations; preferred
                                      input -- lets us re-smooth at any δE)

If the per-state files are missing (older runs), we re-convolve the
existing dP/dE with an additional Gaussian to get the requested δE
(σ_extra² = δE² − δE_existing²; only works if δE > δE_existing).

Outputs (PDF) in --outdir (default = PREFIX directory):
    {stem}_dPdE_per_block_dE{delta}.pdf
    {stem}_dPdE_overlay_dE{delta}.pdf
    {stem}_block_totals.pdf
"""
import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Recipe defaults at B = 155.04 G (rotating-frame zero = M_F=-4 entrance).
# Used as fallback when summary doesn't have block thresholds.
DEFAULT_E_TH_kHz = {
    -5: +2683180.0,
    -4: 0.0,
    -3:  -77417.2,
    -2: -154834.0,
}


def parse_summary(path: Path) -> dict:
    out = {}
    with path.open() as f:
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


def load_dPdE(path: Path):
    E_kHz, y = [], []
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            E_kHz.append(float(row[0]))
            y.append(float(row[1]))
    return np.array(E_kHz), np.array(y)


def load_states(path: Path):
    """Returns (E_kHz_lab, E_kHz_block, prob, prob_pt) numpy arrays.
    The prob_pt column was added when the closed-form PerturbationTheory
    module went live; older states.csv files (4 columns) are still
    accepted -- prob_pt is then None."""
    if not path.exists():
        return None
    n_arr, El, Eb, p, p_pt = [], [], [], [], []
    has_pt = False
    with path.open() as f:
        r = csv.reader(f)
        header = next(r)
        has_pt = (len(header) >= 5 and header[4].strip() == "prob_pt")
        for row in r:
            n_arr.append(int(row[0]))
            El.append(float(row[1]))
            Eb.append(float(row[2]))
            p.append(float(row[3]))
            if has_pt:
                p_pt.append(float(row[4]))
    if has_pt:
        return np.array(El), np.array(Eb), np.array(p), np.array(p_pt)
    return np.array(El), np.array(Eb), np.array(p), None


def adaptive_sigma(E_kHz, floor_kHz=1e-3):
    """Per-state Gaussian width = local level spacing.

    σ_α = ½ (E_{α+1} − E_{α−1})   (central difference)
    Edges use one-sided spacing.  Result is clipped to ≥ floor_kHz so
    near-degenerate states don't produce singular Gaussians."""
    E = np.asarray(E_kHz, dtype=float)
    order = np.argsort(E)
    Es = E[order]
    sig = np.empty_like(Es)
    if len(Es) >= 2:
        sig[1:-1] = 0.5 * (Es[2:] - Es[:-2])
        sig[0]    = Es[1] - Es[0]
        sig[-1]   = Es[-1] - Es[-2]
    else:
        sig[:] = floor_kHz
    sig = np.maximum(sig, floor_kHz)
    out = np.empty_like(sig)
    out[order] = sig
    return out


def smooth_states(E_kHz, prob, E_grid_kHz, sigma_kHz, *, only_continuum=False):
    """Σ_α prob_α · g(E - E_α; σ).  Returns dP/dE in 1/kHz.

    sigma_kHz may be a scalar or a per-state array (shape == E_kHz).
    only_continuum=True drops bound states (E_α < 0 in block-relative
    coords).  Photoelectron spectra are continuum-only by definition;
    the halo bound state is the initial state, not a photoelectron."""
    E_kHz = np.asarray(E_kHz)
    prob  = np.asarray(prob)
    sig_arr = np.broadcast_to(np.asarray(sigma_kHz, dtype=float), E_kHz.shape).copy()
    if only_continuum:
        m       = E_kHz >= 0.0
        E_kHz   = E_kHz[m]
        prob    = prob[m]
        sig_arr = sig_arr[m]
    out = np.zeros_like(E_grid_kHz)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    for E_a, p, s in zip(E_kHz, prob, sig_arr):
        out += (p * inv_sqrt_2pi / s) * np.exp(-(E_grid_kHz - E_a) ** 2 / (2.0 * s * s))
    return out


def reconvolve_dPdE(E_kHz, y, sigma_extra_kHz):
    """Convolve y(E) with an additional Gaussian of std σ_extra (kHz).
    Uses uniform-grid FFT; works only for the uniform E_kHz produced
    by run_tdse."""
    if sigma_extra_kHz <= 0.0:
        return y
    dE = E_kHz[1] - E_kHz[0]
    n = len(E_kHz)
    # Build kernel centered at 0
    half = (n - 1) // 2
    x = (np.arange(n) - half) * dE
    g = np.exp(-(x ** 2) / (2.0 * sigma_extra_kHz ** 2))
    g /= g.sum() * dE   # normalize so ∫ g dE = 1
    # FFT-based circular convolution; use 2N pad for linear conv
    M = 2 * n
    Yf = np.fft.fft(y, M)
    Gf = np.fft.fft(np.roll(g, -half), M)
    return np.fft.ifft(Yf * Gf).real[:n] * dE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", type=str,
                    help="path stem, e.g. ../../tdse/run_L5e5_E5m20G")
    ap.add_argument("--delta_E_kHz", type=float, default=5.0,
                    help="Gaussian smoothing width (default 5.0 kHz; "
                         "recipe says 1-2x local level spacing).  "
                         "Ignored when --adaptive is set.")
    ap.add_argument("--adaptive", action="store_true",
                    help="Use per-state σ_α = local level spacing "
                         "(½(E_{α+1}-E_{α-1})) instead of a fixed δE. "
                         "Tracks the changing DOS automatically.")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    prefix = Path(args.prefix).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else prefix.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = prefix.name

    summary = parse_summary(prefix.parent / f"{stem}_summary.txt")
    omega_kHz   = summary.get("omega_kHz", 80.896)
    delta_existing = summary.get("delta_E_kHz", 0.5)
    delta = args.delta_E_kHz
    adaptive = args.adaptive
    fname_tag = "dEadap" if adaptive else f"dE{delta:.1f}"

    # Block thresholds (kHz, recipe origin).  Prefer summary; fallback to defaults.
    E_th = {}
    for mf in [-5, -4, -3, -2]:
        key = f"E_th_M{mf}_kHz"
        E_th[mf] = summary.get(key, DEFAULT_E_TH_kHz[mf])

    # ---- Load each block's data: prefer states.csv (re-smooth), fallback dPdE ----
    block_data = {}
    for mf in [-5, -4, -3, -2]:
        states_path = prefix.parent / f"{stem}_block_M{mf:+d}_states.csv"
        dPdE_path   = prefix.parent / f"{stem}_block_M{mf:+d}_dPdE.csv"
        E_block, dP = None, None
        E_states_block, prob_states = None, None
        sigma_per_state = None
        prob_pt_states = None
        if states_path.exists():
            E_lab, E_block_states, prob, prob_pt = load_states(states_path)
            E_states_block = E_block_states
            prob_states    = prob
            prob_pt_states = prob_pt
            # Per-state σ for adaptive smoothing (local level spacing).
            sigma_per_state = adaptive_sigma(E_block_states)
            sigma_use = sigma_per_state if adaptive else delta
            # Pad zoom by 5× a representative width.
            pad = 5.0 * (np.median(sigma_per_state) if adaptive else delta)
            E_lo = E_block_states.min() - pad
            E_hi = E_block_states.max() + pad
            E_block = np.linspace(E_lo, E_hi, 4000)
            dP = smooth_states(E_block_states, prob, E_block, sigma_use)
        elif dPdE_path.exists():
            E_lab, y = load_dPdE(dPdE_path)
            E_block = E_lab - E_th[mf]   # convert to block-relative
            sigma_extra = np.sqrt(max(delta**2 - delta_existing**2, 0.0))
            dP = reconvolve_dPdE(E_lab, y, sigma_extra)
        if E_block is not None:
            block_data[mf] = {
                "E_block_kHz":   E_block,
                "dPdE":          dP,
                "states_E_block_kHz": E_states_block,
                "states_prob":   prob_states,
                "states_prob_pt": prob_pt_states,
                "sigma_per_state": sigma_per_state,
            }

    colors = {-5: "tab:purple", -4: "tab:blue", -3: "tab:green", -2: "tab:red"}
    labels = {-5: r"$M_F=-5$  (virtual — not observable, theory only)",
              -4: r"$M_F=-4$  (ZEPE channel)",
              -3: r"$M_F=-3$  (1$\gamma$ channel)",
              -2: r"$M_F=-2$  (2$\gamma$ channel)"}

    delta_label = (r"$\delta E = $ adaptive (local spacing)" if adaptive
                   else rf"$\delta E = {delta:.1f}$ kHz")
    suptitle = (
        f"{stem}    "
        rf"$\omega = {omega_kHz:.2f}$ kHz   "
        rf"$\Omega_R/2\pi = {summary.get('Omega_R_kHz',179):.0f}$ kHz   "
        rf"$\tau = {summary.get('tau_us',30):.0f}\,\mu$s   "
        + delta_label
    )

    # ---- helper: physically meaningful zoom window per block, plus
    # the position of the "expected" peak in block-relative E.
    #
    # E_h^block (M_F=-4 frame) = -E_b   (halo is below entrance threshold).
    # For halo+ω → continuum in M_F=-3:  E_final^block = -E_b + ω + E_th_diff_compensated
    # but in the rotating-frame convention the resonance simply lives at
    # ω in M_F=-3 block-relative E.  Likewise (2ω−E_b) for 2γ in M_F=-2.
    E_b_kHz = abs(omega_kHz / 8.0) if "E_h_kHz" not in summary else abs(summary["E_h_kHz"])
    expected_peak_kHz = {
        -4:  +abs(E_b_kHz) * 0.5,    # ZEPE near threshold, ~ E_b/2
        -3:  +omega_kHz,             # 1γ resonance
        -2:  +(2 * omega_kHz - E_b_kHz),
        -5:  None,                    # no real peak; just show density
    }
    # Default zoom window per block.  Picked to capture the dominant
    # peak and a sane neighborhood; we then INTERSECT with the actual
    # data range so we don't draw empty white space.
    # Photoelectron-spectrum zoom windows: start at threshold (E_block = 0)
    # since the bound halo at E_block = -E_b is the initial state, NOT
    # part of the dP/dE.
    default_zoom = {
        -4: (  0.0, +200.0),
        -3: (  0.0, +250.0),
        -2: (  0.0, +500.0),
        -5: (  0.0, +200.0),    # virtual; no negative E states
    }

    # Paper Eq. (5) ZEPE PT prediction (Azizi/Saalmann/Rost 2024).
    tau_us     = summary.get("tau_us", 30.0)
    T_paper_us = tau_us * np.sqrt(2.0)
    DeltaE_kHz = 1000.0 / (np.pi * T_paper_us)
    beta       = E_b_kHz / DeltaE_kHz
    def E_max_PT(ell):
        return 0.5 * E_b_kHz * (np.sqrt(1.0 + (2*ell + 1)/beta**2) - 1.0)

    # ---- 1) 4-panel dP/dE per block.  Three OBSERVABLE channels (M_F=-4
    # ZEPE, M_F=-3 1γ, M_F=-2 2γ) plus M_F=-5 which is the VIRTUAL block
    # (kept for theoretical interest -- never produces photoelectrons,
    # but its population indicates how complete the basis is).
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, mf in zip(axes.flat, [-5, -4, -3, -2]):
        if mf not in block_data:
            ax.text(0.5, 0.5, f"no data for M_F={mf}",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        d = block_data[mf]
        zlo, zhi = default_zoom[mf]
        # Representative width: per-block median of adaptive σ, or scalar δE.
        sig_typ = (float(np.median(d["sigma_per_state"]))
                   if (adaptive and d["sigma_per_state"] is not None) else delta)
        sigma_use = (d["sigma_per_state"]
                     if (adaptive and d["sigma_per_state"] is not None) else delta)
        # Clamp to actual data range.
        if d["states_E_block_kHz"] is not None:
            zlo = max(zlo, d["states_E_block_kHz"].min() - 5*sig_typ)
            zhi = min(zhi, d["states_E_block_kHz"].max() + 5*sig_typ)
        # Re-smooth on a FINE zoom-local grid (grid spacing ≪ σ) so that
        # narrow features inside this zoom are properly resolved.
        # only_continuum=True for the open blocks (M_F=-4,-3,-2) so the
        # photoelectron spectrum starts at threshold.  For M_F=-5 there
        # is no bound state in the block, so the flag is harmless.
        if d["states_E_block_kHz"] is not None:
            E = np.linspace(zlo, zhi, 4000)
            y = smooth_states(d["states_E_block_kHz"], d["states_prob"],
                              E, sigma_use, only_continuum=True)
        else:
            E = d["E_block_kHz"]
            y = d["dPdE"]
        m_zoom = (E >= zlo) & (E <= zhi)

        # The smoothed dP/dE curve.
        ax.plot(E[m_zoom], y[m_zoom], color=colors[mf], lw=1.4)

        # Rug ticks at discrete state positions, but ONLY those inside the
        # zoom AND above a population threshold (prevent the "solid black bar"
        # of thousands of overlapping ticks).
        if d["states_E_block_kHz"] is not None:
            E_st = d["states_E_block_kHz"]
            p_st = d["states_prob"]
            sel  = (E_st >= zlo) & (E_st <= zhi) & (p_st > 0)
            if sel.any():
                # Only show top 50 by population in the zoom window.
                idx_sorted = np.argsort(p_st[sel])[::-1][:50]
                E_show = E_st[sel][idx_sorted]
                p_show = p_st[sel][idx_sorted]
                top_y = y[m_zoom].max() if m_zoom.any() else 1.0
                # Height proportional to log(probability) so we see the spread.
                h_norm = (np.log10(p_show) - np.log10(p_show.min()) + 1.0)
                h_norm = h_norm / h_norm.max() * 0.10 * top_y
                for E_a, h in zip(E_show, h_norm):
                    ax.plot([E_a, E_a], [0, h], color="k", alpha=0.4, lw=0.4)

        # Mark the peak we found in the data (vertical line, solid).
        if m_zoom.any():
            i_peak = np.argmax(y[m_zoom])
            E_peak_data = E[m_zoom][i_peak]
            ax.axvline(E_peak_data, color=colors[mf], ls="-", lw=0.6, alpha=0.5)
            ax.text(E_peak_data, y[m_zoom][i_peak] * 1.02,
                    rf"  peak @ {E_peak_data:+.1f} kHz",
                    color=colors[mf], fontsize=8, ha="left", va="bottom")

        # Mark the "expected" resonance position (kinematic).
        ep = expected_peak_kHz[mf]
        if ep is not None and zlo <= ep <= zhi:
            if mf == -4:
                # For ZEPE we overlay the paper Eq. (5) PT prediction (ℓ=0).
                E_PT = E_max_PT(0)
                ax.axvline(E_PT, color="k", ls="--", lw=0.8, alpha=0.7,
                           label=rf"PT $\ell{{=}}0$: {E_PT:+.2f} kHz")
                ax.legend(loc="upper right", fontsize=8)
            elif mf == -3:
                ax.axvline(ep, color="k", ls="--", lw=0.8, alpha=0.7,
                           label=rf"$\omega = {ep:.1f}$ kHz")
                ax.legend(loc="upper right", fontsize=8)
            elif mf == -2:
                ax.axvline(ep, color="k", ls="--", lw=0.8, alpha=0.7,
                           label=rf"$2\omega-E_b = {ep:.1f}$ kHz")
                ax.legend(loc="upper right", fontsize=8)

        ax.set_xlim(zlo, zhi)
        ax.set_xlabel("E above block threshold  [kHz]")
        ax.set_ylabel(r"$dP/dE$  [1/kHz]")
        # Show n_states inside the zoom — if it's small relative to total,
        # the plot is undersampling the relevant region (basis-edge sentinel).
        n_states_total = len(d["states_E_block_kHz"]) if d["states_E_block_kHz"] is not None else 0
        n_states_zoom = sel.sum() if d["states_E_block_kHz"] is not None else 0
        ax.set_title(labels[mf] +
                     f"  ({n_states_zoom}/{n_states_total} states in window)")
        ax.grid(alpha=0.3)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = outdir / f"{stem}_dPdE_per_block_{fname_tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")

    # ---- 2) OBSERVABLE photoelectron spectrum.  Only M_F = -4, -3, -2
    # produce real outgoing population (M_F=-5 is virtual; halo is the
    # initial state, observable as DEPLETION).  Style follows
    # Azizi/Saalmann/Rost 2024 Fig. 2: log-log scaled spectrum with
    # kinematic landmarks.
    P_halo = summary.get("P_halo", 0.0)
    P_M4c  = summary.get("P_M-4_continuum", 0.0)
    P_M3   = summary.get("P_M-3_total", 0.0)
    P_M2   = summary.get("P_M-2_total", 0.0)
    P_ion  = P_M4c + P_M3 + P_M2          # total ionization probability

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Photoelectron spectrum starts at threshold (E_block = 0).  No
    # negative-E tail from the halo bound state contaminating the
    # observable region.
    overlay_lo, overlay_hi = 0.0, 500.0
    E_grid = np.linspace(overlay_lo, overlay_hi, 4000)
    obs_labels = {
        -4: rf"$M_F=-4$ (ZEPE):  $P={P_M4c:.4f}$",
        -3: rf"$M_F=-3$ (1$\gamma$):  $P={P_M3:.4f}$",
        -2: rf"$M_F=-2$ (2$\gamma$):  $P={P_M2:.4f}$",
    }
    for mf in [-4, -3, -2]:
        if mf not in block_data:
            continue
        d = block_data[mf]
        sigma_use = (d["sigma_per_state"]
                     if (adaptive and d["sigma_per_state"] is not None) else delta)
        if d["states_E_block_kHz"] is not None:
            y_o = smooth_states(d["states_E_block_kHz"], d["states_prob"],
                                E_grid, sigma_use, only_continuum=True)
        else:
            y_o = np.interp(E_grid, d["E_block_kHz"], d["dPdE"])
        m = y_o > 0
        ax.semilogy(E_grid[m], y_o[m],
                    color=colors[mf], lw=1.6, label=obs_labels[mf])
    ax.axvline(0.0, color="k", ls=":", lw=0.6, alpha=0.7)
    ax.axvline(E_max_PT(0), color=colors[-4], ls=":", lw=0.8, alpha=0.7)
    ax.axvline(omega_kHz, color=colors[-3], ls=":", lw=0.8, alpha=0.7)
    ax.axvline(2 * omega_kHz - E_b_kHz, color=colors[-2], ls=":", lw=0.8, alpha=0.7)
    # Inline kinematic-landmark labels at the top of the panel.
    y_top = ax.get_ylim()[1]
    ax.text(E_max_PT(0), y_top, "  ZEPE PT", color=colors[-4],
            fontsize=8, ha="left", va="top")
    ax.text(omega_kHz, y_top, r"  $\omega$", color=colors[-3],
            fontsize=8, ha="left", va="top")
    ax.text(2*omega_kHz - E_b_kHz, y_top, r"  $2\omega{-}E_b$", color=colors[-2],
            fontsize=8, ha="left", va="top")
    ax.set_xlim(overlay_lo, overlay_hi)
    ax.set_xlabel("photoelectron kinetic energy above each $M_F$ threshold  [kHz]")
    ax.set_ylabel(r"$dP/dE$  [1/kHz]")
    ax.set_title("Photoelectron spectrum (observable channels) — "
                 + suptitle.split('   ',1)[0],
                 fontsize=10)
    pulse_area = (2*np.pi*summary.get('Omega_R_kHz',179)*1e3
                  *np.sqrt(2*np.pi)*tau_us*1e-6)
    txt = (
        rf"$P_{{\rm halo}} = {P_halo:.3e}$"  + "\n"
        rf"$P_{{\rm ion}} = {1-P_halo:.4f}$"  + "\n"
        rf"$\beta = E_b/\Delta E = {beta:.2f}$" + "\n"
        rf"pulse area $= {pulse_area:.1f}$ rad"
    )
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="lightgray", alpha=0.85))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = outdir / f"{stem}_observable_spectrum_{fname_tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")

    # ---- 2b) ZEPE close-up (single panel zoomed on M_F=-4 around the
    # near-threshold region, with the paper's Eq. (5) PT prediction
    # marked for ℓ=0,2 — the s and d wave channels accessible from a
    # halo s-wave initial state via two-photon absorb-emit.
    if -4 in block_data:
        # Two-panel close-up of the ZEPE channel: linear (peak shape)
        # on the left, log-log (Wigner threshold tail) on the right.
        fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5))
        d = block_data[-4]
        sigma_use = (d["sigma_per_state"]
                     if (adaptive and d["sigma_per_state"] is not None) else delta)
        prob_pt = d.get("states_prob_pt")
        # Continuum-only smoothing — drop the halo (E_block < 0).
        if d["states_E_block_kHz"] is not None:
            E_lin = np.linspace(0.0, 80.0, 4000)
            y_lin = smooth_states(d["states_E_block_kHz"], d["states_prob"],
                                  E_lin, sigma_use, only_continuum=True)
            y_lin_pt = (smooth_states(d["states_E_block_kHz"], prob_pt,
                                       E_lin, sigma_use, only_continuum=True)
                        if prob_pt is not None else None)
            # Log-log: span 4 decades from a fraction of the peak position.
            E_log = np.logspace(-2, 3, 4000)    # 0.01 .. 1000 kHz
            y_log = smooth_states(d["states_E_block_kHz"], d["states_prob"],
                                  E_log, sigma_use, only_continuum=True)
            y_log_pt = (smooth_states(d["states_E_block_kHz"], prob_pt,
                                       E_log, sigma_use, only_continuum=True)
                        if prob_pt is not None else None)
        else:
            E_lin = d["E_block_kHz"]; y_lin = d["dPdE"]
            mlin = E_lin >= 0
            E_lin = E_lin[mlin]; y_lin = y_lin[mlin]
            E_log = E_lin; y_log = y_lin
            y_lin_pt = y_log_pt = None

        # ---- left panel (linear) ----------------------------------------
        ax_lin.plot(E_lin, y_lin, color=colors[-4], lw=1.6,
                    label=(r"TDSE  $\delta E=$ adaptive" if adaptive
                           else rf"TDSE  $\delta E={delta:.1f}$ kHz"))
        if y_lin_pt is not None:
            ax_lin.plot(E_lin, y_lin_pt, color="tab:purple", lw=1.2,
                        ls="--", alpha=0.85,
                        label="2nd-order PT (closed-form)")
        # PT prediction (paper Eq. 5).  Our problem keeps ℓ=0 throughout:
        # the σ⁺ photon raises M_F (spin) but does NOT change the orbital
        # angular momentum, so the halo (s-wave) returns to s-wave after
        # the two-photon absorb-emit pathway.  The d-wave (ℓ=2) channel
        # of the original paper requires an orbital-changing dipole and
        # is absent here.
        x_PT = E_max_PT(0)
        ax_lin.axvline(x_PT, color="tab:orange", ls="--", lw=1.0,
                       label=rf"PT $\ell=0$ (s-wave): $E_{{\max}}={x_PT:+.2f}$ kHz")
        # Rug ticks for the most populated states (continuum only).
        if d["states_E_block_kHz"] is not None:
            E_st = d["states_E_block_kHz"]
            p_st = d["states_prob"]
            sel = (E_st >= 0.0) & (E_st <= 80.0) & (p_st > 0)
            if sel.any():
                top = y_lin.max() if y_lin.size else 1.0
                idx = np.argsort(p_st[sel])[::-1][:50]
                E_show = E_st[sel][idx]; p_show = p_st[sel][idx]
                h_norm = (np.log10(p_show) - np.log10(p_show.min()) + 1.0)
                h_norm = h_norm / h_norm.max() * 0.10 * top
                for E_a, h in zip(E_show, h_norm):
                    ax_lin.plot([E_a, E_a], [0, h], color="k", alpha=0.4, lw=0.4)
        ax_lin.set_xlim(0.0, 80.0)
        ax_lin.set_xlabel("photoelectron kinetic energy  [kHz]")
        ax_lin.set_ylabel(r"$dP/dE$  [1/kHz]")
        ax_lin.set_title("linear scale (peak shape)")
        ax_lin.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_lin.grid(alpha=0.3)

        # ---- right panel (log-log) -------------------------------------
        m_pos = y_log > 0
        ax_log.loglog(E_log[m_pos], y_log[m_pos], color=colors[-4], lw=1.6,
                      label=(r"TDSE  $\delta E=$ adaptive" if adaptive
                           else rf"TDSE  $\delta E={delta:.1f}$ kHz"))
        if y_log_pt is not None:
            mp = y_log_pt > 0
            ax_log.loglog(E_log[mp], y_log_pt[mp], color="tab:purple", lw=1.2,
                          ls="--", alpha=0.85,
                          label="2nd-order PT (closed-form)")
        # PT line (s-wave only; see comment on linear panel).
        ax_log.axvline(x_PT, color="tab:orange", ls="--", lw=1.0,
                       label=rf"PT $\ell=0$: $E_{{\max}}={x_PT:+.2f}$ kHz")
        # Wigner threshold law guide for the s-wave channel: dP/dE ∝ E^{1/2}.
        if y_log.max() > 0:
            E_ref = 1.0   # kHz, anchor point for the power-law guide
            i_ref = np.argmin(np.abs(E_log - E_ref))
            if y_log[i_ref] > 0:
                y_ref = y_log[i_ref] / 3.0
                yguide = y_ref * (E_log / E_ref) ** 0.5
                mg = (yguide > 1e-30) & (E_log < 50.0)
                ax_log.loglog(E_log[mg], yguide[mg],
                              color="gray", ls=":", lw=0.8, alpha=0.6,
                              label=r"Wigner s-wave  $\propto E^{1/2}$")
        ax_log.set_xlim(0.05, 1000.0)
        ax_log.set_xlabel("photoelectron kinetic energy  [kHz]")
        ax_log.set_ylabel(r"$dP/dE$  [1/kHz]")
        ax_log.set_title("log-log (threshold law + tail)")
        ax_log.legend(loc="lower center", fontsize=8, framealpha=0.9)
        ax_log.grid(alpha=0.3, which="both")

        pulse_area = (2*np.pi*summary.get('Omega_R_kHz',179)*1e3
                      *np.sqrt(2*np.pi)*tau_us*1e-6)
        fig.suptitle(rf"ZEPE channel $M_F=-4$ — {stem}     "
                     rf"$\beta=E_b/\Delta E = {beta:.2f}$     "
                     rf"pulse area $= {pulse_area:.1f}$ rad",
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = outdir / f"{stem}_ZEPE_closeup_{fname_tag}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")

    # ---- 3) Population summary.  Observable: halo retention + the
    # three M_F-resolved continuum populations (= partial photoelectron
    # yields).  Plus M_F=-5 (virtual) shown separately for theoretical
    # interest as a basis-completeness diagnostic.
    keys = ["halo retention",
            "ZEPE (M_F=-4)",
            "1γ (M_F=-3)",
            "2γ (M_F=-2)",
            "virtual (M_F=-5)\n(NOT observable)"]
    vals = [P_halo,
            summary.get("P_M-4_continuum", 0.0),
            summary.get("P_M-3_total", 0.0),
            summary.get("P_M-2_total", 0.0),
            summary.get("P_M-5_total", 0.0)]
    cols = ["tab:blue", "tab:cyan", "tab:green", "tab:red", "lightgray"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(keys, vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v * 1.05 if v > 0 else 1e-3,
                f"{v:.3e}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(max(min(vals) / 5, 1e-7), 2)
    ax.set_ylabel(r"population at $t=T$")
    ax.set_title(f"Block populations  —  {stem}\n"
                 rf"$\|c\|-1 = {summary.get('err_unitary', 0):.1e}$")
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.autofmt_xdate(rotation=20, ha="right")
    fig.tight_layout()
    out = outdir / f"{stem}_block_totals.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
