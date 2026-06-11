#!/usr/bin/env python3
"""
plot_polar.py
=============
Render a 2 × 3 polar figure from the two .dat files written by
compute_polar_data.py:

  ┌──────────────────────────────────────────────────────────┐
  │  Row A — FIXED Euler angles (α, β) (defaults α=β=0)      │
  │    A1: σ(E, θ; α, β)                                      │
  │    A2: τ(E, θ; α, β)                                      │
  │    A3: σ·τ(E, θ; α, β)                                    │
  ├──────────────────────────────────────────────────────────┤
  │  Row B — ORIENTATION (polarisation) AVERAGE              │
  │    B1: σ_pol(E, θ)                                        │
  │    B2: τ_pol(E, θ)                                        │
  │    B3: σ_pol·τ_pol(E, θ)                                  │
  └──────────────────────────────────────────────────────────┘

Polar convention: angle = θ (emission polar angle in molecular frame),
radius = E_kin in eV (other choices: k, ω -- see --radius-mode).  The
data is at φ=0 only; we mirror to [0, 2π] to fill the disk (this assumes
left/right symmetry — true for any target with O_h or with an σ_v plane
containing the polarisation axis; for general low-symmetry targets you
should compute φ=π separately and concatenate).

Usage
-----
    python plot_polar.py --fixed  polar_fixed_alpha0_beta0_len.dat \
                         --avg    polar_pol_avg_len.dat \
                         --output polar_panels.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
try:
    from matplotlib.colors import AsinhNorm  # matplotlib ≥ 3.6
    _HAS_ASINH = True
except Exception:
    _HAS_ASINH = False
import numpy as np


HA_TO_EV = 27.211386245988
A02_TO_MB = 28.002852      # 1 bohr² = 28.003 Mb


# --------------------------- I/O ---------------------------------
def load_polar_dat(path: Path):
    """Load a polar_*.dat file and reshape into (k, E_kin, omega, theta,
    sigma, tau, sigma_tau) where sigma/tau/sigma_tau are (nE, nT)."""
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"{path}: expected 7 columns, got {data.shape}")
    k_u  = np.unique(data[:, 0])
    th_u = np.unique(data[:, 3])
    nE, nT = k_u.size, th_u.size
    if data.shape[0] != nE * nT:
        raise ValueError(
            f"{path}: rows {data.shape[0]} ≠ nE × nT = {nE}×{nT}")
    order = np.lexsort((data[:, 3], data[:, 0]))
    d = data[order]
    k     = d[:, 0].reshape(nE, nT)[:, 0]
    e_kin = d[:, 1].reshape(nE, nT)[:, 0]
    omega = d[:, 2].reshape(nE, nT)[:, 0]
    theta = d[:, 3].reshape(nE, nT)[0, :]
    sigma     = d[:, 4].reshape(nE, nT)
    tau       = d[:, 5].reshape(nE, nT)
    sigma_tau = d[:, 6].reshape(nE, nT)
    return k, e_kin, omega, theta, sigma, tau, sigma_tau


def mirror_theta(theta_half, *arrays):
    """θ ∈ [0, π] → [0, 2π] by mirror, excluding endpoint duplicates."""
    theta = np.asarray(theta_half, float)
    order = np.argsort(theta)
    theta = theta[order]
    arrays = [np.asarray(a)[:, order] for a in arrays]
    core_t   = theta[1:-1]
    core_a   = [a[:, 1:-1] for a in arrays]
    theta_m  = 2.0 * np.pi - core_t[::-1]
    arrs_m   = [a[:, ::-1] for a in core_a]
    theta_full = np.concatenate([theta, theta_m])
    arrs_full  = [np.concatenate([a0, am], axis=1)
                  for a0, am in zip(arrays, arrs_m)]
    return (theta_full,) + tuple(arrs_full)


def autoscale_log(x: np.ndarray, lo_pct=1.0, hi_pct=99.0) -> Tuple[float, float]:
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return 1e-12, 1.0
    vmin = float(np.percentile(x, lo_pct))
    vmax = float(np.percentile(x, hi_pct))
    vmin = max(vmin, float(np.min(x)))
    vmax = max(vmax, vmin * 10.0)
    return vmin, vmax


def autoscale_sym(x: np.ndarray, hi_pct=99.0) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    return float(np.percentile(np.abs(x), hi_pct)) or 1.0


def autoscale_signed(x: np.ndarray, hi_pct=99.0, lo_pct=1.0
                     ) -> Tuple[float, float]:
    """Return (vlim, linthresh) tuned to *show structure across the full
    dynamic range* of a signed dataset.

    vlim       = `hi_pct`-th percentile of |x|  (saturate top 1%).
    linthresh  = `lo_pct`-th percentile of |x|  (only the bottom 1% sits
                 in the near-zero band; the rest is log-resolved).

    Data-driven: linthresh follows the actual distribution rather than
    being a fraction of vlim.  For data with 10-decade dynamic range,
    this gives ~9-10 decades of visible log resolution on each side of
    zero.
    """
    x = x[np.isfinite(x) & (x != 0)]
    if x.size == 0:
        return 1.0, 1e-3
    ax = np.abs(x)
    vlim = float(np.percentile(ax, hi_pct))
    lin  = float(np.percentile(ax, lo_pct))
    if not np.isfinite(lin) or lin <= 0:
        lin = max(vlim * 1e-8, 1e-30)
    lin = min(lin, vlim * 0.5)
    return vlim, lin


# --------------------------- plot one panel ----------------------
def panel_log(ax, Theta, R, Z_pos, title, cbar_label, vmin=None, vmax=None,
              cmap="viridis"):
    Zp = np.ma.masked_where(~np.isfinite(Z_pos) | (Z_pos <= 0), Z_pos)
    if vmin is None or vmax is None:
        vlo, vhi = autoscale_log(Zp.compressed())
        vmin = vmin if vmin is not None else vlo
        vmax = vmax if vmax is not None else vhi
    pcm = ax.pcolormesh(Theta, R, Zp,
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        cmap=cmap, shading="auto")
    ax.set_title(title, fontsize=10)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_thetagrids(np.arange(0, 360, 45),
                      labels=[f"{a}°" for a in np.arange(0, 360, 45)])
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    cb = plt.colorbar(pcm, ax=ax, pad=0.10, shrink=0.78)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=6)


def panel_sym(ax, Theta, R, Z_signed, title, cbar_label, vlim=None,
              cmap="bwr"):
    Zp = np.ma.masked_where(~np.isfinite(Z_signed), Z_signed)
    if vlim is None:
        vlim = autoscale_sym(Zp.compressed())
    pcm = ax.pcolormesh(Theta, R, Zp,
                        cmap=cmap, vmin=-vlim, vmax=+vlim,
                        shading="auto")
    ax.set_title(title, fontsize=10)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_thetagrids(np.arange(0, 360, 45),
                      labels=[f"{a}°" for a in np.arange(0, 360, 45)])
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    cb = plt.colorbar(pcm, ax=ax, pad=0.10, shrink=0.78)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=6)


def panel_symlog(ax, Theta, R, Z_signed, title, cbar_label,
                 vlim=None, linthresh=None,
                 lo_pct=1.0, linscale=0.5,
                 norm_kind="asinh", cmap="bwr"):
    """Render a signed polar field with structure-visible defaults.

    `norm_kind`:
      "asinh"  - smooth signed-log via arcsinh (matplotlib ≥ 3.6).  No
                 linear band, no white centre stripe.  Default.
      "symlog" - classic linear-near-zero + log-outside.  Use only when
                 the linear band is meant to absorb a known noise floor;
                 `linscale` then controls how much colormap budget the
                 band consumes.
    """
    Zp = np.ma.masked_where(~np.isfinite(Z_signed), Z_signed)
    if vlim is None or linthresh is None:
        v_auto, lin_auto = autoscale_signed(Zp.compressed(), lo_pct=lo_pct)
        vlim      = v_auto   if vlim      is None else vlim
        linthresh = lin_auto if linthresh is None else linthresh
    if norm_kind == "asinh" and _HAS_ASINH:
        norm = AsinhNorm(linear_width=linthresh, vmin=-vlim, vmax=+vlim)
    else:
        if norm_kind == "asinh":
            print("  [warn] AsinhNorm unavailable; falling back to SymLogNorm")
        norm = SymLogNorm(linthresh=linthresh, linscale=linscale,
                          vmin=-vlim, vmax=+vlim, base=10)
    pcm = ax.pcolormesh(Theta, R, Zp, norm=norm, cmap=cmap, shading="auto")
    ax.set_title(title, fontsize=10)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_thetagrids(np.arange(0, 360, 45),
                      labels=[f"{a}°" for a in np.arange(0, 360, 45)])
    ax.tick_params(axis="x", labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    cb = plt.colorbar(pcm, ax=ax, pad=0.10, shrink=0.78)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=6)


# --------------------------- main --------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixed",  required=True, type=Path,
        help="Fixed-(α,β) .dat from compute_polar_data.py")
    ap.add_argument("--avg",    required=True, type=Path,
        help="Polarisation-averaged .dat from compute_polar_data.py")
    ap.add_argument("--output", required=True, type=Path,
        help="Output PNG path")
    ap.add_argument("--radius-mode", choices=("Ekin_eV", "omega_eV", "k_au"),
                    default="Ekin_eV")
    ap.add_argument("--no-mirror", action="store_true",
        help="Do NOT mirror θ ∈ [0,π] to [0,2π]; show a half-disc.  Set this "
             "if your target lacks the σ_v(xz) reflection.")
    ap.add_argument("--sigma-cmap", default="viridis")
    ap.add_argument("--tau-cmap",   default="bwr")
    ap.add_argument("--sigtau-cmap", default="seismic")
    # explicit colour limits (optional; otherwise auto-scale by percentile)
    ap.add_argument("--sigma-vmin", type=float, default=None,
        help="Lower limit of σ colourbar in Mb (LogNorm).")
    ap.add_argument("--sigma-vmax", type=float, default=None)
    ap.add_argument("--tau-vlim",   type=float, default=None,
        help="Symmetric τ colourbar limit in as (default: 99th percentile).")
    ap.add_argument("--sigtau-vlim", type=float, default=None,
        help="Symmetric σ·τ colourbar limit (Mb·as) -- outer extent of the "
             "SymLogNorm colourbar.")
    ap.add_argument("--sigtau-linthresh", type=float, default=None,
        help="Near-zero half-width of the σ·τ colour scale (Mb·as).  "
             "Default: data-driven via --sigtau-lo-pct.")
    ap.add_argument("--sigtau-lo-pct", type=float, default=1.0,
        help="Percentile of |σ·τ| used as the linear/transition threshold "
             "(default: 1.0 → only the bottom 1%% sits in the near-zero "
             "band; 99%% gets log/asinh resolution).")
    ap.add_argument("--sigtau-norm", choices=("asinh", "symlog"), default="asinh",
        help="σ·τ colour-scale mapping (default: asinh, smooth signed-log "
             "via arcsinh — NO white linear band).  Use 'symlog' only to "
             "absorb a known noise floor into the central white stripe.")
    ap.add_argument("--sigtau-linscale", type=float, default=0.5,
        help="(SymLogNorm only) colormap-budget share for the linear band "
             "(default: 0.5 = half a decade of colour reserved for the "
             "linear region).")
    args = ap.parse_args()

    print("=" * 72)
    print(" plot_polar.py")
    print("=" * 72)

    # --- load ---
    kF, EF, wF, thF, sigF, tauF, stF = load_polar_dat(args.fixed)
    kA, EA, wA, thA, sigA, tauA, stA = load_polar_dat(args.avg)
    if not (np.allclose(kF, kA) and np.allclose(thF, thA)):
        raise SystemExit("ERROR: fixed and avg dat files have different grids")

    # --- unit conversions ---
    if args.radius_mode == "Ekin_eV":
        r = EF * HA_TO_EV; r_label = r"$E_{\rm kin}$ [eV]"
    elif args.radius_mode == "omega_eV":
        r = wF * HA_TO_EV; r_label = r"$\hbar\omega$ [eV]"
    else:
        r = kF; r_label = r"$k$ [a.u.]"
    # convert σ → Mb, σ·τ → Mb·as
    sigF_mb  = sigF * A02_TO_MB
    sigA_mb  = sigA * A02_TO_MB
    stF_mbas = stF  * A02_TO_MB
    stA_mbas = stA  * A02_TO_MB

    # --- mirror θ ---
    if not args.no_mirror:
        out = mirror_theta(thF, sigF_mb, tauF, stF_mbas,
                           sigA_mb, tauA, stA_mbas)
        theta = out[0]
        (sigF_mb, tauF, stF_mbas, sigA_mb, tauA, stA_mbas) = out[1:]
    else:
        theta = thF

    Theta, R = np.meshgrid(theta, r)

    # --- header attrs (Euler angles) — read from the .dat file header ---
    alpha_deg = 0.0; beta_deg = 0.0
    try:
        with open(args.fixed) as f:
            for line in f:
                if line.startswith("# alpha"):
                    alpha_deg = float(line.split("=")[1].strip().split()[0])
                if line.startswith("# beta"):
                    beta_deg = float(line.split("=")[1].strip().split()[0])
    except Exception:
        pass

    print(f"  fixed (α, β) = ({alpha_deg}°, {beta_deg}°)")
    print(f"  energies    : {kF.size}    radius mode: {args.radius_mode}")
    print(f"  theta points: {theta.size}   "
          f"({'mirrored to 2π' if not args.no_mirror else 'half-disc'})")

    # --- figure ---
    fig, axes = plt.subplots(2, 3,
                             figsize=(13.5, 9.5),
                             subplot_kw={"projection": "polar"})

    # ---- ROW A: fixed (α, β) ----
    panel_log(axes[0, 0], Theta, R, sigF_mb,
              title=fr"$\sigma(E, \theta)$  fixed  $(\alpha,\beta)=({alpha_deg:g}°,{beta_deg:g}°)$,  $\phi=0$",
              cbar_label="σ [Mb]",
              vmin=args.sigma_vmin, vmax=args.sigma_vmax,
              cmap=args.sigma_cmap)
    panel_sym(axes[0, 1], Theta, R, tauF,
              title=fr"$\tau(E, \theta)$  fixed  $(\alpha,\beta)=({alpha_deg:g}°,{beta_deg:g}°)$,  $\phi=0$",
              cbar_label="τ [as]",
              vlim=args.tau_vlim,
              cmap=args.tau_cmap)
    panel_symlog(axes[0, 2], Theta, R, stF_mbas,
              title=fr"$\sigma\,\tau(E, \theta)$  fixed  $(\alpha,\beta)=({alpha_deg:g}°,{beta_deg:g}°)$  ({args.sigtau_norm})",
              cbar_label="σ·τ [Mb·as]",
              vlim=args.sigtau_vlim, linthresh=args.sigtau_linthresh,
              lo_pct=args.sigtau_lo_pct,
              linscale=args.sigtau_linscale,
              norm_kind=args.sigtau_norm,
              cmap=args.sigtau_cmap)

    # ---- ROW B: orientation average (Eq. S38) ----
    panel_log(axes[1, 0], Theta, R, sigA_mb,
              title=r"$\langle\sigma(E,\theta)\rangle_{\rm pol}$  isotropic avg [Eq. S38]",
              cbar_label="σ [Mb]",
              vmin=args.sigma_vmin, vmax=args.sigma_vmax,
              cmap=args.sigma_cmap)
    panel_sym(axes[1, 1], Theta, R, tauA,
              title=r"$\langle\tau(E,\theta)\rangle_{\rm pol}$  isotropic avg [Eq. S38]",
              cbar_label="τ [as]",
              vlim=args.tau_vlim,
              cmap=args.tau_cmap)
    panel_symlog(axes[1, 2], Theta, R, stA_mbas,
              title=fr"$\langle\sigma\,\tau\rangle_{{\rm pol}}$  isotropic avg  ({args.sigtau_norm})",
              cbar_label="σ·τ [Mb·as]",
              vlim=args.sigtau_vlim, linthresh=args.sigtau_linthresh,
              lo_pct=args.sigtau_lo_pct,
              linscale=args.sigtau_linscale,
              norm_kind=args.sigtau_norm,
              cmap=args.sigtau_cmap)

    # --- radius label (place on bottom-left panel) ---
    for ax in axes.flat:
        ax.set_rlabel_position(135)

    fig.suptitle(
        f"Molecular-frame polar plots at $\\phi=0$  "
        f"(top: fixed Euler $(\\alpha,\\beta)=({alpha_deg:g}°,{beta_deg:g}°)$;  "
        f"bottom: isotropic polarisation average)\n"
        f"radius = {r_label}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
