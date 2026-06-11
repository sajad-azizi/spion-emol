#!/usr/bin/env python3
"""
plot_sigma_tau_only.py
======================
Standalone σ·τ-only figure: ONE row, TWO columns (side-by-side polar
panels), reading the same .dat files written by compute_polar_data.py.

  ┌────────────────────────────────────────────────────────────┐
  │  Left : σ·τ(E, θ; α, β)  fixed Euler                       │
  │  Right: ⟨σ·τ⟩_pol(E, θ)   isotropic polarisation average    │
  └────────────────────────────────────────────────────────────┘

Default energy axis: kinetic energy in eV, capped at --e-max (default
100 eV).  Uses the same SymLogNorm colour scaling as plot_polar.py.

Usage
-----
    python plot_sigma_tau_only.py \
        --fixed  polar_fixed_alpha0_beta0_len.dat \
        --avg    polar_pol_avg_len.dat            \
        --output sigma_tau_only.png               \
        --e-max  100                              \
        --radius-mode Ekin_eV
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
try:
    from matplotlib.colors import AsinhNorm  # matplotlib ≥ 3.6
    _HAS_ASINH = True
except Exception:
    _HAS_ASINH = False
import numpy as np


HA_TO_EV = 27.211386245988
A02_TO_MB = 28.002852


# --------------------------- I/O ---------------------------------
def load_polar_dat(path: Path):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"{path}: expected 7 columns, got {data.shape}")
    k_u = np.unique(data[:, 0]); th_u = np.unique(data[:, 3])
    nE, nT = k_u.size, th_u.size
    if data.shape[0] != nE * nT:
        raise ValueError(f"{path}: rows {data.shape[0]} ≠ nE × nT")
    order = np.lexsort((data[:, 3], data[:, 0]))
    d = data[order]
    k     = d[:, 0].reshape(nE, nT)[:, 0]
    e_kin = d[:, 1].reshape(nE, nT)[:, 0]
    omega = d[:, 2].reshape(nE, nT)[:, 0]
    theta = d[:, 3].reshape(nE, nT)[0, :]
    sigma_tau = d[:, 6].reshape(nE, nT)
    return k, e_kin, omega, theta, sigma_tau


def mirror_theta(theta_half, *arrays):
    theta = np.asarray(theta_half, float)
    order = np.argsort(theta)
    theta = theta[order]
    arrays = [np.asarray(a)[:, order] for a in arrays]
    core_t = theta[1:-1]
    core_a = [a[:, 1:-1] for a in arrays]
    theta_m = 2.0 * np.pi - core_t[::-1]
    arrs_m  = [a[:, ::-1] for a in core_a]
    theta_full = np.concatenate([theta, theta_m])
    arrs_full  = [np.concatenate([a0, am], axis=1) for a0, am in zip(arrays, arrs_m)]
    return (theta_full,) + tuple(arrs_full)


def autoscale_signed(x, hi_pct=99.0, lo_pct=1.0):
    """Return (vlim, linthresh) tuned to *show structure across the full
    dynamic range* of a signed dataset.

    vlim       = `hi_pct`-th percentile of |x|.  Saturates the top 1% so a
                 few extreme spikes don't burn the colormap budget.
    linthresh  = `lo_pct`-th percentile of |x|.  The 1st percentile is the
                 boundary between "bottom-1% near-zero noise" and "actual
                 small structure" -- only the noise sits in the white
                 linear band of SymLogNorm.

    Compared with the earlier `linthresh = vlim × fixed_fraction` heuristic
    this is data-driven: if the dynamic range is 8 decades (some |x| down
    near FP-eps), linthresh follows it down; if the dynamic range is only
    2 decades, linthresh stays close to vlim and the linear band remains
    a tiny strip.
    """
    x = x[np.isfinite(x) & (x != 0)]
    if x.size == 0:
        return 1.0, 1e-3
    ax = np.abs(x)
    vlim = float(np.percentile(ax, hi_pct))
    lin  = float(np.percentile(ax, lo_pct))
    if not np.isfinite(lin) or lin <= 0:
        lin = max(vlim * 1e-8, 1e-30)
    lin = min(lin, vlim * 0.5)   # safety: linthresh < vlim
    return vlim, lin


def panel_symlog(ax, Theta, R, Z_signed, title, cbar_label,
                 vlim=None, linthresh=None, linscale=0.5,
                 norm_kind="asinh", cmap="seismic"):
    """Render a signed polar field on either SymLogNorm or AsinhNorm.

    `norm_kind`:
      "asinh"  - smooth signed-log via arcsinh.  NO linear band; values
                 below linthresh roll smoothly toward zero in colour
                 space instead of mapping to a wide near-white stripe.
                 This is the right default for σ·τ-type data with wide
                 dynamic range, where the "near-zero" region is just
                 small-but-real structure, not protected noise.  Needs
                 matplotlib ≥ 3.6.
      "symlog" - classic linear-near-zero + log-outside.  Use when you
                 explicitly want to suppress a known noise floor by
                 sliding it into the linear band.
    """
    Zp = np.ma.masked_where(~np.isfinite(Z_signed), Z_signed)
    if vlim is None or linthresh is None:
        v_auto, lin_auto = autoscale_signed(Zp.compressed())
        vlim = v_auto if vlim is None else vlim
        linthresh = lin_auto if linthresh is None else linthresh
    if norm_kind == "asinh":
        if not _HAS_ASINH:
            print("  [warn] AsinhNorm unavailable on this matplotlib; "
                  "falling back to SymLogNorm")
        else:
            norm = AsinhNorm(linear_width=linthresh, vmin=-vlim, vmax=+vlim)
    if norm_kind != "asinh" or not _HAS_ASINH:
        norm = SymLogNorm(linthresh=linthresh, linscale=linscale,
                          vmin=-vlim, vmax=+vlim, base=10)
    pcm = ax.pcolormesh(Theta, R, Zp, norm=norm, cmap=cmap, shading="auto")
    ax.set_title(title, fontsize=11)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_thetagrids(np.arange(0, 360, 45),
                      labels=[f"{a}°" for a in np.arange(0, 360, 45)])
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    cb = plt.colorbar(pcm, ax=ax, pad=0.10, shrink=0.85)
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=7)


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
    ap.add_argument("--e-max", type=float, default=100.0,
        help="Upper energy cap for the radial axis (default 100; units of "
             "the chosen --radius-mode, i.e. eV for Ekin/omega, a.u. for k).")
    ap.add_argument("--no-mirror", action="store_true",
        help="Do NOT mirror θ ∈ [0,π] to [0,2π].")
    ap.add_argument("--cmap", default="seismic")
    ap.add_argument("--vlim", type=float, default=None,
        help="Outer extent of the colourbar (Mb·as).  Default: 99th "
             "percentile of |σ·τ| in the cropped energy window.")
    ap.add_argument("--linthresh", type=float, default=None,
        help="Half-width of the central near-zero band (Mb·as).  Default: "
             "data-driven via --lo-pct.")
    ap.add_argument("--lo-pct", type=float, default=1.0,
        help="Percentile of |σ·τ| used as the linear/transition threshold "
             "when --linthresh is not given (default: 1.0).  Only the "
             "bottom 1%% of |σ·τ| values sit inside the near-zero band; "
             "the remaining 99%% gets log/asinh colour resolution.  Set "
             "to 0.1 to push even more structure into the log range, or "
             "up to 5 if FP noise around zero is leaking into colour.")
    ap.add_argument("--norm", choices=("asinh", "symlog"), default="asinh",
        help="Colour-scale mapping (default: asinh).  'asinh' is smooth "
             "signed-log via arcsinh -- NO white linear band, structure "
             "across the full dynamic range visible.  'symlog' is the "
             "classic linear-near-zero + log-outside (the previous "
             "default); use it only when you specifically want the linear "
             "band to absorb a known noise floor.")
    ap.add_argument("--linscale", type=float, default=0.5,
        help="(SymLogNorm only) fraction of the colormap dedicated to the "
             "linear band, default 0.5 = half a decade of colour budget.")
    args = ap.parse_args()

    print("=" * 72)
    print(" plot_sigma_tau_only.py")
    print("=" * 72)

    # --- load ---
    kF, EF, wF, thF, stF = load_polar_dat(args.fixed)
    kA, EA, wA, thA, stA = load_polar_dat(args.avg)
    if not (np.allclose(kF, kA) and np.allclose(thF, thA)):
        raise SystemExit("ERROR: fixed and avg dat files have different grids")

    # --- radius choice ---
    if args.radius_mode == "Ekin_eV":
        r_full = EF * HA_TO_EV; r_label = r"$E_{\rm kin}$ [eV]"
    elif args.radius_mode == "omega_eV":
        r_full = wF * HA_TO_EV; r_label = r"$\hbar\omega$ [eV]"
    else:
        r_full = kF; r_label = r"$k$ [a.u.]"

    # --- crop to e-max ---
    mask = r_full <= args.e_max
    if not mask.any():
        raise SystemExit(f"ERROR: no energies ≤ {args.e_max} in input.  "
                         f"r range = [{r_full.min():.3f}, {r_full.max():.3f}]")
    r   = r_full[mask]
    stF_mbas = (stF[mask, :]) * A02_TO_MB
    stA_mbas = (stA[mask, :]) * A02_TO_MB
    print(f"  energies in window: {r.size} / {r_full.size}  "
          f"(r ∈ [{r.min():.3f}, {r.max():.3f}])")

    # --- mirror θ ---
    if not args.no_mirror:
        theta, stF_mbas, stA_mbas = mirror_theta(thF, stF_mbas, stA_mbas)
    else:
        theta = thF

    Theta, R = np.meshgrid(theta, r)

    # --- Euler angles from header (for title) ---
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

    # Match the colour scaling between the two panels so they're directly
    # comparable: use the JOINT distribution to pick vlim, linthresh.
    if args.vlim is None or args.linthresh is None:
        joint = np.concatenate([stF_mbas[np.isfinite(stF_mbas)],
                                stA_mbas[np.isfinite(stA_mbas)]])
        v_auto, lin_auto = autoscale_signed(joint, lo_pct=args.lo_pct)
        vlim      = v_auto   if args.vlim      is None else args.vlim
        linthresh = lin_auto if args.linthresh is None else args.linthresh
        # Report what bracket of |σ·τ| each tenth-percentile decile sits
        # in, so the user can see *why* the linear band is where it is.
        ax_joint = np.abs(joint[np.isfinite(joint) & (joint != 0)])
        if ax_joint.size > 0:
            ptiles = [1, 5, 25, 50, 75, 95, 99]
            vals = np.percentile(ax_joint, ptiles)
            print("  |σ·τ| distribution: " + "  ".join(
                f"p{p}={v:.2e}" for p, v in zip(ptiles, vals)))
    else:
        vlim = args.vlim; linthresh = args.linthresh
    print(f"  norm={args.norm}  vlim=±{vlim:.3e}  linthresh={linthresh:.3e} Mb·as"
          f"  (lo_pct={args.lo_pct:g}, linscale={args.linscale:g})")
    if linthresh > 0:
        print(f"  log decades visible on each side of zero: "
              f"{np.log10(vlim / linthresh):.2f}")

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4),
                             subplot_kw={"projection": "polar"})

    panel_symlog(axes[0], Theta, R, stF_mbas,
        title=fr"$\sigma\,\tau(E, \theta)$  fixed  $(\alpha,\beta)=({alpha_deg:g}°,{beta_deg:g}°)$,  $\phi=0$",
        cbar_label="σ·τ [Mb·as]",
        vlim=vlim, linthresh=linthresh, linscale=args.linscale,
        norm_kind=args.norm, cmap=args.cmap)
    panel_symlog(axes[1], Theta, R, stA_mbas,
        title=r"$\langle\sigma\,\tau(E, \theta)\rangle_{\rm pol}$  isotropic avg [Eq. S38]",
        cbar_label="σ·τ [Mb·as]",
        vlim=vlim, linthresh=linthresh, linscale=args.linscale,
        norm_kind=args.norm, cmap=args.cmap)

    for ax in axes:
        ax.set_rlabel_position(135)
        ax.set_rmin(r.min()); ax.set_rmax(r.max())

    fig.suptitle(
        f"σ·τ angular pattern  ({r_label} ≤ {args.e_max:g})\n"
        f"left: fixed Euler $(\\alpha,\\beta)=({alpha_deg:g}°,{beta_deg:g}°)$,   "
        f"right: isotropic polarisation average",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
