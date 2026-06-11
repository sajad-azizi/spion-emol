#!/usr/bin/env python3
r"""
gauge_panels.py
===============
Per-gauge summary figures for the C₈F₈ photoionization observables, built
from the gathered per-channel dipole .dat files (gather_dipoles.py).

It writes **two separate figures** — one for the LENGTH gauge, one for the
VELOCITY gauge — each laid out as a 2-column grid:

    row 1 :  left  cross section  σ(E)        |  right  time delay  τ_W(E)
    row 2 :  left  asymmetry      β(E)        |  right  RTD residual η(E)
   [row 3]:  left  two-photon amplitude       |  right  two-photon delay
             (optional; only if a two_photon_delay.dat is supplied)

Sign convention (IMPORTANT — "no minus sign on the time delay")
---------------------------------------------------------------
The Wigner delay is the energy derivative of the matrix-element phase,

    τ_W(E) = ∂_E arg D = + Im[ Σ d*_{lm} ∂_E d_{lm} ] / Σ |d_{lm}|² ,

with **no leading minus sign**.  This is the definition in
postprocessing/docs/density_current.pdf (τ_W = ∂_E Φ) and the one used by
postprocessing/polar_plots/rtd_content.py for 𝒥 = σ τ_W and η.  For the
C₈F₈ production dipoles it yields a POSITIVE delay at the trapping
resonance (≈ +170 as near k ≈ 1.3 a.u.), the physically correct sign.

(Note: cross_section_delay.py's time_delay() carries an explicit minus —
that matched an older fixture's arg D = −δ_ℓ phase convention.  We do NOT
reuse it here; everything in this figure uses the no-minus τ_W so τ, 𝒥,
and η share one convention.)

Reused, already-tested machinery
--------------------------------
  * σ(E)   : cross_section_delay.sigma_from_dipole
  * β(E)   : cross_section_delay.compute_beta  (full SO(3) orientation
             average; bounded β ∈ [−1, 2])
  * η(E)   : rtd_content.{build_F_grid, sigma_J_fixed, sigma_J_pol_average,
             rtd_content}  (full-sphere solid-angle inner products)
  * loaders/constants from cross_section_delay.

η ensemble: by default the FIXED orientation (α=β=0, ẑ polarisation) to
match the molecular-frame focus; pass --eta-mode pol for the isotropic
polarisation average, or --eta-mode both.

Usage
-----
    python gauge_panels.py <gathered_dir> [--output-dir DIR]
        [--xaxis k|E|omega] [--idx-start N]
        [--eta-mode fixed|pol|both] [--n-theta NT] [--n-phi NP]
        [--with-two-photon] [--two-photon-dat PATH]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- import the tested computational pieces from the sibling modules ------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                               # cross_section_delay
sys.path.insert(0, os.path.join(_HERE, "polar_plots"))  # rtd_content, compute_polar_data

from cross_section_delay import (  # noqa: E402
    AU_TO_AS,
    BOHR2_TO_MB,
    DIPOLE_PREFIX_LEN,
    DIPOLE_PREFIX_VEL,
    HA_TO_EV,
    compute_beta,
    get_xvals,
    infer_common_mu,
    load_dipole,
    load_two_photon_delay,
    resolve_input_dir,
    sigma_from_dipole,
)
from rtd_content import (  # noqa: E402
    DipoleDataset,
    build_F_grid,
    sigma_J_fixed,
    sigma_J_pol_average,
)
from rtd_content import rtd_content as compute_rtd_scalars  # noqa: E402


# ---------------------------------------------------------------------------
# Wigner delay — NO minus sign (τ_W = ∂_E arg D, the density_current.pdf /
# rtd_content convention).  See the module docstring.
# ---------------------------------------------------------------------------
def time_delay(k: np.ndarray, D: np.ndarray,
               with_minus: bool = False) -> np.ndarray:
    """τ_W(E) = ∂_E arg D = + Im[ Σ d* ∂_E d ] / Σ |d|²   [a.u. of time].

    Default (with_minus=False) carries NO minus sign — the
    density_current.pdf / rtd_content.py convention (positive delay at
    the trapping resonance for the C8F8 production dipoles).  Pass
    with_minus=True for the legacy −Im sign (kept in lock-step with
    cross_section_delay.py's --with-minus-sign).

    ∂_E = (1/k) ∂_k since E = k²/2.  Returns 0 where the strength vanishes.
    """
    dD_dE = np.gradient(D, k, axis=0) / k[:, None]
    num = np.sum(np.conj(D) * dD_dE, axis=1).imag
    den = np.sum(np.abs(D) ** 2, axis=1)
    sign = -1.0 if with_minus else 1.0
    return sign * np.divide(num, den, out=np.zeros_like(num), where=den > 1e-30)


# ---------------------------------------------------------------------------
# η(E) per gauge, reusing the full-sphere machinery of rtd_content.
# ---------------------------------------------------------------------------
def compute_eta_per_gauge(k, E_kin, omega, D_dict, mu_list,
                          theta, phi, alpha_rad, beta_rad, mode):
    """Return {tag: (eta, normJ)} for tag in the requested ensemble(s)."""
    def _ds(d):
        return DipoleDataset(k, E_kin, omega, mu_list, D_dict[d], "", d)

    Fx, dFx = build_F_grid(_ds("x"), theta, phi)
    Fy, dFy = build_F_grid(_ds("y"), theta, phi)
    Fz, dFz = build_F_grid(_ds("z"), theta, phi)

    out: Dict[str, tuple] = {}
    if mode in ("fixed", "both"):
        sig, J = sigma_J_fixed(Fx, dFx, Fy, dFy, Fz, dFz, alpha_rad, beta_rad)
        eta, _taubar, _lam, normJ = compute_rtd_scalars(sig, J, theta, phi)
        out["fixed"] = (eta, normJ)
    if mode in ("pol", "both"):
        sig, J = sigma_J_pol_average(Fx, dFx, Fy, dFy, Fz, dFz)
        eta, _taubar, _lam, normJ = compute_rtd_scalars(sig, J, theta, phi)
        out["pol"] = (eta, normJ)
    return out


# ---------------------------------------------------------------------------
# Plot a single gauge figure.
# ---------------------------------------------------------------------------
_ETA_STYLE = {
    "fixed": ("#1f77b4", "-", r"$\eta$ fixed $(\alpha,\beta){=}0$"),
    "pol":   ("#2ca02c", "--", r"$\eta$ pol-avg"),
}


def make_gauge_figure(gauge_label, xvals, xlabel,
                      sigma, tau, beta, eta_results,
                      idx_start, out_path,
                      two_photon=None, tp_xaxis="k", mask_frac=0.02,
                      with_minus=False):
    sl = slice(idx_start, None)
    has_tp = two_photon is not None
    n_rows = 2 + (1 if has_tp else 0)

    fig = plt.figure(figsize=(13, 4.6 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.34, wspace=0.22)
    comp_colors = {"x": "C0", "y": "C1", "z": "C2", "avg": "k"}

    # ---- row 0 left: cross section ----
    a = fig.add_subplot(gs[0, 0])
    for d in ("x", "y", "z", "avg"):
        a.plot(xvals[sl], sigma[d][sl] * BOHR2_TO_MB, color=comp_colors[d],
               lw=2.0 if d == "avg" else 1.0,
               label=(r"$\bar\sigma$" if d == "avg" else rf"$\sigma_{d}$"))
    a.set_xlabel(xlabel)
    a.set_ylabel(r"$\sigma$ (Mb)")
    a.set_yscale("log")
    a.grid(True, alpha=0.3)
    a.legend(fontsize=8, ncol=2)
    a.set_title(rf"Cross section $\sigma(E)$ — {gauge_label}")

    # ---- row 0 right: time delay (NO minus sign) ----
    a = fig.add_subplot(gs[0, 1])
    for d in ("x", "y", "z", "avg"):
        a.plot(xvals[sl], tau[d][sl] * AU_TO_AS, color=comp_colors[d],
               lw=2.0 if d == "avg" else 1.0,
               label=(r"$\bar\tau$" if d == "avg" else rf"$\tau_{d}$"))
    a.axhline(0.0, color="0.6", lw=0.7, ls=":")
    a.set_xlabel(xlabel)
    a.set_ylabel(r"$\tau_W$ (as)")
    a.grid(True, alpha=0.3)
    a.legend(fontsize=8, ncol=2)
    tau_rhs = r"-\partial_E\arg D" if with_minus else r"\partial_E\arg D"
    a.set_title(rf"Wigner delay $\tau_W={tau_rhs}$ — {gauge_label}")

    # ---- row 1 left: β ----
    a = fig.add_subplot(gs[1, 0])
    a.plot(xvals[sl], beta[sl], "-", color="#9467bd", lw=2.0, label=r"$\beta$")
    a.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.7)
    a.axhline(2.0, color="gray", ls=":", lw=0.5, alpha=0.4)
    a.axhline(-1.0, color="gray", ls=":", lw=0.5, alpha=0.4)
    a.set_xlabel(xlabel)
    a.set_ylabel(r"$\beta(E)$")
    a.grid(True, alpha=0.3)
    a.legend(fontsize=9)
    a.set_title(rf"Asymmetry $\beta(E)$ — {gauge_label}"
                r"  (orientation-avg; $\beta\in[-1,2]$)")

    # ---- row 1 right: η (faint full + bold reliable) ----
    a = fig.add_subplot(gs[1, 1])
    for tag, (eta, normJ) in eta_results.items():
        color, lsty, lab = _ETA_STYLE[tag]
        peak = np.max(normJ) if np.max(normJ) > 0 else 1.0
        mask = normJ < mask_frac * peak
        a.plot(xvals[sl], eta[sl], lsty, color=color, lw=1.0, alpha=0.25)
        a.plot(xvals[sl], np.where(mask, np.nan, eta)[sl], lsty,
               color=color, lw=2.2, label=lab)
    a.set_ylim(0.0, 1.0)
    a.set_xlabel(xlabel)
    a.set_ylabel(r"orthogonal residual  $\eta(E)$")
    a.grid(True, alpha=0.3)
    a.legend(fontsize=9)
    a.set_title(rf"RTD content $\eta(E)$ — {gauge_label}"
                r"  ($\eta{=}0$: delay $\propto$ DCS)")

    # ---- row 2 (optional): two-photon amplitude | delay ----
    if has_tp:
        x_tp = two_photon["E_eV"]
        xl_tp = xlabel
        if tp_xaxis == "k":
            x_tp = np.sqrt(2.0 * two_photon["E_au"])
            xl_tp = r"$k=\sqrt{2E}$  (a.u.)"
        elif tp_xaxis == "E":
            xl_tp = r"Kinetic energy $E$ (eV)"

        # left: two-photon amplitude
        a = fig.add_subplot(gs[2, 0])
        a.plot(x_tp, two_photon["M_amp"], "s-", color="C4", lw=1.8, ms=6,
               label=r"$|\langle M_<^*\,M_>\rangle|$")
        a.set_xlabel(xl_tp)
        a.set_ylabel(r"$|\langle M_<^*\,M_>\rangle|$ (a.u.)")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=9)
        a.set_title("Two-photon amplitude (resonance check)")

        # right: two-photon delay (plotted as stored; no sign flip)
        a = fig.add_subplot(gs[2, 1])
        a.plot(x_tp, two_photon["tau_as"], "o-", color="C3", lw=1.8, ms=6,
               label=r"$\tau_{2\hbar\omega}$")
        a.axhline(0.0, color="0.6", lw=0.7, ls=":")
        a.set_xlabel(xl_tp)
        a.set_ylabel(r"$\tau_{2\hbar\omega}$ (as)")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=9)
        a.set_title("Effective two-photon delay")

    fig.suptitle(f"C$_8$F$_8$ photoionization observables — {gauge_label} gauge",
                 y=1.005, fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def _sigma_weighted_avg_tau(sigma, tau):
    num = sum(sigma[d] * tau[d] for d in ("x", "y", "z"))
    den = sum(sigma[d] for d in ("x", "y", "z"))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-30)


def process_gauge(gauge_label, prefix, input_dir, mu_list,
                  theta, phi, alpha_rad, beta_rad, eta_mode,
                  with_minus=False):
    """Load one gauge's dipoles and return everything needed to plot."""
    D: Dict[str, np.ndarray] = {}
    k = E_kin = omega = None
    for d in ("x", "y", "z"):
        k2, e2, o2, D[d] = load_dipole(input_dir, prefix, d, mu_list)
        if k is None:
            k, E_kin, omega = k2, e2, o2
        elif not (np.allclose(k, k2) and np.allclose(E_kin, e2)
                  and np.allclose(omega, o2)):
            raise ValueError(f"grid mismatch: {gauge_label}/{d}")

    g = "length" if prefix == DIPOLE_PREFIX_LEN else "velocity"
    sigma = {d: sigma_from_dipole(k, D[d], omega, g) for d in ("x", "y", "z")}
    sigma["avg"] = (sigma["x"] + sigma["y"] + sigma["z"]) / 3.0

    tau = {d: time_delay(k, D[d], with_minus=with_minus) for d in ("x", "y", "z")}
    tau["avg"] = _sigma_weighted_avg_tau(sigma, tau)

    print(f"    β(E) for {gauge_label} ...")
    beta = compute_beta(D["x"], D["y"], D["z"], mu_list)

    print(f"    η(E) for {gauge_label} (mode={eta_mode}) ...")
    eta_results = compute_eta_per_gauge(
        k, E_kin, omega, D, mu_list, theta, phi, alpha_rad, beta_rad, eta_mode)

    return k, E_kin, omega, sigma, tau, beta, eta_results


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-gauge σ | τ | β | η (| two-photon) panel figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="gathered_* directory (gather_dipoles.py).")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write the figures. Default: input_dir.")
    ap.add_argument("--xaxis", choices=["k", "E", "omega"], default="k")
    ap.add_argument("--idx-start", type=int, default=2,
                    help="Drop first N energy points (gradient edge artefacts).")
    ap.add_argument("--eta-mode", choices=["fixed", "pol", "both"],
                    default="fixed",
                    help="η ensemble: fixed (α=β=0), isotropic pol-avg, or both.")
    ap.add_argument("--alpha-deg", type=float, default=0.0)
    ap.add_argument("--beta-deg", type=float, default=0.0)
    ap.add_argument("--n-theta", type=int, default=121,
                    help="θ samples for the η solid-angle integral.")
    ap.add_argument("--n-phi", type=int, default=96,
                    help="φ samples for the η solid-angle integral.")
    ap.add_argument("--mask-frac", type=float, default=0.02,
                    help="η drawn bold where ‖𝒥‖ ≥ frac·max‖𝒥‖, faint below.")
    ap.add_argument("--with-minus-sign", action="store_true",
                    help="Apply a leading minus to the time delay: "
                         "tau = -Im[sum d* dd/dE]/sum|d|^2 (legacy convention, "
                         "matches cross_section_delay.py --with-minus-sign). "
                         "Default is the no-minus tau = +d arg(D)/dE.")
    ap.add_argument("--with-two-photon", action="store_true",
                    help="Add the two-photon row from two_photon_delay.dat.")
    ap.add_argument("--two-photon-dat", type=Path, default=None,
                    help="Path to two_photon_delay.dat (default <out>/two_photon_delay.dat).")
    args = ap.parse_args()

    input_dir = resolve_input_dir(args.input_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("gauge_panels.py")
    print("=" * 72)
    print(f"  input_dir : {input_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  xaxis     : {args.xaxis}   eta-mode: {args.eta_mode}")

    mu_list = infer_common_mu(
        input_dir, (DIPOLE_PREFIX_LEN, DIPOLE_PREFIX_VEL), ("x", "y", "z"))
    print(f"  channels  : {len(mu_list)}   mu range [{mu_list[0]}, {mu_list[-1]}]")

    theta = np.linspace(0.0, np.pi, args.n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, args.n_phi)
    a_rad, b_rad = np.deg2rad(args.alpha_deg), np.deg2rad(args.beta_deg)

    # two-photon (optional, shared by both gauge figures)
    two_photon = None
    if args.with_two_photon:
        tp_path = (args.two_photon_dat if args.two_photon_dat is not None
                   else output_dir / "two_photon_delay.dat")
        two_photon = load_two_photon_delay(tp_path)
        if two_photon is None:
            print(f"  WARN: --with-two-photon set but {tp_path} not found; "
                  f"skipping the two-photon row.")
        else:
            print(f"  two-photon: {len(two_photon['ik'])} sidebands from {tp_path}")

    for gauge_label, prefix, tag in (("Length", DIPOLE_PREFIX_LEN, "len"),
                                     ("Velocity", DIPOLE_PREFIX_VEL, "vel")):
        print(f"  [{gauge_label}]")
        k, E_kin, omega, sigma, tau, beta, eta_results = process_gauge(
            gauge_label, prefix, input_dir, mu_list,
            theta, phi, a_rad, b_rad, args.eta_mode,
            with_minus=args.with_minus_sign)
        if len(k) < 3:
            raise SystemExit("need ≥3 energy points for ∂_E.")

        xvals, xlabel = get_xvals(k, E_kin, omega, args.xaxis)
        out_path = output_dir / f"gauge_panels_{tag}.png"
        make_gauge_figure(
            gauge_label, xvals, xlabel, sigma, tau, beta, eta_results,
            args.idx_start, out_path,
            two_photon=two_photon, tp_xaxis=args.xaxis,
            mask_frac=args.mask_frac, with_minus=args.with_minus_sign)

        sl = slice(args.idx_start, None)
        print(f"    σ̄ max = {np.max(sigma['avg'][sl]) * BOHR2_TO_MB:.3e} Mb;  "
              f"τ̄ ∈ [{(tau['avg'][sl]*AU_TO_AS).min():.1f}, "
              f"{(tau['avg'][sl]*AU_TO_AS).max():.1f}] as;  "
              f"β ∈ [{beta[sl].min():.2f}, {beta[sl].max():.2f}]")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
