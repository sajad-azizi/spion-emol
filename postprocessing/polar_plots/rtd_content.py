#!/usr/bin/env python3
r"""
rtd_content.py
==============
Quantify "what does the angle-resolved time delay (RTD) add beyond the
differential cross section (DCS)?" as a curve in energy, following
``postprocessing/docs/density_current.pdf``.

Background (atomic units; see the PDF)
--------------------------------------
For the one-photon dipole amplitude D(E, k̂) = |D| e^{iΦ} to the
energy-normalised incoming-wave continuum state, the PDF works with the
*undivided* bilinear

    σ(E, k̂)  ≡ |D|²                            (dipole intensity ∝ DCS)
    𝒥(E, k̂)  ≡ Im[D* ∂_E D] = σ τ_W            (spectral phase-current density)

where τ_W = ∂_E Φ is the Wigner delay.  𝒥 — not the normalised delay
τ_W = 𝒥/σ — is the natural RTD-content variable: it is bilinear in D, so
it is finite at amplitude zeros (Cooper minima / multi-centre nodes where
τ_W diverges) and it is *linear* under incoherent (polarisation /
orientation) averaging, while τ_W is not.

The PI's "subtract DCS and RTD" question is made precise as: at fixed E,
is the *angular pattern* of 𝒥 just proportional to that of σ?  With the
solid-angle inner product ⟨f, g⟩_E = ∫ dΩ f(E, k̂) g(E, k̂), define

    λ(E) = ⟨𝒥, σ⟩_E / ⟨σ, σ⟩_E                          (PDF Eq. 5)
    η(E) = ‖ 𝒥 − λ(E) σ ‖_E / ‖ 𝒥 ‖_E                    (PDF Eq. 5)

η(E) is the fraction of the angular structure of 𝒥 orthogonal to that of
σ — i.e. the part of the delay the cross section cannot predict.  η = 0
iff τ_W(E, k̂) is angle-independent (RTD carries no angular info beyond
DCS); η → 1 iff the two angular patterns are uncorrelated.

This script also reports the companion, physically intuitive

    τ̄(E) = ∫ 𝒥 dΩ / ∫ σ dΩ      (yield / σ-weighted mean delay, in as)

Note λ(E) is a σ²-weighted mean delay (the right reference for the
orthogonality decomposition) whereas τ̄(E) is the σ-weighted mean delay
(the intuitive "mean delay").  They are different objects; we plot τ̄ as
the interpretable curve and keep λ implicit inside η.

Two angular ensembles are produced (cf. the two rows of PDF Fig. 1):
  * FIXED Euler orientation (α, β) of the linear polarisation
        D = c_x D_x + c_y D_y + c_z D_z,
        c = (cos α sin β, sin α sin β, cos β),    default α=β=0 ⇒ ẑ.
  * ISOTROPIC POLARISATION AVERAGE, using the linearity of σ and 𝒥:
        σ_pol = (1/3) Σ_q |D_q|²,
        𝒥_pol = (1/3) Σ_q Im[D_q* ∂_E D_q].
    (𝒥 averages linearly — PDF property 2 — so this is the correct
     averaged observable; τ_W would not average this way.)

All scalar outputs (λ, η, τ̄) are ratios in which the smooth
k̂-independent kinematic prefactor C(E)=4π²ω/c and the 1/k² of D=F/k
cancel exactly, so they are computed from F = k·D directly.

Inputs / conventions
--------------------
Reuses the *proven* loader and numerically-stable real-Y_{ℓm} recurrence
from ``compute_polar_data.py`` (same directory).  Gathered channel files
``dipole_{gauge}_homo_{x,y,z}_{mu}.dat`` with μ = ℓ²+ℓ+m, columns
``k E_kin omega Re(d) Im(d) ...``.  Unlike compute_polar_data.py (which
slices φ=0 for the polar plots), the inner products here require the full
sphere, so D is built on a 2-D (θ, φ) grid via the separable real harmonic
Y^R_{ℓm}(θ,φ) = Θ_{ℓm}(θ)·Φ_m(φ),  Φ_0=1, Φ_{m>0}=cos(mφ),
Φ_{m<0}=sin(|m|φ).

Outputs
-------
  rtd_content_{gauge}.dat   columns:
      E_kin[au]  E_kin[eV]  eta_fixed  taubar_fixed[as]  lambda_fixed[as]
      normJ_fixed  eta_pol  taubar_pol[as]  lambda_pol[as]  normJ_pol
  rtd_content_{gauge}.png   twin-axis figure: η(E) (left) and τ̄(E) (right).

Run ``--selftest`` to verify the full-sphere harmonics (orthonormality +
φ=0 consistency with compute_polar_data) before trusting any numbers.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Reuse the tested loader, channel index helper, stable real-Y recurrence,
# and physical constants from the polar-data engine in this same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_polar_data import (  # noqa: E402
    AU_TO_AS,
    C_AU,
    HA_TO_EV,
    DipoleDataset,
    idx_to_lm,
    load_dipole_dataset,
    real_Ylm_phi0,
)

try:
    from scipy.integrate import simpson as _simpson  # SciPy >= 1.6
except Exception:  # pragma: no cover - very old scipy
    from scipy.integrate import simps as _simpson


# ---------------------------------------------------------------------------
# Full-sphere real spherical harmonic, separable form.
#
# real_Ylm_phi0(ℓ, |m|, θ) already returns the *θ-part* with the correct
# √2·(-1)^{|m|} normalisation folded in (it is the value of Y^R_{ℓ,|m|} at
# φ=0, where cos(|m|·0)=1).  We multiply by the azimuthal factor Φ_m(φ).
# ---------------------------------------------------------------------------
def real_Ylm_grid(ell: int, m: int, theta: np.ndarray, phi: np.ndarray
                  ) -> np.ndarray:
    """Y^R_{ℓm}(θ, φ) on the outer grid, shape (nθ, nφ)."""
    theta_part = real_Ylm_phi0(ell, abs(m), theta)      # (nθ,)
    if m == 0:
        azi = np.ones_like(phi)
    elif m > 0:
        azi = np.cos(m * phi)
    else:
        azi = np.sin(abs(m) * phi)
    return theta_part[:, None] * azi[None, :]


# ---------------------------------------------------------------------------
# Build F_q(E, θ, φ) and ∂_E F_q for one Cartesian polarisation q.
#
#   F_q(E,θ,φ)     = 4π Σ_{ℓm} i^{-ℓ} d^q_{ℓm}(E) Y^R_{ℓm}(θ,φ)
#   ∂_E F_q(E,θ,φ) = 4π Σ_{ℓm} i^{-ℓ} ∂_E d^q_{ℓm}(E) Y^R_{ℓm}(θ,φ)
#
# D_q = F_q / k; the 1/k and the kinematic prefactor cancel in every ratio
# below, so we carry F (= k·D) directly.  We use the azimuthal
# factorisation A_m(E,θ) = Σ_ℓ i^{-ℓ} d_{ℓm} Θ_{ℓm}(θ) so the (expensive)
# channel sum is done once per m and only then expanded over φ.
# ---------------------------------------------------------------------------
def build_F_grid(ds: DipoleDataset, theta: np.ndarray, phi: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (F, dF) of shape (nE, nθ, nφ), complex."""
    k = ds.k
    D = ds.dipole                       # (nE, nCh) reduced d_{ℓm}(k)
    nE, nT, nP = k.size, theta.size, phi.size

    ells = np.array([idx_to_lm(mu)[0] for mu in ds.mu_list], dtype=int)
    ms = np.array([idx_to_lm(mu)[1] for mu in ds.mu_list], dtype=int)
    phase = (-1j) ** ells               # i^{-ℓ}

    # ∂_E d = (∂_k d)/k   (E = k²/2 ⇒ ∂_E = (1/k) ∂_k)
    dD_dE = np.gradient(D, k, axis=0) / k[:, None]

    F = np.zeros((nE, nT, nP), dtype=np.complex128)
    dF = np.zeros((nE, nT, nP), dtype=np.complex128)

    for m in sorted(set(int(v) for v in ms)):
        cols = np.where(ms == m)[0]
        if cols.size == 0:
            continue
        # azimuthal factor Φ_m(φ)
        if m == 0:
            azi = np.ones_like(phi)
        elif m > 0:
            azi = np.cos(m * phi)
        else:
            azi = np.sin(abs(m) * phi)

        A_m = np.zeros((nE, nT), dtype=np.complex128)
        dA_m = np.zeros((nE, nT), dtype=np.complex128)
        for col in cols:
            theta_part = real_Ylm_phi0(ells[col], abs(m), theta)   # (nθ,)
            ph = phase[col]
            A_m += (ph * D[:, col])[:, None] * theta_part[None, :]
            dA_m += (ph * dD_dE[:, col])[:, None] * theta_part[None, :]

        F += A_m[:, :, None] * azi[None, None, :]
        dF += dA_m[:, :, None] * azi[None, None, :]

    pre = 4.0 * np.pi
    return pre * F, pre * dF


# ---------------------------------------------------------------------------
# σ and 𝒥 on the (θ, φ) grid (without the cancelling C/k² prefactor).
# ---------------------------------------------------------------------------
def sigma_J_fixed(Fx, dFx, Fy, dFy, Fz, dFz, alpha_rad, beta_rad):
    cx = np.cos(alpha_rad) * np.sin(beta_rad)
    cy = np.sin(alpha_rad) * np.sin(beta_rad)
    cz = np.cos(beta_rad)
    F = cx * Fx + cy * Fy + cz * Fz
    dF = cx * dFx + cy * dFy + cz * dFz
    sigma = np.abs(F) ** 2
    J = np.imag(np.conj(F) * dF)
    return sigma, J


def sigma_J_pol_average(Fx, dFx, Fy, dFy, Fz, dFz):
    sigma = (np.abs(Fx) ** 2 + np.abs(Fy) ** 2 + np.abs(Fz) ** 2) / 3.0
    J = (np.imag(np.conj(Fx) * dFx)
         + np.imag(np.conj(Fy) * dFy)
         + np.imag(np.conj(Fz) * dFz)) / 3.0
    return sigma, J


# ---------------------------------------------------------------------------
# Solid-angle inner products and the RTD-content scalars per energy.
# ---------------------------------------------------------------------------
def _omega_integral(field: np.ndarray, theta: np.ndarray, phi: np.ndarray
                    ) -> np.ndarray:
    """∫ field(E,θ,φ) sinθ dθ dφ  →  (nE,).  Simpson in φ then θ."""
    over_phi = _simpson(field, x=phi, axis=2)               # (nE, nθ)
    integrand = over_phi * np.sin(theta)[None, :]
    return _simpson(integrand, x=theta, axis=1)             # (nE,)


def rtd_content(sigma: np.ndarray, J: np.ndarray,
                theta: np.ndarray, phi: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (eta, taubar_as, lambda_as, normJ) per energy.

        λ(E)   = ⟨𝒥,σ⟩/⟨σ,σ⟩
        η(E)   = ‖𝒥 − λσ‖/‖𝒥‖,  ‖𝒥−λσ‖² = ⟨𝒥,𝒥⟩ − ⟨𝒥,σ⟩²/⟨σ,σ⟩
        τ̄(E)   = ∫𝒥 dΩ / ∫σ dΩ
        normJ  = ‖𝒥‖ = √⟨𝒥,𝒥⟩   (for masking noisy bins)
    """
    I_ss = _omega_integral(sigma * sigma, theta, phi)   # ⟨σ,σ⟩
    I_sJ = _omega_integral(sigma * J, theta, phi)       # ⟨σ,𝒥⟩
    I_JJ = _omega_integral(J * J, theta, phi)           # ⟨𝒥,𝒥⟩
    S_int = _omega_integral(sigma, theta, phi)          # ∫σ dΩ
    J_int = _omega_integral(J, theta, phi)              # ∫𝒥 dΩ

    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(I_ss > 0.0, I_sJ / I_ss, 0.0)
        # residual² = ⟨𝒥,𝒥⟩ − ⟨σ,𝒥⟩²/⟨σ,σ⟩  (≥0 by Cauchy–Schwarz)
        resid_sq = I_JJ - np.where(I_ss > 0.0, I_sJ ** 2 / I_ss, 0.0)
        resid_sq = np.maximum(resid_sq, 0.0)
        normJ = np.sqrt(np.maximum(I_JJ, 0.0))
        eta = np.where(normJ > 0.0, np.sqrt(resid_sq) / normJ, 0.0)
        taubar = np.where(np.abs(S_int) > 0.0, J_int / S_int, 0.0)

    return eta, taubar * AU_TO_AS, lam * AU_TO_AS, normJ


# ---------------------------------------------------------------------------
# Self-test of the full-sphere harmonics (run before trusting physics).
# ---------------------------------------------------------------------------
def selftest() -> int:
    print("=== rtd_content self-test ===")
    ok = True

    # (1) φ=0 column must reproduce compute_polar_data.real_Ylm_phi0 exactly.
    theta = np.linspace(0.0, np.pi, 37)
    phi = np.array([0.0, 0.3, 1.1, 2.0, 4.5, 6.0])
    for (ell, m) in [(0, 0), (1, 0), (1, 1), (2, -1), (3, 2), (5, -4), (8, 7)]:
        Yg = real_Ylm_grid(ell, m, theta, phi)
        ref0 = real_Ylm_phi0(ell, m, theta)             # full Y at φ=0
        if m < 0:
            ref0 = np.zeros_like(theta)                 # sin(|m|·0)=0
        err = np.max(np.abs(Yg[:, 0] - ref0))
        flag = "ok " if err < 1e-12 else "FAIL"
        ok &= err < 1e-12
        print(f"  [{flag}] φ=0 slice (ℓ={ell}, m={m:+d}) err={err:.2e}")

    # (2) Orthonormality ∫ Y_a Y_b dΩ = δ_ab on a fine grid (low ℓ).
    nt, nph = 200, 240
    th = np.linspace(0.0, np.pi, nt)
    ph = np.linspace(0.0, 2.0 * np.pi, nph)
    pairs = [(0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, 0), (2, 1), (3, -1)]
    Ys = {p: real_Ylm_grid(p[0], p[1], th, ph) for p in pairs}
    worst_diag = 0.0
    worst_off = 0.0
    for a in pairs:
        for b in pairs:
            integ = _omega_integral((Ys[a] * Ys[b])[None, :, :], th, ph)[0]
            if a == b:
                worst_diag = max(worst_diag, abs(integ - 1.0))
            else:
                worst_off = max(worst_off, abs(integ))
    print(f"  [{'ok ' if worst_diag < 2e-3 else 'FAIL'}] "
          f"max |⟨Y,Y⟩-1|  = {worst_diag:.2e}")
    print(f"  [{'ok ' if worst_off < 2e-3 else 'FAIL'}] "
          f"max |⟨Y,Y'⟩|    = {worst_off:.2e}  (off-diagonal)")
    ok &= worst_diag < 2e-3 and worst_off < 2e-3

    # (3) Sanity of the RTD scalars on a synthetic field.
    #     σ = |F|², 𝒥 = Im[F* dF].  Choose F real ⇒ 𝒥≡0 ⇒ η undefined→0,
    #     τ̄=0; then F = (1+iE)·g(θ,φ) (global E-phase) ⇒ τ_W angle-indep ⇒
    #     η must be ~0 though 𝒥≠0.
    nE = 9
    g = (real_Ylm_grid(1, 0, th, ph) + 0.4 * real_Ylm_grid(2, 1, th, ph))
    Eax = np.linspace(0.5, 1.3, nE)
    F = (1.0 + 1j * Eax)[:, None, None] * g[None, :, :]
    dF = (1j * np.ones(nE))[:, None, None] * g[None, :, :]
    sig = np.abs(F) ** 2
    J = np.imag(np.conj(F) * dF)
    eta, taubar, lam, nJ = rtd_content(sig, J, th, ph)
    print(f"  [{'ok ' if np.max(eta) < 1e-6 else 'FAIL'}] "
          f"global-phase field ⇒ η≈0 : max η={np.max(eta):.2e}")
    ok &= np.max(eta) < 1e-6

    print("=== self-test", "PASSED ===" if ok else "FAILED ===")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Plot.
# ---------------------------------------------------------------------------
def make_figure(k, eta_fix, tau_fix, eta_pol, tau_pol,
                mask_fix, mask_pol, out_png, gauge, fixed_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_eta, ax_tau) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), sharex=True)

    cF = "#1f77b4"   # fixed-orientation colour
    cP = "#2ca02c"   # pol-average colour
    xlabel = r"$k=\sqrt{2E}$  [a.u.]"

    # --- left panel: η(k) ---
    # The full curve is drawn faintly so nothing is hidden; the reliable
    # part (where the delay weight ‖𝒥‖ is well above the finite-difference
    # noise floor) is overdrawn in bold.  η is a scale-free ratio defined at
    # every energy — the faint segments are simply where a small ‖𝒥‖ makes
    # the ∂_E estimate noisier, so treat them with more caution.
    ax_eta.plot(k, eta_fix, "-", color=cF, lw=1.0, alpha=0.25, zorder=2)
    ax_eta.plot(k, np.where(mask_fix, np.nan, eta_fix), "-",
                color=cF, lw=2.2, zorder=3,
                label=r"fixed $(\alpha,\beta)$")
    if not fixed_only:
        ax_eta.plot(k, eta_pol, "--", color=cP, lw=1.0, alpha=0.25,
                    zorder=2)
        ax_eta.plot(k, np.where(mask_pol, np.nan, eta_pol), "--",
                    color=cP, lw=2.2, zorder=3, label="pol-avg")
    ax_eta.set_ylim(0.0, 1.0)
    ax_eta.set_xlabel(xlabel)
    ax_eta.set_ylabel(r"orthogonal residual  $\eta(k)$")
    ax_eta.set_title(r"$\eta(k)$ — RTD content beyond DCS")
    ax_eta.grid(True, alpha=0.25)
    ax_eta.legend(loc="best", fontsize=9, framealpha=0.9)

    # --- right panel: τ̄(k) ---
    ax_tau.plot(k, tau_fix, "-", color=cF, lw=1.8,
                label=r"fixed $(\alpha,\beta)$")
    if not fixed_only:
        ax_tau.plot(k, tau_pol, "--", color=cP, lw=1.8, label="pol-avg")
    ax_tau.axhline(0.0, color="0.6", lw=0.7, ls=":", zorder=0)
    ax_tau.set_xlabel(xlabel)
    ax_tau.set_ylabel(r"yield-averaged delay  $\bar\tau(k)$ [as]")
    ax_tau.set_title(r"$\bar\tau(k)=\int\sigma\tau\,d\Omega/\int\sigma\,d\Omega$")
    ax_tau.grid(True, alpha=0.25)
    ax_tau.legend(loc="best", fontsize=9, framealpha=0.9)

    suptitle = "RTD content beyond DCS"
    if fixed_only:
        suptitle += r" — fixed $(\alpha,\beta)$"
    fig.suptitle(suptitle + f"  (gauge: {gauge})", y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"  wrote {out_png}")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path,
                    help="Directory with dipole_{gauge}_homo_{q}_<mu>.dat")
    ap.add_argument("--output-dir", type=Path, default=Path("."),
                    help="Where to write rtd_content_{gauge}.{dat,png}")
    ap.add_argument("--gauge", choices=("len", "vel"), default="len")
    ap.add_argument("--alpha-deg", type=float, default=0.0,
                    help="Euler α of the fixed orientation (deg).")
    ap.add_argument("--beta-deg", type=float, default=0.0,
                    help="Euler β of the fixed orientation (deg); "
                         "α=β=0 ⇒ ẑ polarisation.")
    ap.add_argument("--n-theta", type=int, default=181,
                    help="θ samples in [0, π] for the solid-angle integral.")
    ap.add_argument("--n-phi", type=int, default=120,
                    help="φ samples in [0, 2π] for the solid-angle integral.")
    ap.add_argument("--k-min", type=float, default=None)
    ap.add_argument("--k-max", type=float, default=None)
    ap.add_argument("--mask-frac", type=float, default=0.02,
                    help="η is drawn bold where ‖𝒥‖(E) >= frac·max‖𝒥‖ and "
                         "faint below (weak-delay, noisier).  0 ⇒ all bold.")
    ap.add_argument("--fixed-only", action="store_true",
                    help="Plot only the fixed-(α,β) curves (drop pol-avg).")
    ap.add_argument("--no-plot", action="store_true",
                    help="Write only the .dat (skip matplotlib).")
    ap.add_argument("--selftest", action="store_true",
                    help="Validate the full-sphere harmonics and exit.")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.input_dir is None:
        ap.error("--input-dir is required (or pass --selftest)")

    print("=" * 72)
    print(" rtd_content.py — η(E) and τ̄(E): RTD content beyond DCS")
    print("=" * 72)
    print(f"  input-dir : {args.input_dir}")
    print(f"  gauge     : {args.gauge}")
    print(f"  fixed α,β : {args.alpha_deg}, {args.beta_deg} deg")
    print(f"  grid      : nθ={args.n_theta}, nφ={args.n_phi}  (full sphere)")

    ds_x = load_dipole_dataset(args.input_dir, args.gauge, "x")
    ds_y = load_dipole_dataset(args.input_dir, args.gauge, "y")
    ds_z = load_dipole_dataset(args.input_dir, args.gauge, "z")
    if not (np.allclose(ds_x.k, ds_y.k) and np.allclose(ds_y.k, ds_z.k)):
        raise SystemExit("ERROR: k-grids of x/y/z dipoles disagree")
    if not (ds_x.mu_list == ds_y.mu_list == ds_z.mu_list):
        raise SystemExit("ERROR: channel sets of x/y/z dipoles disagree")

    if args.k_min is not None or args.k_max is not None:
        lo = ds_x.k.min() if args.k_min is None else args.k_min
        hi = ds_x.k.max() if args.k_max is None else args.k_max
        mask = (ds_x.k >= lo) & (ds_x.k <= hi)

        def _slice(ds):
            return DipoleDataset(ds.k[mask], ds.e_kin[mask], ds.omega[mask],
                                 ds.mu_list, ds.dipole[mask, :],
                                 ds.gauge, ds.component)

        ds_x, ds_y, ds_z = _slice(ds_x), _slice(ds_y), _slice(ds_z)

    if ds_x.k.size < 3:
        raise SystemExit("ERROR: need ≥3 energy points for ∂_E (np.gradient).")

    k_grid = ds_x.k                 # photoelectron momentum, k = √(2E) [a.u.]
    e_kin = ds_x.e_kin
    E_eV = e_kin * HA_TO_EV
    print(f"  energies  : {ds_x.k.size}   "
          f"k = [{k_grid[0]:.3f}, {k_grid[-1]:.3f}] a.u.   "
          f"E_kin = [{E_eV[0]:.3f}, {E_eV[-1]:.3f}] eV")
    print(f"  channels  : {len(ds_x.mu_list)}   "
          f"ℓ_max = {max(idx_to_lm(mu)[0] for mu in ds_x.mu_list)}")

    theta = np.linspace(0.0, np.pi, args.n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, args.n_phi)

    print("  building F(E,θ,φ) for x, y, z ...")
    Fx, dFx = build_F_grid(ds_x, theta, phi)
    Fy, dFy = build_F_grid(ds_y, theta, phi)
    Fz, dFz = build_F_grid(ds_z, theta, phi)

    # FIXED orientation.
    a_rad, b_rad = np.deg2rad(args.alpha_deg), np.deg2rad(args.beta_deg)
    sig_f, J_f = sigma_J_fixed(Fx, dFx, Fy, dFy, Fz, dFz, a_rad, b_rad)
    eta_f, tau_f, lam_f, nJ_f = rtd_content(sig_f, J_f, theta, phi)

    # ISOTROPIC POLARISATION AVERAGE.
    sig_p, J_p = sigma_J_pol_average(Fx, dFx, Fy, dFy, Fz, dFz)
    eta_p, tau_p, lam_p, nJ_p = rtd_content(sig_p, J_p, theta, phi)

    # Mask η where the global delay weight ‖𝒥‖ is tiny (η is noisy there).
    def _mask(nJ):
        peak = np.max(nJ) if np.max(nJ) > 0 else 1.0
        return nJ < args.mask_frac * peak

    mask_f, mask_p = _mask(nJ_f), _mask(nJ_p)

    # ---- write .dat ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_dat = args.output_dir / f"rtd_content_{args.gauge}.dat"
    table = np.column_stack([
        k_grid, e_kin, E_eV,
        eta_f, tau_f, lam_f, nJ_f,
        eta_p, tau_p, lam_p, nJ_p,
    ])
    header = (
        "RTD content beyond DCS — see postprocessing/docs/density_current.pdf\n"
        f"gauge = {args.gauge}\n"
        f"fixed orientation: alpha = {args.alpha_deg} deg, "
        f"beta = {args.beta_deg} deg\n"
        f"solid-angle grid: n_theta = {args.n_theta}, n_phi = {args.n_phi}\n"
        "Definitions (atomic units; C/k^2 prefactor cancels in all ratios):\n"
        "  sigma = |F|^2,  J = Im[F* dE F]  (F = k*D)\n"
        "  lambda(E) = <J,sigma>/<sigma,sigma>      (sigma^2-weighted delay)\n"
        "  eta(E)    = ||J - lambda*sigma|| / ||J||  in [0,1]\n"
        "  taubar(E) = int J dOmega / int sigma dOmega = int sigma*tau dOmega"
        " / int sigma dOmega  (yield-weighted delay)\n"
        "  normJ(E)  = sqrt(<J,J>)   (delay weight; small => eta noisy)\n"
        "eta = 0  <=> tau_W angle-independent (RTD adds no angular info);\n"
        "eta -> 1 <=> RTD angular pattern uncorrelated with DCS.\n"
        "Columns:\n"
        "  1 k[au]=sqrt(2E)  2 E_kin[au]  3 E_kin[eV]\n"
        "  4 eta_fixed  5 taubar_fixed[as]  6 lambda_fixed[as]  7 normJ_fixed\n"
        "  8 eta_pol    9 taubar_pol[as]   10 lambda_pol[as]   11 normJ_pol"
    )
    np.savetxt(out_dat, table, header=header, fmt="%.10e")
    print(f"  wrote {out_dat}")

    # ---- report magnitudes (the PI's headline number) ----
    def _summ(tag, eta, mask):
        good = eta[~mask]
        if good.size:
            print(f"    {tag}: eta in [{good.min():.3f}, {good.max():.3f}], "
                  f"mean {good.mean():.3f}  ({mask.sum()} bins masked)")
        else:
            print(f"    {tag}: all bins masked (||J|| tiny everywhere)")

    print("  RTD-content summary (η over un-masked energies):")
    _summ("fixed ", eta_f, mask_f)
    _summ("pol-av", eta_p, mask_p)

    if not args.no_plot:
        try:
            make_figure(k_grid, eta_f, tau_f, eta_p, tau_p,
                        mask_f, mask_p,
                        args.output_dir / f"rtd_content_{args.gauge}.png",
                        args.gauge, fixed_only=args.fixed_only)
        except Exception as exc:  # pragma: no cover
            print(f"  [warn] plotting skipped: {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
