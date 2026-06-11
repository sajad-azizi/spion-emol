#!/usr/bin/env python3
"""
test_polar_phi0_match.py
========================
Validate that `polar_fft.py`'s full-(θ, φ) reconstruction reproduces
`compute_polar_data.py`'s φ=0 reference exactly (to FP roundoff) when
evaluated at φ=0.

Why this test matters
---------------------
polar_fft.py builds D_q(k, θ, φ) by summing real-Y_{ℓ, m}(θ, φ) over
ALL m, including m < 0 (which are non-zero at φ ≠ 0 but vanish at
φ = 0 by the sin(|m|φ) factor).  At φ = 0 the m < 0 terms drop, and
the result must reduce to compute_polar_data.py's φ=0 path.

Test
----
1. Build σ_pol, τ_pol on the SAME (k, θ) grid that compute_polar_data
   used to write polar_pol_avg_len.dat (φ=0 implicit there).
2. Use polar_fft.py to compute σ_pol(k_i, θ_j, φ=0), τ_pol(k_i, θ_j, φ=0)
   via the full-(θ,φ) machinery with φ explicitly set to 0.
3. Element-wise compare.

PASS criterion: rel_err < 1e-12 absolute or 1e-10 relative on every
element of σ_pol, τ_pol, σ·τ.  Both implementations use the same input
dipole files and the SAME normalised-Legendre recurrence; the only
difference is whether m<0 is summed (with sin(|m|·0)=0 it should drop)
or skipped.  Agreement at FP roundoff is the expected outcome.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# tests/  -> polar_fft/ -> postprocessing/ -> repo root
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "postprocessing" / "polar_fft"))
sys.path.insert(0, str(REPO / "postprocessing" / "polar_plots"))

from polar_fft import (load_dipole_dataset, idx_to_lm, real_Ylm,  # noqa: E402
                       AU_TO_AS, C_AU)
from compute_polar_data import (per_cartesian_amplitude_at_phi0,    # noqa: E402
                                 compute_polarization_average)


def compute_pol_avg_polar_phi0(ds_x, ds_y, ds_z, k_grid, theta_grid):
    """Evaluate σ_pol, τ_pol at (k_i, θ_j, φ=0) using polar_fft.py's
    full-(θ,φ) machinery: sum real_Y_{ℓ,m}(θ, 0) over ALL m, then form
    σ_pol per Eq. S38.  m<0 terms are sin(|m|·0)=0 in the no-CS
    convention, so they contribute exactly zero -- but we don't skip
    them, we evaluate them.  This is the codepath that runs at φ ≠ 0
    in real usage; here we exercise the φ=0 special case.
    """
    nE = k_grid.size
    nT = theta_grid.size
    mu_list = ds_x.mu_list
    nCh = len(mu_list)

    # Precompute Y^R_{ℓm}(θ, φ=0) for every channel.
    Y = np.zeros((nCh, nT), dtype=np.float64)
    phase = np.zeros(nCh, dtype=np.complex128)
    phi_zero = np.zeros_like(theta_grid)
    for j, mu in enumerate(mu_list):
        ell, m = idx_to_lm(mu)
        Y[j] = real_Ylm(ell, m, theta_grid, phi_zero)
        phase[j] = (-1j) ** ell

    # ∂_E d = (1/k) ∂_k d  on the original k-grid.
    def deriv(D):
        return np.gradient(D, k_grid, axis=0) / k_grid[:, None]

    Dx, Dy, Dz = ds_x.dipole, ds_y.dipole, ds_z.dipole
    dDx, dDy, dDz = deriv(Dx), deriv(Dy), deriv(Dz)

    def F_and_dF(D_q, dD_q):
        F = np.zeros((nE, nT), dtype=np.complex128)
        dF = np.zeros((nE, nT), dtype=np.complex128)
        for j in range(nCh):
            F  += (phase[j] * D_q [:, j])[:, None] * Y[j][None, :]
            dF += (phase[j] * dD_q[:, j])[:, None] * Y[j][None, :]
        return 4.0 * np.pi * F, 4.0 * np.pi * dF

    Fx, dFx = F_and_dF(Dx, dDx)
    Fy, dFy = F_and_dF(Dy, dDy)
    Fz, dFz = F_and_dF(Dz, dDz)

    omega = ds_x.omega
    F2 = np.abs(Fx) ** 2 + np.abs(Fy) ** 2 + np.abs(Fz) ** 2
    Im = (np.imag(np.conj(Fx) * dFx)
          + np.imag(np.conj(Fy) * dFy)
          + np.imag(np.conj(Fz) * dFz))
    sigma_pol = (4.0 * np.pi ** 2 * omega / C_AU)[:, None] / 3.0 \
                * F2 / (k_grid[:, None] ** 2)
    tau_au = np.where(F2 > 1e-30, Im / np.where(F2 > 1e-30, F2, 1.0), 0.0)
    tau_pol = tau_au * AU_TO_AS
    return sigma_pol, tau_pol


def main() -> int:
    GATHERED = REPO / "modle_spherical3d" / "gathered"
    if not GATHERED.exists():
        print(f"  gather dir not found: {GATHERED}")
        print("  Run gather_dipoles.py first to produce per-μ .dat files.")
        return 1
    print("--- polar_fft.py vs compute_polar_data.py at φ=0 ---")
    ds_x = load_dipole_dataset(GATHERED, "len", "x")
    ds_y = load_dipole_dataset(GATHERED, "len", "y")
    ds_z = load_dipole_dataset(GATHERED, "len", "z")

    # Match compute_polar_data's default: n_theta = 181 over [0, π].
    theta_grid = np.linspace(0.0, np.pi, 181)
    k_grid = ds_x.k

    # REFERENCE: compute_polar_data.py's φ=0 path.
    k_ref, _, _, sig_ref, tau_ref = compute_polarization_average(
        ds_x, ds_y, ds_z, theta_grid)
    if not np.allclose(k_ref, k_grid):
        raise RuntimeError("k-axis mismatch between datasets")

    # POLAR_FFT: our full-(θ,φ) path evaluated at φ=0.
    sig_new, tau_new = compute_pol_avg_polar_phi0(
        ds_x, ds_y, ds_z, k_grid, theta_grid)

    # Compare element-wise.
    def report(name, A, B):
        d = np.abs(A - B)
        mx = np.abs(A).max()
        rel = d / np.maximum(np.abs(A), 1e-30)
        # Restrict relative comparison to where reference has signal.
        mask = np.abs(A) > 1e-12 * mx
        rel_masked = rel[mask] if mask.any() else np.array([0.0])
        print(f"  {name:10s}  max|Δ|={d.max():.3e}  "
              f"max|ref|={mx:.3e}  "
              f"max|Δ/ref| (where |ref|>1e-12 max)={rel_masked.max():.3e}")
        return d.max(), rel_masked.max(), mx

    d_sig, r_sig, _ = report("sigma_pol", sig_ref, sig_new)
    d_tau, r_tau, _ = report("tau_pol  ", tau_ref, tau_new)
    d_st,  r_st,  _ = report("sigma*tau",
                              sig_ref * tau_ref, sig_new * tau_new)

    TOL_ABS = 1e-12
    TOL_REL = 1e-10
    fail = 0
    for name, (d, r) in {"sigma_pol": (d_sig, r_sig),
                         "tau_pol":   (d_tau, r_tau),
                         "sigma*tau": (d_st,  r_st)}.items():
        ok = (d < TOL_ABS) or (r < TOL_REL)
        print(f"  {name:10s}  {'PASS' if ok else 'FAIL'}  "
              f"(tol abs={TOL_ABS:.0e} OR rel={TOL_REL:.0e})")
        if not ok:
            fail += 1
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
