"""
phase_d_long_pulse_limit.py
---------------------------
Phase D, validation 1.

As T_L grows, the two-pulse Dawson kernel K_LX(β, α; τ, 0, T_L, T_X)
should select the IR-on-shell intermediate energy

    ε_ν^*  =  ε_κ − s_IR · ω

(β = 0 there) and the sideband delay τ_2hω should converge to a finite
T_L-independent limit -- the BW17 / Sokhotski-Plemelj result.

Test design
-----------
* Single-channel synthetic (1×1 matrices) so the angular machinery is
  trivial and the result is purely the kernel-times-amplitudes integral.
* Discrete sum over a dense ε_ν grid that brackets ε_ν^*  for both paths.
* Smooth, complex M_in(ε_ν) and D(ε_ν) (analytic phase variation so the
  delay is nonzero and identifiable).
* Sweep T_L over 1 -- 200 au^-1 (≈ 0.66 fs -- 130 fs).
* Report τ vs T_L: should plateau as T_L → large.

This validates the FORMULA, not the full m-resolved pipeline -- single-
channel reduces compute_M to a trivial angular wrapper, so the result
checks both compute_T and the kernel.
"""
from __future__ import annotations

import numpy as np

from dawson_kernel_rabbitt import HBAR_EV_FS
from compute_T import compute_T_path
from tau_2homega import tau_pointwise


def synthetic_amplitudes(eps_nu: np.ndarray, eps_kappa: float):
    """Return list of (1,1) M_in and (1,) D arrays, one per ε_ν.

    Phase variation chosen so that arg{M_in · D*} has a clear linear
    energy dependence -- this MAKES the delay nonzero and easy to read.
    """
    # M_in(ε_ν)[0,0] : complex, "Wigner-Smith-like" phase that varies smoothly
    M_in_phase = 0.7 * (eps_nu - eps_kappa)
    M_in_mag   = 1.0 + 0.1 * (eps_nu - eps_kappa)
    M_list = [np.array([[m * np.exp(1j * p)]], dtype=np.complex128)
              for m, p in zip(M_in_mag, M_in_phase)]
    # D(ε_ν)[0] : real-ish (constant phase) to keep the delay attributable
    D_list = [np.array([1.0 + 0.0j], dtype=np.complex128)
              for _ in eps_nu]
    return M_list, D_list


def main():
    # Physical parameters (au).  1.55 eV IR, 23.25 eV XUV (H15), E_EA=1.2 eV.
    omega_IR  = 1.55  / 27.2114
    E_EA      = 1.2   / 27.2114
    Omega_15  = 15.0 * omega_IR
    Omega_17  = 17.0 * omega_IR

    # Sideband 2q=16: final kinetic energy.
    eps_kappa = -E_EA + Omega_15 + omega_IR    # = -E_EA + Omega_17 - omega_IR

    # XUV pulse fixed at 0.35 fs FWHM.  Long XUV would smear; this matches
    # typical RABBITT geometry (short XUV, longer IR).
    T_X_au = (0.35 / HBAR_EV_FS) / np.sqrt(2 * np.log(2)) * 0.6582119514
    # ^ messy: T_X_au should be in atomic units.  Convert via
    #   t[au] = t[fs] / (ħ[eV·fs] / E_h[eV]) = t[fs] * E_h[eV] / ħ[eV·fs]
    #         = t[fs] * 27.2114 / 0.6582119514 = t[fs] * 41.341
    T_X_au = (0.35 * 41.341) / np.sqrt(2 * np.log(2))   # 0.35 fs FWHM → au
    print(f"T_X = {T_X_au:.4g} au  (FWHM = 0.35 fs)")

    # IR delay τ: pick a few fs to put the sideband oscillation at a
    # non-trivial point; the answer for τ_2hω is independent of this τ.
    tau_au = 1.0 * 41.341     # 1 fs in au

    # ε_ν grid is ADAPTIVE per T_L: the Gaussian peak in β has σ_β = √2/T_L,
    # so we cover ±half_window_sigma · σ_β around the on-shell point and
    # use n_grid points to resolve it.  Wider/coarser grids cause erf
    # overflow at large T_L (|Im z| ≈ |β| T_L²/(2√(T_X²+T_L²)) ≈ |β| T_L/2
    # blows up beyond the relevant peak).
    half_window_sigma = 8.0     # ±8 σ_β captures 1 - 2·erfc(8) ≈ 1 - 1e-29
    n_grid = 1001               # ~ 125 points per σ_β
    print(f"ε_ν grid: n = {n_grid}, half-window = ±{half_window_sigma} σ_β")

    def build_grid(eps_center: float, T_L_au_local: float):
        sigma_beta = np.sqrt(2.0) / T_L_au_local
        half_w = half_window_sigma * sigma_beta
        eps_grid = np.linspace(eps_center - half_w, eps_center + half_w, n_grid)
        deps = eps_grid[1] - eps_grid[0]
        return eps_grid, deps

    print("\n--- τ_2hω vs T_L (adaptive grid; should plateau) ---")
    print(f"{'T_L (fs)':>10s}  {'T_L·ω (rad)':>12s}  {'τ_2hω (au)':>14s}  "
          f"{'τ_2hω (as)':>14s}  {'|M_<|':>12s}  {'|M_>|':>12s}")
    AS_PER_AU = 24.188843   # 1 au of time = 24.189 as
    T_L_fs_list = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 80.0, 130.0]
    results = []
    for T_L_fs in T_L_fs_list:
        T_L_au = (T_L_fs * 41.341) / np.sqrt(2 * np.log(2))

        # Adaptive grids per path: on-shell at ε_κ - s_IR·ω.
        eps_nu_abs, deps_abs = build_grid(eps_kappa - omega_IR, T_L_au)
        eps_nu_emi, deps_emi = build_grid(eps_kappa + omega_IR, T_L_au)
        M_abs, D_abs = synthetic_amplitudes(eps_nu_abs, eps_kappa)
        M_emi, D_emi = synthetic_amplitudes(eps_nu_emi, eps_kappa)

        M_less = compute_T_path(
            M_in_list=M_abs, D_list=D_abs, eps_nu=eps_nu_abs,
            eps_kappa=eps_kappa, E_EA=E_EA, Omega_XUV=Omega_15,
            omega_IR=omega_IR, ir_action="absorb",
            T_X=T_X_au, T_L=T_L_au, tau=tau_au,
            weights=np.full(n_grid, deps_abs),
        )
        M_greater = compute_T_path(
            M_in_list=M_emi, D_list=D_emi, eps_nu=eps_nu_emi,
            eps_kappa=eps_kappa, E_EA=E_EA, Omega_XUV=Omega_17,
            omega_IR=omega_IR, ir_action="emit",
            T_X=T_X_au, T_L=T_L_au, tau=tau_au,
            weights=np.full(n_grid, deps_emi),
        )

        # Single-channel: T̃ is (1,1).  M = ang_mu[0] · T̃[0,0] for L_max=0.
        # ang_mu[0] is the SAME constant for both paths, so it CANCELS in
        # arg{M_< M_>*}.  No need to wrap through compute_M.
        m_less    = M_less[0, 0]
        m_greater = M_greater[0, 0]
        tau = tau_pointwise(m_less, m_greater, omega_IR)
        results.append((T_L_fs, T_L_au, tau, m_less, m_greater))
        T_L_omega = T_L_au * omega_IR
        print(f"{T_L_fs:>10.2f}  {T_L_omega:>12.3g}  "
              f"{tau:>14.6f}  {tau * AS_PER_AU:>14.4f}  "
              f"{abs(m_less):>12.4e}  {abs(m_greater):>12.4e}")

    # Convergence diagnostic: relative change in τ between last 4 entries.
    taus = np.array([r[2] for r in results])
    rel_change_last = abs(taus[-1] - taus[-2]) / max(abs(taus[-1]), 1e-30)
    rel_change_mid  = abs(taus[-3] - taus[-4]) / max(abs(taus[-1]), 1e-30)
    print(f"\nConvergence: |Δτ/τ| between last two T_L = {rel_change_last:.2e}")
    print(f"             |Δτ/τ| between [-3] and [-4]  = {rel_change_mid:.2e}")
    if rel_change_last < rel_change_mid:
        print("  → τ is still decreasing in magnitude of change -- converging.")
    else:
        print("  → τ change is NOT shrinking -- not yet in the plateau, or "
              "the synthetic problem doesn't have a finite limit.")

    # For this synthetic example we have a constant linear phase
    # arg{M_in(ε_ν)} = 0.7·(ε_ν − ε_κ), so the IR-on-shell selection picks
    # phases at ε_ν^* = ε_κ ∓ ω, giving arg M_< = -0.7ω,  arg M_> = +0.7ω.
    # With τ = (arg M_> - arg M_<) / (2ω) = (+0.7ω − (−0.7ω))/(2ω) = +0.7 au.
    tau_expected_long_pulse = +0.7
    print(f"\nAnalytic long-pulse limit (synthetic phase 0.7·(ε_ν−ε_κ)):")
    print(f"  τ → -0.7 au = {tau_expected_long_pulse * AS_PER_AU:.4f} as")
    err = abs(taus[-1] - tau_expected_long_pulse)
    print(f"  Numerical at largest T_L (={T_L_fs_list[-1]} fs): "
          f"τ = {taus[-1]:.6f} au   error = {err:.3e} au "
          f"= {err * AS_PER_AU:.4f} as")


if __name__ == "__main__":
    main()
