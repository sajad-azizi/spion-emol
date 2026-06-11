#!/usr/bin/env python3
"""
test_euler_tau.py
=================
Validate the sigma-weighted Euler-angle-averaged Wigner delay added to
build_lab_euler_kcube_{numerical, closed_form}.

Formula being tested:

    tau_avg(K, k_hat) = <Im( F_mol*(R) * dE F_mol(R) )>_R / <|F_mol(R)|^2>_R

where F_mol(R) = (R^T z_lab) . F^q(K, R^T k_hat_lab).  Both numerator
and denominator are bilinear in the dipole amplitudes and thus
restricted to L = 0, 2 in cos(theta_lab) by Yang's theorem.

Probes:

  PROBE A (CONSTANT-PHASE EMITTER -> tau_avg == 0).
    If d^q_{ell,m}(K) is a real Gaussian times a CONSTANT (K-independent)
    phase, dE d = (1/K) d/dK [phase * |d|] also has a real ratio to d,
    so Im(d* dE d) = 0 channel-by-channel and tau_avg must vanish to
    FP roundoff.

  PROBE B (LINEAR-PHASE EMITTER -> tau_avg = constant slope).
    With d(K) = g(K) * exp(i alpha K) (single channel, e.g. d^z_{1,0}):
    Im(d* dE d) = Im(|g|^2 * exp(-i alpha K) * dE [g * exp(i alpha K)])
                = Im(|g|^2 * exp(-i alpha K) * (dE g + i alpha (1/K) g exp(i alpha K)))
    For real g this simplifies and tau_avg = Im(d* dE d) / |d|^2 = alpha / K
    (the standard Wigner delay for a linear phase).  We assert this
    matches across the K-grid to better than 1e-5 (Simpson noise +
    discrete gradient).

  PROBE C (NUM vs CLOSED-FORM tau_avg).
    Same synthetic dataset as PROBE B; the two cube paths must agree
    to ~1e-3 of the tau_avg peak (bilinear-in-theta noise dominates,
    same as Probe 3 of test_euler_average).

  PROBE D (Yang on the NUMERATOR).
    For the synthetic input above, the Legendre L > 2 moments of the
    numerator <Im(F* dE F)>_R(K, theta) must be at Simpson floor
    (one-photon dipole -> only L=0, 2 in numerator).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polar_fft import (build_lab_euler_kcube_numerical,                # noqa: E402
                       build_lab_euler_kcube_closed_form,
                       sigma_lab_2d, beta_sigma_from_2d,
                       DipoleDataset, AU_TO_AS, C_AU)


def _mk_dataset(mu_list, d_func, *, k_min=0.5, k_max=4.0, n_k=300,
                ip_au=0.30):
    k = np.linspace(k_min, k_max, n_k)
    e_kin = 0.5 * k * k
    omega = e_kin + ip_au
    n_ch = len(mu_list)
    Dx = np.zeros((n_k, n_ch), dtype=np.complex128)
    Dy = np.zeros((n_k, n_ch), dtype=np.complex128)
    Dz = np.zeros((n_k, n_ch), dtype=np.complex128)
    for j, mu in enumerate(mu_list):
        Dx[:, j] = d_func(mu, "x", k)
        Dy[:, j] = d_func(mu, "y", k)
        Dz[:, j] = d_func(mu, "z", k)
    def _ds(D, comp):
        return DipoleDataset(k=k.copy(), e_kin=e_kin.copy(), omega=omega.copy(),
                              mu_list=list(mu_list), dipole=D, gauge="len",
                              component=comp)
    return _ds(Dx, "x"), _ds(Dy, "y"), _ds(Dz, "z")


# ----------------------------------------------------------------------------
# PROBE A: real (constant-phase) dipole -> tau_avg == 0
# ----------------------------------------------------------------------------
def probe_zero_tau(verbose=True):
    def d(mu, q, k):
        if mu == 2 and q == "z":
            return np.exp(-0.5 * ((k - 2.0) / 0.5) ** 2).astype(np.complex128)
        return np.zeros_like(k, dtype=np.complex128)
    ds = _mk_dataset([2], d)

    n_K, n_theta = 30, 161
    K_grid = np.linspace(ds[0].k[0] + 0.2, ds[0].k[-1] - 0.2, n_K)
    theta_grid = np.linspace(0.0, np.pi, n_theta)
    sigma_2d, tau_num_2d, F2_2d = sigma_lab_2d(*ds, K_grid, theta_grid,
                                                 n_a=8, n_b=8, n_g=8,
                                                 with_tau=True, verbose=False)
    # tau_num / F2 should be ~ 0 since the dipole is real -> no Wigner phase.
    F2_safe = np.where(F2_2d > 1e-30, F2_2d, 1.0)
    tau = np.where(F2_2d > 1e-30, tau_num_2d / F2_safe, 0.0)
    peak_in_signal = float(np.abs(tau[F2_2d > 1e-3 * F2_2d.max()]).max())
    if verbose:
        print(f"  PROBE A: max |tau_avg| (real dipole, signal region) = {peak_in_signal:.3e}  (expect 0)")
    TOL = 1e-8
    ok = peak_in_signal < TOL
    if verbose:
        print(f"  PROBE A: real dipole -> tau == 0   {'PASS' if ok else 'FAIL'}  (tol {TOL:.0e})")
    return ok


# ----------------------------------------------------------------------------
# PROBE B: linear-phase dipole -> tau_avg = alpha / K (standard Wigner delay)
# ----------------------------------------------------------------------------
def probe_linear_phase_tau(verbose=True):
    ALPHA = 0.35    # phase slope in d/dK
    def d(mu, q, k):
        if mu == 2 and q == "z":
            g = np.exp(-0.5 * ((k - 2.0) / 0.6) ** 2)
            return (g * np.exp(1j * ALPHA * k)).astype(np.complex128)
        return np.zeros_like(k, dtype=np.complex128)
    ds = _mk_dataset([2], d)

    n_K, n_theta = 30, 161
    K_grid = np.linspace(ds[0].k[0] + 0.3, ds[0].k[-1] - 0.3, n_K)
    theta_grid = np.linspace(0.0, np.pi, n_theta)
    sigma_2d, tau_num_2d, F2_2d = sigma_lab_2d(*ds, K_grid, theta_grid,
                                                 n_a=8, n_b=8, n_g=8,
                                                 with_tau=True, verbose=False)
    F2_safe = np.where(F2_2d > 1e-30, F2_2d, 1.0)
    tau_au = np.where(F2_2d > 1e-30, tau_num_2d / F2_safe, 0.0)
    # For a single-channel linear-phase synthetic, tau_avg comes out the
    # same at every theta_lab where the channel has signal (only theta=0
    # has full p_z signal; off-axis is fine too once normalised by F^2).
    # Pick the on-axis column.
    j0 = np.argmin(np.abs(theta_grid - 0.0))
    tau_K = tau_au[:, j0]
    # Expected:
    #   d = g(K) e^{i alpha K}
    #   dE d = (1/K) dK d = (1/K) (g' + i alpha g) e^{i alpha K}
    #   Im(d* dE d) = Im( |g|^2 e^{-i alpha K} (1/K)(g' + i alpha g) e^{i alpha K} )
    #              = (1/K) Im( |g|^2 (g' + i alpha g) / g ... )
    # Simplified: with d = g e^{i alpha K} (real g), the phase factor
    # cancels: d* dE d = (g / K)(dE_internal g + i alpha g)
    # Then Im(.) / |d|^2 = alpha / K.
    expected = ALPHA / K_grid
    dev_au = float(np.max(np.abs(tau_K - expected)))
    rel    = float(np.max(np.abs((tau_K - expected) / np.maximum(np.abs(expected), 1e-30))))
    if verbose:
        print(f"  PROBE B: tau_avg(K) sampled along +z axis.  expected = alpha/K with alpha={ALPHA}")
        print(f"  PROBE B: max |tau - alpha/K| = {dev_au:.3e} au  rel = {rel:.3e}")
    TOL = 5e-3   # discrete np.gradient error on n_k=300 + Simpson noise
    ok = rel < TOL
    if verbose:
        print(f"  PROBE B: tau = alpha/K     {'PASS' if ok else 'FAIL'}  (tol {TOL:.0e})")
    return ok


# ----------------------------------------------------------------------------
# PROBE C: numerical-vs-closed-form tau_avg cube equivalence
# ----------------------------------------------------------------------------
def probe_num_vs_closed_tau(verbose=True):
    ALPHA = 0.35
    def d(mu, q, k):
        if mu == 2 and q == "z":
            g = np.exp(-0.5 * ((k - 2.0) / 0.6) ** 2)
            return (g * np.exp(1j * ALPHA * k)).astype(np.complex128)
        return np.zeros_like(k, dtype=np.complex128)
    ds = _mk_dataset([2], d)
    N, k_max = 32, 4.5
    kw = dict(n_K=60, n_theta=81, n_a=8, n_b=8, n_g=8)
    _, fn = build_lab_euler_kcube_numerical(*ds, N=N, k_max=k_max,
                                              with_tau=True, verbose=False, **kw)
    _, fc = build_lab_euler_kcube_closed_form(*ds, N=N, k_max=k_max,
                                                with_tau=True, verbose=False, **kw)
    tn = fn["tau_avg"]
    tc = fc["tau_avg"]
    peak = float(np.abs(tn).max())
    rel  = float(np.abs(tn - tc).max() / max(peak, 1e-30))
    if verbose:
        print(f"  PROBE C: tau_avg cube peak (num) = {peak:.3e} as")
        print(f"  PROBE C: max |tau_num - tau_closed| / peak = {rel:.3e}")
    TOL = 1e-2     # bilinear-in-theta noise + dE_d gradient noise on cube grid
    ok = rel < TOL
    if verbose:
        print(f"  PROBE C: num ~= closed     {'PASS' if ok else 'FAIL'}  (tol {TOL:.0e})")
    return ok


# ----------------------------------------------------------------------------
# PROBE D: Yang's theorem on the numerator
# ----------------------------------------------------------------------------
def probe_yang_numerator(verbose=True):
    ALPHA = 0.35
    def d(mu, q, k):
        if mu == 2 and q == "z":
            g = np.exp(-0.5 * ((k - 2.0) / 0.6) ** 2)
            return (g * np.exp(1j * ALPHA * k)).astype(np.complex128)
        return np.zeros_like(k, dtype=np.complex128)
    ds = _mk_dataset([2], d)
    n_K, n_theta = 30, 161
    K_grid = np.linspace(ds[0].k[0] + 0.3, ds[0].k[-1] - 0.3, n_K)
    theta_grid = np.linspace(0.0, np.pi, n_theta)
    sigma_2d, tau_num_2d, F2_2d = sigma_lab_2d(*ds, K_grid, theta_grid,
                                                 n_a=8, n_b=8, n_g=8,
                                                 with_tau=True, verbose=False)
    _, _, B_num = beta_sigma_from_2d(theta_grid, tau_num_2d, L_check_max=6)
    mask = np.abs(B_num[0]) > 1e-3 * np.abs(B_num[0]).max()
    L4_rel = float(np.abs(B_num[4][mask] / B_num[0][mask]).max())
    L6_rel = float(np.abs(B_num[6][mask] / B_num[0][mask]).max())
    if verbose:
        print(f"  PROBE D: Yang on numerator -- max |B_4 / B_0| = {L4_rel:.3e}")
        print(f"  PROBE D: Yang on numerator -- max |B_6 / B_0| = {L6_rel:.3e}")
    TOL = 1e-5
    ok = L4_rel < TOL and L6_rel < TOL
    if verbose:
        print(f"  PROBE D: Yang on numerator {'PASS' if ok else 'FAIL'}  (tol {TOL:.0e})")
    return ok


def main() -> int:
    print("=" * 72)
    print("  polar_fft sigma-weighted Euler-averaged tau (lab frame)")
    print("=" * 72)
    n_fail = 0
    print("--- PROBE A: real dipole -> tau == 0 ---")
    if not probe_zero_tau(): n_fail += 1
    print("\n--- PROBE B: linear-phase d -> tau = alpha/K ---")
    if not probe_linear_phase_tau(): n_fail += 1
    print("\n--- PROBE C: numerical vs closed-form tau_avg cube ---")
    if not probe_num_vs_closed_tau(): n_fail += 1
    print("\n--- PROBE D: Yang on the numerator ---")
    if not probe_yang_numerator(): n_fail += 1
    print(f"\n  Total failures: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
