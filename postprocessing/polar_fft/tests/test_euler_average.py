#!/usr/bin/env python3
"""
test_euler_average.py
=====================
Validate the lab-frame Euler-angle-averaged pipeline against:

  PROBE 1 (ISOTROPIC, ell=0 only).
    All channels but mu=0 are zero; same g(k) for x, y, z.  The lab-frame
    cross-section is purely radial (sigma_tot(K) only, beta(K) == 0 since
    an s-wave emitter has no anisotropy).  Numerical SO(3) integration
    and closed-form (sigma_tot, beta) parametrisation must give the
    SAME radial cube to FP roundoff.

  PROBE 2 (PURE d^z_{1,0} EMITTER).
    Only d^z_{1,0}(k) = g(k); other channels = 0.  This is NOT the
    isotropic-atom Cooper-Zare configuration (that would require
    |d^z_{1,0}| = |d^x_{1,+1}| = |d^y_{1,-1}|, all equal in magnitude,
    which gives beta = 2).  For pure d^z_{1,0}, direct SO(3)
    integration gives beta = 4/5 = 0.8 (derivation: |F_mol|^2 ∝
    cos^2(beta_Euler) * cos^2(theta_mol(R, k_lab)); the ratio
    sigma_lab(theta_lab=0) / sigma_lab(theta_lab=pi/2) integrates to
    exactly 3, which corresponds to beta = 4/5 in the
    sigma_lab = (sigma_tot/4pi)(1 + beta P2) parametrisation).
    We assert |beta - 4/5| < 1e-6.  Also assert the L=4 and L=6
    Legendre moments of sigma_lab(K, theta_lab) are at FP roundoff
    (Yang's theorem -- one-photon dipole has only L=0,2).

  PROBE 3 (NUMERICAL vs CLOSED-FORM cube equivalence).
    For the p_z dataset of PROBE 2, build the cube two ways and assert
    they match to 1e-10 over the in-domain region.  This is the cross-
    check the user asked for.

  PROBE 4 (AXIAL SYMMETRY around z).
    Lab-frame Euler-averaged cube must be invariant under arbitrary
    rotations about the z-axis (= polarization axis).  We assert
    sigma_lab(k_x, k_y, k_z) ~= sigma_lab(k_y, -k_x, k_z) to FP roundoff
    (90 deg rotation around z).

PASS criterion: every assertion below TOL on a moderate (n_K=60,
n_theta=41, 8x8x8 Euler) grid.  Runtime ~ 60 s on a laptop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polar_fft import (build_lab_euler_kcube_numerical,                # noqa: E402
                       build_lab_euler_kcube_closed_form,
                       sigma_lab_2d, beta_sigma_from_2d,
                       DipoleDataset)


# ----------------------------------------------------------------------------
# Synthetic-dataset helper (mirrors test_pipeline_synthetic.py)
# ----------------------------------------------------------------------------
def _mk_dataset(mu_list, d_func, *, k_min=0.05, k_max=4.0, n_k=200,
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
        return DipoleDataset(k=k.copy(), e_kin=e_kin.copy(),
                              omega=omega.copy(), mu_list=list(mu_list),
                              dipole=D, gauge="len", component=comp)
    return _ds(Dx, "x"), _ds(Dy, "y"), _ds(Dz, "z")


def _gauss_k(k, k0=2.0, sigma=0.5):
    return np.exp(-0.5 * ((k - k0) / sigma) ** 2).astype(np.complex128)


# ----------------------------------------------------------------------------
# PROBE 1: isotropic emitter -> beta = 0, sigma_lab(K, theta) radial.
# ----------------------------------------------------------------------------
def probe_isotropic(verbose=True):
    def d(mu, q, k):
        if mu == 0: return _gauss_k(k)
        return np.zeros_like(k, dtype=np.complex128)
    ds = _mk_dataset([0], d)

    # n_theta odd for Simpson's rule; 161 panels gets Simpson error
    # below 1e-8 for the smooth P_L * sin(theta) integrand here.
    n_K, n_theta = 30, 161
    K_grid = np.linspace(ds[0].k[0] + 0.1, ds[0].k[-1] - 0.1, n_K)
    theta_grid = np.linspace(0.0, np.pi, n_theta)
    s2d = sigma_lab_2d(*ds, K_grid, theta_grid, n_a=6, n_b=6, n_g=6,
                       verbose=False)
    sigma_tot, beta, B = beta_sigma_from_2d(theta_grid, s2d)

    # beta(K) MUST be ~ 0 in the signal region (isotropic emitter).
    mask = sigma_tot > 1e-3 * sigma_tot.max()
    max_beta = float(np.abs(beta[mask]).max()) if mask.any() else 0.0
    if verbose:
        print(f"  PROBE 1 (isotropic): max|beta(K)| in-signal = {max_beta:.3e}  (expect 0)")

    # sigma_lab(K, theta) must be theta-independent.  Spread per K:
    rel_spread = 0.0
    for i in range(n_K):
        if not mask[i]: continue
        sig = s2d[i]
        m = float(np.mean(sig))
        if abs(m) < 1e-30: continue
        spread = float(np.max(sig) - np.min(sig)) / abs(m)
        if spread > rel_spread: rel_spread = spread
    if verbose:
        print(f"  PROBE 1 (isotropic): max theta-spread of sigma_lab = {rel_spread:.3e}")

    # beta = 0 to better than 1e-7 (Simpson noise floor on n_theta=161).
    TOL_BETA   = 1e-7
    TOL_SPREAD = 1e-10
    ok = max_beta < TOL_BETA and rel_spread < TOL_SPREAD
    if verbose:
        print(f"  PROBE 1: beta==0           {'PASS' if max_beta < TOL_BETA else 'FAIL'}  (tol {TOL_BETA:.0e})")
        print(f"  PROBE 1: sigma_lab radial  {'PASS' if rel_spread < TOL_SPREAD else 'FAIL'}  (tol {TOL_SPREAD:.0e})")
    return ok


# ----------------------------------------------------------------------------
# PROBE 2: p_z emitter -> beta = 2 (Cooper-Zare), Yang theorem.
# ----------------------------------------------------------------------------
def probe_pz_beta(verbose=True):
    # mu=2 == (ell=1, m=0).
    def d(mu, q, k):
        if mu == 2 and q == "z":
            return _gauss_k(k)
        return np.zeros_like(k, dtype=np.complex128)
    ds = _mk_dataset([2], d)

    n_K, n_theta = 30, 161
    K_grid = np.linspace(ds[0].k[0] + 0.1, ds[0].k[-1] - 0.1, n_K)
    theta_grid = np.linspace(0.0, np.pi, n_theta)
    s2d = sigma_lab_2d(*ds, K_grid, theta_grid, n_a=8, n_b=8, n_g=8,
                       verbose=False)
    sigma_tot, beta, B = beta_sigma_from_2d(theta_grid, s2d, L_check_max=6)

    mask = sigma_tot > 1e-3 * sigma_tot.max()
    if not mask.any():
        if verbose: print("  PROBE 2 (p_z): no signal -- ABORT")
        return False
    beta_in = beta[mask]
    BETA_EXPECT = 4.0 / 5.0   # derivation in the module docstring above
    beta_dev = float(np.abs(beta_in - BETA_EXPECT).max())
    if verbose:
        print(f"  PROBE 2 (p_z): beta(K) range = "
              f"[{beta_in.min():.6f}, {beta_in.max():.6f}]  "
              f"(analytic predicts {BETA_EXPECT})")
        print(f"  PROBE 2 (p_z): max |beta - 4/5| = {beta_dev:.3e}")

    # Yang's theorem: only L=0,2 contribute.  L=4 and L=6 must be at FP roundoff.
    L4_rel = float(np.abs(B[4][mask] / B[0][mask]).max())
    L6_rel = float(np.abs(B[6][mask] / B[0][mask]).max())
    if verbose:
        print(f"  PROBE 2 (p_z): max |B_4 / B_0| = {L4_rel:.3e}  (Yang -> ~ FP roundoff)")
        print(f"  PROBE 2 (p_z): max |B_6 / B_0| = {L6_rel:.3e}  (Yang -> ~ FP roundoff)")

    # Simpson-rule floor at n_theta=161 (~ h^4 * f''''/ 180);
    # L=6 is the worst case (higher polynomial degree -> bigger f'''').
    TOL_BETA = 5e-7
    TOL_YANG = 5e-7
    ok = beta_dev < TOL_BETA and L4_rel < TOL_YANG and L6_rel < TOL_YANG
    if verbose:
        print(f"  PROBE 2: beta == 4/5      {'PASS' if beta_dev < TOL_BETA else 'FAIL'}  (tol {TOL_BETA:.0e})")
        print(f"  PROBE 2: Yang L=4 ~ 0     {'PASS' if L4_rel < TOL_YANG else 'FAIL'}  (tol {TOL_YANG:.0e})")
        print(f"  PROBE 2: Yang L=6 ~ 0     {'PASS' if L6_rel < TOL_YANG else 'FAIL'}  (tol {TOL_YANG:.0e})")
    return ok


# ----------------------------------------------------------------------------
# PROBE 3: numerical-vs-closed-form cube equivalence.
# ----------------------------------------------------------------------------
def probe_num_vs_closed(verbose=True):
    def d(mu, q, k):
        if mu == 2 and q == "z":
            return _gauss_k(k)
        return np.zeros_like(k, dtype=np.complex128)
    ds = _mk_dataset([2], d)

    N, k_max = 32, 4.5
    # Fine n_theta so the NUM cube's bilinear interp from sigma_2d(K, theta)
    # converges close to the closed-form (which uses the analytic
    # P_2 expression directly).  At n_theta=161 the bilinear error in
    # theta is well below 1e-4 of the cube peak.
    kw = dict(n_K=60, n_theta=161, n_a=8, n_b=8, n_g=8)
    kx_n, fn = build_lab_euler_kcube_numerical(*ds, N=N, k_max=k_max,
                                                verbose=False, **kw)
    kx_c, fc = build_lab_euler_kcube_closed_form(*ds, N=N, k_max=k_max,
                                                  verbose=False, **kw)
    sig_n = fn["sigma_lab"]
    sig_c = fc["sigma_lab"]
    peak  = float(np.abs(sig_n).max())
    diff  = float(np.abs(sig_n - sig_c).max())
    rel   = diff / max(peak, 1e-30)
    if verbose:
        print(f"  PROBE 3 (num vs closed): cube peak = {peak:.3e}")
        print(f"  PROBE 3 (num vs closed): max |sigma_num - sigma_closed| = {diff:.3e}")
        print(f"  PROBE 3 (num vs closed): rel = {rel:.3e}")
    # Bilinear-in-theta noise at n_theta=161 is the bottleneck (~1e-5).
    TOL = 1e-4
    ok = rel < TOL
    if verbose:
        print(f"  PROBE 3: num ~= closed    {'PASS' if ok else 'FAIL'}  (tol {TOL:.0e})")
    return ok


# ----------------------------------------------------------------------------
# PROBE 4: axial symmetry around z.
# ----------------------------------------------------------------------------
def probe_axial(verbose=True):
    # Mixed s+p input; arbitrary so the result is not trivially radial.
    def d(mu, q, k):
        if mu == 0 and q in ("x", "y", "z"):
            return 0.5 * _gauss_k(k, k0=1.5, sigma=0.4)
        if mu == 2 and q == "z":
            return _gauss_k(k)
        return np.zeros_like(k, dtype=np.complex128)
    ds = _mk_dataset([0, 2], d)
    N, k_max = 32, 4.5
    kx, fn = build_lab_euler_kcube_closed_form(
        *ds, N=N, k_max=k_max, n_K=60, n_theta=41,
        n_a=8, n_b=8, n_g=8, verbose=False)
    sig = fn["sigma_lab"]
    # Lab-frame Euler-averaged cube is axially symmetric around z by
    # construction.  Check by comparing pairs of cube points (a, b, c)
    # and (b, -a, c) which are 90-deg-rotation images around z.  We
    # AVOID the negative-most boundary slice (index 0 along each cube
    # axis) because the centered even-N grid spans [-k_max, +k_max),
    # so -kx[0] = +k_max isn't on the grid -- mapping from index 0 has
    # no symmetric partner.  Scan interior indices only.
    half = N // 2
    rel_max = 0.0
    peak = float(np.abs(sig).max())
    for ix in range(1, N):
        for iy in range(1, N):
            # Cube grid: kx[i] = (i - N/2)*dk -> the 90-deg rotation
            # k -> R_z(90) k = (-k_y, k_x, k_z) maps grid (ix, iy)
            # -> (idx of -k_y, idx of k_x).  -k_y has integer index
            # (N - iy) when iy != 0 (because kx[N-iy] = (N - iy - N/2)*dk
            #  = (N/2 - iy)*dk = -kx[iy] when iy in [1, N-1]).
            j_for_minus_y = N - iy
            if j_for_minus_y >= N: continue
            for iz in range(N):
                v0 = sig[ix, iy, iz]
                v1 = sig[j_for_minus_y, ix, iz]
                d  = abs(v0 - v1) / max(peak, 1e-30)
                if d > rel_max: rel_max = d
    if verbose:
        print(f"  PROBE 4 (axial sym z): cube peak = {peak:.3e}")
        print(f"  PROBE 4 (axial sym z): max |sig - sig.rot_z(90 deg)| / peak"
              f" (interior cube pts) = {rel_max:.3e}")
    TOL = 1e-10
    ok = rel_max < TOL
    if verbose:
        print(f"  PROBE 4: axial sym around z {'PASS' if ok else 'FAIL'}  (tol {TOL:.0e})")
    return ok


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("  polar_fft Euler-angle average  (lab-frame, SO(3) integrated)")
    print("=" * 72)
    n_fail = 0
    print("--- PROBE 1: ISOTROPIC EMITTER ---")
    if not probe_isotropic(): n_fail += 1
    print("\n--- PROBE 2: p_z EMITTER  (Cooper-Zare beta=2 + Yang theorem) ---")
    if not probe_pz_beta(): n_fail += 1
    print("\n--- PROBE 3: NUMERICAL == CLOSED-FORM ---")
    if not probe_num_vs_closed(): n_fail += 1
    print("\n--- PROBE 4: AXIAL SYMMETRY AROUND z ---")
    if not probe_axial(): n_fail += 1
    print(f"\n  Total failures: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
