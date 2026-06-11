#!/usr/bin/env python3
"""
test_validation_T5_onshell.py
------------------------------
Direct on-shell (T_L → ∞) evaluation of τ_2ℏω WITHOUT any integration.

In the strict long-pulse limit the Dawson kernel collapses to a δ-function
in β at the path's on-shell intermediate energy.  The two-photon amplitude
then reduces to

    M_path  ∝  M_in[κ, ν*_path]  ·  D[ν*_path]

(times a real, path-independent normalization) where ν*_< = ε_κ − ω and
ν*_> = ε_κ + ω.  Linearly interpolating M_in and D from the saved scan
grid to those exact ν* points gives the "T_L = ∞" prediction for τ_2ℏω:

    τ_2ℏω^onshell  =  (1/2ω) · arg[ (M·D)_>* ⊕ (M·D)_<  evaluated at ν* ]

This is a SEPARATE code path -- no kernel, no integration, no trim --
and gives an absolute reference for the trimmed-Dawson result at large
T_L.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "postprocessing/two_photon_delay/python")
from compute_M    import compute_M_one_path, angular_grid
from m_p_polarization import cart_to_spherical
from rotate_to_instate import _Uinv
from tau_2homega  import tau_angle_averaged


def _energy(ik, dk):
    return (ik*dk)**2/2.0


def interp_along_nu(values: np.ndarray, eps_grid: np.ndarray, eps_target: float):
    """Linearly interpolate complex `values[i, ...]` at ε_target in ε_grid."""
    if eps_target <= eps_grid[0] or eps_target >= eps_grid[-1]:
        # extrapolation -- shouldn't happen if scan covers the on-shell point
        raise ValueError(
            f"ε_target = {eps_target:.4f} outside ε_grid "
            f"[{eps_grid[0]:.4f}, {eps_grid[-1]:.4f}]")
    j = int(np.searchsorted(eps_grid, eps_target) - 1)
    a = (eps_target - eps_grid[j]) / (eps_grid[j+1] - eps_grid[j])
    return (1.0 - a) * values[j] + a * values[j+1]


def build_T_onshell(f, ik_kappa, available_nus, omega_IR, s_IR: int):
    """T̃[3,3,N_psi,N_psi] = M_in[κ, ν*] · D[ν*]  with interpolated ν*."""
    dk    = float(f.attrs["dk"])
    N_psi = int(f.attrs["N_psi"])
    eps_kappa = _energy(ik_kappa, dk)
    eps_nu_star = eps_kappa - s_IR * omega_IR

    # Load all M_in and D for available ν.
    A_k = f[f"per_ik/ik{ik_kappa:04d}/A"][()]
    B_k = f[f"per_ik/ik{ik_kappa:04d}/B"][()]
    U_k = _Uinv(A_k, B_k)
    eps_nu = np.array([_energy(ikn, dk) for ikn in available_nus])

    M_in_stack = {q: np.zeros((len(available_nus), N_psi, N_psi), dtype=np.complex128)
                  for q in ("x","y","z")}
    D_stack    = {q: np.zeros((len(available_nus), N_psi), dtype=np.complex128)
                  for q in ("x","y","z")}
    for i, ikn in enumerate(available_nus):
        pair = f[f"pairs/pair_k{ik_kappa:04d}_n{ikn:04d}"]
        A_n = f[f"per_ik/ik{ikn:04d}/A"][()]
        B_n = f[f"per_ik/ik{ikn:04d}/B"][()]
        U_n = _Uinv(A_n, B_n)
        for q in ("x","y","z"):
            cc = pair[f"cc_raw_len_{q}"][()]
            M_in_stack[q][i] = U_k.conj().T @ cc @ U_n
            D_stack[q][i]    = (f[f"per_ik/ik{ikn:04d}/D_ortho_len_{q}_re"][()]
                                + 1j*f[f"per_ik/ik{ikn:04d}/D_ortho_len_{q}_im"][()])

    # Interpolate to ν*
    M_in_at_star = {q: interp_along_nu(M_in_stack[q], eps_nu, eps_nu_star) for q in ("x","y","z")}
    D_at_star    = {q: interp_along_nu(D_stack[q],    eps_nu, eps_nu_star) for q in ("x","y","z")}

    # Spherical-tensor convert
    M_m1, M_0, M_p1 = cart_to_spherical(M_in_at_star["x"], M_in_at_star["y"], M_in_at_star["z"])
    M_sph = (M_m1, M_0, M_p1)
    D_m1, D_0, D_p1 = cart_to_spherical(D_at_star["x"], D_at_star["y"], D_at_star["z"])
    D_sph = (D_m1, D_0, D_p1)

    T = np.zeros((3, 3, N_psi, N_psi), dtype=np.complex128)
    for i_IR in range(3):
        for i_XUV in range(3):
            T[i_IR, i_XUV] = M_sph[i_IR] * D_sph[i_XUV][None, :]
    return T


def compute_M_avg(T_tilde, L_max, theta_grid, phi_grid, W_k, a_R, b_R, w_b, g_R):
    n_th, n_ph = len(theta_grid), len(phi_grid)
    n_a, n_b, n_g = len(a_R), len(b_R), len(g_R)
    M_arr = np.empty((n_th, n_ph, n_a, n_b, n_g), dtype=np.complex128)
    W = np.empty_like(M_arr, dtype=float)
    for it in range(n_th):
        for ip in range(n_ph):
            for ia in range(n_a):
                for ib in range(n_b):
                    for ig in range(n_g):
                        M_arr[it,ip,ia,ib,ig] = compute_M_one_path(
                            T_tilde, L_max,
                            theta_grid[it], phi_grid[ip],
                            a_R[ia], b_R[ib], g_R[ig], 0, 0)
                        W[it,ip,ia,ib,ig] = W_k[it,ip]/n_a*w_b[ib]/n_g
    return M_arr, W


def main():
    phase_a = "h2o_test/two_photon_me_h2o_cc_delay.h5"
    f = h5py.File(phase_a, "r")
    omega_IR = 1.55/27.211386
    AU_PER_AS = 24.188843
    L_max = int(f.attrs["l_cont"])

    theta_k, phi_k, W_k = angular_grid(6, 6)
    a_R = np.linspace(0, 2*np.pi, 6, endpoint=False)
    cb, wb = np.polynomial.legendre.leggauss(6); b_R = np.arccos(cb); wb = wb/2.0
    g_R = np.linspace(0, 2*np.pi, 6, endpoint=False)

    print("=== T5: ON-SHELL (T_L=∞) reference vs Dawson(trim, finite T_L) ===")
    print(f"{'κ':>4s}  {'E_κ (eV)':>10s}  {'τ on-shell (as)':>16s}  {'τ Dawson@5fs (as)':>20s}")
    for ik in (55, 60):
        nus = sorted({int(k[12:16]) for k in f["pairs"].keys()
                      if k.startswith(f"pair_k{ik:04d}")})
        if not nus:
            continue
        # on-shell
        T_less    = build_T_onshell(f, ik, nus, omega_IR, s_IR=+1)
        T_greater = build_T_onshell(f, ik, nus, omega_IR, s_IR=-1)
        M_l, W = compute_M_avg(T_less,    L_max, theta_k, phi_k, W_k, a_R, b_R, wb, g_R)
        M_g, _ = compute_M_avg(T_greater, L_max, theta_k, phi_k, W_k, a_R, b_R, wb, g_R)
        tau_onshell = tau_angle_averaged(M_l, M_g, omega_IR, weights=W) * AU_PER_AS

        # Dawson trimmed at T_L = 5 fs (production setting)
        import subprocess
        subprocess.check_call([
            "python3","postprocessing/two_photon_delay/python/run_cc_delay.py",
            "--phase-a", phase_a, "--out", "/tmp/dawson_compare.h5",
            "--sidebands", str(ik), "--omega-IR-eV","1.55",
            "--T-X-fs","0.35","--T-L-fs","5.0","--tau-delay-fs","0.0",
            "--angle-n-theta","6","--angle-n-phi","6",
            "--orient-n-alpha","6","--orient-n-beta","6","--orient-n-gamma","6",
            "--quiet"], stdout=subprocess.DEVNULL)
        with h5py.File("/tmp/dawson_compare.h5","r") as g:
            tau_dawson = float(g["sideband_000/tau_avg"][()]) * AU_PER_AS

        dk = float(f.attrs["dk"])
        E_kin = _energy(ik, dk) * 27.2114
        print(f"  {ik:>4d}  {E_kin:>10.3f}  {tau_onshell:>+16.3f}  {tau_dawson:>+20.3f}")


if __name__ == "__main__":
    main()
