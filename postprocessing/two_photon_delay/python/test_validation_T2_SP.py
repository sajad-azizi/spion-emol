#!/usr/bin/env python3
"""
test_validation_T2_SP.py
-------------------------
Validation T2: cross-check the finite-pulse Dawson kernel against a
DIRECT Sokhotski-Plemelj evaluation as an independent code path.

BW17 amplitude in the long-pulse / monochromatic limit:

    T̃_SP[μ_κ, μ_ν]^path  =  ε→0+  ∫ dε_ν'  M_in[κ,ν'][μ_κ, μ_ν'] · D[ν'][μ_ν']
                                          / (ε_κ − ε_ν' − s·ω + iε)

The integral is evaluated numerically with a small but finite ε (regulator)
that we sweep down to verify ε→0 convergence.  Then we compare the
resulting τ_2ℏω against the finite-pulse Dawson result at the same
sideband for a sweep of T_L values.

Predict: τ_Dawson(T_L) → τ_SP as T_L → ∞.

This validates the FULL pipeline (rotate_to_instate → compute_T →
compute_M → tau) end-to-end on REAL H₂O data using a second
independent kernel formulation.
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


def _load_per_ik_AB_D(f, ik, gauge="len"):
    g = f[f"/per_ik/ik{ik:04d}"]
    A = g["A"][()]; B = g["B"][()]
    D = {}
    for q in ("x","y","z"):
        D[q] = g[f"D_ortho_{gauge}_{q}_re"][()] + 1j*g[f"D_ortho_{gauge}_{q}_im"][()]
    return A, B, D


def _energy(ik, dk):
    return (ik*dk)**2/2.0


def build_T_SP(f, ik_kappa, ik_nu_list, omega_IR, s_IR: int, eps_regularize: float):
    """Direct SP integration.  Returns T̃[3,3,N_psi,N_psi] for one path.

    s_IR = +1 (absorb, '<')  or  −1 (emit, '>').
    """
    dk = float(f.attrs["dk"])
    N_psi = int(f.attrs["N_psi"])
    A_k, B_k, _ = _load_per_ik_AB_D(f, ik_kappa)
    U_k = _Uinv(A_k, B_k)

    eps_kappa = _energy(ik_kappa, dk)
    # SP denominator pole at ε_ν = ε_κ − s·ω.
    # We do a TRAPEZOIDAL integral on the available ν grid; the small
    # iε regulator makes the integrand finite and well-behaved.
    eps_nu = np.array([_energy(ikn, dk) for ikn in ik_nu_list])
    if len(eps_nu) >= 2:
        w_full = np.zeros_like(eps_nu)
        w_full[0]    = (eps_nu[1]-eps_nu[0])/2
        w_full[-1]   = (eps_nu[-1]-eps_nu[-2])/2
        w_full[1:-1] = (eps_nu[2:]-eps_nu[:-2])/2
    else:
        raise ValueError("need ≥ 2 ν points")

    T = np.zeros((3, 3, N_psi, N_psi), dtype=np.complex128)
    for i, ik_nu in enumerate(ik_nu_list):
        # M_in (Cartesian)
        pair = f[f"/pairs/pair_k{ik_kappa:04d}_n{ik_nu:04d}"]
        cc = {q: pair[f"cc_raw_len_{q}"][()] for q in ("x","y","z")}
        A_n, B_n, D_xyz = _load_per_ik_AB_D(f, ik_nu)
        U_n = _Uinv(A_n, B_n)
        M_in_xyz = {q: U_k.conj().T @ cc[q] @ U_n for q in ("x","y","z")}

        # Convert M_in_xyz → M_in_sph[m]
        M_m1, M_0, M_p1 = cart_to_spherical(M_in_xyz["x"], M_in_xyz["y"], M_in_xyz["z"])
        M_sph = (M_m1, M_0, M_p1)
        # Convert D_xyz → D_sph[m]
        D_m1, D_0, D_p1 = cart_to_spherical(D_xyz["x"], D_xyz["y"], D_xyz["z"])
        D_sph = (D_m1, D_0, D_p1)

        # SP-style denominator with small regulator
        denom = (eps_kappa - eps_nu[i] - s_IR * omega_IR + 1j * eps_regularize)
        scale = (1.0 / denom) * w_full[i]
        for i_IR in range(3):
            for i_XUV in range(3):
                T[i_IR, i_XUV] += M_sph[i_IR] * D_sph[i_XUV][None, :] * scale
    return T


def compute_M_avg_from_T(T_tilde, L_max, theta_grid, phi_grid, W_k,
                         alpha_grid, beta_grid, w_b, gamma_grid):
    """Wrap compute_M over the angle grid and return ⟨M⟩_coherent and weights."""
    n_th, n_ph = len(theta_grid), len(phi_grid)
    n_a, n_b, n_g = len(alpha_grid), len(beta_grid), len(gamma_grid)
    M_arr = np.empty((n_th, n_ph, n_a, n_b, n_g), dtype=np.complex128)
    W = np.empty_like(M_arr, dtype=float)
    w_a = 1.0/n_a; w_g = 1.0/n_g
    for it in range(n_th):
        for ip in range(n_ph):
            for ia in range(n_a):
                for ib in range(n_b):
                    for ig in range(n_g):
                        M_arr[it,ip,ia,ib,ig] = compute_M_one_path(
                            T_tilde, L_max,
                            theta_grid[it], phi_grid[ip],
                            alpha_grid[ia], beta_grid[ib], gamma_grid[ig],
                            0, 0)
                        W[it,ip,ia,ib,ig] = W_k[it,ip]*w_a*w_b[ib]*w_g
    return M_arr, W


def main():
    phase_a = "h2o_test/two_photon_me_h2o_cc_delay.h5"
    f = h5py.File(phase_a, "r")
    ik_kappa = 60
    omega_IR = 1.55/27.211386
    AU_PER_AS = 24.188843

    # Available ν list for this κ:
    nus = sorted({int(k[12:16]) for k in f["pairs"].keys()
                  if k.startswith(f"pair_k{ik_kappa:04d}")})
    print(f"ν list for κ={ik_kappa}: {nus[:5]} ... {nus[-3:]} ({len(nus)} total)")

    L_max = int(f.attrs["l_cont"])
    theta_k, phi_k, W_k = angular_grid(3, 3)
    a_R = np.linspace(0, 2*np.pi, 3, endpoint=False)
    cb, wb = np.polynomial.legendre.leggauss(3)
    b_R = np.arccos(cb); wb = wb/2.0
    g_R = np.linspace(0, 2*np.pi, 3, endpoint=False)

    print("\n=== SP integration: ε regulator sweep (κ=60) ===")
    for eps in (5e-3, 2e-3, 1e-3, 5e-4, 2e-4):
        T_less    = build_T_SP(f, ik_kappa, nus, omega_IR, s_IR=+1, eps_regularize=eps)
        T_greater = build_T_SP(f, ik_kappa, nus, omega_IR, s_IR=-1, eps_regularize=eps)
        M_l, W = compute_M_avg_from_T(T_less,    L_max, theta_k, phi_k, W_k, a_R, b_R, wb, g_R)
        M_g, _ = compute_M_avg_from_T(T_greater, L_max, theta_k, phi_k, W_k, a_R, b_R, wb, g_R)
        tau_au = tau_angle_averaged(M_l, M_g, omega_IR, weights=W)
        print(f"  ε = {eps:.0e}   τ_SP = {tau_au * AU_PER_AS:+.3f} as   "
              f"|⟨M_<⟩|={abs((W*M_l).sum()/W.sum()):.3e}  "
              f"|⟨M_>⟩|={abs((W*M_g).sum()/W.sum()):.3e}")

    # Now compare to Dawson finite-pulse at same κ, sweeping T_L
    print("\n=== Dawson finite-pulse τ vs T_L for SAME κ=60 (trim ON) ===")
    import subprocess
    for TL in (1.0, 5.0, 10.0, 30.0, 60.0):
        out = "/tmp/scan_T_L.h5"
        subprocess.check_call([
            "python3","postprocessing/two_photon_delay/python/run_cc_delay.py",
            "--phase-a", phase_a, "--out", out,
            "--sidebands", str(ik_kappa),
            "--omega-IR-eV", "1.55", "--T-X-fs", "0.35",
            "--T-L-fs", str(TL), "--tau-delay-fs", "0.0",
            "--angle-n-theta", "3", "--angle-n-phi", "3",
            "--orient-n-alpha", "3", "--orient-n-beta", "3",
            "--orient-n-gamma", "3", "--quiet"],
            stdout=subprocess.DEVNULL)
        with h5py.File(out, "r") as g:
            t = float(g["sideband_000/tau_avg"][()])
        print(f"  T_L = {TL:5.1f} fs   τ_Dawson = {t*AU_PER_AS:+.3f} as")


if __name__ == "__main__":
    main()
