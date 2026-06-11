#!/usr/bin/env python3
"""
run_cc_delay.py
---------------
Top-level orchestrator: consolidated Phase A HDF5 + RABBITT pulse params
→ τ_2hω(ε_κ; k̂, R̂_γ) plus the angle-averaged τ_2hω(ε_κ).

Calling sequence per sideband:

    1. For each pol q ∈ {x, y, z}:
         - Rotate cc_raw[ik_κ, ik_ν] to in-state basis (rotate_to_instate).
         - Pull D_ortho_len_q at each ik_ν.
    2. Convert (M_in^x, M_in^y, M_in^z) → spherical M_in^{m_p^IR_mol}.
       Same for (D^x, D^y, D^z) → D^{m_p^XUV_mol}.
    3. For each path (< = absorb-XUV-then-absorb-IR, > = absorb-then-emit):
         For each (m_p^IR_mol, m_p^XUV_mol) ∈ (-1, 0, +1)²:
           T̃[m_IR, m_XUV, μ_κ, μ_ν]
              = Σ_ν M_in[m_IR][μ_κ, μ_ν] · D[m_XUV][μ_ν]
                    · K_LX(β_path, α_path; τ, 0, T_L, T_X) · dε_ν
       → compute_M_one_path → M_<, M_>.
    4. τ_2hω(k̂, R̂_γ) = (1/2ω) arg{M_< M_>*}.
    5. Angle-averaged: coherent average  arg<M_< M_>*>_(k̂, R̂_γ).

The intermediate-energy ν integration uses the trapezoid weight `dk·k`
(uniform-in-k scan converted to uniform-in-ε). Adapt the window per κ.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

from compute_M import (
    angular_grid,
    compute_M_one_path,
)
from compute_T import compute_T_path
from m_p_polarization import cart_to_spherical
from rotate_to_instate import _Uinv, rotate_pair
from tau_2homega import tau_angle_averaged, tau_pointwise


# ---------------------------------------------------------------------------
# Phase A HDF5 helpers
# ---------------------------------------------------------------------------
def _list_ik(f: h5py.File) -> List[int]:
    """Return sorted list of available ik in /per_ik/."""
    return sorted(int(k[2:]) for k in f["per_ik"].keys() if k.startswith("ik"))


def _list_pairs(f: h5py.File) -> List[Tuple[int, int]]:
    """Return sorted list of (ik_kappa, ik_nu) pairs in /pairs/."""
    out = []
    for k in f["pairs"].keys():
        # pair_kNNNN_nMMMM
        ks = int(k[6:10])
        ns = int(k[12:16])
        out.append((ks, ns))
    return sorted(out)


def _load_per_ik_AB_D(f: h5py.File, ik: int, gauge: str = "len"
                      ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Return (A, B, {pol: D_complex}) for one ik."""
    g = f[f"/per_ik/ik{ik:04d}"]
    A = g["A"][()]
    B = g["B"][()]
    D = {}
    for q in ("x", "y", "z"):
        re = g[f"D_ortho_{gauge}_{q}_re"][()]
        im = g[f"D_ortho_{gauge}_{q}_im"][()]
        D[q] = re + 1j * im
    return A, B, D


def _energy_au(ik: int, dk: float) -> float:
    """ε(ik) = (ik·dk)² / 2."""
    return (ik * dk) ** 2 / 2.0


# ---------------------------------------------------------------------------
# Per-sideband driver
# ---------------------------------------------------------------------------
def build_T_tilde_one_path(
    M_in_list_xyz: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    D_list_xyz:    List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    eps_nu: np.ndarray,
    weights: np.ndarray,
    eps_kappa: float,
    E_EA: float,
    Omega_XUV: float,
    omega_IR: float,
    ir_action: str,
    T_X: float,
    T_L: float,
    tau_delay: float,
    symmetric_trim_sigma: float | None = 8.0,
) -> np.ndarray:
    """Return T̃[3, 3, N_psi, N_psi] with axes (m_IR_mol, m_XUV_mol, μ_κ, μ_ν).

    Inputs:
      M_in_list_xyz[i_nu]  = (M^x, M^y, M^z)   each (N_psi, N_psi) complex
      D_list_xyz[i_nu]     = (D^x, D^y, D^z)   each (N_psi,)        complex
    """
    n_nu = len(eps_nu)
    N_psi = M_in_list_xyz[0][0].shape[0]

    # Spherical-tensor lists per (m_IR_mol or m_XUV_mol).
    # Index 0 = m=-1, 1 = m=0, 2 = m=+1.
    M_in_sph_per_nu = [None] * n_nu
    D_sph_per_nu    = [None] * n_nu
    for inu in range(n_nu):
        Mx, My, Mz = M_in_list_xyz[inu]
        M_m1, M_0, M_p1 = cart_to_spherical(Mx, My, Mz)
        M_in_sph_per_nu[inu] = (M_m1, M_0, M_p1)
        Dx, Dy, Dz = D_list_xyz[inu]
        D_m1, D_0, D_p1 = cart_to_spherical(Dx, Dy, Dz)
        D_sph_per_nu[inu]   = (D_m1, D_0, D_p1)

    T_tilde = np.zeros((3, 3, N_psi, N_psi), dtype=np.complex128)
    for i_IR in range(3):       # m_p^IR_mol = i_IR - 1
        for i_XUV in range(3):  # m_p^XUV_mol = i_XUV - 1
            M_in_list = [M_in_sph_per_nu[inu][i_IR] for inu in range(n_nu)]
            D_list    = [D_sph_per_nu[inu][i_XUV]   for inu in range(n_nu)]
            T_tilde[i_IR, i_XUV] = compute_T_path(
                M_in_list=M_in_list, D_list=D_list, eps_nu=eps_nu,
                eps_kappa=eps_kappa, E_EA=E_EA,
                Omega_XUV=Omega_XUV, omega_IR=omega_IR, ir_action=ir_action,
                T_X=T_X, T_L=T_L, tau=tau_delay, weights=weights,
                symmetric_trim_sigma=symmetric_trim_sigma,
            )
    return T_tilde


def sideband_one(
    f: h5py.File,
    ik_kappa: int,
    ik_nu_list: List[int],
    omega_IR: float,
    T_X: float,
    T_L: float,
    tau_delay: float,
    angle_n_theta: int,
    angle_n_phi: int,
    orient_n_alpha: int,
    orient_n_beta: int,
    orient_n_gamma: int,
    m_p_IR_lab: int = 0,
    m_p_XUV_lab: int = 0,
    Omega_XUV_less: float | None = None,
    Omega_XUV_greater: float | None = None,
    symmetric_trim_sigma: float | None = 8.0,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute τ_2hω(k̂, R̂_γ) at a single sideband.  Returns a dict with
    τ array, angle grids, |M_<|, |M_>|."""
    dk = f.attrs["dk"]
    E_HOMO = f.attrs["E_HOMO"]
    L_max = int(f.attrs["l_cont"])
    N_psi = int(f.attrs["N_psi"])

    E_EA = -float(E_HOMO)      # binding-energy magnitude
    eps_kappa = _energy_au(ik_kappa, dk)

    # XUV harmonic energies: < path absorbs Ω_{2q-1}, > path absorbs Ω_{2q+1}.
    # Defaults: pin them so the on-shell intermediate sits at ε_κ ∓ ω_IR
    # (i.e. Ω_(2q±1) = ε_κ + E_EA ∓ ω_IR).  This is the conventional choice
    # for a RABBITT sideband and means the user only has to pass ω_IR.
    if Omega_XUV_less is None:
        Omega_XUV_less = eps_kappa + E_EA - omega_IR        # (2q-1)·ω
    if Omega_XUV_greater is None:
        Omega_XUV_greater = eps_kappa + E_EA + omega_IR     # (2q+1)·ω

    # --- collect all required (κ, ν) pairs and the per-ν b-c dipoles ----
    # Verify all needed pairs exist + load.
    pair_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    A_kappa, B_kappa, _ = _load_per_ik_AB_D(f, ik_kappa)
    U_kappa = _Uinv(A_kappa, B_kappa)
    for ik_nu in ik_nu_list:
        pair_path = f"/pairs/pair_k{ik_kappa:04d}_n{ik_nu:04d}"
        if pair_path not in f:
            raise KeyError(f"Phase A HDF5 missing pair {pair_path}")
        pg = f[pair_path]
        cc_x = pg["cc_raw_len_x"][()]
        cc_y = pg["cc_raw_len_y"][()]
        cc_z = pg["cc_raw_len_z"][()]
        A_nu, B_nu, _ = _load_per_ik_AB_D(f, ik_nu)
        U_nu = _Uinv(A_nu, B_nu)
        M_in_x = U_kappa.conj().T @ cc_x @ U_nu
        M_in_y = U_kappa.conj().T @ cc_y @ U_nu
        M_in_z = U_kappa.conj().T @ cc_z @ U_nu
        pair_data[ik_nu] = (M_in_x, M_in_y, M_in_z)

    # Per-ν b-c dipoles (D_ortho_len).
    D_per_nu: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ik_nu in ik_nu_list:
        _, _, D_xyz = _load_per_ik_AB_D(f, ik_nu)
        D_per_nu[ik_nu] = (D_xyz["x"], D_xyz["y"], D_xyz["z"])

    # ε_ν grid and integration weights (uniform-in-k → dε = k·dk).
    ik_nu_sorted = sorted(ik_nu_list)
    eps_nu_full = np.array([_energy_au(ikn, dk) for ikn in ik_nu_sorted])
    k_nu_full   = np.array([ikn * dk for ikn in ik_nu_sorted])
    # Trapezoid weights on a uniform-in-ε rule (since dk is uniform, dε is
    # NOT uniform; use trapezoidal in ε).
    if len(eps_nu_full) >= 2:
        w_full = np.zeros_like(eps_nu_full)
        w_full[0]    = (eps_nu_full[1] - eps_nu_full[0]) / 2.0
        w_full[-1]   = (eps_nu_full[-1] - eps_nu_full[-2]) / 2.0
        w_full[1:-1] = (eps_nu_full[2:] - eps_nu_full[:-2]) / 2.0
    else:
        w_full = np.ones_like(eps_nu_full)  # 1-pt fallback (smoke test only)

    M_in_xyz_list = [pair_data[ikn] for ikn in ik_nu_sorted]
    D_xyz_list    = [D_per_nu[ikn]  for ikn in ik_nu_sorted]

    # T̃ for each path.
    if verbose:
        print(f"  sideband κ: ik={ik_kappa}  ε_κ={eps_kappa:.6f} au "
              f"({eps_kappa*27.2114:.4f} eV)")
        print(f"    ν grid: {len(eps_nu_full)} points  ε_ν ∈ "
              f"[{eps_nu_full.min():.4f}, {eps_nu_full.max():.4f}] au")
        print(f"    Ω_<: {Omega_XUV_less:.6f} au   Ω_>: {Omega_XUV_greater:.6f} au")
    T_less = build_T_tilde_one_path(
        M_in_xyz_list, D_xyz_list, eps_nu_full, w_full,
        eps_kappa, E_EA, Omega_XUV_less, omega_IR, "absorb",
        T_X, T_L, tau_delay, symmetric_trim_sigma=symmetric_trim_sigma,
    )
    T_greater = build_T_tilde_one_path(
        M_in_xyz_list, D_xyz_list, eps_nu_full, w_full,
        eps_kappa, E_EA, Omega_XUV_greater, omega_IR, "emit",
        T_X, T_L, tau_delay, symmetric_trim_sigma=symmetric_trim_sigma,
    )

    # Angle grids.
    theta_k, phi_k, W_k = angular_grid(angle_n_theta, angle_n_phi)
    alpha_R = np.linspace(0, 2 * np.pi, orient_n_alpha, endpoint=False)
    cos_b, w_b = np.polynomial.legendre.leggauss(orient_n_beta)
    beta_R    = np.arccos(cos_b)
    w_b       = w_b / 2.0
    gamma_R = np.linspace(0, 2 * np.pi, orient_n_gamma, endpoint=False)
    w_alpha = 1.0 / orient_n_alpha
    w_gamma = 1.0 / orient_n_gamma

    # M arrays of shape (n_th, n_ph, n_a, n_b, n_g)
    n_th, n_ph = len(theta_k), len(phi_k)
    n_a, n_b, n_g = orient_n_alpha, orient_n_beta, orient_n_gamma
    M_less_arr    = np.empty((n_th, n_ph, n_a, n_b, n_g), dtype=np.complex128)
    M_greater_arr = np.empty_like(M_less_arr)
    W_full        = np.empty_like(M_less_arr, dtype=float)

    for it in range(n_th):
        for ip in range(n_ph):
            for ia in range(n_a):
                for ib in range(n_b):
                    for ig in range(n_g):
                        Ml = compute_M_one_path(
                            T_less, L_max, theta_k[it], phi_k[ip],
                            alpha_R[ia], beta_R[ib], gamma_R[ig],
                            m_p_IR_lab, m_p_XUV_lab,
                        )
                        Mg = compute_M_one_path(
                            T_greater, L_max, theta_k[it], phi_k[ip],
                            alpha_R[ia], beta_R[ib], gamma_R[ig],
                            m_p_IR_lab, m_p_XUV_lab,
                        )
                        M_less_arr[it, ip, ia, ib, ig]    = Ml
                        M_greater_arr[it, ip, ia, ib, ig] = Mg
                        W_full[it, ip, ia, ib, ig] = (
                            W_k[it, ip] * w_alpha * w_b[ib] * w_gamma)

    tau_grid = tau_pointwise(M_less_arr, M_greater_arr, omega_IR)
    tau_avg  = tau_angle_averaged(M_less_arr, M_greater_arr, omega_IR,
                                  weights=W_full)

    return {
        "eps_kappa":      np.array(eps_kappa),
        "tau_avg":        np.array(tau_avg),
        "tau_grid":       tau_grid,
        "M_less":         M_less_arr,
        "M_greater":      M_greater_arr,
        "W_grid":         W_full,
        "theta_k":        theta_k,
        "phi_k":          phi_k,
        "alpha_R":        alpha_R,
        "beta_R":         beta_R,
        "gamma_R":        gamma_R,
        "ik_kappa":       np.array(ik_kappa),
        "ik_nu_list":     np.array(ik_nu_sorted),
        "eps_nu":         eps_nu_full,
        "w_nu":           w_full,
        "Omega_XUV_less":    np.array(Omega_XUV_less),
        "Omega_XUV_greater": np.array(Omega_XUV_greater),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase-a", required=True, type=Path,
                    help="consolidated Phase A HDF5")
    ap.add_argument("--out", required=True, type=Path,
                    help="output HDF5 with τ arrays")
    ap.add_argument("--sidebands",
                    help="comma-separated list of ik_kappa values "
                         "(default: every ik present in /pairs/)")
    ap.add_argument("--nu-window", type=int, default=10,
                    help="number of ik_ν points on EACH side of the on-shell "
                         "ν for each path (default 10)")
    ap.add_argument("--omega-IR-eV", type=float, default=1.55,
                    help="IR carrier frequency in eV (default 1.55)")
    ap.add_argument("--T-X-fs", type=float, default=0.35,
                    help="XUV pulse FWHM in fs (default 0.35)")
    ap.add_argument("--T-L-fs", type=float, default=5.0,
                    help="IR pulse FWHM in fs (default 5.0)")
    ap.add_argument("--tau-delay-fs", type=float, default=1.0,
                    help="IR-XUV delay τ in fs (default 1.0)")
    ap.add_argument("--angle-n-theta", type=int, default=4)
    ap.add_argument("--angle-n-phi",   type=int, default=4)
    ap.add_argument("--orient-n-alpha", type=int, default=4)
    ap.add_argument("--orient-n-beta",  type=int, default=4)
    ap.add_argument("--orient-n-gamma", type=int, default=4)
    ap.add_argument("--m-p-IR-lab",  type=int, default=0, choices=(-1, 0, 1))
    ap.add_argument("--m-p-XUV-lab", type=int, default=0, choices=(-1, 0, 1))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    HBAR_EV_FS = 0.6582119514
    EV_PER_AU = 27.2114
    AU_PER_FS = 41.341
    omega_IR_au = args.omega_IR_eV / EV_PER_AU
    T_X_au = (args.T_X_fs * AU_PER_FS) / np.sqrt(2 * np.log(2))
    T_L_au = (args.T_L_fs * AU_PER_FS) / np.sqrt(2 * np.log(2))
    tau_delay_au = args.tau_delay_fs * AU_PER_FS

    if not args.phase_a.exists():
        print(f"error: phase-a HDF5 not found: {args.phase_a}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.phase_a, "r") as f:
        all_pairs = _list_pairs(f)
        kappas_in_pairs = sorted({k for k, _ in all_pairs})

        if args.sidebands:
            sidebands = [int(s) for s in args.sidebands.split(",")]
        else:
            sidebands = kappas_in_pairs

        if not args.quiet:
            print(f"# Phase A: {args.phase_a}")
            print(f"#   ik present in /per_ik/  : {_list_ik(f)}")
            print(f"#   pairs present           : {len(all_pairs)}")
            print(f"#   sidebands chosen        : {sidebands}")
            print(f"#   ω_IR = {args.omega_IR_eV} eV  = {omega_IR_au:.6f} au")
            print(f"#   T_X  = {args.T_X_fs} fs ({T_X_au:.4g} au)   "
                  f"T_L = {args.T_L_fs} fs ({T_L_au:.4g} au)")
            print(f"#   IR delay τ = {args.tau_delay_fs} fs  "
                  f"angle grid θ×φ×α×β×γ = "
                  f"{args.angle_n_theta}×{args.angle_n_phi}×{args.orient_n_alpha}"
                  f"×{args.orient_n_beta}×{args.orient_n_gamma}")

        # For each sideband, figure out the ν window that is ACTUALLY
        # present in /pairs/ AND covers both paths' on-shell points.
        results = []
        for ik_k in sidebands:
            available_nus = sorted({n for k, n in all_pairs if k == ik_k})
            if not available_nus:
                print(f"  WARN: ik_κ={ik_k} has no pairs; skipping")
                continue
            res = sideband_one(
                f, ik_k, available_nus, omega_IR_au,
                T_X_au, T_L_au, tau_delay_au,
                args.angle_n_theta, args.angle_n_phi,
                args.orient_n_alpha, args.orient_n_beta, args.orient_n_gamma,
                m_p_IR_lab=args.m_p_IR_lab, m_p_XUV_lab=args.m_p_XUV_lab,
                verbose=not args.quiet,
            )
            tau_avg_au = float(res["tau_avg"])
            tau_avg_as = tau_avg_au * 24.188843
            if not args.quiet:
                print(f"    -> τ_2hω(avg) = {tau_avg_au:+.6e} au  "
                      f"= {tau_avg_as:+.4f} as")
            results.append(res)

        # Write output.
        with h5py.File(args.out, "w") as g:
            for key in ("E_HOMO", "dk", "l_cont", "N_psi", "molecule_name",
                         "scan_id"):
                if key in f.attrs:
                    g.attrs[key] = f.attrs[key]
            g.attrs["omega_IR_eV"]   = args.omega_IR_eV
            g.attrs["omega_IR_au"]   = omega_IR_au
            g.attrs["T_X_fs"]        = args.T_X_fs
            g.attrs["T_L_fs"]        = args.T_L_fs
            g.attrs["T_X_au"]        = T_X_au
            g.attrs["T_L_au"]        = T_L_au
            g.attrs["tau_delay_fs"]  = args.tau_delay_fs
            g.attrs["m_p_IR_lab"]    = args.m_p_IR_lab
            g.attrs["m_p_XUV_lab"]   = args.m_p_XUV_lab
            for i, res in enumerate(results):
                sg = g.create_group(f"sideband_{i:03d}")
                for k, v in res.items():
                    sg.create_dataset(k, data=v)

        if not args.quiet:
            print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
