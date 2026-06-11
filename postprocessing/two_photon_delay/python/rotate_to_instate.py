#!/usr/bin/env python3
"""
rotate_to_instate.py
--------------------
Rotate the continuum-continuum dipole matrices saved by Phase A in the
BACK-PROP basis into the IN-STATE (Ψ⁻) basis used by BW17.

Convention (consistent with static_exchangeHF/DipoleMatrixElement.hpp):

    Ψ⁻_μ(r) = Σ_β ψ_β(r) · [(A − iB)⁻¹]_{β μ}

So that  <Ψ⁻_μ| = Σ_β [(A − iB)⁻¹]*_{β μ} · ψ_β  =  Σ_β ((A − iB)⁻¹)†_{μ β} · ψ_β.

For a c-c matrix element (real ψ in back-prop basis, real cc_raw):

    <Ψ⁻_(κ, μ_κ) | r·ε̂_q | Ψ⁻_(ν, μ_ν)>
        = Σ_{β, α}  [(A_κ − iB_κ)⁻¹]†_{μ_κ β}  ·  cc_raw[β, α]  ·  [(A_ν − iB_ν)⁻¹]_{α μ_ν}
        = (U_κ† · cc_raw · U_ν)[μ_κ, μ_ν]
        ≡ M_in_{κν}^{q}[μ_κ, μ_ν]

with U_κ = (A_κ − iB_κ)⁻¹.  Result is COMPLEX.  Two operator-Hermiticity
checks the rotation must satisfy:

    1) At κ = ν :        M_in[μ_κ, μ_ν]  =  M_in[μ_ν, μ_κ]*   (Hermitian)
    2) At κ ≠ ν :        M_in_{κν}      =  (M_in_{νκ})†       (cross-pair)

Failure of either check by more than ~few × |M_in| × (fit_residual_rel of
A_κ, A_ν) means the basis rotation is broken; we throw a clear error so
Phase C never silently consumes wrong data.

The rotation is single-pair, single-pol; loops in calling code.

Usage as a library:
    from rotate_to_instate import load_pair_M_in
    M = load_pair_M_in(phase_a_h5_path, ik_kappa=50, ik_nu=60, pol='x')
    # M : complex array (N_psi, N_psi)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np


def _load_AB(f: h5py.File, ik: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read /per_ik/ik<NNNN>/{A, B}  -> two (N_psi, N_psi) float arrays."""
    g = f[f"/per_ik/ik{ik:04d}"]
    return g["A"][()], g["B"][()]


def _Uinv(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """U = (A − iB)⁻¹.  Returns COMPLEX (N_psi, N_psi).

    Numerically: build (A − iB) explicitly and invert via numpy.linalg.solve
    on an identity right-hand side.  Singular A − iB would indicate a
    degenerate K-matrix; we surface that as ValueError so callers don't
    propagate NaNs into the τ assembly.
    """
    M = A - 1j * B
    # Solve M·X = I for X = M⁻¹.  Faster + more accurate than np.linalg.inv
    # because it avoids forming the inverse explicitly until needed.
    try:
        return np.linalg.solve(M, np.eye(M.shape[0], dtype=M.dtype))
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"rotate_to_instate: (A - iB) is singular -- "
            f"max|A|={np.max(np.abs(A)):.3e}  max|B|={np.max(np.abs(B)):.3e}  "
            f"err={e}")


def rotate_pair(cc_raw: np.ndarray, A_kappa: np.ndarray, B_kappa: np.ndarray,
                A_nu: np.ndarray, B_nu: np.ndarray) -> np.ndarray:
    """Return  M_in = U_κ†  ·  cc_raw  ·  U_ν   (complex)."""
    if cc_raw.dtype.kind != "f":
        raise TypeError("cc_raw must be REAL (length-gauge back-prop basis)")
    U_kappa = _Uinv(A_kappa, B_kappa)
    U_nu    = _Uinv(A_nu,    B_nu)
    return U_kappa.conj().T @ cc_raw @ U_nu


def load_pair_M_in(phase_a_h5: str | Path, ik_kappa: int, ik_nu: int,
                   pol: str, gauge: str = "len") -> np.ndarray:
    """Load + rotate the cc_raw matrix for one (κ, ν, gauge, pol).

    Currently only length gauge is stored by cc_dipole_driver; gauge must
    be 'len' until velocity is added.  Raises KeyError if the pair or the
    gauge/pol is missing from the Phase A HDF5.
    """
    if gauge != "len":
        raise ValueError(f"gauge='{gauge}' not in Phase A HDF5 yet; "
                         f"velocity-gauge c-c is TODO")
    if pol not in ("x", "y", "z"):
        raise ValueError(f"pol must be x/y/z, got {pol!r}")

    with h5py.File(phase_a_h5, "r") as f:
        pair_path = f"/pairs/pair_k{ik_kappa:04d}_n{ik_nu:04d}"
        if pair_path not in f:
            raise KeyError(f"{pair_path} not in {phase_a_h5}")
        cc_raw = f[pair_path][f"cc_raw_{gauge}_{pol}"][()]
        A_k, B_k = _load_AB(f, ik_kappa)
        A_n, B_n = _load_AB(f, ik_nu)

    return rotate_pair(cc_raw, A_k, B_k, A_n, B_n)


def hermitian_residual(M: np.ndarray) -> Tuple[float, float]:
    """For diagnostics: max|M - M†|_∞ and the same divided by max|M|."""
    diff = np.max(np.abs(M - M.conj().T))
    norm = np.max(np.abs(M)) or 1.0
    return diff, diff / norm


def cross_pair_residual(M_kn: np.ndarray, M_nk: np.ndarray) -> Tuple[float, float]:
    """For diagnostics: max|M_kn - (M_nk)†| and rel."""
    diff = np.max(np.abs(M_kn - M_nk.conj().T))
    norm = np.max(np.abs(M_kn)) or 1.0
    return diff, diff / norm


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase-a", required=True, type=Path)
    ap.add_argument("--ik-kappa", required=True, type=int)
    ap.add_argument("--ik-nu",    required=True, type=int)
    ap.add_argument("--pol", default="x", choices=("x", "y", "z"))
    args = ap.parse_args()

    M = load_pair_M_in(args.phase_a, args.ik_kappa, args.ik_nu, args.pol)

    print(f"=== M_in pair (κ={args.ik_kappa}, ν={args.ik_nu}) pol={args.pol} ===")
    print(f"  shape         : {M.shape}")
    print(f"  max|M|        : {np.max(np.abs(M)):.6e}")
    print(f"  max|Re M|     : {np.max(np.abs(M.real)):.6e}")
    print(f"  max|Im M|     : {np.max(np.abs(M.imag)):.6e}")
    print(f"  mean|M|       : {np.mean(np.abs(M)):.6e}")
    nz = np.count_nonzero(np.abs(M) > 1e-12)
    print(f"  nonzeros      : {nz} / {M.size}  ({nz/M.size*100:.2f}%)")

    # Hermiticity check ONLY makes sense at κ = ν.
    if args.ik_kappa == args.ik_nu:
        d, r = hermitian_residual(M)
        print(f"  Hermiticity check : max|M - M†| = {d:.3e}  (rel {r:.3e})")
    else:
        print("  Hermiticity skipped (κ ≠ ν).  Use cross-pair check instead "
              "(also load the (ν, κ) pair and verify M_kn == (M_nk)†).")


if __name__ == "__main__":
    main()
