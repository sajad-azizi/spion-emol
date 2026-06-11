"""
compute_T.py
------------
m-resolved RABBITT kernel summation per derivation §(D7h):

    T̃_{μ_κ, μ_ν}^{m_p^IR, m_p^XUV}(κ; τ, path)
        =  Σ_{ν ∈ scan}  M_in^{m_p^IR}_{κν}[μ_κ, μ_ν]
                       · D^{m_p^XUV}_{ν, μ_ν}
                       · K_LX(β_path, α_path; τ, 0, T_L, T_X)
                       · w_ν

where the two-pulse Gaussian kernel  K_LX  is the closed form from the
user's derivation in ``docs/a test derivation/`` and is implemented in
``dawson_kernel_rabbitt.kernel_erf``.

Detunings (user derivation §2, §3):
    XUV step (absorb):    α = ε_ν + E_EA − Ω_XUV
    IR step  (absorb):    β = ε_κ − ε_ν − ω        ('<' path, sideband 2q)
    IR step  (emit):      β = ε_κ − ε_ν + ω        ('>' path, sideband 2q)

The IR pulse is centred at delay τ (label `tau_b = τ`); the XUV pulse is
centred at τ = 0 (label `tau_a = 0`).  T_a = T_X (XUV), T_b = T_L (IR).

This module does NOT do any HDF5 IO -- the caller is expected to load
M_in and D from the consolidated Phase A file (see
``phase_a_assembler.py`` and ``docs/phase_a_hdf5_layout.md``) and rotate
to the in-state basis using ``rotate_to_instate.rotate_pair``.

Sanity-check entry: ``_self_test()``.  Reduces to a single-channel
analytic check that matches the user's adaptive-quadrature reference
(``dawson_kernel_rabbitt.kernel_quad``) to machine precision.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from dawson_kernel_rabbitt import (
    kernel_erf,
    rabbitt_detunings,
)


# ---------------------------------------------------------------------------
# Kernel values on the scan ε_ν grid
# ---------------------------------------------------------------------------
def kernel_grid(
    eps_nu: np.ndarray,
    eps_kappa: float,
    E_EA: float,
    Omega_XUV: float,
    omega_IR: float,
    ir_action: str,
    T_X: float,
    T_L: float,
    tau: float,
) -> np.ndarray:
    """Evaluate K_LX(β, α; τ, 0, T_L, T_X) at every ε_ν on the scan grid.

    Parameters
    ----------
    eps_nu      : (n_nu,) intermediate kinetic energies [au]
    eps_kappa   : final kinetic energy [au]
    E_EA        : binding-energy magnitude (= -ε_i > 0) [au]
    Omega_XUV   : XUV harmonic energy for this pathway [au]
    omega_IR    : IR carrier frequency [au]
    ir_action   : 'absorb' (sideband '<') or 'emit' (sideband '>')
    T_X, T_L    : XUV and IR pulse duration parameters (= FWHM/√(2 ln 2))
                  [au^-1]
    tau         : IR delay [au^-1] (XUV centred at 0)

    Returns
    -------
    K(ε_ν) : (n_nu,) complex array
    """
    eps_nu = np.asarray(eps_nu, dtype=float)
    K = np.empty(eps_nu.shape, dtype=np.complex128)
    for i, eps in enumerate(eps_nu):
        alpha, beta = rabbitt_detunings(
            E=eps_kappa, E_k=eps, E_EA=E_EA,
            omega_IR=omega_IR, Omega_XUV=Omega_XUV,
            ir_action=ir_action,
        )
        K[i] = kernel_erf(
            alpha=alpha, beta=beta,
            T_a=T_X, T_b=T_L,
            tau_a=0.0, tau_b=tau,
        )
    return K


# ---------------------------------------------------------------------------
# Main: T̃ matrix for one (κ, path, m_p^IR, m_p^XUV, τ) combo
# ---------------------------------------------------------------------------
def compute_T_path(
    M_in_list: Iterable[np.ndarray],
    D_list: Iterable[np.ndarray],
    eps_nu: np.ndarray,
    eps_kappa: float,
    E_EA: float,
    Omega_XUV: float,
    omega_IR: float,
    ir_action: str,
    T_X: float,
    T_L: float,
    tau: float,
    weights: np.ndarray,
    symmetric_trim_sigma: float | None = None,
) -> np.ndarray:
    """Compute T̃[μ_κ, μ_ν] = Σ_ν M_in[μ_κ, μ_ν] · D[μ_ν] · K_ν · w_ν.

    M_in_list and D_list must have the same length as eps_nu (one entry
    per scan point in increasing-ε order).  The c-c matrices in M_in_list
    must already be rotated to the in-state basis (see
    ``rotate_to_instate.rotate_pair``).

    Returns
    -------
    T_tilde : (N_psi, N_psi) complex array
              T_tilde[μ_κ, μ_ν]
    """
    M_in_list = list(M_in_list)
    D_list    = list(D_list)
    n_nu = len(eps_nu)
    if len(M_in_list) != n_nu:
        raise ValueError(
            f"M_in_list length {len(M_in_list)} != n_nu {n_nu}")
    if len(D_list) != n_nu:
        raise ValueError(
            f"D_list length {len(D_list)} != n_nu {n_nu}")
    if weights.shape != (n_nu,):
        raise ValueError(
            f"weights shape {weights.shape} must be ({n_nu},)")
    # Validate shapes against the first entry.
    N_psi = M_in_list[0].shape[0]
    for i, (M, D) in enumerate(zip(M_in_list, D_list)):
        if M.shape != (N_psi, N_psi):
            raise ValueError(
                f"M_in_list[{i}] shape {M.shape}, expected ({N_psi},{N_psi})")
        if D.shape != (N_psi,):
            raise ValueError(
                f"D_list[{i}] shape {D.shape}, expected ({N_psi},)")

    # ------------------------------------------------------------------
    # PER-PATH SYMMETRIC INTEGRATION centered on the on-shell ν*.
    #
    # The kernel's IMAGINARY part is the principal-value piece P/β of the
    # Sokhotski-Plemelj decomposition.  P/β has odd symmetry around β=0,
    # so its integral vanishes for a symmetric integration window.  But:
    #   1. our scan grid is UNIFORM IN k → NON-UNIFORM IN ε,
    #   2. the path's on-shell ν* = ε_κ ∓ ω falls BETWEEN grid points,
    # so a naive grid-based "symmetric trim" leaves a residual P-shift.
    #
    # FIX: build a UNIFORM-IN-ε sub-grid of (2N+1) points spanning
    # [ν* − N·σ_β·factor,  ν* + N·σ_β·factor]  with ν* exactly at the
    # centre.  Interpolate M_in and D linearly onto that sub-grid.
    # The kernel and trapezoid weight are then EXACTLY symmetric around
    # ν*; the imaginary P-tail cancels by construction.
    # ------------------------------------------------------------------
    s_IR = +1.0 if ir_action == "absorb" else -1.0
    eps_nu_on_shell = eps_kappa - s_IR * omega_IR
    sigma_beta = np.sqrt(2.0) / T_L
    if symmetric_trim_sigma is not None and symmetric_trim_sigma > 0:
        half_window = symmetric_trim_sigma * sigma_beta
        n_sub = 401                              # odd → ν* at exact centre
        sub_eps = np.linspace(eps_nu_on_shell - half_window,
                              eps_nu_on_shell + half_window, n_sub)
        d_sub = sub_eps[1] - sub_eps[0]
        # Trapezoid weights on the uniform sub-grid:
        sub_w = np.full(n_sub, d_sub); sub_w[0] *= 0.5; sub_w[-1] *= 0.5

        # Linearly interpolate M_in and D to the sub-grid (element-wise).
        # Stack M_in (n_nu, N_psi, N_psi) and D (n_nu, N_psi) for vector ops.
        M_stack = np.stack(M_in_list, axis=0)              # (n_nu, N_psi, N_psi)
        D_stack = np.stack(D_list,   axis=0)               # (n_nu, N_psi)
        # np.interp works on real values; treat re/im separately.
        eps_nu_sorted = eps_nu
        M_sub = (np.array([[np.interp(sub_eps, eps_nu_sorted, M_stack[:, a, b].real)
                              + 1j * np.interp(sub_eps, eps_nu_sorted, M_stack[:, a, b].imag)
                            for b in range(N_psi)] for a in range(N_psi)])
                  .transpose(2, 0, 1))                      # (n_sub, N_psi, N_psi)
        D_sub = np.array([np.interp(sub_eps, eps_nu_sorted, D_stack[:, a].real)
                           + 1j * np.interp(sub_eps, eps_nu_sorted, D_stack[:, a].imag)
                          for a in range(N_psi)]).T          # (n_sub, N_psi)

        K_sub = kernel_grid(
            eps_nu=sub_eps, eps_kappa=eps_kappa, E_EA=E_EA,
            Omega_XUV=Omega_XUV, omega_IR=omega_IR, ir_action=ir_action,
            T_X=T_X, T_L=T_L, tau=tau,
        )
        T_tilde = np.zeros((N_psi, N_psi), dtype=np.complex128)
        for i in range(n_sub):
            T_tilde += M_sub[i] * D_sub[i][np.newaxis, :] * K_sub[i] * sub_w[i]
        return T_tilde

    # Untrimmed: original behaviour, integrate over the supplied ν grid as-is.
    K = kernel_grid(
        eps_nu=eps_nu, eps_kappa=eps_kappa, E_EA=E_EA,
        Omega_XUV=Omega_XUV, omega_IR=omega_IR, ir_action=ir_action,
        T_X=T_X, T_L=T_L, tau=tau,
    )
    T_tilde = np.zeros((N_psi, N_psi), dtype=np.complex128)
    for i in range(n_nu):
        T_tilde += M_in_list[i] * D_list[i][np.newaxis, :] * K[i] * weights[i]
    return T_tilde


# ---------------------------------------------------------------------------
# Convenience: detuning grid for inspection
# ---------------------------------------------------------------------------
def detuning_grid(
    eps_nu: np.ndarray,
    eps_kappa: float,
    E_EA: float,
    Omega_XUV: float,
    omega_IR: float,
    ir_action: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (α_ν, β_ν) arrays for diagnostic plots."""
    eps_nu = np.asarray(eps_nu, dtype=float)
    n = eps_nu.shape[0]
    alpha = np.empty(n)
    beta  = np.empty(n)
    for i, eps in enumerate(eps_nu):
        a, b = rabbitt_detunings(
            E=eps_kappa, E_k=eps, E_EA=E_EA,
            omega_IR=omega_IR, Omega_XUV=Omega_XUV,
            ir_action=ir_action,
        )
        alpha[i] = a
        beta[i]  = b
    return alpha, beta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _self_test() -> int:
    """Reduce to a single-channel scalar case and check against the user's
    adaptive-quadrature reference (kernel_quad)."""
    from dawson_kernel_rabbitt import kernel_quad, HBAR_EV_FS

    fails = 0
    def _check(cond: bool, what: str):
        nonlocal fails
        if cond:
            print(f"  [ok  ] {what}")
        else:
            print(f"  [FAIL] {what}")
            fails += 1

    # Physical parameters identical to the test in dawson_kernel_rabbitt.
    T_X = (0.35 / HBAR_EV_FS) / np.sqrt(2 * np.log(2))   # fs → eV^-1
    T_L = (5.0  / HBAR_EV_FS) / np.sqrt(2 * np.log(2))
    tau = 0.45 / HBAR_EV_FS

    # All energies in eV (consistent with T being eV^-1).  E_EA = 1.2 eV,
    # ω_IR = 1.55 eV (Ti:S photon), Ω_XUV = 23.25 eV (H15 of 1.55 eV).
    # E_κ at sideband 2q=16 path '<': E_κ = -E_EA + Ω_XUV + ω_IR = 23.6 eV
    E_EA      = 1.2
    omega_IR  = 1.55
    Omega_XUV = 15.0 * omega_IR
    eps_kappa = -E_EA + Omega_XUV + omega_IR  # path '<'

    print("=== (T1) single-channel reduction matches scalar quadrature ===")
    # Build a 1×1 channel space and a 3-point ν grid.
    eps_nu = np.array([eps_kappa - 0.4, eps_kappa, eps_kappa + 0.3])
    weights = np.array([0.2, 0.35, 0.45])
    # Random scalar M_in and D values.
    rng = np.random.default_rng(2)
    M_in_list = [np.array([[rng.normal() + 1j * rng.normal()]]) for _ in eps_nu]
    D_list    = [np.array([rng.normal() + 1j * rng.normal()]) for _ in eps_nu]

    T = compute_T_path(
        M_in_list=M_in_list, D_list=D_list, eps_nu=eps_nu,
        eps_kappa=eps_kappa, E_EA=E_EA,
        Omega_XUV=Omega_XUV, omega_IR=omega_IR, ir_action="absorb",
        T_X=T_X, T_L=T_L, tau=tau, weights=weights,
    )

    # Direct loop reference using kernel_quad (independent of kernel_erf).
    T_ref = 0.0 + 0.0j
    for i, eps in enumerate(eps_nu):
        alpha, beta = rabbitt_detunings(
            E=eps_kappa, E_k=eps, E_EA=E_EA,
            omega_IR=omega_IR, Omega_XUV=Omega_XUV, ir_action="absorb")
        Kq = kernel_quad(alpha, beta, T_X, T_L, 0.0, tau)
        T_ref += M_in_list[i][0, 0] * D_list[i][0] * Kq * weights[i]

    rel = abs(T[0, 0] - T_ref) / abs(T_ref)
    _check(rel < 1e-10, f"scalar T̃ matches kernel_quad reference: rel = {rel:.2e}")

    print("\n=== (T2) detuning grid sanity ===")
    alpha, beta = detuning_grid(eps_nu, eps_kappa, E_EA, Omega_XUV, omega_IR, "absorb")
    # For ε_ν=ε_κ: at sideband (path '<' absorb) we have
    #     β = ε_κ - ε_ν - ω = -ω
    # and α = ε_ν + E_EA - Ω_XUV.  Both checkable directly.
    expected_beta_on_shell = -omega_IR
    _check(abs(beta[1] - expected_beta_on_shell) < 1e-15,
           f"β at ε_ν=ε_κ (absorb path) = -ω: got {beta[1]:.6g} vs -ω = {-omega_IR:.6g}")
    expected_alpha = eps_nu[1] + E_EA - Omega_XUV
    _check(abs(alpha[1] - expected_alpha) < 1e-15,
           f"α at ε_ν=ε_κ: ε_ν + E_EA - Ω_XUV = {expected_alpha:.6g} vs {alpha[1]:.6g}")

    print("\n=== (T3) emit path flips β sign of ω ===")
    a_abs, b_abs = detuning_grid(eps_nu, eps_kappa, E_EA, Omega_XUV, omega_IR, "absorb")
    a_emi, b_emi = detuning_grid(eps_nu, eps_kappa, E_EA, Omega_XUV, omega_IR, "emit")
    # α unchanged across paths (XUV step is the same -- well, in RABBITT
    # the > path uses Ω_{2q+1} not Ω_{2q-1}; in detuning_grid we passed
    # the SAME Ω_XUV so α should match.)
    err_a = np.max(np.abs(a_abs - a_emi))
    _check(err_a < 1e-15, f"α(absorb) == α(emit) when Ω_XUV the same: max err {err_a:.2e}")
    # β: absorb-emit = -2ω → ω = -(b_abs - b_emi)/2
    omega_inferred = -(b_abs - b_emi) / 2.0
    _check(np.allclose(omega_inferred, omega_IR),
           f"β(absorb) - β(emit) = -2ω: inferred ω = {omega_inferred}")

    print("\n=== (T4) zero IR pulse-length suppression ===")
    # With T_L extremely SHORT (i.e. very wide IR spectrum), the kernel
    # peak gets very broad -- not a useful sharp test.  Use the OPPOSITE
    # limit instead: very LONG T_L → β = 0 (on-shell) needed for any
    # large kernel.  At β_<= -ω (always nonzero for IR carrier > 0)
    # the |K| at fixed T_X should DECREASE as T_L grows large because
    # the Gaussian penalty exp(-(β T_L)²/4) kills off-shell weight.
    # That's a clean, physically-motivated test.
    K_Tx_short = kernel_grid(eps_nu, eps_kappa, E_EA, Omega_XUV, omega_IR,
                             "absorb", T_X, T_L * 0.5, tau)
    K_Tx_long  = kernel_grid(eps_nu, eps_kappa, E_EA, Omega_XUV, omega_IR,
                             "absorb", T_X, T_L * 2.0, tau)
    # At ν=κ (i=1) we have β=-ω; |K| should be smaller for longer T_L.
    _check(np.abs(K_Tx_long[1]) < np.abs(K_Tx_short[1]),
           f"|K(β=-ω)| decreases with T_L: short={abs(K_Tx_short[1]):.3e}, "
           f"long={abs(K_Tx_long[1]):.3e}")

    print(f"\n  Total failures: {fails}\n")
    return fails


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    sys.exit(_self_test())
