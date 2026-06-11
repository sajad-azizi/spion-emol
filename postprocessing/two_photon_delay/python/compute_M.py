"""
compute_M.py
------------
m-resolved RABBITT angular assembly per derivation §(D6):

    M(k̂; ε_i + Ω; R̂_γ)
       = (1/i) √(2/π)
         Σ_L  (−i)^L
         Σ_{M_R}  μ_κ = L² + L + M_R
              [ Σ_{M_C} Y^C_(LM_C)(k̂)  U(L, M_R, M_C) ]
         Σ_{μ_ν}
         Σ_{m_p^IR_mol, m_p^XUV_mol}
              D^(1)_{m_p^IR_mol,  m_p^IR_lab}(R̂_γ)
              · D^(1)_{m_p^XUV_mol, m_p^XUV_lab}(R̂_γ)
              · T̃_{(L, M_R), μ_ν}^{m_p^IR_mol, m_p^XUV_mol}(κ)

The amplitudes for the two RABBITT pathways at sideband 2q are obtained
by calling ``compute_M_one_path`` with the path-specific T̃; the
sideband delay follows from

    τ_2hω(2q; k̂, R̂_γ)  =  (1/2ω) · arg [ M_< · M_>* ]    (D8)

(Implemented in ``tau_2homega.py``, the next file.)

Wigner-D convention: active rotation
    D^(1)_{m', m}(α, β, γ) = e^{-i α m'}  d^(1)_{m', m}(β)  e^{-i γ m}

with the small-d j=1 matrix:

                m=-1            m=0             m=+1
    m'=-1   (1+cosβ)/2       sinβ/√2          (1-cosβ)/2
    m'= 0   -sinβ/√2          cosβ            sinβ/√2
    m'=+1   (1-cosβ)/2       -sinβ/√2         (1+cosβ)/2

This is the Varshalovich / Edmonds convention, matched to BW17 usage.

Complex spherical harmonics use the Condon-Shortley phase via
``scipy.special.sph_harm_y(L, M, theta, phi)`` with θ = polar (colatitude),
φ = azimuthal.  This is consistent with the U_real_to_complex matrix
implemented in ``src/angular/Gaunt.hpp`` and exported by
``complex_Y_transform.U_block``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import sph_harm_y

from complex_Y_transform import U_block


# ---------------------------------------------------------------------------
# Complex spherical harmonics
# ---------------------------------------------------------------------------
def Y_complex(L: int, M: int, theta: float, phi: float) -> complex:
    """Y^C_LM(θ, φ) with θ = polar (∈ [0, π]) and φ = azimuthal (∈ [0, 2π]).
    Condon-Shortley convention."""
    return complex(sph_harm_y(L, M, theta, phi))


def Y_complex_row(L: int, theta: float, phi: float) -> np.ndarray:
    """Row vector  Y^C_LM(θ, φ)  for M = -L..+L, length 2L+1."""
    Ms = np.arange(-L, L + 1)
    return np.array([Y_complex(L, int(M), theta, phi) for M in Ms],
                    dtype=np.complex128)


# ---------------------------------------------------------------------------
# Wigner small-d / D for j=1
# ---------------------------------------------------------------------------
def wigner_d1(beta: float) -> np.ndarray:
    """3×3 real small-d matrix d^(1)(β).  Indices [i, j] correspond to
    (m'=−1, 0, +1) × (m=−1, 0, +1) in that order."""
    c = np.cos(beta)
    s = np.sin(beta)
    sq2 = np.sqrt(2.0)
    return np.array([
        [ (1.0 + c) / 2.0,  s / sq2,         (1.0 - c) / 2.0 ],
        [-s / sq2,          c,                s / sq2        ],
        [ (1.0 - c) / 2.0, -s / sq2,         (1.0 + c) / 2.0 ],
    ], dtype=float)


def wigner_D1(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """3×3 complex D^(1)(α, β, γ) Wigner matrix, active convention.

    D[i, j] = D^(1)_{m', m}(α, β, γ) with (m', m) ∈ {(−1, 0, +1)}².
    Use D[:, idx_lab] to project a lab-frame spherical-tensor component
    onto its three molecular-frame components.
    """
    d = wigner_d1(beta)
    ms = np.array([-1, 0, 1])
    phase_alpha = np.exp(-1j * alpha * ms).reshape(3, 1)   # rows: m'
    phase_gamma = np.exp(-1j * gamma * ms).reshape(1, 3)   # cols: m
    return phase_alpha * d * phase_gamma


# ---------------------------------------------------------------------------
# Per-channel angular factor  ang_mu[μ_κ]
# ---------------------------------------------------------------------------
def angular_factor_per_channel(
    L_max: int,
    theta_k: float,
    phi_k: float,
) -> np.ndarray:
    """Return the (N_psi,) array

         ang_mu[μ_κ = L² + L + M_R]
              =  (1/i) √(2/π)  (−i)^L  Σ_{M_C} Y^C_(LM_C)(k̂) U(L, M_R, M_C)

    that multiplies T̃[μ_κ, ·] in (D6).  This is the only place where
    k̂ enters the assembly.
    """
    N_psi = (L_max + 1) ** 2
    ang = np.zeros(N_psi, dtype=np.complex128)
    pref_base = (1.0 / 1j) * np.sqrt(2.0 / np.pi)
    for L in range(L_max + 1):
        U = U_block(L)                              # (2L+1, 2L+1)
        Y = Y_complex_row(L, theta_k, phi_k)        # (2L+1,)
        # A[M_R + L] = Σ_{M_C} U(L, M_R, M_C) · Y(L, M_C)
        A = U @ Y                                   # (2L+1,)
        pref = pref_base * ((-1j) ** L)
        for M_R in range(-L, L + 1):
            mu = L * L + L + M_R
            ang[mu] = pref * A[M_R + L]
    return ang


# ---------------------------------------------------------------------------
# Final assembly for ONE RABBITT path
# ---------------------------------------------------------------------------
def compute_M_one_path(
    T_tilde: np.ndarray,
    L_max: int,
    theta_k: float,
    phi_k: float,
    alpha_R: float,
    beta_R: float,
    gamma_R: float,
    m_p_IR_lab: int,
    m_p_XUV_lab: int,
) -> complex:
    """Assemble M(k̂; R̂_γ) for one path per (D6).

    Parameters
    ----------
    T_tilde     : shape (3, 3, N_psi, N_psi) complex.
                  T_tilde[i_IR, i_XUV, μ_κ, μ_ν]
                  with (i_IR, i_XUV) indexing m_p^mol ∈ (−1, 0, +1).
                  Produced by ``compute_T.compute_T_path`` for each
                  (m_p^IR_mol, m_p^XUV_mol) combination.
    L_max       : highest L (must satisfy (L_max+1)² == N_psi).
    theta_k     : k̂ polar angle in lab frame  [rad, 0..π]
    phi_k       : k̂ azimuthal angle in lab frame  [rad, 0..2π]
    alpha_R, beta_R, gamma_R : Euler angles MOL → LAB.
    m_p_IR_lab, m_p_XUV_lab  : ±1 or 0 (linear-Z lab → m_p_lab = 0).

    Returns
    -------
    M  : complex scalar amplitude.
    """
    if T_tilde.shape[0] != 3 or T_tilde.shape[1] != 3:
        raise ValueError(
            f"T_tilde must have shape (3, 3, N, N); got {T_tilde.shape}")
    N_psi = T_tilde.shape[2]
    if T_tilde.shape[2] != N_psi or T_tilde.shape[3] != N_psi:
        raise ValueError(f"T_tilde last two axes must be (N, N); got {T_tilde.shape}")
    if (L_max + 1) ** 2 != N_psi:
        raise ValueError(
            f"L_max={L_max} inconsistent with N_psi={N_psi}; expected "
            f"(L_max+1)² = {(L_max + 1) ** 2}")
    if m_p_IR_lab not in (-1, 0, 1) or m_p_XUV_lab not in (-1, 0, 1):
        raise ValueError(
            f"m_p_*_lab must be -1, 0, +1; got IR={m_p_IR_lab}, XUV={m_p_XUV_lab}")

    ang_mu = angular_factor_per_channel(L_max, theta_k, phi_k)

    D1 = wigner_D1(alpha_R, beta_R, gamma_R)
    idx_IR  = m_p_IR_lab  + 1
    idx_XUV = m_p_XUV_lab + 1
    coef = np.outer(D1[:, idx_IR], D1[:, idx_XUV])     # (3, 3)

    # Σ_μ_ν T̃[i_IR, i_XUV, μ_κ, μ_ν]:
    T_mu_summed = T_tilde.sum(axis=3)                   # (3, 3, N_psi)
    # M = Σ_{i_IR, i_XUV, μ_κ} coef · ang_mu · T̃
    return complex(np.einsum('ij, ija, a -> ', coef, T_mu_summed, ang_mu))


# ---------------------------------------------------------------------------
# Convenience: angular average of |M|² over k̂
# ---------------------------------------------------------------------------
def angular_grid(n_theta: int, n_phi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Legendre × uniform-φ grid on the unit sphere.  Returns
    (theta, phi, weights) such that  Σ_(i,j) w(i,j) f(θ_i, φ_j)
    ≈ (1/4π) ∫ f(k̂) dΩ.  Weights sum to 1."""
    cos_th, w_th = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(cos_th)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    w_th = w_th / 2.0                       # ∫ d(cosθ) /2 = 1
    w_phi = np.full(n_phi, 1.0 / n_phi)     # ∫ dφ / (2π) = 1
    W = np.outer(w_th, w_phi)               # (n_theta, n_phi), Σ W = 1
    return theta, phi, W


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _self_test() -> int:
    fails = 0
    def _check(cond: bool, what: str):
        nonlocal fails
        if cond:
            print(f"  [ok  ] {what}")
        else:
            print(f"  [FAIL] {what}")
            fails += 1

    print("=== (T1) Y_complex known values ===")
    # Y_00 = 1/(2√π)
    Y = Y_complex(0, 0, 0.7, 1.2)
    _check(abs(Y - 1.0 / (2.0 * np.sqrt(np.pi))) < 1e-14,
           f"Y_00(any) = 1/(2√π): got {Y!r}")
    # Y_1,0(θ, φ) = √(3/4π) cos θ
    for theta in (0.0, 0.5, 1.3):
        Y = Y_complex(1, 0, theta, 0.7)
        ref = np.sqrt(3.0 / (4 * np.pi)) * np.cos(theta)
        _check(abs(Y - ref) < 1e-14,
               f"Y_10(θ={theta}) = √(3/4π) cosθ: got {Y.real:.10g}, ref={ref:.10g}")
    # Y_1,1(θ, φ) = -√(3/8π) sin θ e^{iφ}
    Y = Y_complex(1, 1, np.pi / 3, np.pi / 4)
    ref = -np.sqrt(3.0 / (8 * np.pi)) * np.sin(np.pi / 3) * np.exp(1j * np.pi / 4)
    _check(abs(Y - ref) < 1e-14, f"Y_11(π/3, π/4)  err={abs(Y - ref):.2e}")

    print("\n=== (T2) Wigner d^(1) unitarity ===")
    for beta in (0.0, 0.3, 0.7, np.pi / 2, np.pi - 0.1):
        d = wigner_d1(beta)
        err = np.max(np.abs(d @ d.T - np.eye(3)))
        _check(err < 1e-14, f"d^(1)({beta:.3f}) orthogonal: ||d d^T - I|| = {err:.2e}")

    print("\n=== (T3) Wigner D^(1) unitarity ===")
    rng = np.random.default_rng(3)
    for _ in range(5):
        a, b, g = rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
        D = wigner_D1(a, b, g)
        err = np.max(np.abs(D @ D.conj().T - np.eye(3)))
        _check(err < 1e-14,
               f"D^(1)({a:.2f},{b:.2f},{g:.2f}) unitary: err={err:.2e}")

    print("\n=== (T4) D^(1)(0, 0, 0) = I ===")
    D = wigner_D1(0.0, 0.0, 0.0)
    err = np.max(np.abs(D - np.eye(3)))
    _check(err < 1e-15, f"D(0,0,0) = I: err={err:.2e}")

    print("\n=== (T5) D^(1) composition: D(R1) · D(R2) = D(R1∘R2) ===")
    # Easiest: D(α₁+α₂, 0, 0) = D(α₁,0,0) D(α₂,0,0) about z-axis only
    a1, a2 = 0.7, 1.1
    D12 = wigner_D1(a1, 0, 0) @ wigner_D1(a2, 0, 0)
    D_sum = wigner_D1(a1 + a2, 0, 0)
    err = np.max(np.abs(D12 - D_sum))
    _check(err < 1e-14, f"about-z group law: err={err:.2e}")

    print("\n=== (T6) angular_factor_per_channel reduces to L=0 only ===")
    # If T̃ is supported only in (μ_κ, μ_ν) = (0, 0), M = ang_mu[0] * sum
    # over m_p × D^(1) coefs × T̃[i_IR, i_XUV, 0, 0].
    L_max = 0
    N = 1
    T = np.zeros((3, 3, N, N), dtype=np.complex128)
    T[1, 1, 0, 0] = 1.0 + 0.5j                          # m_p_IR_mol = 0, m_p_XUV_mol = 0
    # No rotation, lab = mol.
    M = compute_M_one_path(T, L_max, theta_k=0.4, phi_k=0.8,
                           alpha_R=0, beta_R=0, gamma_R=0,
                           m_p_IR_lab=0, m_p_XUV_lab=0)
    # D^(1)(0,0,0) = I, so coef[1, 1] = 1 (m_p=0 → m_p=0).
    # ang_mu[0] = (1/i)√(2/π) · (-i)^0 · Y^C(0,0)(k̂) · U(0,0,0)
    #          = -i√(2/π) · 1/(2√π) · 1
    #          = -i / (2π) · √(2/π) · √π   wait let's just compute directly
    ang_mu = angular_factor_per_channel(L_max, 0.4, 0.8)
    M_ref = ang_mu[0] * (1.0 + 0.5j)
    err = abs(M - M_ref)
    _check(err < 1e-15,
           f"L=0 reduction: M = ang_mu[0] · T̃[0,0,0,0]  (err={err:.2e})")

    # Numerically: ang_mu[0] = (1/i)√(2/π) · 1 · (1/(2√π)) = -i / (2π·... ) wait
    # = (1/i) · sqrt(2/π) · (1/(2 sqrt(π))) = -i · sqrt(2) / (2π)
    ang_ref = -1j * np.sqrt(2.0) / (2.0 * np.pi)
    err = abs(ang_mu[0] - ang_ref)
    _check(err < 1e-14, f"ang_mu[0] = -i√2/(2π): got {ang_mu[0]!r}, ref {ang_ref!r}")

    print("\n=== (T7) full pipeline smoke test (L_max=2) ===")
    L_max = 2
    N = (L_max + 1) ** 2
    T = (rng.normal(size=(3, 3, N, N))
         + 1j * rng.normal(size=(3, 3, N, N)))
    M = compute_M_one_path(T, L_max,
                           theta_k=0.7, phi_k=1.3,
                           alpha_R=0.4, beta_R=1.0, gamma_R=2.1,
                           m_p_IR_lab=0, m_p_XUV_lab=0)
    _check(np.isfinite(M.real) and np.isfinite(M.imag),
           f"M_one_path finite at L_max=2: M = {M!r}")
    # Hermiticity-type test: complex-conjugating T̃ and conjugating M shouldn't agree
    # (M has explicit i factors); but flipping the sign of γ on a real-T̃ should give
    # a clean transformation.  Skipping; smoke test is enough.

    print("\n=== (T8) angular_grid weights sum to 1 ===")
    theta, phi, W = angular_grid(8, 12)
    _check(abs(W.sum() - 1.0) < 1e-14, f"Σ W = {W.sum():.10g}")

    # Reproduce ∫ Y_00² / 4π = (1/(2√π))² ≈ const → sum = const.
    # More useful: ∫ |Y_LM|² dΩ / 4π should give 1/(4π).
    # Σ W |Y_LM|² ≈ 1/(4π).
    L, M_idx = 2, 1
    val = 0.0
    for it, th in enumerate(theta):
        for ip, ph in enumerate(phi):
            val += W[it, ip] * abs(Y_complex(L, M_idx, th, ph)) ** 2
    err = abs(val - 1.0 / (4 * np.pi))
    _check(err < 1e-10,
           f"Σ W |Y_{L}{M_idx}|² = 1/(4π): got {val:.10g} ref {1.0/(4*np.pi):.10g}")

    print(f"\n  Total failures: {fails}\n")
    return fails


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    sys.exit(_self_test())
