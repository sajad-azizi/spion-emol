#!/usr/bin/env python3
"""
polar_fft.py
============
Build σ_pol, τ_pol, σ·τ_pol on a 3-D Cartesian (k_x, k_y, k_z) cube
from the per-(ℓ, m) dipole matrix elements in a gather dir, then
inverse-FFT each to r-space and write three Gaussian .cube files.

Critical: this does *NOT* read the polar_pol_avg_*.dat file (that file
is at φ=0 only, and real-Y_{ℓ,m<0}(θ, φ=0) ≡ 0 — so it misses the m<0
channels that are non-zero at φ ≠ 0).  Instead we sum the per-channel
dipoles directly:

    F_q(k, θ, φ) = 4π · Σ_{ℓm}  i^{-ℓ} · d^q_{ℓ,m}(k) · Y^R_{ℓ,m}(θ, φ)
    σ_pol = (4π² ω(k) / c) · (1/3) · Σ_q |F_q|² / k²
    τ_pol = Σ_q Im[F_q* · ∂_E F_q] / Σ_q |F_q|²

(Eq. S38 of the Azizi et al. supplementary; ω(k) = E_kin + Ip from the
gather files; ∂_E = (1/k) ∂_k via finite differences in k.)

Y^R uses the no-CS real-spherical-harmonic convention identical to
`src/angular/Gaunt.hpp::real_Ylm`:
    Y^R_{ℓ,0}(θ, φ)   = S_{ℓ,0}(θ)
    Y^R_{ℓ,m>0}(θ, φ) = √2  S_{ℓ,|m|}(θ) · cos(m φ)
    Y^R_{ℓ,m<0}(θ, φ) = √2  S_{ℓ,|m|}(θ) · sin(|m| φ)
where S is the normalised associated Legendre (Press et al. NR §6.7,
no Condon-Shortley phase).

Output layout: 3-D inverse FFT with centered input AND centered output
(via ifftshift → ifftn → fftshift).  Cube grid spans [-r_max, +r_max-dr)
per axis where dr = 2π / (N · dk).
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


HA_TO_EV = 27.211386245988
AU_TO_AS = 24.188843265857
C_AU     = 137.035999139         # speed of light, atomic units

FILE_RE = re.compile(r"^dipole_(len|vel)_homo_([xyz])_(\d+)\.dat$")


# ============================================================================
# Real spherical harmonics  (matches src/angular/Gaunt.hpp::real_Ylm exactly)
# ============================================================================
def idx_to_lm(idx: int) -> Tuple[int, int]:
    """flat-index ↔ (ℓ, m), with index = ℓ² + ℓ + m, m ∈ [-ℓ, +ℓ]."""
    ell = int(np.sqrt(idx))
    if (ell + 1) * (ell + 1) <= idx:
        ell += 1
    return ell, idx - ell * ell - ell


def _sph_legendre_CS(ell: int, m: int, theta: np.ndarray) -> np.ndarray:
    """Normalised associated Legendre P̃_ℓ^m(cos θ) WITH Condon-Shortley
    phase.  BIT-IDENTICAL to the recurrence used in
    `postprocessing/polar_plots/compute_polar_data.py::real_Ylm_phi0`.

    Recurrence (Press et al. NR §6.7, CS-phased):
        P̃_0^0     = 1/(2√π)
        P̃_k^k     = -√((2k+1)/(2k)) · sin θ · P̃_{k-1}^{k-1}      (k ≥ 1)
        P̃_{m+1}^m = √(2m+3) · cos θ · P̃_m^m
        P̃_ℓ^m     = a_ℓ · cos θ · P̃_{ℓ-1}^m − b_ℓ · P̃_{ℓ-2}^m   (ℓ ≥ m+2)
        a_ℓ = √((4ℓ² − 1)/(ℓ² − m²))
        b_ℓ = √((2ℓ+1)/(2ℓ−3) · ((ℓ−1)² − m²)/(ℓ² − m²))

    Relation to the no-CS Legendre S_{ℓ,m}:  P̃_ℓ^m = (−1)^m · S_{ℓ,m}.

    Using THIS recurrence (not the no-CS one) is a deliberate choice:
    it makes our φ=0 path bit-identical to compute_polar_data.py's
    reference output (validated by test_polar_phi0_match.py).
    """
    theta = np.asarray(theta, dtype=np.float64)
    if ell < 0 or m < 0 or m > ell:
        return np.zeros_like(theta)
    s = np.sin(theta)
    c = np.cos(theta)
    P_mm = np.full_like(theta, 1.0 / (2.0 * np.sqrt(np.pi)))
    for k in range(1, m + 1):
        P_mm = -np.sqrt((2.0 * k + 1.0) / (2.0 * k)) * s * P_mm
    if ell == m:
        return P_mm
    P_p2 = P_mm
    P_p1 = np.sqrt(2.0 * m + 3.0) * c * P_mm
    if ell == m + 1:
        return P_p1
    for ll in range(m + 2, ell + 1):
        a = np.sqrt((4.0 * ll * ll - 1.0) / (ll * ll - m * m))
        b = np.sqrt((2.0 * ll + 1.0) / (2.0 * ll - 3.0)
                    * ((ll - 1.0) * (ll - 1.0) - m * m)
                    / (ll * ll - m * m))
        P_cur = a * c * P_p1 - b * P_p2
        P_p2 = P_p1
        P_p1 = P_cur
    return P_p1


def real_Ylm(ell: int, m: int, theta: np.ndarray, phi: np.ndarray
             ) -> np.ndarray:
    """Real spherical harmonic Y^R_{ℓ,m}(θ, φ).

    Mathematically identical to src/angular/Gaunt.hpp::real_Ylm (no-CS
    final form), but built from the CS-phase Legendre so that at φ = 0
    it is BIT-IDENTICAL to compute_polar_data.py::real_Ylm_phi0.

        Y^R_{ℓ, m=0}(θ, φ) =          P̃_{ℓ,0}(θ)
        Y^R_{ℓ, m>0}(θ, φ) = √2 (−1)^m P̃_{ℓ,m}(θ) · cos(m φ)
        Y^R_{ℓ, m<0}(θ, φ) = √2 (−1)^|m| P̃_{ℓ,|m|}(θ) · sin(|m| φ)

    (The (−1)^m at output combines with the (−1)^m inside P̃ to give the
    no-CS spherical harmonic Y^R = √2 · S · cos/sin, matching Gaunt.hpp
    element-for-element.)
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi   = np.asarray(phi,   dtype=np.float64)
    am = abs(m)
    if ell < 0 or am > ell:
        return np.zeros(np.broadcast(theta, phi).shape, dtype=np.float64)
    P = _sph_legendre_CS(ell, am, theta)
    if m == 0:
        return P
    sign = -1.0 if (am & 1) else 1.0   # (-1)^|m|
    if m > 0:
        return np.sqrt(2.0) * sign * P * np.cos(m * phi)
    return np.sqrt(2.0) * sign * P * np.sin(am * phi)


# ============================================================================
# Gather-dir loader  (mirrors postprocessing/polar_plots/compute_polar_data.py)
# ============================================================================
@dataclass
class DipoleDataset:
    k:       np.ndarray              # (Nk,) photoelectron k in a.u.
    e_kin:   np.ndarray              # (Nk,) kinetic energy
    omega:   np.ndarray              # (Nk,) photon energy = E_kin + Ip
    mu_list: List[int]               # channel indices μ = ℓ² + ℓ + m
    dipole:  np.ndarray              # (Nk, Nch) complex d^q_{ℓm}(k)
    gauge:   str
    component: str                   # 'x' / 'y' / 'z'


def _discover_mu_files(input_dir: Path, gauge: str, comp: str
                        ) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for p in sorted(input_dir.glob(f"dipole_{gauge}_homo_{comp}_*.dat")):
        m = FILE_RE.match(p.name)
        if not m: continue
        g, c, mu = m.groups()
        if g == gauge and c == comp:
            out.append((int(mu), p))
    if not out:
        raise FileNotFoundError(
            f"no dipole_{gauge}_homo_{comp}_*.dat in {input_dir}")
    out.sort(key=lambda x: x[0])
    return out


def _load_one_channel(p: Path):
    a = np.loadtxt(p)
    if a.ndim == 1: a = a[None, :]
    if a.shape[1] < 5:
        raise ValueError(f"{p}: expected ≥5 columns, got {a.shape[1]}")
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3] + 1j * a[:, 4]


def load_dipole_dataset(input_dir: Path, gauge: str, component: str
                         ) -> DipoleDataset:
    files = _discover_mu_files(input_dir, gauge, component)
    mus = [mu for mu, _ in files]
    k0, e0, w0, d0 = _load_one_channel(files[0][1])
    D = np.zeros((k0.size, len(mus)), dtype=np.complex128)
    D[:, 0] = d0
    for j, (mu, p) in enumerate(files[1:], start=1):
        k_j, e_j, w_j, d_j = _load_one_channel(p)
        if not (np.allclose(k_j, k0) and np.allclose(e_j, e0)
                and np.allclose(w_j, w0)):
            raise ValueError(f"energy-grid mismatch in {p}")
        D[:, j] = d_j
    return DipoleDataset(k0, e0, w0, mus, D, gauge, component)


# ============================================================================
# Cube builder: σ_pol, τ_pol, σ·τ_pol on a centered Cartesian k-cube
# ============================================================================
def _interp_complex(k_axis: np.ndarray, D_col: np.ndarray,
                     K_flat: np.ndarray) -> np.ndarray:
    """1-D linear interpolation of a complex sequence; values outside
    [k_axis[0], k_axis[-1]] are set to zero (no extrapolation)."""
    out_re = np.interp(K_flat, k_axis, D_col.real, left=0.0, right=0.0)
    out_im = np.interp(K_flat, k_axis, D_col.imag, left=0.0, right=0.0)
    return out_re + 1j * out_im


def build_pol_avg_kcube(ds_x: DipoleDataset, ds_y: DipoleDataset,
                         ds_z: DipoleDataset, N: int, k_max: float,
                         verbose: bool = False
                         ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build σ_pol, τ_pol, σ·τ_pol on an N×N×N centered Cartesian k-cube
    spanning [-k_max, +k_max) per axis.

    Returns
    -------
    kx     : (N,) centered k-axis with kx[N/2] = 0 for even N.
    fields : dict with keys 'sigma' (bohr²), 'tau' (as), 'sigma_tau'
             (bohr²·as), each (N, N, N).
    """
    if N % 2 != 0:
        raise ValueError(f"N must be even (centered FFT); got {N}")
    if not (np.allclose(ds_x.k, ds_y.k) and np.allclose(ds_x.k, ds_z.k)):
        raise ValueError("ds_x / ds_y / ds_z must share the same k-axis")
    if ds_x.mu_list != ds_y.mu_list or ds_x.mu_list != ds_z.mu_list:
        raise ValueError("x / y / z must share the same μ channel list")

    k_axis = ds_x.k
    omega_axis = ds_x.omega
    mu_list = ds_x.mu_list

    # ∂_E d = (1/k) ∂_k d, computed on the original k-grid by gradient.
    dD = {
        "x": np.gradient(ds_x.dipole, k_axis, axis=0) / k_axis[:, None],
        "y": np.gradient(ds_y.dipole, k_axis, axis=0) / k_axis[:, None],
        "z": np.gradient(ds_z.dipole, k_axis, axis=0) / k_axis[:, None],
    }
    D = {"x": ds_x.dipole, "y": ds_y.dipole, "z": ds_z.dipole}

    # Cartesian k-cube.
    dk = 2.0 * k_max / N
    kx = (np.arange(N) - N // 2) * dk
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K = np.sqrt(KX * KX + KY * KY + KZ * KZ)
    # Domain mask: only points inside the input (k_min, k_max) carry
    # valid d^q_{ℓm}(k); outside we zero σ, τ, σ·τ.
    in_domain = (K >= k_axis[0]) & (K <= k_axis[-1])
    with np.errstate(invalid="ignore"):
        THETA = np.arccos(np.clip(np.divide(KZ, K, where=(K > 0)), -1.0, 1.0))
    THETA = np.where(K > 0, THETA, 0.0)
    PHI = np.arctan2(KY, KX)
    K_flat   = K.ravel()
    TH_flat  = THETA.ravel()
    PHI_flat = PHI.ravel()

    F  = {q: np.zeros(K_flat.size, dtype=np.complex128) for q in "xyz"}
    dF = {q: np.zeros(K_flat.size, dtype=np.complex128) for q in "xyz"}

    # Channel-by-channel accumulation in flat arrays.
    for j, mu in enumerate(mu_list):
        ell, m = idx_to_lm(mu)
        Y_flat = real_Ylm(ell, m, TH_flat, PHI_flat)
        phase_l = (-1j) ** ell
        for q in "xyz":
            d_interp  = _interp_complex(k_axis, D [q][:, j], K_flat)
            dd_interp = _interp_complex(k_axis, dD[q][:, j], K_flat)
            F [q] += phase_l * d_interp  * Y_flat
            dF[q] += phase_l * dd_interp * Y_flat
        if verbose and (j % 32 == 0 or j == len(mu_list) - 1):
            print(f"  channel {j+1:4d} / {len(mu_list)}  (ℓ={ell}, m={m})")
    PRE = 4.0 * np.pi
    for q in "xyz":
        F [q] *= PRE
        dF[q] *= PRE
        F [q]  = F [q].reshape(K.shape)
        dF[q]  = dF[q].reshape(K.shape)

    # Photon energy at every cube point.
    omega_K = np.interp(K_flat, k_axis, omega_axis,
                        left=0.0, right=0.0).reshape(K.shape)

    F2 = np.abs(F["x"]) ** 2 + np.abs(F["y"]) ** 2 + np.abs(F["z"]) ** 2
    Im = (np.imag(np.conj(F["x"]) * dF["x"])
          + np.imag(np.conj(F["y"]) * dF["y"])
          + np.imag(np.conj(F["z"]) * dF["z"]))
    K2_safe = np.where(K > 0, K * K, 1.0)
    sigma_pol = (4.0 * np.pi ** 2 * omega_K / C_AU) / 3.0 * F2 / K2_safe
    sigma_pol = np.where(in_domain & (K > 0), sigma_pol, 0.0)
    F2_safe = np.where(F2 > 1e-30, F2, 1.0)
    tau_au  = np.where(F2 > 1e-30, Im / F2_safe, 0.0)
    tau_pol = tau_au * AU_TO_AS
    tau_pol = np.where(in_domain, tau_pol, 0.0)
    sigma_tau = sigma_pol * tau_pol

    fields = {"sigma": sigma_pol, "tau": tau_pol, "sigma_tau": sigma_tau}
    return kx, fields


# ============================================================================
# Lab-frame Euler-angle average  (SO(3) integration + closed-form path)
# ============================================================================
#
# What this section computes
# --------------------------
# For an ensemble of RANDOMLY ORIENTED molecules under LINEARLY POLARIZED
# light with the polarization vector along the lab +z axis, the
# lab-frame photoelectron differential cross-section is
#
#     dsigma_lab(k_e)/dOmega_e = ( 4*pi^2 * omega / (c * K^2) ) *
#                                 < |F_mol(R^T k_e; R^T z_lab)|^2 >_R
#
# where the average is over R ∈ SO(3) with Haar measure
# dR = (1/8*pi^2) sin(beta) dalpha dbeta dgamma and
#
#     F_mol(k_mol; eps_mol) = sum_q (eps_mol)_q * F^q(k_mol)
#     F^q(k_mol)            = 4*pi * sum_{ell,m} (-i)^ell d^q_{ell,m}(K)
#                                                   Y^R_{ell,m}(k_mol)
#
# By Yang's theorem for one-photon dipole ionization the result is
# axially symmetric around z_lab and reduces to
#
#     dsigma_lab/dOmega = (sigma_tot(K)/4pi) * (1 + beta(K) * P2(cos theta_lab))
#
# Two implementation paths, validated against each other:
#
#   * NUMERICAL.  Build sigma_lab(K, theta_lab) on a (n_K, n_theta) grid
#     by directly evaluating the SO(3) integral with Gauss-Legendre
#     quadrature in cos(beta) and trapezoid in (alpha, gamma).  Makes
#     no one-photon assumption.  ~ seconds for typical N.
#
#   * CLOSED-FORM.  Extract sigma_tot(K) and beta(K) from the numerical
#     2-D sigma_lab(K, theta_lab) via Legendre projection onto P0 and P2,
#     then rebuild the cube from the parametrisation above.  Identical
#     to NUMERICAL up to FP roundoff when Yang's theorem holds; the
#     L > 2 Legendre residuals quantify any one-photon-violation
#     (must be near machine epsilon for honest 1-photon input).
# ----------------------------------------------------------------------------


def _rot_zyz(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray
             ) -> np.ndarray:
    """ZYZ Euler rotation matrices, shape (..., 3, 3).

    Convention: a molecular-frame vector v_mol becomes
        v_lab = R(alpha, beta, gamma) @ v_mol
    so the lab-to-molecular rotation is R^T (which is what we use here
    for k_lab -> k_mol).
    """
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta,  dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    # R = Rz(alpha) @ Ry(beta) @ Rz(gamma)
    R = np.empty(a.shape + (3, 3), dtype=np.float64)
    R[..., 0, 0] = ca * cb * cg - sa * sg
    R[..., 0, 1] = -ca * cb * sg - sa * cg
    R[..., 0, 2] = ca * sb
    R[..., 1, 0] = sa * cb * cg + ca * sg
    R[..., 1, 1] = -sa * cb * sg + ca * cg
    R[..., 1, 2] = sa * sb
    R[..., 2, 0] = -sb * cg
    R[..., 2, 1] = sb * sg
    R[..., 2, 2] = cb
    return R


def _euler_grid(n_a: int, n_b: int, n_g: int
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SO(3) quadrature grid + weights normalised to integrate 1 -> 1.

    cos(beta) on Gauss-Legendre nodes over [-1, +1] (n_b nodes integrate
    polynomials of degree <= 2 n_b - 1 exactly).  alpha and gamma on a
    periodic trapezoid grid over [0, 2 pi).  For one-photon dipole
    transitions the integrand is degree <= 2 in each Wigner-D, so the
    grid is exact for n_a >= 5, n_b >= 3, n_g >= 5 (Yang's theorem L=2).

    Returns
    -------
    alphas, betas, gammas, weights : 3-D arrays of shape (n_a, n_b, n_g),
        flattened to size n_a*n_b*n_g.  sum(weights) == 1.0 (Haar measure).
    """
    # alpha, gamma : periodic trapezoid on [0, 2 pi) - exact for any
    # trigonometric polynomial of order < n.
    alpha = (np.arange(n_a) + 0.5) * (2.0 * np.pi / n_a)   # midpoint avoids 0=2pi double-count
    gamma = (np.arange(n_g) + 0.5) * (2.0 * np.pi / n_g)
    w_alpha = np.full(n_a, 1.0 / n_a)
    w_gamma = np.full(n_g, 1.0 / n_g)
    # cos(beta) : Gauss-Legendre on [-1, +1].  beta = arccos(cb).
    cb, w_cb = np.polynomial.legendre.leggauss(n_b)
    # Weights w_cb integrate functions of cos(beta) over [-1, +1];
    # sin(beta) dbeta = -d(cos beta), so int sin(beta) f(beta) dbeta
    # = int_{-1}^{+1} f(arccos cb) dcb -- the GL weights are already
    # the d cos(beta) measure on [-1, +1] (sum == 2).  Normalise so
    # the Haar measure ( sin beta / 4 pi ) integrates to 1:
    #   int dalpha (2pi) * int dgamma (2pi) * int_-1^1 dcb / (8 pi^2) = 1
    # -> w_b = w_cb / 2 so int dcb -> sum w_b == 1.
    beta_arr = np.arccos(np.clip(cb, -1.0, 1.0))
    w_beta = 0.5 * w_cb
    A, B, G = np.meshgrid(alpha, beta_arr, gamma, indexing="ij")
    WA, WB, WG = np.meshgrid(w_alpha, w_beta, w_gamma, indexing="ij")
    W = WA * WB * WG
    return (A.ravel(), B.ravel(), G.ravel(), W.ravel())


def _build_F_q_at_directions(ds_x: "DipoleDataset",
                              ds_y: "DipoleDataset",
                              ds_z: "DipoleDataset",
                              K_val: float,
                              theta: np.ndarray,
                              phi: np.ndarray,
                              with_dE: bool = False,
                              dD_x: np.ndarray = None,
                              dD_y: np.ndarray = None,
                              dD_z: np.ndarray = None,
                              ) -> Tuple[np.ndarray, ...]:
    """Evaluate the molecular-frame amplitudes
        F^q(K_val, theta_i, phi_i)  for q in (x, y, z)
    at a vector of (theta, phi) directions (shape (M,)).

    When `with_dE=False`: returns (Fx, Fy, Fz).
    When `with_dE=True`:  returns (Fx, Fy, Fz, dE_Fx, dE_Fy, dE_Fz)
        where dE_F^q = sum_{ell,m} (-i)^ell * (∂_E d^q_{ell,m})(K) * Y^R_{ell,m}
        The caller must pre-compute dD_x, dD_y, dD_z = ∂_E d^q_{ell,m}(k) on
        the dataset's k-grid (typically np.gradient(D, k, axis=0) / k[:, None]
        since ∂_E = (1/K) ∂_K).

    Linearly interpolates each channel's d^q_{ell,m}(K) (and ∂_E d^q_{ell,m})
    onto K_val.  K_val outside the dataset's [k_min, k_max] yields 0
    amplitudes (matches the molecular pipeline's in-domain mask).
    """
    M = theta.size
    if not (K_val >= ds_x.k[0] and K_val <= ds_x.k[-1]):
        zero = np.zeros(M, dtype=np.complex128)
        if with_dE:
            return zero.copy(), zero.copy(), zero.copy(), zero.copy(), zero.copy(), zero.copy()
        return zero.copy(), zero.copy(), zero.copy()
    Fx = np.zeros(M, dtype=np.complex128)
    Fy = np.zeros(M, dtype=np.complex128)
    Fz = np.zeros(M, dtype=np.complex128)
    if with_dE:
        dFx = np.zeros(M, dtype=np.complex128)
        dFy = np.zeros(M, dtype=np.complex128)
        dFz = np.zeros(M, dtype=np.complex128)
    for j, mu in enumerate(ds_x.mu_list):
        ell, m = idx_to_lm(mu)
        Y = real_Ylm(ell, m, theta, phi)
        ph = (-1j) ** ell
        dx = np.interp(K_val, ds_x.k, ds_x.dipole[:, j].real) \
             + 1j * np.interp(K_val, ds_x.k, ds_x.dipole[:, j].imag)
        dy = np.interp(K_val, ds_y.k, ds_y.dipole[:, j].real) \
             + 1j * np.interp(K_val, ds_y.k, ds_y.dipole[:, j].imag)
        dz = np.interp(K_val, ds_z.k, ds_z.dipole[:, j].real) \
             + 1j * np.interp(K_val, ds_z.k, ds_z.dipole[:, j].imag)
        Fx += ph * dx * Y
        Fy += ph * dy * Y
        Fz += ph * dz * Y
        if with_dE:
            ddx = np.interp(K_val, ds_x.k, dD_x[:, j].real) \
                  + 1j * np.interp(K_val, ds_x.k, dD_x[:, j].imag)
            ddy = np.interp(K_val, ds_y.k, dD_y[:, j].real) \
                  + 1j * np.interp(K_val, ds_y.k, dD_y[:, j].imag)
            ddz = np.interp(K_val, ds_z.k, dD_z[:, j].real) \
                  + 1j * np.interp(K_val, ds_z.k, dD_z[:, j].imag)
            dFx += ph * ddx * Y
            dFy += ph * ddy * Y
            dFz += ph * ddz * Y
    PRE = 4.0 * np.pi
    if with_dE:
        return Fx * PRE, Fy * PRE, Fz * PRE, dFx * PRE, dFy * PRE, dFz * PRE
    return Fx * PRE, Fy * PRE, Fz * PRE


def sigma_lab_2d(ds_x: "DipoleDataset",
                  ds_y: "DipoleDataset",
                  ds_z: "DipoleDataset",
                  K_grid: np.ndarray,
                  theta_lab_grid: np.ndarray,
                  n_a: int = 8, n_b: int = 8, n_g: int = 8,
                  *, with_tau: bool = False,
                  verbose: bool = False
                  ) -> Tuple[np.ndarray, ...]:
    """Build sigma_lab(K, theta_lab) and (optionally) the sigma-weighted
    lab-frame tau numerator <Im(F_mol* dE F_mol)>_R by SO(3) integration.

    For each (K, theta_lab) the sum over an Euler-angle grid is:
        sigma_lab(K, theta) = (4 pi^2 omega / c K^2)  * <|F_mol|^2>_R
        tau_num_lab (K, theta) =                        <Im(F_mol* dE F_mol)>_R
    With this normalisation the sigma-weighted Euler-averaged Wigner
    delay (your formula) is
        tau_avg(K, theta) = tau_num_lab(K, theta) / <|F_mol|^2>_R
                          = tau_num_lab(K, theta) / (sigma_lab / (4 pi^2 omega / c K^2))
    i.e. tau_avg in a.u. has the (4 pi^2 omega / c K^2) prefactor
    cancel between numerator and denominator -- it is a pure ratio of
    Euler averages.  See module docstring eq. (5).

    omega(K) is interpolated from ds_x.omega(k).  phi_lab is taken = 0
    by axial symmetry.

    Returns
    -------
    with_tau == False : sigma_2d                    (n_K, n_theta)
    with_tau == True  : sigma_2d, tau_num_2d        each (n_K, n_theta)
                         where tau_num is the raw <Im(F* dE F)> Euler average
                         (no prefactor); divide by <|F|^2>_R to get tau_avg.
    """
    alphas, betas, gammas, W = _euler_grid(n_a, n_b, n_g)
    R = _rot_zyz(alphas, betas, gammas)                  # (M, 3, 3)
    Rt = np.swapaxes(R, -1, -2)                           # (M, 3, 3)
    # eps_mol(R) = R^T z_lab = R^T[:, 2] -> shape (M, 3)
    eps_mol = Rt[..., :, 2]                                # (M, 3)

    # Pre-compute energy derivatives of every channel on the dataset's
    # k-grid: dE = (1/k) d/dk.  Reused inside the K-loop via interp.
    if with_tau:
        kvec = ds_x.k.copy()
        kvec_inv = 1.0 / kvec[:, None]
        dD_x = np.gradient(ds_x.dipole, kvec, axis=0) * kvec_inv
        dD_y = np.gradient(ds_y.dipole, ds_y.k, axis=0) * kvec_inv
        dD_z = np.gradient(ds_z.dipole, ds_z.k, axis=0) * kvec_inv
    else:
        dD_x = dD_y = dD_z = None

    nK = K_grid.size
    nT = theta_lab_grid.size
    sigma_2d = np.zeros((nK, nT), dtype=np.float64)
    tau_num_2d = np.zeros((nK, nT), dtype=np.float64) if with_tau else None
    F2_avg_2d  = np.zeros((nK, nT), dtype=np.float64) if with_tau else None
    for i, K in enumerate(K_grid):
        if not (K >= ds_x.k[0] and K <= ds_x.k[-1]):
            continue
        omega_K = float(np.interp(K, ds_x.k, ds_x.omega))
        prefactor = 4.0 * np.pi ** 2 * omega_K / (C_AU * K * K)
        for j, th_lab in enumerate(theta_lab_grid):
            k_lab = np.array([np.sin(th_lab), 0.0, np.cos(th_lab)])
            k_mol = Rt @ k_lab                              # (M, 3)
            r = np.linalg.norm(k_mol, axis=-1)
            theta_mol = np.arccos(np.clip(k_mol[..., 2] / np.where(r > 0, r, 1.0), -1, 1))
            phi_mol = np.arctan2(k_mol[..., 1], k_mol[..., 0])
            if with_tau:
                Fx, Fy, Fz, dFx, dFy, dFz = _build_F_q_at_directions(
                    ds_x, ds_y, ds_z, K, theta_mol, phi_mol,
                    with_dE=True, dD_x=dD_x, dD_y=dD_y, dD_z=dD_z)
                dF_mol = (eps_mol[..., 0] * dFx
                          + eps_mol[..., 1] * dFy
                          + eps_mol[..., 2] * dFz)
            else:
                Fx, Fy, Fz = _build_F_q_at_directions(
                    ds_x, ds_y, ds_z, K, theta_mol, phi_mol)
            F_mol = (eps_mol[..., 0] * Fx
                     + eps_mol[..., 1] * Fy
                     + eps_mol[..., 2] * Fz)
            F2 = F_mol.real ** 2 + F_mol.imag ** 2
            sigma_2d[i, j] = prefactor * float(np.sum(W * F2))
            if with_tau:
                # Im(F* dF) = F.re * dF.im - F.im * dF.re
                Im_FdF = F_mol.real * dF_mol.imag - F_mol.imag * dF_mol.real
                tau_num_2d[i, j] = float(np.sum(W * Im_FdF))
                F2_avg_2d[i, j]  = float(np.sum(W * F2))
        if verbose and (i % 10 == 0 or i == nK - 1):
            print(f"    sigma_lab_2d: K[{i+1}/{nK}] = {K:.4f}  "
                  f"max sigma over theta = {sigma_2d[i].max():.3e}"
                  + (f"  max |tau_num| = {np.abs(tau_num_2d[i]).max():.3e}"
                     if with_tau else ""))
    if with_tau:
        return sigma_2d, tau_num_2d, F2_avg_2d
    return sigma_2d


def beta_sigma_from_2d(theta_lab_grid: np.ndarray, sigma_2d: np.ndarray,
                       L_check_max: int = 4
                       ) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Extract sigma_tot(K) and beta(K) from sigma_lab(K, theta_lab) via
    Legendre projection.  Also compute L = 0..L_check_max moments so the
    caller can sanity-check Yang's theorem (L > 2 must be ~ 0 for
    one-photon dipole input).

    Parametrisation:
        sigma_lab(K, theta) = (sigma_tot(K) / 4 pi) * (1 + beta(K) * P_2(cos theta))
    Multiplying by P_L(cos theta) and integrating over solid angle gives
        B_L(K) = 2 pi * int_{-1}^{+1} sigma_lab(K, theta) P_L d(cos theta)
              = 2 pi * int_0^pi sigma_lab(K, theta) P_L(cos theta) sin theta dtheta
    with B_0 = sigma_tot, B_2 = sigma_tot * beta / 5, B_L = 0 for L = 1
    and L > 2 (Yang's theorem).  Hence
        beta(K) = 5 * B_2(K) / B_0(K).
    The factor of 5 = 1 / (∫ P_2^2 d cos theta / 2) = 1 / (1/5) is the
    Legendre orthogonality constant for L = 2; missing it is the most
    common sign / scale error in beta extractions.

    Quadrature.  theta_lab_grid is assumed uniform in theta over [0, pi]
    (which is what sigma_lab_2d builds).  We integrate over theta using
    composite Simpson's rule (requires an ODD number of grid points;
    falls back to trapezoid otherwise).  The sin(theta) Jacobian
    converts d cos theta -> sin theta dtheta and is folded into the
    integrand explicitly.

    Returns
    -------
    sigma_tot : (n_K,) array
    beta      : (n_K,) array  (NaN where sigma_tot < 1e-30)
    B_L       : dict { L : (n_K,) array } for L = 0..L_check_max
    """
    theta = np.asarray(theta_lab_grid, dtype=np.float64)
    n_theta = theta.size
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    # Composite Simpson weights for uniform grid (requires odd count);
    # otherwise trapezoid.
    h = theta[1] - theta[0]
    if (n_theta % 2) == 1 and n_theta >= 3:
        w = np.ones(n_theta) * (h / 3.0)
        w[1:-1:2] *= 4.0
        w[2:-1:2] *= 2.0
    else:
        w = np.ones(n_theta) * h
        w[0] *= 0.5; w[-1] *= 0.5

    nK = sigma_2d.shape[0]
    B = {L: np.zeros(nK) for L in range(L_check_max + 1)}
    for L in range(L_check_max + 1):
        PL = np.polynomial.legendre.legval(cos_th, [0] * L + [1.0])
        kernel = w * PL * sin_th       # combined quadrature factor
        for i in range(nK):
            B[L][i] = 2.0 * np.pi * float(np.sum(sigma_2d[i] * kernel))
    sigma_tot = B[0].copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(np.abs(sigma_tot) > 1e-30,
                        5.0 * B[2] / sigma_tot, np.nan)
    return sigma_tot, beta, B


def build_lab_euler_kcube_numerical(ds_x: "DipoleDataset",
                                     ds_y: "DipoleDataset",
                                     ds_z: "DipoleDataset",
                                     N: int, k_max: float,
                                     *, n_K: int = 80, n_theta: int = 41,
                                     n_a: int = 8, n_b: int = 8, n_g: int = 8,
                                     with_tau: bool = True,
                                     verbose: bool = False
                                     ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build lab-frame Euler-angle-averaged cubes on the same
    N x N x N k-cube as build_pol_avg_kcube.  Heavy step is sigma_lab_2d
    over (n_K, n_theta); the cube is filled by bilinear interp.

    Returns
    -------
    kx     : (N,) k-axis (same as build_pol_avg_kcube)
    fields : dict.  Always contains 'sigma_avg' : (N, N, N) cube
             (= <sigma>_R, the orientation-averaged differential cross-section
              == "sigma_lab" in earlier docs; the rename is for symmetry with
              tau_avg below; both are Euler-angle averages).
             When with_tau=True (default) ALSO contains:
                'tau_avg'      : sigma-weighted Euler-averaged Wigner delay,
                                 = <Im(F* dE F)>_R / <|F|^2>_R, in attoseconds
                'sigma_tau_avg': sigma_avg * tau_avg  (cube of the
                                 photoelectron-yield-weighted delay,
                                 cheap convenience output)
    """
    if N % 2 != 0: raise ValueError("N must be even")
    K_grid = np.linspace(ds_x.k[0], ds_x.k[-1], n_K)
    theta_grid = np.linspace(0.0, np.pi, n_theta)
    if verbose:
        print(f"  sigma_lab_2d: n_K={n_K}  n_theta={n_theta}  "
              f"Euler grid {n_a}x{n_b}x{n_g}={n_a*n_b*n_g}  with_tau={with_tau}")
    if with_tau:
        sigma_2d, tau_num_2d, F2_avg_2d = sigma_lab_2d(
            ds_x, ds_y, ds_z, K_grid, theta_grid,
            n_a=n_a, n_b=n_b, n_g=n_g, with_tau=True, verbose=verbose)
        # tau_avg [au] = <Im(F* dE F)> / <|F|^2>;  threshold below noise floor.
        F2_safe = np.where(F2_avg_2d > 1e-30, F2_avg_2d, 1.0)
        tau_avg_2d_au = np.where(F2_avg_2d > 1e-30, tau_num_2d / F2_safe, 0.0)
        tau_avg_2d_as = tau_avg_2d_au * AU_TO_AS
    else:
        sigma_2d = sigma_lab_2d(ds_x, ds_y, ds_z, K_grid, theta_grid,
                                 n_a=n_a, n_b=n_b, n_g=n_g, verbose=verbose)

    # Build the 3-D cube via bilinear interp.
    dk = 2.0 * k_max / N
    kx = (np.arange(N) - N // 2) * dk
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K = np.sqrt(KX * KX + KY * KY + KZ * KZ)
    in_domain = (K >= ds_x.k[0]) & (K <= ds_x.k[-1])
    with np.errstate(invalid="ignore"):
        theta_cube = np.arccos(np.clip(np.divide(KZ, K, where=(K > 0)), -1.0, 1.0))
    theta_cube = np.where(K > 0, theta_cube, 0.0)
    K_flat = K.ravel(); th_flat = theta_cube.ravel()
    dK_grid = K_grid[1] - K_grid[0]
    dth     = theta_grid[1] - theta_grid[0]
    i_K = (K_flat - K_grid[0]) / dK_grid
    i_T = (th_flat - theta_grid[0]) / dth
    i_K = np.clip(i_K, 0.0, n_K - 1.0001)
    i_T = np.clip(i_T, 0.0, n_theta - 1.0001)
    iK0 = i_K.astype(np.int64); iK1 = iK0 + 1
    iT0 = i_T.astype(np.int64); iT1 = iT0 + 1
    fK = i_K - iK0; fT = i_T - iT0

    def _bilinear(grid_2d):
        flat = ((1 - fK) * (1 - fT) * grid_2d[iK0, iT0]
                + fK       * (1 - fT) * grid_2d[iK1, iT0]
                + (1 - fK) * fT       * grid_2d[iK0, iT1]
                + fK       * fT       * grid_2d[iK1, iT1])
        return flat.reshape(K.shape)

    sig_cube = _bilinear(sigma_2d)
    sig_cube = np.where(in_domain, sig_cube, 0.0)
    fields = {"sigma_avg":      sig_cube,
              "sigma_lab":      sig_cube,           # backward-compat alias
              "_sigma_avg_2d":  sigma_2d,
              "_K_grid":        K_grid,
              "_theta_grid":    theta_grid}
    if with_tau:
        tau_cube = _bilinear(tau_avg_2d_as)
        tau_cube = np.where(in_domain, tau_cube, 0.0)
        fields["tau_avg"]       = tau_cube
        fields["sigma_tau_avg"] = sig_cube * tau_cube
        fields["_tau_avg_2d"]   = tau_avg_2d_as
    return kx, fields


def build_lab_euler_kcube_closed_form(ds_x: "DipoleDataset",
                                       ds_y: "DipoleDataset",
                                       ds_z: "DipoleDataset",
                                       N: int, k_max: float,
                                       *, n_K: int = 80, n_theta: int = 41,
                                       n_a: int = 8, n_b: int = 8, n_g: int = 8,
                                       with_tau: bool = True,
                                       verbose: bool = False
                                       ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Same observable as build_lab_euler_kcube_numerical but the cube
    is rebuilt from Legendre-moment parametrisations:

        sigma_lab(K, theta) = (sigma_tot(K) / 4 pi) * (1 + beta(K) * P2(cos theta))

    For the sigma-weighted Euler-averaged tau (your formula), Yang's
    theorem applies to both numerator <Im(F* dE F)>_R and denominator
    <|F|^2>_R separately (both are bilinear in the dipole amplitudes,
    rank 2 in Wigner-D's).  The Legendre expansion of each is L = 0, 2
    only:
        <Im(F* dE F)>_R(K, theta) = (N0(K) / 4 pi) * (1 + b_N(K) * P2(cos theta))
        <|F|^2>_R     (K, theta) = (D0(K) / 4 pi) * (1 + b_D(K) * P2(cos theta))
    so the closed-form tau_avg cube is the POINTWISE RATIO
        tau_avg(K, theta) = (N0/D0) * (1 + b_N P2) / (1 + b_D P2)         [a.u.]
    multiplied by AU_TO_AS for attoseconds.  This is bit-equivalent to
    the numerical path for true one-photon input.

    Returns
    -------
    kx, fields : as in build_lab_euler_kcube_numerical.  fields includes
                  sigma_tot, beta (and when with_tau: tau_N0, tau_bN,
                  tau_D0, tau_bD, all (n_K,) curves) for diagnostics.
    """
    if N % 2 != 0: raise ValueError("N must be even")
    K_grid = np.linspace(ds_x.k[0], ds_x.k[-1], n_K)
    theta_grid = np.linspace(0.0, np.pi, n_theta)
    if with_tau:
        sigma_2d, tau_num_2d, F2_avg_2d = sigma_lab_2d(
            ds_x, ds_y, ds_z, K_grid, theta_grid,
            n_a=n_a, n_b=n_b, n_g=n_g, with_tau=True, verbose=verbose)
    else:
        sigma_2d = sigma_lab_2d(ds_x, ds_y, ds_z, K_grid, theta_grid,
                                 n_a=n_a, n_b=n_b, n_g=n_g, verbose=verbose)
    sigma_tot, beta, B_sigma = beta_sigma_from_2d(theta_grid, sigma_2d)
    if with_tau:
        # Legendre moments of <Im(F* dE F)>_R (numerator of tau_avg).
        _, _, B_num = beta_sigma_from_2d(theta_grid, tau_num_2d)
        # Legendre moments of <|F|^2>_R (denominator of tau_avg);
        # note this differs from sigma_lab by the (4 pi^2 omega / c K^2)
        # prefactor (K-dependent), which cancels in the ratio anyway.
        _, _, B_den = beta_sigma_from_2d(theta_grid, F2_avg_2d)
        tau_N0 = B_num[0].copy()
        tau_bN = np.where(np.abs(B_num[0]) > 1e-30, 5.0 * B_num[2] / B_num[0], 0.0)
        tau_D0 = B_den[0].copy()
        tau_bD = np.where(np.abs(B_den[0]) > 1e-30, 5.0 * B_den[2] / B_den[0], 0.0)

    dk = 2.0 * k_max / N
    kx = (np.arange(N) - N // 2) * dk
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K = np.sqrt(KX * KX + KY * KY + KZ * KZ)
    in_domain = (K >= ds_x.k[0]) & (K <= ds_x.k[-1])
    with np.errstate(invalid="ignore"):
        cos_th = np.where(K > 0, KZ / np.where(K > 0, K, 1.0), 0.0)
    P2 = 0.5 * (3.0 * cos_th * cos_th - 1.0)
    sig_tot_cube = np.interp(K, K_grid, sigma_tot, left=0.0, right=0.0)
    beta_cube    = np.interp(K, K_grid, beta,      left=0.0, right=0.0)
    sig_cube = (sig_tot_cube / (4.0 * np.pi)) * (1.0 + beta_cube * P2)
    sig_cube = np.where(in_domain, sig_cube, 0.0)

    fields = {"sigma_avg":   sig_cube,
              "sigma_lab":   sig_cube,        # backward-compat alias
              "sigma_tot":   sigma_tot,
              "beta":        beta,
              "_K_grid":     K_grid,
              "_theta_grid": theta_grid,
              "_B_L":        B_sigma}
    if with_tau:
        tN0_cube = np.interp(K, K_grid, tau_N0, left=0.0, right=0.0)
        tbN_cube = np.interp(K, K_grid, tau_bN, left=0.0, right=0.0)
        tD0_cube = np.interp(K, K_grid, tau_D0, left=0.0, right=0.0)
        tbD_cube = np.interp(K, K_grid, tau_bD, left=0.0, right=0.0)
        num_2d   = (tN0_cube / (4.0 * np.pi)) * (1.0 + tbN_cube * P2)
        den_2d   = (tD0_cube / (4.0 * np.pi)) * (1.0 + tbD_cube * P2)
        # threshold where the denominator (<|F|^2>) has no signal
        thresh = 1e-12 * float(np.abs(tD0_cube).max() if tD0_cube.size else 0.0)
        den_safe = np.where(np.abs(den_2d) > thresh, den_2d, 1.0)
        tau_cube_au = np.where(np.abs(den_2d) > thresh, num_2d / den_safe, 0.0)
        tau_cube_as = tau_cube_au * AU_TO_AS
        tau_cube_as = np.where(in_domain, tau_cube_as, 0.0)
        fields["tau_avg"]      = tau_cube_as
        fields["sigma_tau_avg"] = sig_cube * tau_cube_as
        fields["tau_N0"]       = tau_N0
        fields["tau_bN"]       = tau_bN
        fields["tau_D0"]       = tau_D0
        fields["tau_bD"]       = tau_bD
    return kx, fields


# ============================================================================
# 3-D FFT with centered input AND centered output
# ============================================================================
def fft3_centered(kx: np.ndarray, f_kcube: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Centered 3-D inverse FFT.

    Convention: continuous IFT  F(r) = (1/(2π)^3) ∫ f(k) e^{i k·r} d³k.
    Numerical relation:  F(r_m) ≈ (N dk / 2π)^3 · numpy_ifftn(...)[m]
                                 = (1/dr)^3 · numpy_ifftn(...)[m],
    with the numpy call applied as ifftshift → ifftn → fftshift.
    Returns the RAW shifted DFT; multiply by (1/dr)^3 externally for
    physical scale (see --normalize physical in main()).
    """
    if kx.size != f_kcube.shape[0]:
        raise ValueError("kx length must equal cube axis size")
    if not (f_kcube.shape[0] == f_kcube.shape[1] == f_kcube.shape[2]):
        raise ValueError("cube must be cubic 3-D")
    N  = kx.size
    dk = kx[1] - kx[0]
    dr = 2.0 * np.pi / (N * dk)
    rx = (np.arange(N) - N // 2) * dr
    f_r = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f_kcube)))
    return rx, f_r


# ============================================================================
# Gaussian .cube writer
# ============================================================================
def write_cube(path: Path, rx: np.ndarray, f_real: np.ndarray,
               *, title: str = "", atoms=None) -> None:
    """Write a real (N, N, N) grid to a Gaussian .cube file (atomic units).

    Cube format (per Gaussian docs):
        line 1: title
        line 2: description
        line 3: Natoms  Ox Oy Oz             (origin)
        line 4: Nx  Vxx Vxy Vxz              (voxel basis row 1)
        line 5: Ny  Vyx Vyy Vyz              (         row 2)
        line 6: Nz  Vzx Vzy Vzz              (         row 3)
        atoms (Natoms lines): Z  q  x  y  z
        data: 6 values/line, innermost loop = z

    Centering: origin = rx[0] (= -(N/2)·dr for even N), voxel spacing dr
    along each cardinal axis.  Innermost-loop z matches numpy C-order
    ravel of a (N, N, N) array indexed (x, y, z).
    """
    if atoms is None:
        atoms = []
    N  = rx.size
    dr = rx[1] - rx[0]
    O  = rx[0]
    with open(path, "w") as fh:
        fh.write(f"{title or 'polar_fft.py output'}\n")
        fh.write("atomic units (bohr); 3-D inverse FFT of polar k-cube\n")
        fh.write(f" {len(atoms):4d}  {O: .6e}  {O: .6e}  {O: .6e}\n")
        fh.write(f" {N:4d}  {dr: .6e}  {0.0: .6e}  {0.0: .6e}\n")
        fh.write(f" {N:4d}  {0.0: .6e}  {dr: .6e}  {0.0: .6e}\n")
        fh.write(f" {N:4d}  {0.0: .6e}  {0.0: .6e}  {dr: .6e}\n")
        for (Z, q, x, y, z) in atoms:
            fh.write(f" {Z:4d}  {q: .6e}  {x: .6e}  {y: .6e}  {z: .6e}\n")
        flat = np.ascontiguousarray(f_real, dtype=np.float64).ravel(order="C")
        for i in range(0, flat.size, 6):
            fh.write(" ".join(f"{v: .6e}" for v in flat[i:i + 6]) + "\n")


# ============================================================================
# CLI driver
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path,
        help="Gather dir with dipole_{gauge}_homo_{q}_<mu>.dat files.")
    ap.add_argument("--output-dir", required=True, type=Path,
        help="Where to write the three .cube files.")
    ap.add_argument("--gauge", choices=("len", "vel"), default="len")
    ap.add_argument("-N", "--n-cube", type=int, default=128,
        help="Cube side length (must be even).  Default 128.")
    ap.add_argument("--k-max", type=float, default=None,
        help="Half-extent of the cube in k-space (a.u.).  Default: "
             "max k from the gather data.")
    ap.add_argument("--part", choices=("real", "imag", "abs"), default="real",
        help="Which part of the complex FFT to write (default: real).")
    ap.add_argument("--normalize", choices=("none", "physical"), default="none",
        help="'none' (default): raw shifted DFT.  'physical': multiply "
             "by (N dk / 2π)^3 so the values approximate the continuous "
             "inverse FT with convention F(r) = (1/(2π)^3) ∫ f e^{ikr} d³k.")
    ap.add_argument("--mode",
        choices=("molecular", "euler-numerical", "euler-closed"),
        default="molecular",
        help="Which observable to FFT: "
             "'molecular' (default) -- molecular-frame polarisation-averaged "
             "(1/3) sum_q |F_q|^2, the original behaviour.  "
             "'euler-numerical' -- lab-frame randomly-oriented-molecule "
             "average via direct SO(3) integration (no one-photon assumption).  "
             "'euler-closed' -- same observable as euler-numerical but the "
             "cube is rebuilt from (sigma_tot(K), beta(K)) via the closed-form "
             "(1 + beta P_2) parametrisation (exact for one-photon by Yang's "
             "theorem).  See module docstring for the physics derivation.")
    ap.add_argument("--euler-n-K", type=int, default=80,
        help="(euler modes) number of K-grid samples for sigma_lab_2d.")
    ap.add_argument("--euler-n-theta", type=int, default=81,
        help="(euler modes) number of theta-grid samples; must be ODD for "
             "Simpson rule in the beta-extraction projection.")
    ap.add_argument("--euler-quad", type=int, nargs=3, default=(8, 8, 8),
        metavar=("N_A", "N_B", "N_G"),
        help="(euler modes) SO(3) integration grid: trapezoid in alpha, gamma "
             "and Gauss-Legendre in cos(beta).  Default 8x8x8 is exact for "
             "one-photon dipole integrands (Yang theorem).")
    ap.add_argument("--fields", default="sigma,tau,sigma_tau",
        help="Comma list of fields to FFT (default: all three for the "
             "molecular mode; the euler modes have just 'sigma_lab').")
    ap.add_argument("--verbose", action="store_true",
        help="Print progress every 32 channels.")
    args = ap.parse_args()

    print("=" * 72)
    print(" polar_fft.py")
    print("=" * 72)
    if args.n_cube % 2 != 0:
        raise SystemExit("N must be even")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  input-dir  : {args.input_dir}")
    print(f"  gauge      : {args.gauge}")
    ds_x = load_dipole_dataset(args.input_dir, args.gauge, "x")
    ds_y = load_dipole_dataset(args.input_dir, args.gauge, "y")
    ds_z = load_dipole_dataset(args.input_dir, args.gauge, "z")
    print(f"  channels   : {len(ds_x.mu_list)}   "
          f"k range : [{ds_x.k[0]:.4f}, {ds_x.k[-1]:.4f}] a.u.   "
          f"N_k = {ds_x.k.size}")
    k_max = float(ds_x.k[-1]) if args.k_max is None else float(args.k_max)
    N = args.n_cube
    dk = 2.0 * k_max / N
    dr = 2.0 * np.pi / (N * dk)
    print(f"  cube       : N = {N}   k_max = {k_max:.4f} a.u.   dk = {dk:.4e}")
    print(f"  r grid     : dr = {dr:.4e} bohr   r_max = {N/2 * dr:.4f} bohr")
    print(f"  normalize  : {args.normalize}    part: {args.part}")

    print(f"  mode       : {args.mode}")
    n_a, n_b, n_g = args.euler_quad
    if args.mode == "molecular":
        kx, fields = build_pol_avg_kcube(ds_x, ds_y, ds_z, N, k_max,
                                         verbose=args.verbose)
        plot_fields = {k: v for k, v in fields.items() if not k.startswith("_")}
    elif args.mode == "euler-numerical":
        print(f"  euler grid : n_K={args.euler_n_K}  n_theta={args.euler_n_theta}  "
              f"SO(3) {n_a}x{n_b}x{n_g}")
        kx, fields = build_lab_euler_kcube_numerical(
            ds_x, ds_y, ds_z, N, k_max,
            n_K=args.euler_n_K, n_theta=args.euler_n_theta,
            n_a=n_a, n_b=n_b, n_g=n_g, verbose=args.verbose)
        plot_fields = {"sigma_avg":     fields["sigma_avg"],
                       "tau_avg":       fields["tau_avg"],
                       "sigma_tau_avg": fields["sigma_tau_avg"]}
    else:  # euler-closed
        print(f"  euler grid : n_K={args.euler_n_K}  n_theta={args.euler_n_theta}  "
              f"SO(3) {n_a}x{n_b}x{n_g}")
        kx, fields = build_lab_euler_kcube_closed_form(
            ds_x, ds_y, ds_z, N, k_max,
            n_K=args.euler_n_K, n_theta=args.euler_n_theta,
            n_a=n_a, n_b=n_b, n_g=n_g, verbose=args.verbose)
        plot_fields = {"sigma_avg":     fields["sigma_avg"],
                       "tau_avg":       fields["tau_avg"],
                       "sigma_tau_avg": fields["sigma_tau_avg"]}
        if args.verbose:
            B = fields["_B_L"]; K_grid = fields["_K_grid"]
            for i in range(0, len(K_grid), max(1, len(K_grid) // 8)):
                print(f"    K={K_grid[i]:.3f}  sigma_tot={fields['sigma_tot'][i]: .3e}  "
                      f"beta={fields['beta'][i]: .4f}  "
                      f"|B4|/B0={abs(B[4][i] / max(B[0][i], 1e-30)): .2e}")

    print("  k-cube built; field ranges:")
    for name, f in plot_fields.items():
        print(f"    {name:10s}  min={f.min(): .3e}  max={f.max(): .3e}  "
              f"|max|={np.abs(f).max(): .3e}")
    fields = plot_fields

    # In the euler modes the field names differ; remap the default CLI
    # --fields setting so the user doesn't have to override it.
    if args.mode != "molecular" and args.fields == "sigma,tau,sigma_tau":
        args.fields = "sigma_avg,tau_avg,sigma_tau_avg"

    want = [s.strip() for s in args.fields.split(",") if s.strip()]
    for name in want:
        if name not in fields:
            raise SystemExit(f"unknown field '{name}'; have {list(fields)}")
        rx, f_r = fft3_centered(kx, fields[name])
        if args.normalize == "physical":
            f_r = f_r * (1.0 / dr) ** 3
        if   args.part == "real": f_out = f_r.real
        elif args.part == "imag": f_out = f_r.imag
        else:                      f_out = np.abs(f_r)
        out_path = args.output_dir / f"{name}_{args.part}.cube"
        write_cube(out_path, rx, f_out,
                   title=f"FFT[{name}]  part={args.part}  norm={args.normalize}")
        print(f"  wrote {out_path}   "
              f"min={f_out.min(): .3e}  max={f_out.max(): .3e}  "
              f"|max|={np.abs(f_out).max(): .3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
