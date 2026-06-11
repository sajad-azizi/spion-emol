#!/usr/bin/env python3
"""
compute_polar_data.py
=====================
Generate angle-resolved photoionization data on a (E, θ) grid at φ=0 for:

  (A) FIXED LAB-FRAME LASER ORIENTATION  (Euler angles α, β; γ irrelevant for
      linear polarization).  Uses the dipole operator of Eq. (S26-S27) of the
      Azizi et al. supplementary:
          µ̂  =  cos α sin β · x̂ + sin α sin β · ŷ + cos β · ẑ
      For each Cartesian component q ∈ {x, y, z} we have per-channel reduced
      dipoles d^q_{ℓm}(E) from the gather step.  At fixed (α, β) the angular
      dipole amplitude is

          D^{(-)}(θ; α, β; E)  =  (4π/k) · Σ_{ℓ m}  i^{-ℓ}  d^{eff}_{ℓm}(E)
                                                          · Y^R_{ℓm}(θ, φ=0),

      with d^{eff}_{ℓm} = cos α sin β · d^x + sin α sin β · d^y + cos β · d^z.

      Then [Eqs. (S22) and (S25)]:
          σ(θ; α, β; E) = (4π² ω / c) · |D^{(-)}|²
          τ(θ; α, β; E) = Im[D^{(-)*} ∂_E D^{(-)}] / |D^{(-)}|²

      Default α = β = 0 ⇒ pure z-polarization (the term  cos β · ẑ  of S27).

  (B) ISOTROPIC POLARIZATION AVERAGE  [Eq. (S38) of the supplementary].
      Define the three per-Cartesian amplitudes
          D_q(θ; E)  =  (4π/k) · Σ_{ℓ m}  i^{-ℓ}  d^q_{ℓm}(E)  Y^R_{ℓm}(θ, 0)
      and use  ⟨F_i F_j⟩_F̂ = δ_{ij}/3  to obtain
          σ_pol(θ; E) = (4π² ω / c) · (1/3) · Σ_q |D_q(θ; E)|²
          τ_pol(θ; E) = Σ_q Im[D_q* ∂_E D_q] / Σ_q |D_q|²

  Each block also outputs the product σ·τ as a third channel (the user-
  requested third panel; this is the unnormalised "delay-weighted intensity"
  σ τ = (4π²ω/c) · Im[D^* ∂_E D]).

Numerics
--------
Real-Y_{ℓm}(θ, φ=0) is computed by a STABLE normalised-Legendre 3-term
recurrence (Press et al.); this avoids the factorial-overflow NaN of
scipy.special.sph_harm at high (ℓ, |m|) on older scipy.  Stable to ℓ ≳ 1000.

The input gathered files are read in the same convention as
postprocessing/angular_resolving/generate_data_fig2_correct.py — your
existing C₈F₈ pipeline output.

Outputs
-------
Two text files in --output-dir:

  polar_fixed_alpha{α_deg}_beta{β_deg}_{gauge}.dat
  polar_pol_avg_{gauge}.dat

Each has columns

  1 k          [a.u.]
  2 E_kin      [a.u.]
  3 omega      [a.u.]
  4 theta      [rad]
  5 sigma      [bohr²]
  6 tau        [as]
  7 sigma_tau  [bohr² · as]   ( = σ·τ )

The plot_polar.py companion reads these and renders the polar panels.
"""
from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ----------------------- physical constants ------------------------
C_AU = 137.035999139           # CODATA 1/α (speed of light, atomic units)
AU_TO_AS = 24.188843265857     # CODATA atomic time unit in attoseconds
HA_TO_EV = 27.211386245988

FILE_RE = re.compile(r"^dipole_(len|vel)_homo_([xyz])_(\d+)\.dat$")


@dataclass
class DipoleDataset:
    k: np.ndarray              # (nE,) momenta in a.u.
    e_kin: np.ndarray          # (nE,) photoelectron kinetic energy [a.u.]
    omega: np.ndarray          # (nE,) photon energy [a.u.]
    mu_list: List[int]         # channel indices μ = ℓ² + ℓ + m
    dipole: np.ndarray         # (nE, nCh) complex per-channel dipole
    gauge: str
    component: str             # 'x' / 'y' / 'z'


# ----------------------- channel index helpers -------------------
def idx_to_lm(idx: int) -> Tuple[int, int]:
    ell = int(np.floor(np.sqrt(idx)))
    while (ell + 1) * (ell + 1) <= idx:
        ell += 1
    return ell, idx - ell * ell - ell


# -------------- numerically-stable real-Y_{ℓm}(θ, φ=0) ------------
def real_Ylm_phi0(ell: int, m: int, theta: np.ndarray) -> np.ndarray:
    """Real spherical harmonic Y^R_{ℓ,m}(θ, φ=0) using a normalised-Legendre
    3-term recurrence.  For m < 0, returns 0 (sin(|m|·0) = 0).

    Recurrence (Press et al., Numerical Recipes, Eq. 6.7.9 normalised form):
        P̃_0^0   = 1/(2 √π)
        P̃_k^k   = -sqrt((2k+1)/(2k)) · sin θ · P̃_{k-1}^{k-1}            (k≥1)
        P̃_{m+1}^m = sqrt(2m+3) · cos θ · P̃_m^m
        P̃_l^m   = a_l · cos θ · P̃_{l-1}^m  −  b_l · P̃_{l-2}^m       (l ≥ m+2)
        a_l = sqrt((4l² − 1)/(l² − m²))
        b_l = sqrt((2l+1)/(2l−3) · ((l−1)² − m²)/(l² − m²))
    Then  Y^R_{ℓ,0}(θ, 0) = P̃_ℓ^0
           Y^R_{ℓ,m}(θ, 0) = √2 · (−1)^m · P̃_ℓ^m   (m > 0; cos(m·0) = 1)
    """
    theta = np.asarray(theta, float)
    if m < 0:
        return np.zeros_like(theta)
    am = m
    if am > ell:
        return np.zeros_like(theta)
    x = np.cos(theta); s = np.sin(theta)
    P_mm = np.full_like(x, 1.0 / (2.0 * np.sqrt(np.pi)))
    for k in range(1, am + 1):
        P_mm = -np.sqrt((2.0 * k + 1.0) / (2.0 * k)) * s * P_mm
    if ell == am:
        P_lm = P_mm
    else:
        P_p2 = P_mm
        P_p1 = np.sqrt(2.0 * am + 3.0) * x * P_mm
        if ell == am + 1:
            P_lm = P_p1
        else:
            for ll in range(am + 2, ell + 1):
                a = np.sqrt((4.0 * ll * ll - 1.0) / (ll * ll - am * am))
                b = np.sqrt((2.0 * ll + 1.0) / (2.0 * ll - 3.0)
                            * ((ll - 1.0) * (ll - 1.0) - am * am)
                            / (ll * ll - am * am))
                P_cur = a * x * P_p1 - b * P_p2
                P_p2 = P_p1; P_p1 = P_cur
            P_lm = P_p1
    if m == 0:
        return P_lm
    # m > 0
    return np.sqrt(2.0) * ((-1) ** m) * P_lm


# ----------------------- I/O -------------------------------------
def discover_mu_files(input_dir: Path, gauge: str, component: str
                      ) -> List[Tuple[int, Path]]:
    matches: List[Tuple[int, Path]] = []
    for path in sorted(input_dir.glob(f"dipole_{gauge}_homo_{component}_*.dat")):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        g, comp, mu_str = m.groups()
        if g == gauge and comp == component:
            matches.append((int(mu_str), path))
    if not matches:
        raise FileNotFoundError(
            f"No files dipole_{gauge}_homo_{component}_*.dat in {input_dir}")
    matches.sort(key=lambda x: x[0])
    return matches


def load_one_channel(path: Path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 5:
        raise ValueError(
            f"{path}: expected ≥5 columns (k, E_kin, omega, Re d, Im d); got {data.shape[1]}")
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3] + 1j * data[:, 4]


def load_dipole_dataset(input_dir: Path, gauge: str, component: str
                        ) -> DipoleDataset:
    mu_files = discover_mu_files(input_dir, gauge, component)
    mu_list = [mu for mu, _ in mu_files]
    ref_k, ref_e, ref_omega, ref_d = load_one_channel(mu_files[0][1])
    nE = ref_k.size; nCh = len(mu_files)
    D = np.zeros((nE, nCh), dtype=np.complex128)
    D[:, 0] = ref_d
    for j, (mu, path) in enumerate(mu_files[1:], start=1):
        k_j, e_j, w_j, d_j = load_one_channel(path)
        if not (np.allclose(k_j, ref_k) and np.allclose(e_j, ref_e)
                and np.allclose(w_j, ref_omega)):
            raise ValueError(f"energy grid mismatch in {path}")
        D[:, j] = d_j
    return DipoleDataset(ref_k, ref_e, ref_omega, mu_list, D, gauge, component)


# ----------------------- physics ---------------------------------
def per_cartesian_amplitude_at_phi0(
        ds: DipoleDataset, theta: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute, for a single Cartesian polarization q ∈ {x,y,z}:

        F_q(E, θ)     = 4π · Σ_{ℓm} i^{-ℓ} · d^q_{ℓm}(E) · Y^R_{ℓm}(θ, φ=0)
        ∂_E F_q(E, θ) = 4π · Σ_{ℓm} i^{-ℓ} · ∂_E d^q_{ℓm}(E) · Y^R_{ℓm}(θ, φ=0)

    Returns (F, dF_dE), each of shape (nE, nT), complex.

    The 1/k factor of D = F/k is applied later in compute_observables(),
    since σ uses |F|²/k² and τ is unaffected by k (cancels).
    """
    k = ds.k
    D = ds.dipole
    mu_list = ds.mu_list
    nE = k.size; nT = theta.size; nCh = len(mu_list)

    # Precompute Y^R(θ, φ=0) and i^{-ℓ}
    Y = np.zeros((nCh, nT), dtype=np.float64)
    phase = np.zeros(nCh, dtype=np.complex128)
    for j, mu in enumerate(mu_list):
        ell, m = idx_to_lm(mu)
        Y[j] = real_Ylm_phi0(ell, m, theta)
        phase[j] = (-1j) ** ell

    # ∂_E d = (∂_k d)/k
    dD_dE = np.zeros_like(D)
    for j in range(nCh):
        dD_dE[:, j] = np.gradient(D[:, j], k) / k

    F = np.zeros((nE, nT), dtype=np.complex128)
    dF = np.zeros((nE, nT), dtype=np.complex128)
    for j, mu in enumerate(mu_list):
        _, m = idx_to_lm(mu)
        if m < 0:
            continue                              # exactly zero at φ=0
        F  += (phase[j] * D[:, j])[:, None] * Y[j][None, :]
        dF += (phase[j] * dD_dE[:, j])[:, None] * Y[j][None, :]
    PRE = 4.0 * np.pi
    return PRE * F, PRE * dF


def compute_fixed_alpha_beta(
        ds_x: DipoleDataset, ds_y: DipoleDataset, ds_z: DipoleDataset,
        theta: np.ndarray, alpha_rad: float, beta_rad: float
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eq. (S26-S27): linear superposition of the three Cartesian amplitudes
    using the Euler-angle field projection coefficients.
       µ̂ = cos α sin β x̂ + sin α sin β ŷ + cos β ẑ.
    Returns (k, E_kin, omega, σ, τ) on the (nE, nT) grid; σ in bohr², τ in as.
    """
    cx = np.cos(alpha_rad) * np.sin(beta_rad)
    cy = np.sin(alpha_rad) * np.sin(beta_rad)
    cz = np.cos(beta_rad)

    Fx, dFx = per_cartesian_amplitude_at_phi0(ds_x, theta)
    Fy, dFy = per_cartesian_amplitude_at_phi0(ds_y, theta)
    Fz, dFz = per_cartesian_amplitude_at_phi0(ds_z, theta)

    F  = cx * Fx  + cy * Fy  + cz * Fz
    dF = cx * dFx + cy * dFy + cz * dFz

    k, omega = ds_x.k, ds_x.omega
    F2 = np.abs(F) ** 2
    sigma = (4.0 * np.pi**2 * omega / C_AU)[:, None] * F2 / (k[:, None] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_au = np.where(F2 > 1e-30,
                          np.imag(np.conj(F) * dF) / F2, 0.0)
    tau_as = tau_au * AU_TO_AS
    return k, ds_x.e_kin, omega, sigma, tau_as


def compute_polarization_average(
        ds_x: DipoleDataset, ds_y: DipoleDataset, ds_z: DipoleDataset,
        theta: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eq. (S38):  σ_pol = (4π²ω/c)(1/3) Σ_q |D_q|² ,
                  τ_pol = Σ_q Im[D_q* ∂_E D_q] / Σ_q |D_q|².
    Returns (k, E_kin, omega, σ_pol, τ_pol)."""
    Fx, dFx = per_cartesian_amplitude_at_phi0(ds_x, theta)
    Fy, dFy = per_cartesian_amplitude_at_phi0(ds_y, theta)
    Fz, dFz = per_cartesian_amplitude_at_phi0(ds_z, theta)

    k, omega = ds_x.k, ds_x.omega
    F2sum  = np.abs(Fx) ** 2 + np.abs(Fy) ** 2 + np.abs(Fz) ** 2
    Imsum  = (np.imag(np.conj(Fx) * dFx)
              + np.imag(np.conj(Fy) * dFy)
              + np.imag(np.conj(Fz) * dFz))

    sigma_pol = (4.0 * np.pi**2 * omega / C_AU)[:, None] / 3.0 \
                * F2sum / (k[:, None] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_au = np.where(F2sum > 1e-30, Imsum / F2sum, 0.0)
    tau_as = tau_au * AU_TO_AS
    return k, ds_x.e_kin, omega, sigma_pol, tau_as


# ----------------------- write data ------------------------------
def _write_grid(out_path: Path, k, e_kin, omega, theta, sigma, tau,
                header: str) -> None:
    nE, nT = sigma.shape
    out = np.zeros((nE * nT, 7), dtype=np.float64)
    row = 0
    for i in range(nE):
        for j in range(nT):
            out[row, 0] = k[i]
            out[row, 1] = e_kin[i]
            out[row, 2] = omega[i]
            out[row, 3] = theta[j]
            out[row, 4] = sigma[i, j]
            out[row, 5] = tau[i, j]
            out[row, 6] = sigma[i, j] * tau[i, j]
            row += 1
    np.savetxt(out_path, out, header=header, fmt="%.12e")


# ----------------------- main driver -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path,
        help="Directory with gather'd dipole_{gauge}_homo_{q}_<mu>.dat")
    ap.add_argument("--output-dir", required=True, type=Path,
        help="Output directory for the two .dat files")
    ap.add_argument("--gauge", choices=("len", "vel"), default="len")
    ap.add_argument("--alpha-deg", type=float, default=0.0,
        help="Euler α in degrees (default 0).  γ is irrelevant for linear pol.")
    ap.add_argument("--beta-deg",  type=float, default=0.0,
        help="Euler β in degrees (default 0; α=β=0 ⇒ pure z polarisation).")
    ap.add_argument("--n-theta",   type=int,   default=181,
        help="Number of θ points in [0, π] (default 181 → 1° spacing).")
    ap.add_argument("--k-min", type=float, default=None,
        help="Optional k-range filter (a.u.)")
    ap.add_argument("--k-max", type=float, default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" compute_polar_data.py")
    print("=" * 72)
    print(f"  input-dir   : {args.input_dir}")
    print(f"  gauge       : {args.gauge}")
    print(f"  α, β (deg)  : {args.alpha_deg}, {args.beta_deg}")
    print(f"  n_theta     : {args.n_theta}   (φ=0)")
    if args.k_min is not None or args.k_max is not None:
        print(f"  k window    : [{args.k_min}, {args.k_max}]")
    print()

    ds_x = load_dipole_dataset(args.input_dir, args.gauge, "x")
    ds_y = load_dipole_dataset(args.input_dir, args.gauge, "y")
    ds_z = load_dipole_dataset(args.input_dir, args.gauge, "z")
    # Verify all three share the same k-grid and channels.
    if not (np.allclose(ds_x.k, ds_y.k) and np.allclose(ds_y.k, ds_z.k)):
        raise SystemExit("ERROR: k-grids of x/y/z dipoles disagree")
    if not (ds_x.mu_list == ds_y.mu_list == ds_z.mu_list):
        raise SystemExit("ERROR: channel sets of x/y/z dipoles disagree")
    print(f"  energies    : {ds_x.k.size}     k = [{ds_x.k[0]:.4f}, {ds_x.k[-1]:.4f}]")
    print(f"  channels    : {len(ds_x.mu_list)}   μ ∈ [{ds_x.mu_list[0]}, {ds_x.mu_list[-1]}]")

    if args.k_min is not None or args.k_max is not None:
        lo = ds_x.k.min() if args.k_min is None else args.k_min
        hi = ds_x.k.max() if args.k_max is None else args.k_max
        mask = (ds_x.k >= lo) & (ds_x.k <= hi)
        def _slice(ds):
            return DipoleDataset(ds.k[mask], ds.e_kin[mask], ds.omega[mask],
                                  ds.mu_list, ds.dipole[mask, :],
                                  ds.gauge, ds.component)
        ds_x, ds_y, ds_z = _slice(ds_x), _slice(ds_y), _slice(ds_z)
        print(f"  after filter: {ds_x.k.size} energies, k=[{ds_x.k[0]:.4f}, {ds_x.k[-1]:.4f}]")

    theta = np.linspace(0.0, np.pi, args.n_theta)
    alpha_rad = np.deg2rad(args.alpha_deg)
    beta_rad  = np.deg2rad(args.beta_deg)

    # --- block A: fixed (α, β) ---
    k, e_kin, omega, sig_fixed, tau_fixed = compute_fixed_alpha_beta(
        ds_x, ds_y, ds_z, theta, alpha_rad, beta_rad)
    fname_fixed = (
        f"polar_fixed_alpha{args.alpha_deg:g}_beta{args.beta_deg:g}_{args.gauge}.dat")
    out_fixed = args.output_dir / fname_fixed
    hdr_fixed = (
        "Polar-plot data, FIXED Euler angles (α, β)  per BW17-style /\n"
        "Azizi et al. supplementary Eq. (S26-S27).\n"
        f"gauge = {args.gauge}\n"
        f"alpha = {args.alpha_deg} deg\n"
        f"beta  = {args.beta_deg} deg\n"
        "(γ has no effect for linear polarisation.)\n"
        "Cartesian field coefficients:\n"
        f"  c_x = cos α sin β = {np.cos(alpha_rad)*np.sin(beta_rad):.10f}\n"
        f"  c_y = sin α sin β = {np.sin(alpha_rad)*np.sin(beta_rad):.10f}\n"
        f"  c_z = cos β       = {np.cos(beta_rad):.10f}\n"
        "Columns:\n"
        "  1 k         [a.u.]\n"
        "  2 E_kin     [a.u.]\n"
        "  3 omega     [a.u.]\n"
        "  4 theta     [rad]   (emission polar angle in MF; φ=0)\n"
        "  5 sigma     [bohr²]\n"
        "  6 tau       [as]\n"
        "  7 sigma·tau [bohr²·as]\n"
    )
    _write_grid(out_fixed, k, e_kin, omega, theta, sig_fixed, tau_fixed,
                header=hdr_fixed)
    print(f"  wrote {out_fixed}")
    print(f"    σ range     : [{sig_fixed.min():.3e}, {sig_fixed.max():.3e}] bohr²")
    print(f"    τ range     : [{tau_fixed.min():+.3f}, {tau_fixed.max():+.3f}] as")

    # --- block B: orientation average ---
    k, e_kin, omega, sig_pol, tau_pol = compute_polarization_average(
        ds_x, ds_y, ds_z, theta)
    fname_avg = f"polar_pol_avg_{args.gauge}.dat"
    out_avg = args.output_dir / fname_avg
    hdr_avg = (
        "Polar-plot data, ISOTROPIC POLARISATION-AVERAGED  per Eq. (S38) of\n"
        "Azizi et al. supplementary.\n"
        f"gauge = {args.gauge}\n"
        "  σ_pol = (4π²ω/c)·(1/3)·Σ_q |D_q|²\n"
        "  τ_pol = Σ_q Im[D_q* ∂_E D_q] / Σ_q |D_q|²\n"
        "Columns identical to the fixed-(α,β) file.\n"
        "  1 k         [a.u.]\n"
        "  2 E_kin     [a.u.]\n"
        "  3 omega     [a.u.]\n"
        "  4 theta     [rad]   (emission polar angle in MF; φ=0)\n"
        "  5 sigma_pol [bohr²]\n"
        "  6 tau_pol   [as]\n"
        "  7 sigma_pol·tau_pol [bohr²·as]\n"
    )
    _write_grid(out_avg, k, e_kin, omega, theta, sig_pol, tau_pol,
                header=hdr_avg)
    print(f"  wrote {out_avg}")
    print(f"    σ_pol range : [{sig_pol.min():.3e}, {sig_pol.max():.3e}] bohr²")
    print(f"    τ_pol range : [{tau_pol.min():+.3f}, {tau_pol.max():+.3f}] as")
    print()
    print("Done.  Now run plot_polar.py to render the 2×3 polar figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
