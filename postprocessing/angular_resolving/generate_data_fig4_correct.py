#!/usr/bin/env python3
r"""
Generate Fig. 4 data: polarization-averaged angular distributions at fixed phi.

Supports two averaging modes (selected via --weighting):

  weighted (default):
    tau_pol = [sum_i Im(F_i* dF_i/dE)] / [sum_i |F_i|^2]
    (analytical, cross-section-weighted)

  unweighted:
    <tau> = (1/4pi) int d\hat{F} Im[D* dD/dE] / |D|^2
    (numerical, Gauss-Legendre + trapezoidal quadrature)

The cross section is the same in both cases (analytical):

    sigma_pol = (4pi^2 omega/c) * (16pi^2 / 3k^2) * (|F_x|^2 + |F_y|^2 + |F_z|^2)

Physics
-------
For each Cartesian polarization component i in {x,y,z}, define

    F_i(theta,phi;E) = sum_{lm} (-i)^l d^{(i)}_{lm}(E) Y^R_{lm}(theta,phi)

Including the 4*pi amplitude prefactor:

    D_i(theta,phi;E) = 4*pi * F_i / k
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.special import sph_harm_y as _sph_harm  # SciPy >= 1.15
    def complex_sph_harm(m: int, ell: int, phi, theta):
        return _sph_harm(ell, m, theta, phi)
except Exception:
    from scipy.special import sph_harm as _sph_harm
    def complex_sph_harm(m: int, ell: int, phi, theta):
        return _sph_harm(m, ell, phi, theta)


# =============================================================================
# Physical constants
# =============================================================================
C_AU = 137.035999139
IP = 0.119591
AU_TO_AS = 24.188843265857

INPUT_DIR = Path("/dss/dsshome1/08/di35ker/static-exchange-Hartree-Fock/postProcessing/dipole_data_P10/")
OUTPUT_DIR = Path(".")

FILE_RE = re.compile(r"^dipole_(len|vel)_homo_([xyz])_(\d+)\.dat$")


# =============================================================================
# Index conversion
# =============================================================================
def idx_to_lm(idx: int) -> Tuple[int, int]:
    ell = int(np.floor(np.sqrt(idx)))
    m = idx - ell * ell - ell
    return ell, m


# =============================================================================
# Real spherical harmonics
# =============================================================================
def real_spherical_harmonic(ell: int, m: int, theta: np.ndarray, phi: float) -> np.ndarray:
    theta = np.asarray(theta)
    if m == 0:
        return complex_sph_harm(0, ell, phi, theta).real
    elif m > 0:
        ylm = complex_sph_harm(m, ell, phi, theta)
        return np.sqrt(2.0) * ((-1) ** m) * ylm.real
    else:
        am = abs(m)
        ylm = complex_sph_harm(am, ell, phi, theta)
        return np.sqrt(2.0) * ((-1) ** am) * ylm.imag


# =============================================================================
# File discovery and loading
# =============================================================================
def discover_channel_files(input_dir: Path, gauge: str, component: str) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in sorted(input_dir.glob(f"dipole_{gauge}_homo_{component}_*.dat")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        mu = int(m.group(3))
        out[mu] = p
    if not out:
        raise FileNotFoundError(
            f"No files found matching dipole_{gauge}_homo_{component}_*.dat in {input_dir}"
        )
    return out


def _read_channel_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    ncol = data.shape[1]
    if ncol >= 5:
        k = data[:, 0]
        e_kin = data[:, 1]
        omega = data[:, 2]
        d = data[:, 3] + 1j * data[:, 4]
    elif ncol == 3:
        k = data[:, 0]
        e_kin = 0.5 * k**2
        omega = e_kin + IP
        d = data[:, 1] + 1j * data[:, 2]
    else:
        raise ValueError(
            f"Unsupported file format in {path}: expected at least 5 columns or legacy 3 columns, got {ncol}"
        )
    return k, e_kin, omega, d


def common_mu_list(input_dir: Path, gauge: str) -> List[int]:
    files_x = discover_channel_files(input_dir, gauge, "x")
    files_y = discover_channel_files(input_dir, gauge, "y")
    files_z = discover_channel_files(input_dir, gauge, "z")
    common = sorted(set(files_x) & set(files_y) & set(files_z))
    if not common:
        raise RuntimeError("No common mu channels found across x, y, z components.")
    return common


def load_dipole_component(
    input_dir: Path, gauge: str, component: str, mu_list: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = discover_channel_files(input_dir, gauge, component)
    ref_k = ref_e = ref_omega = None
    dipoles: List[np.ndarray] = []
    for mu in mu_list:
        path = files[mu]
        k, e_kin, omega, d = _read_channel_file(path)
        if ref_k is None:
            ref_k, ref_e, ref_omega = k, e_kin, omega
        else:
            if len(k) != len(ref_k) or not np.allclose(k, ref_k, rtol=0.0, atol=1e-12):
                raise ValueError(f"Inconsistent k grid in {path}")
            if not np.allclose(e_kin, ref_e, rtol=0.0, atol=1e-12):
                raise ValueError(f"Inconsistent E_kin grid in {path}")
            if not np.allclose(omega, ref_omega, rtol=0.0, atol=1e-12):
                raise ValueError(f"Inconsistent omega grid in {path}")
        dipoles.append(d)
    dipole = np.column_stack(dipoles)
    return ref_k, ref_e, ref_omega, dipole


def load_all_components(
    input_dir: Path, gauge: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], List[int]]:
    mu_list = common_mu_list(input_dir, gauge)
    kx, ex, ox, dx = load_dipole_component(input_dir, gauge, "x", mu_list)
    ky, ey, oy, dy = load_dipole_component(input_dir, gauge, "y", mu_list)
    kz, ez, oz, dz = load_dipole_component(input_dir, gauge, "z", mu_list)
    if not (np.allclose(kx, ky) and np.allclose(kx, kz)):
        raise ValueError("k grids differ across x, y, z")
    if not (np.allclose(ex, ey) and np.allclose(ex, ez)):
        raise ValueError("E_kin grids differ across x, y, z")
    if not (np.allclose(ox, oy) and np.allclose(ox, oz)):
        raise ValueError("omega grids differ across x, y, z")
    dipole = {"x": dx, "y": dy, "z": dz}
    return kx, ex, ox, dipole, mu_list


# =============================================================================
# Build F_i(theta,phi) and dF_i/dE(theta,phi)
# =============================================================================
def build_F_and_dF(
    dipole: np.ndarray, k: np.ndarray, theta: np.ndarray, phi: float, mu_list: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    nE = len(k)
    nT = len(theta)
    d_dE = np.gradient(dipole, k, axis=0) / k[:, None]
    F = np.zeros((nE, nT), dtype=np.complex128)
    dF = np.zeros((nE, nT), dtype=np.complex128)
    for col, mu in enumerate(mu_list):
        ell, m = idx_to_lm(mu)
        Y = real_spherical_harmonic(ell, m, theta, phi)
        phase = (-1j) ** ell
        F += phase * dipole[:, col][:, None] * Y[None, :]
        dF += phase * d_dE[:, col][:, None] * Y[None, :]
    return F, dF


# =============================================================================
# Polarization average: weighted (analytical) or unweighted (numerical)
# =============================================================================
def compute_polarization_average_at_phi(
    dipole_xyz: Dict[str, np.ndarray],
    k: np.ndarray,
    omega: np.ndarray,
    theta: np.ndarray,
    phi: float,
    mu_list: List[int],
    input_normalization: str,
    weighting: str = "weighted",
    n_beta: int = 32,
    n_alpha: int = 64,
    with_minus: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    Fx, dFx = build_F_and_dF(dipole_xyz["x"], k, theta, phi, mu_list)
    Fy, dFy = build_F_and_dF(dipole_xyz["y"], k, theta, phi, mu_list)
    Fz, dFz = build_F_and_dF(dipole_xyz["z"], k, theta, phi, mu_list)

    nE = len(k)
    nT = len(theta)

    # --- Cross section: analytical in both modes ---
    abs2_sum = np.abs(Fx)**2 + np.abs(Fy)**2 + np.abs(Fz)**2
    pref = 4.0 * np.pi**2 * omega / C_AU
    PREFACTOR_SQ = 16.0 * np.pi**2
    if input_normalization == "reduced":
        sigma_pol = pref[:, None] * PREFACTOR_SQ * abs2_sum / (3.0 * k[:, None]**2)
    elif input_normalization == "energy":
        sigma_pol = pref[:, None] * PREFACTOR_SQ * abs2_sum / 3.0
    else:
        raise ValueError("input_normalization must be 'reduced' or 'energy'")

    # --- Time delay ---
    if weighting == "weighted":
        # Analytical cross-section-weighted average:
        #   tau = sum_i Im[F_i* dF_i/dE] / sum_i |F_i|^2
        num_sum = (
            np.imag(np.conj(Fx) * dFx) +
            np.imag(np.conj(Fy) * dFy) +
            np.imag(np.conj(Fz) * dFz)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_au = np.where(abs2_sum > 1e-30, num_sum / abs2_sum, 0.0)
        tau_pol = tau_au * AU_TO_AS

    elif weighting == "unweighted":
        # Numerical unweighted average:
        #   <tau> = (1/4pi) int d\hat{F} tau(\hat{F})
        u_nodes, w_beta = np.polynomial.legendre.leggauss(n_beta)
        cos_beta = u_nodes
        sin_beta = np.sqrt(1.0 - u_nodes**2)
        alpha_nodes = np.linspace(0.0, 2.0 * np.pi, n_alpha, endpoint=False)
        w_alpha = 2.0 * np.pi / n_alpha

        tau_sum = np.zeros((nE, nT), dtype=np.float64)
        for ia in range(n_alpha):
            cos_a = np.cos(alpha_nodes[ia])
            sin_a = np.sin(alpha_nodes[ia])
            for ib in range(n_beta):
                cx = sin_beta[ib] * cos_a
                cy = sin_beta[ib] * sin_a
                cz = cos_beta[ib]
                D = cx * Fx + cy * Fy + cz * Fz
                dD = cx * dFx + cy * dFy + cz * dFz
                abs2_D = np.abs(D) ** 2
                num_D = np.imag(np.conj(D) * dD)
                with np.errstate(divide="ignore", invalid="ignore"):
                    tau_here = np.where(abs2_D > 1e-30, num_D / abs2_D, 0.0)
                tau_sum += (w_alpha * w_beta[ib] / (4.0 * np.pi)) * tau_here
        tau_pol = tau_sum * AU_TO_AS
    else:
        raise ValueError("weighting must be 'weighted' or 'unweighted'")

    sign = -1.0 if with_minus else 1.0
    return sigma_pol, sign * tau_pol


# =============================================================================
# Helpers
# =============================================================================
def phi_to_tag(phi: float) -> str:
    if np.isclose(phi, 0.0):
        return "0"
    if np.isclose(phi, np.pi / 4):
        return "pi_4"
    if np.isclose(phi, np.pi / 3):
        return "pi_3"
    if np.isclose(phi, np.pi / 2):
        return "pi_2"
    return f"{phi:.6f}".replace(".", "p")


# =============================================================================
# Driver
# =============================================================================
def generate_fig4_data(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    gauge: str = "len",
    phi_values: List[float] = [0.0, np.pi / 4, np.pi / 3, np.pi / 2],
    n_theta_output: int = 181,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    output_prefix: str = "pol_avg",
    input_normalization: str = "reduced",
    weighting: str = "weighted",
    n_beta: int = 32,
    n_alpha: int = 64,
    with_minus: bool = False,
):
    print("=" * 72)
    print("Generating Fig. 4 data")
    print("=" * 72)
    print(f"Input directory       : {input_dir}")
    print(f"Gauge                 : {gauge}")
    print(f"Input normalization   : {input_normalization}")
    print(f"Weighting             : {weighting}")
    if weighting == "unweighted":
        print(f"Quadrature            : n_beta={n_beta} (GL), n_alpha={n_alpha} (trapezoidal)")

    k, e_kin, omega, dipole_xyz, mu_list = load_all_components(input_dir, gauge)

    if len(k) < 3:
        raise ValueError("Need at least 3 energy points to compute d/dE reliably.")

    if k_min is not None or k_max is not None:
        kmn = k.min() if k_min is None else k_min
        kmx = k.max() if k_max is None else k_max
        mask = (k >= kmn) & (k <= kmx)
        k = k[mask]
        e_kin = e_kin[mask]
        omega = omega[mask]
        for comp in ("x", "y", "z"):
            dipole_xyz[comp] = dipole_xyz[comp][mask, :]
        print(f"Restricted k range    : [{kmn:.6f}, {kmx:.6f}]")

    l_max = max(idx_to_lm(mu)[0] for mu in mu_list)
    print(f"Energy points         : {len(k)}")
    print(f"Channels discovered   : {len(mu_list)}")
    print(f"l_max discovered      : {l_max}")
    print(f"k range (a.u.)        : [{k[0]:.6f}, {k[-1]:.6f}]")
    print(f"E_kin range (a.u.)    : [{e_kin[0]:.6f}, {e_kin[-1]:.6f}]")

    theta_output = np.linspace(0.0, np.pi, n_theta_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for phi in phi_values:
        phi_deg = phi * 180.0 / np.pi
        print(f"\nProcessing phi = {phi:.6f} rad ({phi_deg:.1f} deg)")

        sigma_pol, tau_pol = compute_polarization_average_at_phi(
            dipole_xyz=dipole_xyz, k=k, omega=omega, theta=theta_output,
            phi=phi, mu_list=mu_list, input_normalization=input_normalization,
            weighting=weighting, n_beta=n_beta, n_alpha=n_alpha,
            with_minus=with_minus,
        )

        rows = len(k) * len(theta_output)
        out = np.zeros((rows, 4), dtype=float)
        r = 0
        for i in range(len(k)):
            for j in range(len(theta_output)):
                out[r, 0] = e_kin[i]
                out[r, 1] = theta_output[j]
                out[r, 2] = tau_pol[i, j]
                out[r, 3] = sigma_pol[i, j]
                r += 1

        tag = phi_to_tag(phi)
        output_file = output_dir / f"{output_prefix}_phi_{tag}.dat"

        if weighting == "weighted":
            tau_formula = "tau_pol = [sum_i Im(F_i* dF_i/dE)] / [sum_i |F_i|^2]  (cross-section weighted)"
        else:
            tau_formula = f"<tau> = (1/4pi) int d(hat{{F}}) Im[D* dD/dE]/|D|^2  (unweighted, n_beta={n_beta}, n_alpha={n_alpha})"

        header = (
            f"Polarization-averaged data at fixed phi={phi:.6f} rad ({phi_deg:.1f} deg)\n"
            f"Gauge: {gauge}\n"
            f"Input normalization: {input_normalization}\n"
            f"Weighting: {weighting}\n"
            f"tau sign: {'-' if with_minus else '+'}Im[...]  "
            f"({'legacy --with-minus-sign' if with_minus else 'default +d argD/dE'})\n"
            f"Definitions:\n"
            f"  F_i(theta,phi) = sum_lm (-i)^l d_lm^(i) Y_lm^R(theta,phi), i=x,y,z\n"
            f"  sigma_pol = (4pi^2 omega/c) * "
            f"{'16pi^2 * (|F_x|^2+|F_y|^2+|F_z|^2)/(3 k^2)' if input_normalization=='reduced' else '16pi^2 * (|F_x|^2+|F_y|^2+|F_z|^2)/3'}\n"
            f"  {tau_formula}\n"
            f"Columns: E_kin(a.u.)  theta(rad)  tau_pol(as)  sigma_pol(bohr^2)"
        )
        np.savetxt(output_file, out, header=header, fmt="%.10e  %.10e  %.10e  %.10e")

        print(f"  Saved: {output_file}")
        print(f"  sigma range: [{sigma_pol.min():.6e}, {sigma_pol.max():.6e}] bohr^2")
        print(f"  tau range  : [{tau_pol.min():.2f}, {tau_pol.max():.2f}] as")

    print("\nDone.")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate Fig. 4 data")
    parser.add_argument("--input_dir", type=str, default=str(INPUT_DIR))
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--gauge", type=str, default="len", choices=["len", "vel"])
    parser.add_argument(
        "--phi", type=float, nargs="*",
        default=[0.0, np.pi / 4, np.pi / 3, np.pi / 2],
        help="List of fixed phi values in radians",
    )
    parser.add_argument("--n_theta_output", type=int, default=181)
    parser.add_argument("--k_min", type=float, default=None)
    parser.add_argument("--k_max", type=float, default=None)
    parser.add_argument("--prefix", type=str, default="pol_avg")
    parser.add_argument(
        "--input_normalization", type=str, default="reduced",
        choices=["reduced", "energy"],
    )
    parser.add_argument(
        "--weighting", type=str, default="weighted",
        choices=["weighted", "unweighted"],
        help="'weighted' = cross-section-weighted (analytical), 'unweighted' = democratic average (numerical quadrature)",
    )
    parser.add_argument("--n_beta", type=int, default=32, help="GL nodes for beta (unweighted only)")
    parser.add_argument("--n_alpha", type=int, default=64, help="Trapezoidal points for alpha (unweighted only)")
    parser.add_argument(
        "--with-minus-sign", action="store_true",
        help="Apply a leading minus to the time delay (legacy convention, "
             "matches cross_section_delay.py --with-minus-sign). "
             "Default is the no-minus tau = +d arg(D)/dE.",
    )
    args = parser.parse_args()

    generate_fig4_data(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        gauge=args.gauge,
        phi_values=list(args.phi),
        n_theta_output=args.n_theta_output,
        k_min=args.k_min,
        k_max=args.k_max,
        output_prefix=args.prefix,
        input_normalization=args.input_normalization,
        weighting=args.weighting,
        n_beta=args.n_beta,
        n_alpha=args.n_alpha,
        with_minus=args.with_minus_sign,
    )


if __name__ == "__main__":
    main()