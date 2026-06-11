#!/usr/bin/env python3
r"""
Generate Fig. 5 data: azimuthal- and polarization-averaged photoionization data.

Supports two averaging modes (selected via --weighting):

  weighted (default):
    tau_pol = [sum_m c_m sum_i Im(A_m^(i)* dA_m^(i)/dE)]
             / [sum_m c_m sum_i |A_m^(i)|^2]
    (analytical, cross-section-weighted)

  unweighted:
    <tau_tilde> = (1/4pi) int d\hat{F} tau_tilde(\hat{F})
    where tau_tilde(\hat{F}) = [sum_m c_m Im(A_m* dA_m/dE)] / [sum_m c_m |A_m|^2]
    and A_m(\hat{F}) = c_x A_m^(x) + c_y A_m^(y) + c_z A_m^(z)
    (numerical, Gauss-Legendre + trapezoidal quadrature)

The cross section is the same in both cases (analytical):

    sigma_pol = (4pi^2 omega/c) * (16pi^2 / 3k^2) * sum_m c_m sum_i |A_m^(i)|^2

Note: for systems with cubic symmetry, both modes give identical results
because the azimuthally-averaged tau_tilde is constant over the polarization
sphere.
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
# Theta part of real spherical harmonics
# =============================================================================
def theta_part_real_sph_harm(ell: int, m: int, theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta)
    if m == 0:
        return complex_sph_harm(0, ell, 0.0, theta).real
    if m > 0:
        ylm = complex_sph_harm(m, ell, 0.0, theta)
        return np.sqrt(2.0) * ((-1) ** m) * ylm.real
    am = abs(m)
    phi_test = np.pi / (2.0 * am)
    ylm = complex_sph_harm(am, ell, phi_test, theta)
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
            f"Unsupported file format in {path}: expected >=5 columns or legacy 3 columns, got {ncol}"
        )
    return k, e_kin, omega, d


def common_mu_list(input_dir: Path, gauge: str) -> List[int]:
    files_x = discover_channel_files(input_dir, gauge, "x")
    files_y = discover_channel_files(input_dir, gauge, "y")
    files_z = discover_channel_files(input_dir, gauge, "z")
    common = sorted(set(files_x) & set(files_y) & set(files_z))
    if not common:
        raise RuntimeError("No common mu channels found across x, y, z.")
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
# Precompute A_m^(i)(theta) and dA_m^(i)/dE(theta)
# =============================================================================
def precompute_Am_arrays(
    dipole_xyz: Dict[str, np.ndarray],
    k: np.ndarray,
    theta: np.ndarray,
    mu_list: List[int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int, np.ndarray]:
    nE = len(k)
    nT = len(theta)
    l_vals = np.array([idx_to_lm(mu)[0] for mu in mu_list], dtype=int)
    m_vals_ch = np.array([idx_to_lm(mu)[1] for mu in mu_list], dtype=int)
    l_max = int(np.max(l_vals))
    nM = 2 * l_max + 1

    Theta = np.zeros((len(mu_list), nT), dtype=float)
    phase = np.zeros(len(mu_list), dtype=np.complex128)
    for col, mu in enumerate(mu_list):
        ell, m = idx_to_lm(mu)
        Theta[col, :] = theta_part_real_sph_harm(ell, m, theta)
        phase[col] = (-1j) ** ell

    Am = {}
    dAm_dE = {}
    for comp in ("x", "y", "z"):
        d_dE = np.gradient(dipole_xyz[comp], k, axis=0) / k[:, None]
        A = np.zeros((nE, nM, nT), dtype=np.complex128)
        dA = np.zeros((nE, nM, nT), dtype=np.complex128)
        for col, mu in enumerate(mu_list):
            ell, m = idx_to_lm(mu)
            m_idx = m + l_max
            ph = phase[col]
            th = Theta[col][None, :]
            A[:, m_idx, :] += ph * dipole_xyz[comp][:, col][:, None] * th
            dA[:, m_idx, :] += ph * d_dE[:, col][:, None] * th
        Am[comp] = A
        dAm_dE[comp] = dA

    m_values = np.arange(-l_max, l_max + 1)
    return Am, dAm_dE, l_max, m_values


# =============================================================================
# Azimuthal + polarization average: weighted or unweighted
# =============================================================================
def compute_azimuthal_polarization_average(
    dipole_xyz: Dict[str, np.ndarray],
    k: np.ndarray,
    omega: np.ndarray,
    theta: np.ndarray,
    mu_list: List[int],
    input_normalization: str,
    weighting: str = "weighted",
    n_beta: int = 32,
    n_alpha: int = 64,
    with_minus: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    Am, dAm_dE, l_max, m_values = precompute_Am_arrays(dipole_xyz, k, theta, mu_list)

    nE = len(k)
    nT = len(theta)
    nM = 2 * l_max + 1
    c_m_arr = np.array([1.0 if m == 0 else 0.5 for m in m_values])  # shape (nM,)

    # --- Cross section: analytical in both modes ---
    sum_cm_abs2_all = np.zeros((nE, nT), dtype=float)
    for im, m in enumerate(m_values):
        c_m = c_m_arr[im]
        abs2_m = (
            np.abs(Am["x"][:, im, :]) ** 2 +
            np.abs(Am["y"][:, im, :]) ** 2 +
            np.abs(Am["z"][:, im, :]) ** 2
        )
        sum_cm_abs2_all += c_m * abs2_m

    pref = 4.0 * np.pi**2 * omega / C_AU
    PREFACTOR_SQ = 16.0 * np.pi**2
    if input_normalization == "reduced":
        sigma_avg = pref[:, None] * PREFACTOR_SQ * sum_cm_abs2_all / (3.0 * k[:, None] ** 2)
    elif input_normalization == "energy":
        sigma_avg = pref[:, None] * PREFACTOR_SQ * sum_cm_abs2_all / 3.0
    else:
        raise ValueError("input_normalization must be 'reduced' or 'energy'")

    # --- Time delay ---
    if weighting == "weighted":
        # Analytical cross-section-weighted average:
        #   tau = [sum_m c_m sum_i Im(A_m^(i)* dA_m^(i)/dE)]
        #       / [sum_m c_m sum_i |A_m^(i)|^2]
        sum_cm_num_all = np.zeros((nE, nT), dtype=np.complex128)
        for im, m in enumerate(m_values):
            c_m = c_m_arr[im]
            num_m = (
                np.conj(Am["x"][:, im, :]) * dAm_dE["x"][:, im, :] +
                np.conj(Am["y"][:, im, :]) * dAm_dE["y"][:, im, :] +
                np.conj(Am["z"][:, im, :]) * dAm_dE["z"][:, im, :]
            )
            sum_cm_num_all += c_m * num_m

        with np.errstate(divide="ignore", invalid="ignore"):
            tau_au = np.where(sum_cm_abs2_all > 1e-30,
                              np.imag(sum_cm_num_all) / sum_cm_abs2_all, 0.0)
        tau_avg = tau_au * AU_TO_AS

    elif weighting == "unweighted":
        # Numerical unweighted average:
        #   <tau_tilde> = (1/4pi) int d\hat{F} tau_tilde(\hat{F})
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

                # A_m(\hat{F}) = cx*A_m^(x) + cy*A_m^(y) + cz*A_m^(z)
                A_m = cx * Am["x"] + cy * Am["y"] + cz * Am["z"]
                dA_m = cx * dAm_dE["x"] + cy * dAm_dE["y"] + cz * dAm_dE["z"]

                abs2 = np.abs(A_m) ** 2
                num = np.imag(np.conj(A_m) * dA_m)

                S = np.einsum("m,emt->et", c_m_arr, abs2)
                N = np.einsum("m,emt->et", c_m_arr, num)

                with np.errstate(divide="ignore", invalid="ignore"):
                    tau_here = np.where(S > 1e-30, N / S, 0.0)

                tau_sum += (w_alpha * w_beta[ib] / (4.0 * np.pi)) * tau_here

        tau_avg = tau_sum * AU_TO_AS
    else:
        raise ValueError("weighting must be 'weighted' or 'unweighted'")

    sign = -1.0 if with_minus else 1.0
    return sigma_avg, sign * tau_avg


# =============================================================================
# Driver
# =============================================================================
def generate_averaged_data(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    gauge: str = "len",
    n_theta_output: int = 91,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    output_prefix: str = "avg_phi_pol",
    input_normalization: str = "reduced",
    weighting: str = "weighted",
    n_beta: int = 32,
    n_alpha: int = 64,
    with_minus: bool = False,
):
    print("=" * 72)
    print("Generating azimuthal + polarization averaged data (Fig. 5)")
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

    print(f"\nComputing azimuthal (analytical) + {weighting} polarization average ...")
    sigma_avg, tau_avg = compute_azimuthal_polarization_average(
        dipole_xyz=dipole_xyz, k=k, omega=omega, theta=theta_output,
        mu_list=mu_list, input_normalization=input_normalization,
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
            out[r, 2] = tau_avg[i, j]
            out[r, 3] = sigma_avg[i, j]
            r += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{output_prefix}_averaged.dat"

    if weighting == "weighted":
        tau_formula = (
            "tau_pol = [sum_m c_m sum_i Im(A_m^(i)* dA_m^(i)/dE)] "
            "/ [sum_m c_m sum_i |A_m^(i)|^2]  (cross-section weighted)"
        )
    else:
        tau_formula = (
            f"<tau_tilde> = (1/4pi) int d(hat{{F}}) tau_tilde(hat{{F}})  "
            f"(unweighted, n_beta={n_beta}, n_alpha={n_alpha})"
        )

    header = (
        f"Azimuthal + polarization averaged data for gauge={gauge}\n"
        f"Input files: dipole_{gauge}_homo_[x,y,z]_mu.dat\n"
        f"Input normalization: {input_normalization}\n"
        f"Weighting: {weighting}\n"
        f"tau sign: {'-' if with_minus else '+'}Im[...]  "
        f"({'legacy --with-minus-sign' if with_minus else 'default +d argD/dE'})\n"
        f"Definitions:\n"
        f"  A_m^(i)(theta) = sum_l (-i)^l d_lm^(i) Theta_lm(theta)\n"
        f"  sigma_avg = (4pi^2 omega/c) * "
        f"{'16pi^2 * sum_m c_m sum_i |A_m^(i)|^2 / (3 k^2)' if input_normalization=='reduced' else '16pi^2 * sum_m c_m sum_i |A_m^(i)|^2 / 3'}\n"
        f"  {tau_formula}\n"
        f"  c_0=1, c_m=1/2 for m!=0\n"
        f"Columns: E_kin(a.u.)  theta(rad)  tau_avg(as)  sigma_avg(bohr^2)"
    )
    np.savetxt(out_file, out, header=header, fmt="%.10e  %.10e  %.10e  %.10e")

    print(f"\nSaved: {out_file}")
    print(f"Cross section range   : [{sigma_avg.min():.6e}, {sigma_avg.max():.6e}] bohr^2")
    print(f"Delay range           : [{tau_avg.min():.2f}, {tau_avg.max():.2f}] as")
    print("Done.")

    return out


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate Fig. 5 data")
    parser.add_argument("--input_dir", type=str, default=str(INPUT_DIR))
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--gauge", type=str, default="len", choices=["len", "vel"])
    parser.add_argument("--n_theta_output", type=int, default=91)
    parser.add_argument("--k_min", type=float, default=None)
    parser.add_argument("--k_max", type=float, default=None)
    parser.add_argument("--prefix", type=str, default="avg_phi_pol")
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

    generate_averaged_data(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        gauge=args.gauge,
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