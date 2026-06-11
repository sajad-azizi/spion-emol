#!/usr/bin/env python3
"""
Generate averaged photoionization data for Fig. 3 from gathered channel files.

This corrected version is consistent with the current pipeline where gathered files
store reduced channel amplitudes d_{lm}(k) in the format

    k   E_kin   omega   Re[d_lm]   Im[d_lm]   |d_lm|^2   arg(d_lm)

for files named like
    dipole_len_homo_z_<mu>.dat

It computes, for a fixed polarization component (default: z) and gauge (default: len):

1) Azimuthal average over phi at fixed theta:
       sigma_tilde(E,theta)
       tau_tilde(E,theta)

2) Full emission average over theta and phi:
       sigma_bar(E)
       tau_bar(E)

For reduced channel amplitudes d_{lm}(k), the molecular-frame amplitude is

    D(theta,phi;E) = (4*pi/k) sum_{lm} (-i)^l d_{lm}(E) Y^R_{lm}(theta,phi)

Hence the cross section carries an explicit (4*pi)^2/k^2 factor from the amplitude
squared, while the Wigner-type angle-resolved delay is independent of that real
prefactor.
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

from scipy.integrate import simpson

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


def lm_to_idx(ell: int, m: int) -> int:
    return ell * ell + ell + m


# =============================================================================
# Real spherical harmonics: theta part only
# =============================================================================
def theta_part_real_sph_harm(ell: int, m: int, theta: np.ndarray) -> np.ndarray:
    """
    Theta_{lm}(theta) for the real spherical-harmonic convention used in the codebase:

        Y^R_{l,0}   = Y_{l,0}
        Y^R_{l,m>0} = sqrt(2) (-1)^m Re[Y_{l,m}]
        Y^R_{l,m<0} = sqrt(2) (-1)^{|m|} Im[Y_{l,|m|}]

    with
        Y^R_{lm}(theta,phi) = Theta_{lm}(theta) * Phi_m(phi)
        Phi_0 = 1,
        Phi_{m>0} = cos(m phi),
        Phi_{m<0} = sin(|m| phi).
    """
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
        # legacy fallback: k, Re, Im
        k = data[:, 0]
        e_kin = 0.5 * k**2
        omega = e_kin + IP
        d = data[:, 1] + 1j * data[:, 2]
    else:
        raise ValueError(
            f"Unsupported file format in {path}: expected at least 5 columns or legacy 3 columns, got {ncol}"
        )
    return k, e_kin, omega, d


def load_dipole_data(
    input_dir: Path,
    gauge: str,
    component: str,
    mu_list: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    files = discover_channel_files(input_dir, gauge, component)
    if mu_list is None:
        mu_list = sorted(files)
    else:
        missing = [mu for mu in mu_list if mu not in files]
        if missing:
            raise FileNotFoundError(f"Missing channel files for mu={missing[:10]}...")

    ref_k = ref_e = ref_omega = None
    dipoles: List[np.ndarray] = []

    for mu in mu_list:
        path = files[mu]
        k, e_kin, omega, d = _read_channel_file(path)
        if ref_k is None:
            ref_k = k
            ref_e = e_kin
            ref_omega = omega
        else:
            if len(k) != len(ref_k) or not np.allclose(k, ref_k, rtol=0.0, atol=1e-12):
                raise ValueError(f"Inconsistent k grid in {path}")
            if not np.allclose(e_kin, ref_e, rtol=0.0, atol=1e-12):
                raise ValueError(f"Inconsistent E_kin grid in {path}")
            if not np.allclose(omega, ref_omega, rtol=0.0, atol=1e-12):
                raise ValueError(f"Inconsistent omega grid in {path}")
        dipoles.append(d)

    dipole = np.column_stack(dipoles)
    return ref_k, ref_e, ref_omega, dipole, mu_list


# =============================================================================
# Core formulas
# =============================================================================
def compute_full_emission_average(
    dipole: np.ndarray,
    k: np.ndarray,
    omega: np.ndarray,
    input_normalization: str,
    with_minus: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full emission average for one polarization component.

    Including the 4*pi prefactor from Eq. (S5), for reduced inputs d_{lm}(k):

        sigma_bar(E) = (4 pi^2 omega / c) * (4 pi / k^2) * sum |d_lm|^2

    which matches Eq. (S18a) of the supplemental material.

    For energy-normalized inputs d^E_{lm} = d_{lm}/k:

        sigma_bar(E) = (4 pi^2 omega / c) * 4 pi * sum |d^E_lm|^2

    The time delay is the same in both cases (prefactor cancels):

        tau_bar(E) = Im[sum d*_lm dd_lm/dE] / sum |d_lm|^2
    """
    sum_abs2 = np.sum(np.abs(dipole) ** 2, axis=1)

    pref = 4.0 * np.pi**2 * omega / C_AU
    if input_normalization == "reduced":
        # Eq. (S18a): (4pi^2 omega/c) * (4pi/k^2) * sum|d|^2
        sigma_bar = pref * (4.0 * np.pi) * sum_abs2 / k**2
    elif input_normalization == "energy":
        # Same but without the 1/k^2 (already absorbed into d^E)
        sigma_bar = pref * (4.0 * np.pi) * sum_abs2
    else:
        raise ValueError("input_normalization must be 'reduced' or 'energy'")

    d_dE = np.gradient(dipole, k, axis=0) / k[:, None]
    numerator = np.sum(np.conj(dipole) * d_dE, axis=1)

    sign = -1.0 if with_minus else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_au = np.where(sum_abs2 > 1e-30, np.imag(numerator) / sum_abs2, 0.0)
    return sigma_bar, sign * tau_au * AU_TO_AS


def compute_azimuthal_average(
    dipole: np.ndarray,
    mu_list: List[int],
    k: np.ndarray,
    omega: np.ndarray,
    theta: np.ndarray,
    input_normalization: str,
    with_minus: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Azimuthal average over phi at fixed theta.

    Using the real-harmonic separation Y^R_{lm}(theta,phi) = Theta_{lm}(theta) Phi_m(phi)
    and (1/2pi) int Phi_m Phi_m' dphi = c_m delta_{mm'} with c_0=1, c_{m!=0}=1/2,

        A_m(theta) = sum_l (-i)^l d_{lm}(E) Theta_{lm}(theta)

    Including the 4*pi prefactor from Eq. (S5), the azimuthal average (Eq. S24a) is:

        sigma_tilde(E,theta) = (4 pi^2 omega / c) * (16 pi^2 / k^2) * sum_m c_m |A_m|^2

    for reduced inputs. The time delay (Eq. S24b) is unaffected by the prefactor:

        tau_tilde(E,theta) = Im[sum_m c_m A*_m dA_m/dE] / sum_m c_m |A_m|^2
    """
    nE = len(k)
    nT = len(theta)

    l_vals = np.array([idx_to_lm(mu)[0] for mu in mu_list], dtype=int)
    m_vals = np.array([idx_to_lm(mu)[1] for mu in mu_list], dtype=int)
    phases = (-1j) ** l_vals

    d_dE = np.gradient(dipole, k, axis=0) / k[:, None]

    unique_m = sorted(set(int(m) for m in m_vals))
    sum_cm_abs2 = np.zeros((nE, nT), dtype=float)
    sum_cm_num = np.zeros((nE, nT), dtype=np.complex128)

    for m in unique_m:
        cols = np.where(m_vals == m)[0]
        if cols.size == 0:
            continue

        c_m = 1.0 if m == 0 else 0.5
        A_m = np.zeros((nE, nT), dtype=np.complex128)
        dA_m_dE = np.zeros((nE, nT), dtype=np.complex128)

        for col in cols:
            ell = l_vals[col]
            Theta = theta_part_real_sph_harm(ell, m, theta)
            ph = phases[col]
            A_m += ph * dipole[:, col][:, None] * Theta[None, :]
            dA_m_dE += ph * d_dE[:, col][:, None] * Theta[None, :]

        sum_cm_abs2 += c_m * np.abs(A_m) ** 2
        sum_cm_num += c_m * np.conj(A_m) * dA_m_dE

    pref = 4.0 * np.pi**2 * omega / C_AU
    # The (4*pi)^2 = 16*pi^2 comes from the amplitude prefactor in Eq. (S5):
    #   D = 4*pi * (1/k) * sum (-i)^l d_{lm} Y_{lm}
    # so |D|^2 picks up (4*pi)^2 = 16*pi^2.
    PREFACTOR_SQ = 16.0 * np.pi**2
    if input_normalization == "reduced":
        # Eq. (S24a): (4pi^2 omega/c) * (16pi^2/k^2) * sum c_m |A_m|^2
        sigma_tilde = pref[:, None] * PREFACTOR_SQ * sum_cm_abs2 / (k[:, None] ** 2)
    elif input_normalization == "energy":
        sigma_tilde = pref[:, None] * PREFACTOR_SQ * sum_cm_abs2
    else:
        raise ValueError("input_normalization must be 'reduced' or 'energy'")

    sign = -1.0 if with_minus else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_au = np.where(sum_cm_abs2 > 1e-30, np.imag(sum_cm_num) / sum_cm_abs2, 0.0)
    return sigma_tilde, sign * tau_au * AU_TO_AS


# =============================================================================
# Driver
# =============================================================================
def generate_averaged_data(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    gauge: str = "len",
    component: str = "z",
    n_theta_output: int = 91,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    output_prefix: str = "zpol",
    input_normalization: str = "reduced",
    with_minus: bool = False,
):
    print("=" * 72)
    print("Generating Fig. 3 averaged data")
    print("=" * 72)
    print(f"Input directory       : {input_dir}")
    print(f"Gauge / component     : {gauge} / {component}")
    print(f"Input normalization   : {input_normalization}")

    k, e_kin, omega, dipole, mu_list = load_dipole_data(input_dir, gauge, component)
    n_channels = len(mu_list)
    l_max = max(idx_to_lm(mu)[0] for mu in mu_list)

    if len(k) < 3:
        raise ValueError("Need at least 3 energy points to compute d/dE reliably.")

    dk = np.diff(k)
    if not np.allclose(dk, dk[0], rtol=0.0, atol=1e-10):
        warnings.warn("k-grid is not perfectly uniform; np.gradient will still work.")

    if k_min is not None or k_max is not None:
        kmn = k.min() if k_min is None else k_min
        kmx = k.max() if k_max is None else k_max
        mask = (k >= kmn) & (k <= kmx)
        k = k[mask]
        e_kin = e_kin[mask]
        omega = omega[mask]
        dipole = dipole[mask, :]
        print(f"Restricted k range    : [{kmn:.6f}, {kmx:.6f}]")

    print(f"Energy points         : {len(k)}")
    print(f"Channels discovered   : {n_channels}")
    print(f"l_max discovered      : {l_max}")
    print(f"k range (a.u.)        : [{k[0]:.6f}, {k[-1]:.6f}]")
    print(f"E_kin range (a.u.)    : [{e_kin[0]:.6f}, {e_kin[-1]:.6f}]")

    theta = np.linspace(0.0, np.pi, n_theta_output)

    print("\nComputing azimuthal average ...")
    sigma_tilde, tau_tilde = compute_azimuthal_average(
        dipole=dipole,
        mu_list=mu_list,
        k=k,
        omega=omega,
        theta=theta,
        input_normalization=input_normalization,
        with_minus=with_minus,
    )

    print("Computing full emission average ...")
    sigma_bar, tau_bar = compute_full_emission_average(
        dipole=dipole,
        k=k,
        omega=omega,
        input_normalization=input_normalization,
        with_minus=with_minus,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = len(k) * len(theta)
    out_az = np.zeros((rows, 4), dtype=float)
    r = 0
    for i in range(len(k)):
        for j in range(len(theta)):
            out_az[r, 0] = e_kin[i]
            out_az[r, 1] = theta[j]
            out_az[r, 2] = tau_tilde[i, j]
            out_az[r, 3] = sigma_tilde[i, j]
            r += 1

    tau_sign = "-" if with_minus else "+"
    tau_note = (f"tau sign: {tau_sign}Im[...]  "
                f"({'legacy --with-minus-sign' if with_minus else 'default +d argD/dE'})")
    az_path = output_dir / f"{output_prefix}_azimuthal_average.dat"
    header_az = (
        f"Azimuthal average for gauge={gauge}, component={component}\n"
        f"Input files: dipole_{gauge}_homo_{component}_mu.dat\n"
        f"Input normalization: {input_normalization}\n"
        f"{tau_note}\n"
        f"Loaded columns: k, E_kin, omega, Re[d], Im[d], ...\n"
        f"Formula: A_m(theta)=sum_l (-i)^l d_lm Theta_lm(theta)\n"
        f"         tau_tilde = Im[sum_m c_m A*_m dA_m/dE] / sum_m c_m |A_m|^2\n"
        f"         sigma_tilde = (4pi^2 omega/c) * {'16pi^2 * sum_m c_m |A_m|^2 / k^2' if input_normalization=='reduced' else '16pi^2 * sum_m c_m |A_m|^2'}\n"
        f"         c_0=1, c_m=1/2 for m!=0\n"
        f"Channels discovered: {n_channels}, l_max={l_max}\n"
        f"Columns: E_kin(a.u.)  theta(rad)  tau_tilde(as)  sigma_tilde(bohr^2)"
    )
    np.savetxt(az_path, out_az, header=header_az, fmt="%.10e  %.10e  %.10e  %.10e")

    out_full = np.column_stack([e_kin, tau_bar, sigma_bar])
    full_path = output_dir / f"{output_prefix}_full_emission_average.dat"
    header_full = (
        f"Full emission average for gauge={gauge}, component={component}\n"
        f"Input files: dipole_{gauge}_homo_{component}_mu.dat\n"
        f"Input normalization: {input_normalization}\n"
        f"{tau_note}\n"
        f"Formula: tau_bar = Im[sum_lm d*_lm dd_lm/dE] / sum_lm |d_lm|^2\n"
        f"         sigma_bar = (4pi^2 omega/c) * {'4pi * sum_lm |d_lm|^2 / k^2' if input_normalization=='reduced' else '4pi * sum_lm |d_lm|^2'}\n"
        f"Columns: E_kin(a.u.)  tau_bar(as)  sigma_bar(bohr^2)"
    )
    np.savetxt(full_path, out_full, header=header_full, fmt="%.10e  %.10e  %.10e")

    sigma_bar_from_tilde = simpson(sigma_tilde * np.sin(theta)[None, :], x=theta, axis=1) / 2.0
    rel = np.max(np.abs(sigma_bar_from_tilde - sigma_bar) / np.maximum(np.abs(sigma_bar), 1e-30))

    print(f"Saved azimuthal data  : {az_path}")
    print(f"Saved full data       : {full_path}")
    print(f"Max rel. check error  : {rel:.3e}")
    print("Done.")

    return out_az, out_full


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate corrected Fig. 3 averaged data")
    parser.add_argument("--input_dir", type=str, default=str(INPUT_DIR))
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--gauge", type=str, default="len", choices=["len", "vel"])
    parser.add_argument("--component", type=str, default="z", choices=["x", "y", "z"])
    parser.add_argument("--n_theta_output", type=int, default=91)
    parser.add_argument("--k_min", type=float, default=None)
    parser.add_argument("--k_max", type=float, default=None)
    parser.add_argument("--prefix", type=str, default="zpol")
    parser.add_argument(
        "--input_normalization",
        type=str,
        default="reduced",
        choices=["reduced", "energy"],
        help="Use 'reduced' for current gathered dipoles d_lm(k); use 'energy' only if files already contain d_lm/k.",
    )
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
        component=args.component,
        n_theta_output=args.n_theta_output,
        k_min=args.k_min,
        k_max=args.k_max,
        output_prefix=args.prefix,
        input_normalization=args.input_normalization,
        with_minus=args.with_minus_sign,
    )


if __name__ == "__main__":
    main()