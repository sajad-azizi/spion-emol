#!/usr/bin/env python3
"""
Generate molecular-frame angular-resolved photoionization/phodetachment data
in the phi=0 plane for a chosen Cartesian polarization component.

Theory
------
For reduced channel dipoles d_{lm}(k) in the convention

    D(theta,phi;k) = (4*pi/k) * sum_{lm} (-i)^l d_{lm}(k) Y^R_{lm}(theta,phi)

this script computes, for fixed molecular orientation,

    sigma(theta,phi;E) = (4 pi^2 omega / c) |D|^2
    tau(theta,phi;E)   = Im[D* dD/dE] / |D|^2

If the input dipoles are already energy-normalized, the explicit 1/k factor is
not applied again.

At phi = 0 with the real-spherical-harmonic convention used in the C++ code,
all m < 0 channels vanish exactly because the sine-like harmonics are zero.

Expected gathered input format per channel file:
    col 0: k        [a.u.]
    col 1: E_kin    [a.u.]
    col 2: omega    [a.u.]
    col 3: Re d_mu
    col 4: Im d_mu
    col 5: |d_mu|^2
    col 6: arg(d_mu)

Example
-------
python generate_data_fig2_correct.py \
  --input_dir /path/to/dipole_data \
  --output_dir . \
  --gauge len --component z --prefix fig2_z_len
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.special import sph_harm_y as _sph_harm  # SciPy >= 1.15
    def complex_sph_harm(m: int, ell: int, phi, theta):
        return _sph_harm(ell, m, theta, phi)
except Exception:
    from scipy.special import sph_harm as _sph_harm  # SciPy < 1.15
    def complex_sph_harm(m: int, ell: int, phi, theta):
        return _sph_harm(m, ell, phi, theta)

# Physical constants (CODATA; consistent with the rest of the postprocessing suite)
C_AU = 137.035999139
AU_TO_AS = 24.188843265857

DEFAULT_INPUT_DIR = Path(
    "/dss/dsshome1/08/di35ker/static-exchange-Hartree-Fock/postProcessing/dipole_data_P10/"
)
DEFAULT_OUTPUT_DIR = Path(".")

FILE_RE = re.compile(r"^dipole_(len|vel)_homo_([xyz])_(\d+)\.dat$")


@dataclass
class DipoleDataset:
    k: np.ndarray
    e_kin: np.ndarray
    omega: np.ndarray
    mu_list: List[int]
    dipole: np.ndarray  # shape (n_energy, n_channels)
    gauge: str
    component: str


# -----------------------------------------------------------------------------
# Channel indexing
# -----------------------------------------------------------------------------
def idx_to_lm(idx: int) -> Tuple[int, int]:
    ell = int(np.floor(np.sqrt(idx)))
    m = idx - ell * ell - ell
    return ell, m


# -----------------------------------------------------------------------------
# Real spherical harmonics
# -----------------------------------------------------------------------------
def real_spherical_harmonic_phi0(ell: int, m: int, theta: np.ndarray) -> np.ndarray:
    """
    Real spherical harmonics at phi=0 in the same convention as the C++ code:
        Y^R_{l,0}   = Y_{l,0}
        Y^R_{l,m>0} = sqrt(2) (-1)^m Re Y_{l,m}
        Y^R_{l,m<0} = sqrt(2) (-1)^m Im Y_{l,|m|}

    At phi=0, the m<0 terms are exactly zero.
    """
    theta = np.asarray(theta)
    if m < 0:
        return np.zeros_like(theta)
    if m == 0:
        return complex_sph_harm(0, ell, 0.0, theta).real
    return np.sqrt(2.0) * ((-1) ** m) * complex_sph_harm(m, ell, 0.0, theta).real


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def discover_mu_files(input_dir: Path, gauge: str, component: str) -> List[Tuple[int, Path]]:
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
            f"No files found for pattern dipole_{gauge}_homo_{component}_*.dat in {input_dir}"
        )
    return matches



def load_one_channel(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 5:
        raise ValueError(
            f"File {path} has {data.shape[1]} columns; expected at least 5 columns "
            f"with k, E_kin, omega, Re, Im."
        )
    k = data[:, 0]
    e_kin = data[:, 1]
    omega = data[:, 2]
    d = data[:, 3] + 1j * data[:, 4]
    return k, e_kin, omega, d



def load_dipole_dataset(input_dir: Path, gauge: str, component: str) -> DipoleDataset:
    mu_files = discover_mu_files(input_dir, gauge, component)
    mu_list = [mu for mu, _ in mu_files]

    ref_k, ref_e, ref_omega, ref_d = load_one_channel(mu_files[0][1])
    n_energy = ref_k.size
    n_channels = len(mu_files)
    dipole = np.zeros((n_energy, n_channels), dtype=np.complex128)
    dipole[:, 0] = ref_d

    for j, (mu, path) in enumerate(mu_files[1:], start=1):
        k, e_kin, omega, d = load_one_channel(path)
        if not (np.allclose(k, ref_k) and np.allclose(e_kin, ref_e) and np.allclose(omega, ref_omega)):
            raise ValueError(f"Energy grid mismatch in file {path}")
        dipole[:, j] = d

    return DipoleDataset(
        k=ref_k,
        e_kin=ref_e,
        omega=ref_omega,
        mu_list=mu_list,
        dipole=dipole,
        gauge=gauge,
        component=component,
    )


# -----------------------------------------------------------------------------
# Physics
# -----------------------------------------------------------------------------
def compute_angular_resolved_phi0(
    dataset: DipoleDataset,
    theta: np.ndarray,
    input_normalization: str = "reduced",
    with_minus: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    sigma : ndarray, shape (n_energy, n_theta)
        Differential cross section in bohr^2.
    tau : ndarray, shape (n_energy, n_theta)
        Wigner-type delay in attoseconds.  tau = +Im[D* dD/dE]/|D|^2 by
        default (= d arg(D)/dE = d_E Phi, the density_current.pdf /
        cross_section_delay.py convention).  with_minus=True applies the
        legacy -Im sign.
    amplitude : ndarray, shape (n_energy, n_theta)
        Complex angular amplitude D(theta,phi=0;E).
    """
    k = dataset.k
    omega = dataset.omega
    dipole = dataset.dipole
    mu_list = dataset.mu_list

    n_energy = k.size
    n_theta = theta.size
    n_channels = len(mu_list)

    # Precompute angular basis and phase factors
    Y_phi0 = np.zeros((n_channels, n_theta), dtype=np.float64)
    phase = np.zeros(n_channels, dtype=np.complex128)
    for j, mu in enumerate(mu_list):
        ell, m = idx_to_lm(mu)
        Y_phi0[j, :] = real_spherical_harmonic_phi0(ell, m, theta)
        phase[j] = (-1j) ** ell

    # d(dipole)/dE = (1/k) d(dipole)/dk
    dd_dE = np.zeros_like(dipole)
    for j in range(n_channels):
        dd_dk = np.gradient(dipole[:, j], k)
        dd_dE[:, j] = dd_dk / k

    F = np.zeros((n_energy, n_theta), dtype=np.complex128)
    dF_dE = np.zeros((n_energy, n_theta), dtype=np.complex128)

    for j, mu in enumerate(mu_list):
        _, m = idx_to_lm(mu)
        if m < 0:
            continue  # exactly zero at phi=0 in the real-harmonic convention
        Y = Y_phi0[j, :]
        ph = phase[j]
        F += ph * dipole[:, j][:, None] * Y[None, :]
        dF_dE += ph * dd_dE[:, j][:, None] * Y[None, :]

    # Apply the 4*pi prefactor from Eq. (S5) of the paper:
    #   D(k) = 4*pi * (1/k) * sum_{lm} (-i)^l d_{lm} Y_{lm}
    # Both F and dF_dE must carry this factor so that the cross section
    # is consistent with sigma = (4*pi^2 * omega / c) |D|^2.
    # The time delay tau = Im[D* dD/dE] / |D|^2 is unaffected because
    # the constant (4*pi)^2 cancels between numerator and denominator.
    PREFACTOR = 4.0 * np.pi
    F *= PREFACTOR
    dF_dE *= PREFACTOR

    if input_normalization == "reduced":
        amplitude = F / k[:, None]
        sigma = (4.0 * np.pi**2 * omega / C_AU)[:, None] * (np.abs(F) ** 2) / (k[:, None] ** 2)
    elif input_normalization == "energy":
        amplitude = F
        sigma = (4.0 * np.pi**2 * omega / C_AU)[:, None] * (np.abs(F) ** 2)
    else:
        raise ValueError("input_normalization must be 'reduced' or 'energy'")

    F2 = np.abs(F) ** 2
    sign = -1.0 if with_minus else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_au = np.where(F2 > 1e-30, np.imag(np.conj(F) * dF_dE) / F2, 0.0)
    tau = sign * tau_au * AU_TO_AS
    return sigma, tau, amplitude


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
def generate_phi0_data(
    input_dir: Path,
    output_dir: Path,
    gauge: str = "len",
    component: str = "z",
    n_theta_output: int = 181,
    k_min: float | None = None,
    k_max: float | None = None,
    input_normalization: str = "reduced",
    output_prefix: str = "zpol_phi0",
    with_minus: bool = False,
) -> Path:
    dataset = load_dipole_dataset(input_dir, gauge, component)

    if k_min is not None or k_max is not None:
        lo = dataset.k.min() if k_min is None else k_min
        hi = dataset.k.max() if k_max is None else k_max
        mask = (dataset.k >= lo) & (dataset.k <= hi)
        dataset = DipoleDataset(
            k=dataset.k[mask],
            e_kin=dataset.e_kin[mask],
            omega=dataset.omega[mask],
            mu_list=dataset.mu_list,
            dipole=dataset.dipole[mask, :],
            gauge=dataset.gauge,
            component=dataset.component,
        )

    theta = np.linspace(0.0, np.pi, n_theta_output)
    sigma, tau, amplitude = compute_angular_resolved_phi0(
        dataset, theta, input_normalization, with_minus=with_minus)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = np.zeros((dataset.k.size * theta.size, 9), dtype=np.float64)

    row = 0
    for i in range(dataset.k.size):
        for j in range(theta.size):
            out[row, 0] = dataset.k[i]
            out[row, 1] = dataset.e_kin[i]
            out[row, 2] = dataset.omega[i]
            out[row, 3] = theta[j]
            out[row, 4] = amplitude[i, j].real
            out[row, 5] = amplitude[i, j].imag
            out[row, 6] = sigma[i, j]
            out[row, 7] = tau[i, j]
            out[row, 8] = np.abs(amplitude[i, j]) ** 2
            row += 1

    outfile = output_dir / f"{output_prefix}_angular_resolved_phi0_{gauge}_{component}.dat"
    tau_sign = "-" if with_minus else "+"
    header = (
        "Molecular-frame angular-resolved data at phi=0\n"
        f"gauge = {gauge}, polarization component = {component}\n"
        f"input_normalization = {input_normalization}\n"
        f"tau = {tau_sign}Im[D* dD/dE]/|D|^2  "
        f"({'legacy -Im (--with-minus-sign)' if with_minus else 'default +d argD/dE'})\n"
        "Amplitude convention:\n"
        "  reduced input : D(theta,phi) = (4*pi/k) sum_mu (-i)^l d_mu Y^R_mu(theta,phi)\n"
        "  energy input  : D(theta,phi) =  4*pi    sum_mu (-i)^l d_mu Y^R_mu(theta,phi)\n"
        "At phi=0, all m<0 channels vanish exactly in the real-harmonic convention.\n"
        "Columns:\n"
        "  1 k [a.u.]\n"
        "  2 E_kin [a.u.]\n"
        "  3 omega [a.u.]\n"
        "  4 theta [rad]\n"
        "  5 Re D(theta,0)\n"
        "  6 Im D(theta,0)\n"
        "  7 sigma(theta,0) [bohr^2]\n"
        "  8 tau(theta,0) [as]\n"
        "  9 |D(theta,0)|^2\n"
    )
    np.savetxt(outfile, out, header=header, fmt="%.12e")

    print("=" * 72)
    print("Angular-resolved molecular-frame data generated")
    print("=" * 72)
    print(f"Input directory      : {input_dir}")
    print(f"Gauge / component    : {gauge} / {component}")
    print(f"Input normalization  : {input_normalization}")
    print(f"Number of channels   : {len(dataset.mu_list)}")
    print(f"mu range             : {dataset.mu_list[0]} .. {dataset.mu_list[-1]}")
    print(f"k range              : {dataset.k[0]:.6f} .. {dataset.k[-1]:.6f} a.u.")
    print(f"E_kin range          : {dataset.e_kin[0]:.6f} .. {dataset.e_kin[-1]:.6f} a.u.")
    print(f"theta points         : {theta.size}")
    print(f"Output file          : {outfile}")
    print(f"sigma range          : {sigma.min():.6e} .. {sigma.max():.6e} bohr^2")
    print(f"tau range            : {tau.min():.3f} .. {tau.max():.3f} as")

    small_sigma = np.count_nonzero(sigma < 1e-12)
    if small_sigma:
        warnings.warn(
            f"{small_sigma} angular points have sigma < 1e-12; delays there may be numerically unstable."
        )

    return outfile


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate angular-resolved molecular-frame data at phi=0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gauge", choices=["len", "vel"], default="len")
    parser.add_argument("--component", choices=["x", "y", "z"], default="z")
    parser.add_argument("--n_theta_output", type=int, default=181)
    parser.add_argument("--k_min", type=float, default=None)
    parser.add_argument("--k_max", type=float, default=None)
    parser.add_argument(
        "--input_normalization",
        choices=["reduced", "energy"],
        default="reduced",
        help="Use 'reduced' for your current gathered dipoles d_mu(k); use 'energy' only if you already divided amplitudes by k^(1/2) or otherwise converted to energy-normalized convention.",
    )
    parser.add_argument("--prefix", type=str, default="zpol_phi0")
    parser.add_argument(
        "--with-minus-sign", action="store_true",
        help="Apply a leading minus to the time delay: tau = -Im[D* dD/dE]/|D|^2 "
             "(legacy convention, matches cross_section_delay.py --with-minus-sign). "
             "Default is the no-minus tau = +d arg(D)/dE.",
    )
    args = parser.parse_args()

    ## print gauge and component info and k_max/k_min if specified
    print("=" * 72)
    print("Generating angular-resolved molecular-frame data at phi=0")
    print("=" * 72)
    print(f"Gauge                : {args.gauge}")
    print(f"Polarization comp.   : {args.component}")
    if args.k_min is not None or args.k_max is not None:
        print(f"k range filter       : {args.k_min} .. {args.k_max} a.u.")
    print(f"Input normalization  : {args.input_normalization}")
    print(f"Number of theta pts  : {args.n_theta_output}")
    print(f"Input directory      : {args.input_dir}")
    print(f"Output directory     : {args.output_dir}")
    print(f"Output prefix        : {args.prefix}")
    print("=" * 72)

    generate_phi0_data(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        gauge=args.gauge,
        component=args.component,
        n_theta_output=args.n_theta_output,
        k_min=args.k_min,
        k_max=args.k_max,
        input_normalization=args.input_normalization,
        output_prefix=args.prefix,
        with_minus=args.with_minus_sign,
    )


if __name__ == "__main__":
    main()