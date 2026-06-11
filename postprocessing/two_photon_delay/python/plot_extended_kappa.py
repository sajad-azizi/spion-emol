#!/usr/bin/env python3
"""
plot_extended_kappa.py
----------------------
Show τ_2ℏω(E_κ) at many κ values, side by side for H2⁻ and H2O⁻.
The earlier 3-point plot suggested a flat ~300-400 as cc delay; the
extended sweep reveals a resonance-like peak with off-resonance values
that drop to tens of as -- consistent with Lindroth-Dahlström for
closed-shell-residual anions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EV_PER_AU = 27.2114
AU_PER_AS = 24.188843


def single_photon_tau_from_phase_a(phase_a_h5: Path):
    with h5py.File(phase_a_h5, "r") as f:
        dk = float(f.attrs["dk"])
        iks = sorted(int(g[2:]) for g in f["per_ik"].keys() if g.startswith("ik"))
        k = np.array(iks, dtype=float) * dk
        E = k**2/2.0
        N_psi = int(f.attrs["N_psi"])
        D = np.zeros((len(iks), 3*N_psi), dtype=np.complex128)
        for i, ik in enumerate(iks):
            g = f[f"/per_ik/ik{ik:04d}"]
            for qi, q in enumerate(("x","y","z")):
                re = g[f"D_ortho_len_{q}_re"][:]; im = g[f"D_ortho_len_{q}_im"][:]
                D[i, qi*N_psi:(qi+1)*N_psi] = re + 1j*im
    dD = np.empty_like(D)
    for j in range(D.shape[1]):
        dD[:, j] = np.gradient(D[:, j], k)/k
    num = np.sum(np.conj(D)*dD, axis=1).imag
    den = np.sum(np.abs(D)**2, axis=1)
    tau_W = -np.divide(num, den, out=np.zeros_like(num), where=den>1e-30)
    return E, tau_W


def main():
    # Data points from the fine sweeps just run.
    # H2- (fine sweep available)
    h2 = {
      'E_eV': [3.401, 4.116, 4.898, 5.748, 5.927, 6.108, 6.291, 6.478, 6.667],
      'tau_as':[165.42, 306.45, 353.89, 361.26, 339.24, 285.99, 193.58, 94.53, 28.23],
    }
    # H2O- (sparser sweep)
    h2o = {
      'E_eV': [3.401, 4.116, 4.898, 5.748, 6.667],
      'tau_as':[416.20, 480.57, 447.98, 384.20, 165.38],
    }

    E_W_h2,  tau_W_h2  = single_photon_tau_from_phase_a(Path("h2_test/two_photon_me_h2_fine.h5"))
    E_W_h2o, tau_W_h2o = single_photon_tau_from_phase_a(Path("h2o_test/two_photon_me_h2o_cc_delay.h5"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    ax = axes[0]
    ax.plot(E_W_h2 * EV_PER_AU, tau_W_h2 * AU_PER_AS, '-',
            color='tab:blue', lw=2, label=r"$\tau_W$ one-photon")
    ax.plot(h2['E_eV'], h2['tau_as'], 'o-', color='tab:red', lw=2, ms=8,
            label=r"$\tau_{2\hbar\omega}$ (cc)")
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel("photoelectron kinetic energy E$_\\kappa$ (eV)")
    ax.set_ylabel(r"$\tau$  (as)")
    ax.set_title(r"H$_2^-$ photodetachment (THF, B3LYP/aug-cc-pVDZ + SE-HF)")
    ax.grid(alpha=0.3)
    ax.legend()
    # annotate peak
    peak_E, peak_tau = h2['E_eV'][3], h2['tau_as'][3]
    ax.annotate(f"peak: {peak_tau:.0f} as at {peak_E:.2f} eV",
                xy=(peak_E, peak_tau), xytext=(4.5, peak_tau+30),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.4),
                fontsize=9)
    # annotate off-resonance
    off_E, off_tau = h2['E_eV'][-1], h2['tau_as'][-1]
    ax.annotate(f"off-resonance: {off_tau:.0f} as\n(consistent with anion expectation)",
                xy=(off_E, off_tau), xytext=(5.5, -50),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.4),
                fontsize=9)

    ax = axes[1]
    ax.plot(E_W_h2o * EV_PER_AU, tau_W_h2o * AU_PER_AS, '-',
            color='tab:blue', lw=2, label=r"$\tau_W$ one-photon")
    ax.plot(h2o['E_eV'], h2o['tau_as'], 's-', color='tab:red', lw=2, ms=8,
            label=r"$\tau_{2\hbar\omega}$ (cc)")
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel("photoelectron kinetic energy E$_\\kappa$ (eV)")
    ax.set_ylabel(r"$\tau$  (as)")
    ax.set_title(r"H$_2$O$^-$ photodetachment (THF, B3LYP + SE-HF)")
    ax.grid(alpha=0.3)
    ax.legend()
    peak_E, peak_tau = h2o['E_eV'][1], h2o['tau_as'][1]
    ax.annotate(f"peak: {peak_tau:.0f} as at {peak_E:.2f} eV",
                xy=(peak_E, peak_tau), xytext=(4.7, peak_tau+30),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.4),
                fontsize=9)

    fig.suptitle(r"Two-photon RABBITT delay: extended $\kappa$ sweep reveals "
                  "resonance-like structure, not a flat $\\tau_{cc}$",
                  fontsize=12)
    fig.tight_layout()
    out = Path("h2_test/h2_and_h2o_resonance_structure.png")
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
