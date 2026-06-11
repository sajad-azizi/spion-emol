#!/usr/bin/env python3
"""
plot_cc_delay.py
----------------
Plot three quantities side-by-side as a function of photoelectron kinetic
energy:

  1. Single-photon Wigner-Smith delay τ_W(E)  computed from the b-c
     dipole D_ortho stored in the Phase A HDF5 (cross_section_delay.py
     convention, Pazourek/Nagele sign).
  2. CC delay τ_2hω(E_κ)  -- the two-photon RABBITT delay at each
     sideband κ, angle-averaged, computed by ``run_cc_delay.py``.
  3. Sum  τ_W + (τ_2hω − τ_W)  =  τ_2hω  -- shown to make the BW17
     decomposition visible: τ_cc^(BW17) ≡ τ_2hω − τ_W.  For an anion the
     short-range continuum makes the explicit Coulomb-laser τ_cc vanish,
     so the difference here measures non-Coulomb cc effects in the
     finite-pulse pipeline.

Usage:
    python3 plot_cc_delay.py
        --phase-a    h2o_test/two_photon_me_h2o_phaseA.h5
        --cc-delay   h2o_test/two_photon_me_h2o_tau.h5
        --out        h2o_test/cc_delay_compare.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EV_PER_AU = 27.211386245988
AU_PER_FS = 41.341374575751
AU_PER_AS = 24.188843265857


def single_photon_tau_from_phase_a(phase_a_h5: Path
                                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read per-ik D_ortho_len_(x,y,z) and compute τ_W(E) by Pazourek/Nagele:

        τ_W(E) = − Im[ Σ_q,μ D*_q,μ · dD_q,μ/dE ] / Σ_q,μ |D_q,μ|²

    where the q,μ sum is over Cartesian polarisation and channel index.
    Returns (E_kin, k, τ_W) all in atomic units.
    """
    with h5py.File(phase_a_h5, "r") as f:
        dk = float(f.attrs["dk"])
        iks = sorted(int(g[2:]) for g in f["per_ik"].keys() if g.startswith("ik"))
        if len(iks) < 3:
            raise ValueError(
                f"Need ≥ 3 ik points to compute τ_W via gradient; got {iks}")

        k = np.array(iks, dtype=float) * dk
        E = k ** 2 / 2.0
        N_psi = int(f.attrs["N_psi"])
        D_stack = np.zeros((len(iks), 3 * N_psi), dtype=np.complex128)
        for i, ik in enumerate(iks):
            g = f[f"/per_ik/ik{ik:04d}"]
            for q_idx, q in enumerate(("x", "y", "z")):
                re = g[f"D_ortho_len_{q}_re"][()]
                im = g[f"D_ortho_len_{q}_im"][()]
                D_stack[i, q_idx * N_psi:(q_idx + 1) * N_psi] = re + 1j * im

    # τ_W using k → E mapping (dE = k dk, dD/dE = dD/dk / k):
    dD_dE = np.empty_like(D_stack)
    for j in range(D_stack.shape[1]):
        dD_dk = np.gradient(D_stack[:, j], k)
        dD_dE[:, j] = dD_dk / k
    num = np.sum(np.conj(D_stack) * dD_dE, axis=1).imag
    den = np.sum(np.abs(D_stack) ** 2, axis=1)
    tau_W = -np.divide(num, den, out=np.zeros_like(num), where=den > 1e-30)
    return E, k, tau_W


def cc_tau_from_output(cc_h5: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read τ_2hω(ε_κ) angle-averaged from run_cc_delay.py output.
    Returns (E_kappa, τ_avg) in au.
    """
    out_E, out_tau = [], []
    with h5py.File(cc_h5, "r") as f:
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            if not isinstance(grp, h5py.Group):
                continue
            if "eps_kappa" not in grp or "tau_avg" not in grp:
                continue
            out_E.append(float(grp["eps_kappa"][()]))
            out_tau.append(float(grp["tau_avg"][()]))
    if not out_E:
        raise ValueError(f"no sideband_* groups found in {cc_h5}")
    order = np.argsort(out_E)
    return np.array(out_E)[order], np.array(out_tau)[order]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase-a", required=True, type=Path)
    ap.add_argument("--cc-delay", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default="H₂O RABBITT delay comparison")
    args = ap.parse_args()

    if not args.phase_a.exists():
        print(f"phase-a HDF5 missing: {args.phase_a}", file=sys.stderr)
        return 1
    if not args.cc_delay.exists():
        print(f"cc-delay HDF5 missing: {args.cc_delay}", file=sys.stderr)
        return 1

    E_W, _, tau_W_au = single_photon_tau_from_phase_a(args.phase_a)
    E_cc, tau_cc_au  = cc_tau_from_output(args.cc_delay)

    # Interpolate τ_W onto the cc grid to compute the difference (BW17 τ_cc).
    if len(E_W) >= 2:
        tau_W_on_cc = np.interp(E_cc, E_W, tau_W_au)
    else:
        tau_W_on_cc = np.full_like(E_cc, np.nan)
    diff = tau_cc_au - tau_W_on_cc          # τ_2hω - τ_W = τ_cc(BW17)

    print(f"# E_W grid (eV): {E_W * EV_PER_AU}")
    print(f"# τ_W (as):       {tau_W_au * AU_PER_AS}")
    print(f"# E_cc (eV):      {E_cc * EV_PER_AU}")
    print(f"# τ_2hω (as):     {tau_cc_au * AU_PER_AS}")
    print(f"# Δτ = τ_2hω − τ_W (as): {diff * AU_PER_AS}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

    ax = axes[0]
    ax.plot(E_W * EV_PER_AU, tau_W_au * AU_PER_AS, "-",
            color="tab:blue", lw=2, label=r"$\tau_W$ (1-photon)")
    ax.set_xlabel("photoelectron kinetic energy  (eV)")
    ax.set_ylabel(r"$\tau$  (as)")
    ax.set_title("Single-photon Wigner-Smith delay")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(E_cc * EV_PER_AU, tau_cc_au * AU_PER_AS, "o-",
            color="tab:red", lw=2, ms=7, label=r"$\tau_{2\hbar\omega}$ (cc)")
    ax.set_xlabel("photoelectron kinetic energy  (eV)")
    ax.set_ylabel(r"$\tau$  (as)")
    ax.set_title("Two-photon (cc) RABBITT delay")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(E_W * EV_PER_AU, tau_W_au * AU_PER_AS, "-",
            color="tab:blue", lw=2, label=r"$\tau_W$")
    ax.plot(E_cc * EV_PER_AU, tau_cc_au * AU_PER_AS, "o-",
            color="tab:red", lw=2, ms=7, label=r"$\tau_{2\hbar\omega}$")
    ax.plot(E_cc * EV_PER_AU, diff * AU_PER_AS, "s--",
            color="tab:green", lw=1.5, ms=6,
            label=r"$\Delta\tau = \tau_{2\hbar\omega} - \tau_W$  (BW17 $\tau_{cc}$)")
    ax.set_xlabel("photoelectron kinetic energy  (eV)")
    ax.set_ylabel(r"$\tau$  (as)")
    ax.set_title("Comparison + decomposition")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(args.title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"# wrote plot: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
