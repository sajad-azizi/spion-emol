#!/usr/bin/env python3
"""
two_photon_delay.py
-------------------
High-level orchestrator for the C₈F₈⁻ (and generally any anion / closed-
shell-residual) RABBITT two-photon delay calculation.  Drop-in companion
to ``cross_section_delay.py``: takes the SAME run directory layout
(consolidated Phase A HDF5 in --phase-a, output goes alongside the
single-photon .dat files in ``--output-dir``) and produces

    two_photon_delay.dat   columns described in the header
    two_photon_delay.png   diagnostic plot (τ vs E and |M|² vs E)

Quantity computed (per Baykusheva & Wörner, J. Chem. Phys. 146 124306
(2017), hereafter BW17):

    τ(2q, k̂, R̂_γ)  ≡  (1/2ω) · arg[ M^{(2q-1)*} · M^{(2q+1)} ]      Eq. 21

This is the EFFECTIVE two-photon RABBITT delay.  By BW17 Eq. 23 it
decomposes into a universal Coulomb-laser piece τ_cc (Eq. 24) and a
target-specific molecular piece τ_mol (Eq. 25):

    τ  =  τ_cc(Z, ω; 2q)  +  τ_mol(2q, k̂, R̂_γ).

For an anion with neutral residual the Coulomb-laser factor A_κk
(Eq. 14) has Z = 0, all its imaginary phases vanish, and τ_cc(2q) is a
real-valued (constant-in-energy) contribution that does NOT enter the
observable.  Hence for C₈F₈⁻, H₂⁻, H₂O⁻ we report directly

    τ_2hω(2q, k̂, R̂_γ)  ≈  τ_mol(2q, k̂, R̂_γ).

The .dat file therefore contains τ_mol (orientation-averaged via BW17
Eqs. 22-26) as the "two-photon delay" column.

PIPELINE
========

Prerequisites (NOT done by this script; do them separately and confirm
__SUCCESS__ markers exist):

  1. preprocess_molden                → <work>/<mol>.preproc.h5
  2. scattering ik_min..ik_max dk     → <work>/dipole_<hash>_<scanid>/
                                          + <scratch>/psi_<hash>_ik*/
  3. cc_dipole_driver --ik_kappa ...  → cc_dipole.h5
                       --ik_nu   ...
  4. phase_a_assembler                → two_photon_me_<run>.h5

This script (5):

  5. two_photon_delay.py              → two_photon_delay.dat
     --phase-a two_photon_me_<run>.h5    + two_photon_delay.png
     --output-dir <gathered_dir>         (--output-dir is normally the
     --omega-IR-eV 1.55                   same dir cross_section_delay.py
     --T-X-fs 0.35  --T-L-fs 5.0          writes its .dat files into)
     --sidebands 30,40,50,60,70,80
     --angle-grid 6,6,6,6,6
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# Allow running from anywhere by adding our python/ to sys.path.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "two_photon_delay" / "python"))

from run_cc_delay import sideband_one    # type: ignore  (added to sys.path above)


EV_PER_AU = 27.211386245988
AU_PER_FS = 41.341374575751
AU_PER_AS = 24.188843265857


# ---------------------------------------------------------------------------
# Output file layout (so cross_section_delay.py can find it)
# ---------------------------------------------------------------------------
OUT_DAT_NAME = "two_photon_delay.dat"
OUT_PNG_NAME = "two_photon_delay.png"

DAT_HEADER = (
"# two_photon_delay.dat -- effective RABBITT two-photon delay for anion\n"
"# photodetachment, computed via BW17 Eq. 21.  For Z=0 residual the\n"
"# universal Coulomb-laser τ_cc (BW17 Eq. 24) is constant in energy\n"
"# (its imaginary phases vanish), so the value below is essentially\n"
"# τ_mol (Eq. 25), the molecular two-photon contribution.\n"
"#\n"
"# Columns:\n"
"#   1: ik_kappa        scan momentum index of the chosen sideband\n"
"#   2: E_kappa_au      photoelectron kinetic energy at sideband [au]\n"
"#   3: E_kappa_eV      same in eV (= 27.2114 * col 2)\n"
"#   4: tau_2hw_au      τ_2hω = (1/2ω) arg(<M_<* M_>>)        [au of time]\n"
"#   5: tau_2hw_as      same in attoseconds (= 24.18884 * col 4)\n"
"#   6: abs_MlsMgs      |<M_<* M_>>|  -- the two-photon amplitude  [a.u.]\n"
"#   7: arg_MlsMgs      arg(<M_<* M_>>) [rad]\n"
"#   8: abs_Mls         |<M_<>|       [a.u., coherent angle average]\n"
"#   9: abs_Mgs         |<M_>>|       [a.u., coherent angle average]\n"
"#\n"
"# Pulse parameters: ω_IR={omega_IR_eV:.4f} eV, T_X={T_X_fs:.3f} fs FWHM,\n"
"#                   T_L={T_L_fs:.3f} fs FWHM, τ_delay={tau_delay_fs:.3f} fs.\n"
"# Angle grid     : θ×φ × α×β×γ = {n_th}×{n_ph} × {n_a}×{n_b}×{n_g}.\n"
"# Lab polarization: m_p^IR_lab={m_IR_lab}, m_p^XUV_lab={m_XUV_lab}.\n"
"# Symmetric ν trim window: ±{trim_sigma}σ_β around each path's on-shell ν*.\n"
)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase-a", required=True, type=Path,
        help="Consolidated Phase A HDF5 (from phase_a_assembler.py).")
    ap.add_argument("--output-dir", required=True, type=Path,
        help="Directory to write two_photon_delay.dat + .png.  "
             "Normally the gathered_dipole_* dir so cross_section_delay.py "
             "sees them.")
    ap.add_argument("--sidebands", required=True,
        help="Comma-separated ik_kappa values to compute (e.g. "
             "'30,40,50,60,70,80').  These are SCAN INDICES; the actual "
             "photoelectron kinetic energy at sideband 2q is "
             "E_κ = (ik_κ · dk)² / 2.")
    ap.add_argument("--omega-IR-eV",   type=float, default=1.55,
        help="IR carrier photon energy in eV (default 1.55 eV = Ti:Sa, "
             "λ_IR = 800 nm; standard in Klünder et al. PRL 2011, Isinger "
             "et al. Science 2017, Mauritsson et al. PRL 2010, "
             "Sabbar et al. PRA 2015, Vos et al. Science 2018, etc.).")
    ap.add_argument("--T-X-fs",        type=float, default=0.30,
        help="XUV single-burst FWHM in fs (default 0.30 = 300 as; "
             "matches typical APT bursts H15-H23 in the cited experiments; "
             "Isinger et al. 2017 measured ~150 as, Mauritsson/Sabbar ~300 as).")
    ap.add_argument("--T-L-fs",        type=float, default=30.0,
        help="IR pulse FWHM in fs (default 30 fs ≈ 11 IR cycles; standard "
             "value across the RABBITT literature -- Klünder/Isinger/"
             "Mauritsson/Sabbar/Cattaneo/Vos/Heuser all use 30-40 fs).")
    ap.add_argument("--tau-delay-fs",  type=float, default=0.0,
        help="IR-XUV delay τ in fs (default 0; in experiment τ is SCANNED "
             "over a few IR periods and the sideband phase extracted by "
             "cos[2ωτ + φ] fitting; the molecular τ_2hω is the φ offset "
             "which is τ-independent at long T_L).")
    ap.add_argument("--angle-grid",
        default="6,6,6,6,6",
        help="θ×φ × α×β×γ grid sizes, comma-separated 5 ints (default 6,6,6,6,6). "
             "For converged orientation average use 8,8,8,8,8 or higher.")
    ap.add_argument("--m-p-IR-lab",    type=int, default=0, choices=(-1, 0, 1),
        help="Lab-frame IR polarization spherical-tensor m (default 0 = linear z).")
    ap.add_argument("--m-p-XUV-lab",   type=int, default=0, choices=(-1, 0, 1),
        help="Lab-frame XUV polarization spherical-tensor m (default 0 = linear z).")
    ap.add_argument("--trim-sigma",    type=float, default=8.0,
        help="Per-path symmetric trim window around the on-shell ν* in units "
             "of σ_β = √2/T_L (default 8 = ±8σ).  Set to 0 or negative to "
             "disable trim and integrate over the full ν grid (NOT recommended "
             "for paper-grade results -- the asymmetric P-tail produces "
             "window-dependent τ shifts).")
    ap.add_argument("--no-plot", action="store_true",
        help="Write only the .dat file, skip the diagnostic plot.")
    ap.add_argument("--quiet", action="store_true",
        help="Reduce stdout chatter from the inner driver.")
    args = ap.parse_args()

    if not args.phase_a.exists():
        print(f"error: --phase-a not found: {args.phase_a}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grid_pieces = [int(s) for s in args.angle_grid.split(",")]
    if len(grid_pieces) != 5:
        print(f"error: --angle-grid must be 5 ints comma-separated, got "
              f"{args.angle_grid!r}", file=sys.stderr)
        return 1
    n_th, n_ph, n_a, n_b, n_g = grid_pieces

    sidebands = [int(s) for s in args.sidebands.split(",")]

    omega_IR_au  = args.omega_IR_eV / EV_PER_AU
    T_X_au       = (args.T_X_fs * AU_PER_FS) / np.sqrt(2.0 * np.log(2.0))
    T_L_au       = (args.T_L_fs * AU_PER_FS) / np.sqrt(2.0 * np.log(2.0))
    tau_delay_au = args.tau_delay_fs * AU_PER_FS
    trim_sigma   = args.trim_sigma if args.trim_sigma > 0 else None

    # ----- run the inner driver one sideband at a time -----
    print(f"# two_photon_delay.py")
    print(f"#   Phase A    : {args.phase_a}")
    print(f"#   Output dir : {args.output_dir}")
    print(f"#   sidebands  : {sidebands}")
    print(f"#   ω_IR       : {args.omega_IR_eV} eV ({omega_IR_au:.6f} au)")
    print(f"#   T_X / T_L  : {args.T_X_fs} / {args.T_L_fs} fs")
    print(f"#   angle grid : θ×φ × α×β×γ = {n_th}×{n_ph} × {n_a}×{n_b}×{n_g}")
    print(f"#   trim       : ±{trim_sigma if trim_sigma else 'OFF'}σ_β")
    print()

    rows = []
    with h5py.File(args.phase_a, "r") as f:
        all_pairs = sorted(
            (int(k[6:10]), int(k[12:16])) for k in f["pairs"].keys())
        for ik_k in sidebands:
            available_nus = sorted({n for kk, n in all_pairs if kk == ik_k})
            if not available_nus:
                print(f"  WARN: no pairs for ik_κ={ik_k}; skipping")
                continue
            res = sideband_one(
                f, ik_k, available_nus, omega_IR_au,
                T_X_au, T_L_au, tau_delay_au,
                n_th, n_ph, n_a, n_b, n_g,
                m_p_IR_lab=args.m_p_IR_lab,
                m_p_XUV_lab=args.m_p_XUV_lab,
                symmetric_trim_sigma=trim_sigma,
                verbose=not args.quiet,
            )
            E_au = float(res["eps_kappa"])
            tau_au = float(res["tau_avg"])
            Ml = res["M_less"]; Mg = res["M_greater"]; W = res["W_grid"]
            ML = (W * Ml).sum() / W.sum()
            MG = (W * Mg).sum() / W.sum()
            cross = (W * np.conj(Ml) * Mg).sum() / W.sum()
            rows.append((
                int(ik_k),
                E_au,
                E_au * EV_PER_AU,
                tau_au,
                tau_au * AU_PER_AS,
                float(abs(cross)),
                float(np.angle(cross)),
                float(abs(ML)),
                float(abs(MG)),
            ))
            if not args.quiet:
                print(f"    => τ_2hω = {tau_au:+.6e} au = "
                      f"{tau_au * AU_PER_AS:+.3f} as")

    if not rows:
        print("error: no sidebands produced output; aborting", file=sys.stderr)
        return 1

    # ----- write .dat -----
    dat_path = args.output_dir / OUT_DAT_NAME
    hdr = DAT_HEADER.format(
        omega_IR_eV=args.omega_IR_eV,
        T_X_fs=args.T_X_fs, T_L_fs=args.T_L_fs,
        tau_delay_fs=args.tau_delay_fs,
        n_th=n_th, n_ph=n_ph, n_a=n_a, n_b=n_b, n_g=n_g,
        m_IR_lab=args.m_p_IR_lab, m_XUV_lab=args.m_p_XUV_lab,
        trim_sigma=trim_sigma if trim_sigma else "OFF",
    )
    arr = np.array(rows, dtype=float)
    fmt = ("%6d", "%.10e", "%.10e", "%.10e", "%.6f",
           "%.10e", "%.10e", "%.10e", "%.10e")
    with open(dat_path, "w") as fh:
        fh.write(hdr)
        np.savetxt(fh, arr, fmt=fmt)
    print(f"\n# wrote {dat_path}")

    # ----- write diagnostic plot -----
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        E_eV  = arr[:, 2]
        tau   = arr[:, 4]
        Mamp  = arr[:, 5]
        ax[0].plot(E_eV, tau, "o-", c="tab:red", ms=7)
        ax[0].axhline(0, c="gray", lw=0.5)
        ax[0].set_xlabel(r"$\varepsilon_\kappa$ (eV)")
        ax[0].set_ylabel(r"$\tau_{2\hbar\omega}$ (as)")
        ax[0].set_title(r"two-photon delay (BW17 Eq. 21, $\approx\tau_{mol}$ for Z=0)")
        ax[0].grid(alpha=0.3)
        ax[1].plot(E_eV, Mamp, "o-", c="tab:purple", ms=7)
        ax[1].set_xlabel(r"$\varepsilon_\kappa$ (eV)")
        ax[1].set_ylabel(r"$|\langle M_<^* M_>\rangle|$ (a.u.)")
        ax[1].set_title(r"two-photon amplitude (resonance check)")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        png_path = args.output_dir / OUT_PNG_NAME
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"# wrote {png_path}")

    print("\n# Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
