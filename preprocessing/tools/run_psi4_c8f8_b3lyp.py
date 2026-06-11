#!/usr/bin/env python3
"""
run_psi4_c8f8_b3lyp.py
----------------------
Run B3LYP/6-311++G(d,p) for octafluorocubane C8F8 in two states:

    1. Neutral, closed-shell (charge=0, multiplicity=1) -- gas phase
    2. Anion, open-shell     (charge=-1, multiplicity=2) -- THF (PCM)

Outputs under <out_dir>/:
    c8f8_b3lyp_neutral.molden
    c8f8_b3lyp_anion_thf.molden
    *_psi4.out           (raw Psi4 logs; kept for provenance)

Geometry: O_h cubane, C at (+-a, +-a, +-a) and F at (+-b, +-b, +-b)
          a = 0.7778 A, b = 1.5812 A (same as preprocessing/tools/gen_reference.py).

Usage:
    conda activate psi4
    python run_psi4_c8f8_b3lyp.py [--out-dir DIR] [--memory GB] [--threads N]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import psi4
except ImportError:
    sys.stderr.write("ERROR: psi4 not importable. Activate the psi4 conda env first.\n")
    sys.exit(1)


# -------------------- geometry --------------------
# Cubane C8F8, O_h point group. Same geometry used by gen_reference.py.
C_A   = 0.7778    # C radius from origin (Angstrom)
F_A   = 1.5812    # F radius from origin (Angstrom)

_GEOM_LINES = []
for sym, R in (("C", C_A), ("F", F_A)):
    for sx in (-1, +1):
        for sy in (-1, +1):
            for sz in (-1, +1):
                _GEOM_LINES.append(f"{sym}  {sx*R:+10.6f}  {sy*R:+10.6f}  {sz*R:+10.6f}")
GEOM = "\n".join(_GEOM_LINES)


def run_one(label: str,
            charge: int,
            multiplicity: int,
            pcm_solvent: str | None,
            basis: str,
            out_dir: Path,
            memory_gb: int,
            threads: int,
            compute_polarizability: bool = False) -> Path:
    """Run B3LYP SCF for C8F8 at given charge/mult (+ optional PCM) and
    write a molden file.  Returns the molden path."""
    tag = f"c8f8_b3lyp_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.set_output_file(str(out_dir / f"{tag}_psi4.out"), False)
    psi4.set_memory(f"{memory_gb} GB")
    psi4.set_num_threads(threads)

    is_open_shell = (multiplicity != 1)
    geom_str = (f"{charge} {multiplicity}\n" + GEOM +
                "\nsymmetry c1\nno_reorient\nno_com\nunits angstrom\n")
    mol = psi4.geometry(geom_str)

    psi4.set_options({
        "basis":          basis,
        "puream":         True,                 # spherical AOs
        "reference":      "uks" if is_open_shell else "rks",
        "scf_type":       "df",                 # density-fitted -- necessary for C8F8
        "e_convergence":  1e-9,
        "d_convergence":  1e-7,
        "maxiter":        200,
        "print_mos":      False,
    })

    if pcm_solvent is not None:
        psi4.set_options({"pcm": True, "pcm_scf_type": "total"})
        pcm_input = (
            "Medium {\n"
            "    SolverType = IEFPCM\n"
            f"    Solvent    = {pcm_solvent}\n"
            "}\n"
            "Cavity {\n"
            "    Type = GePol\n"
            "    RadiiSet = UFF\n"
            "    Scaling = True\n"
            "    Mode = Implicit\n"
            "}\n"
        )
        psi4.pcm_helper(pcm_input)
        print(f"[{label}]  PCM enabled, solvent = {pcm_solvent}")
    else:
        print(f"[{label}]  gas phase")

    print(f"[{label}]  running B3LYP/{basis}  charge={charge:+d} mult={multiplicity} ...")
    e_scf, wfn = psi4.energy("b3lyp", return_wfn=True)

    molden_path = out_dir / f"{tag}.molden"
    psi4.molden(wfn, str(molden_path))
    nbf = int(wfn.basisset().nbf())
    print(f"[{label}]  E_DFT = {e_scf:.10f} Ha   nbf={nbf}")
    print(f"[{label}]  wrote  {molden_path}")

    # ------ finite-field polarizability ------
    # α_ij = -∂²E/∂F_i ∂F_j |_{F=0}  ≈  -(E(+F_i ê_j) + E(-F_i ê_j) - 2 E_0) / F²
    # We compute the three DIAGONAL entries α_xx, α_yy, α_zz here — exact
    # for O_h C8F8 (symmetry forces off-diagonals = 0 and α_xx = α_yy = α_zz,
    # so one field would be enough, but doing three is a cheap symmetry check).
    if compute_polarizability:
        print(f"[{label}]  computing α by finite field (ΔF = 0.001 au, 6 extra SCFs) ...")
        delta = 0.001
        alpha_diag = [0.0, 0.0, 0.0]
        axes = ("x", "y", "z")
        for i_ax, ax in enumerate(axes):
            field_plus  = [0.0, 0.0, 0.0];  field_plus[i_ax]  = +delta
            field_minus = [0.0, 0.0, 0.0];  field_minus[i_ax] = -delta
            psi4.set_options({"perturb_h":      True,
                              "perturb_with":   "dipole",
                              "perturb_dipole": field_plus})
            E_plus = psi4.energy("b3lyp")
            psi4.set_options({"perturb_dipole": field_minus})
            E_minus = psi4.energy("b3lyp")
            alpha_diag[i_ax] = -(E_plus + E_minus - 2.0 * e_scf) / (delta * delta)
            print(f"[{label}]    α_{ax}{ax} = {alpha_diag[i_ax]:+.6f} au")
        # Disable perturbation for subsequent calls.
        psi4.set_options({"perturb_h": False})
        alpha = np.diag(alpha_diag)
        alpha_iso = float(np.trace(alpha) / 3.0)
        alpha_path = out_dir / f"{tag}_polarizability.txt"
        with open(alpha_path, "w") as f:
            f.write(f"# B3LYP/{basis} static dipole polarizability tensor (atomic units)\n")
            f.write(f"# Molecule: C8F8 ({label})\n")
            f.write(f"# Method: finite field, ΔF = {delta} au, central difference of E\n")
            for ii in range(3):
                f.write(f"{alpha[ii,0]:+.8e}  {alpha[ii,1]:+.8e}  {alpha[ii,2]:+.8e}\n")
            f.write(f"# alpha_iso = tr(alpha)/3 = {alpha_iso:.6f} au\n")
        print(f"[{label}]  α_iso = {alpha_iso:.4f} au")
        print(f"[{label}]  (O_h symmetry check: α_xx - α_zz = "
              f"{alpha_diag[0] - alpha_diag[2]:+.3e} au, should be 0)")
        print(f"[{label}]  wrote  {alpha_path}")
    return molden_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="B3LYP/6-311++G(d,p) SCF for C8F8 (neutral + anion/THF) → molden")
    default_out = os.environ.get("WORK", ".") + "/c8f8_b3lyp"
    ap.add_argument("--out-dir", default=default_out,
                    help=f"Output directory (default: {default_out})")
    ap.add_argument("--memory",  type=int, default=4,
                    help="Psi4 memory in GB (default: 4)")
    ap.add_argument("--threads", type=int, default=4,
                    help="Psi4 threads (default: 4)")
    ap.add_argument("--basis", default="6-311++g(d,p)",
                    help='Psi4 basis name (default: "6-311++g(d,p)")')
    ap.add_argument("--skip-neutral", action="store_true",
                    help="Skip the neutral calculation (assumes molden already exists)")
    ap.add_argument("--skip-anion",   action="store_true",
                    help="Skip the anion calculation")
    ap.add_argument("--polarizability", action="store_true",
                    help="Compute static dipole polarizability for the neutral "
                         "(used downstream by --polarizability in preprocess_molden)")
    ap.add_argument("--solvent", default="Tetrahydrofurane",
                    help='PCM solvent name for anion (default: "Tetrahydrofurane")')
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    print(f"Output directory: {out_dir}")
    print(f"Memory: {args.memory} GB   threads: {args.threads}")
    print(f"Basis: {args.basis}\n")

    if not args.skip_neutral:
        run_one(label="neutral",
                charge=0, multiplicity=1,
                pcm_solvent=None,
                basis=args.basis, out_dir=out_dir,
                memory_gb=args.memory, threads=args.threads,
                compute_polarizability=args.polarizability)
        print()

    if not args.skip_anion:
        run_one(label="anion_thf",
                charge=-1, multiplicity=2,
                pcm_solvent=args.solvent,
                basis=args.basis, out_dir=out_dir,
                memory_gb=args.memory, threads=args.threads)
        print()

    print("All done.")


if __name__ == "__main__":
    main()
