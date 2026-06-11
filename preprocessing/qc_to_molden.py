#!/usr/bin/env python3
"""
qc_to_molden.py
===============
Convert a quantum-chemistry output from ANY supported format into a single
canonical Molden file that the static-exchange preprocessing
(`preprocess_molden`) consumes.

WHY THIS EXISTS
---------------
The C++ preprocessing reads exactly ONE format -- a spec-compliant Molden
file in the no-Condon-Shortley real-spherical-harmonic convention with
contracted functions normalised to unit L2 norm (see
preprocessing/src/molden/Molden.hpp and basis/MoldenBasis.hpp).  Different
QC codes disagree on three conventions that, if mishandled, give SILENTLY
WRONG densities and cross sections:

  1. spherical-harmonic m-ordering   (molden: 0,+1,-1,+2,-2,...; PySCF: -l..+l)
  2. Condon-Shortley phase           (present or absent)
  3. contraction normalisation       (raw vs unit-norm; per-shell scale)

Rather than re-implement the convention zoo per format in C++ (the
multi-week-debugging trap), this front-end delegates ALL convention
handling to `iodata` (https://github.com/theochem/iodata), a
battle-tested QC interchange library that normalises every supported
input to ONE documented internal convention and writes a spec-compliant
Molden file.

VALIDATED EQUIVALENCE (2026-05-28)
----------------------------------
Round-trip  reference_molden -> iodata -> canonical_molden -> preprocess
reproduces the direct-preprocess SCE orbital norms, electron count, and
<rho|V_en> to ~1e-10 (molden) / ~1e-8 (fchk path).  See
preprocessing/tests/test_qc_to_molden_roundtrip.sh.  As a SECOND safety
net, the preprocessing itself runs an SCE self-check
(--validate-input, default ON) that aborts loudly if any conversion ever
produces an inconsistent density -- so a future iodata regression or an
unsupported edge case can never silently corrupt a production run.

SUPPORTED INPUT FORMATS  (via iodata unless noted)
--------------------------------------------------
  Gaussian   .fchk / .fch    (formatted checkpoint)         -> iodata 'fchk'
  Gaussian   .log / .out     (output; needs pop=full)       -> iodata 'gaussianlog'
  GAMESS-US  .dat / .gms     (punch / output)               -> iodata 'gamess'
  ORCA       .mkl            (from orca_2mkl)                -> iodata 'molekel'
  ORCA       .out            (output)                       -> iodata 'orcalog'
  Q-Chem     .out            (output)                       -> iodata 'qchemlog'
  Molpro     .molden         (put,molden,file)              -> iodata 'molden'
  Multiwfn   .mwfn                                           -> iodata 'mwfn'
  AIMPAC     .wfn / .wfx                                     -> iodata 'wfn'/'wfx'
  QCSchema   .json                                          -> iodata 'json_qcschema'
  PySCF      .chk            (HDF5 checkpoint)               -> pyscf native (see note)
  Molden     .molden         (any code)                     -> iodata 'molden' (canonicalises)

PySCF NOTE
----------
PySCF's own Molden writer (`pyscf.tools.molden.dump_scf`) is already
spec-compliant, so the *simplest* PySCF path is to call that in your
PySCF script and feed the .molden straight to preprocess_molden.  This
converter ALSO accepts a raw PySCF `.chk` (it imports pyscf, rebuilds
the Mole + SCF object, and dumps a molden) for the case where you only
kept the checkpoint.

USAGE
-----
    python3 qc_to_molden.py INPUT OUTPUT.molden [--format FMT] [--verbose]

      INPUT          path to the QC output (any supported format)
      OUTPUT.molden  canonical molden written here
      --format FMT   override auto-detection: one of
                       fchk gaussianlog gamess molekel orcalog qchemlog
                       molden mwfn wfn wfx json_qcschema pyscf-chk
      --verbose      print the parsed summary (atoms, nbf, nelec, MO kind)

Requires:  pip install qc-iodata   (and optionally pyscf for .chk inputs)
"""
from __future__ import annotations

import argparse
import os
import sys


# Map file extensions -> iodata format name.  Auto-detection fallback;
# always overridable with --format.
_EXT_TO_FMT = {
    ".fchk": "fchk",
    ".fch":  "fchk",
    ".molden": "molden",
    ".input": "molden",       # some molden files use .input
    ".mkl":  "molekel",       # ORCA orca_2mkl
    ".mwfn": "mwfn",
    ".wfn":  "wfn",
    ".wfx":  "wfx",
    ".json": "json_qcschema",
    ".chk":  "pyscf-chk",     # PySCF HDF5 checkpoint (special path)
    # Output-log formats are ambiguous by extension (.log / .out); the
    # user should pass --format for these.
}


def detect_format(path: str, override: str | None) -> str:
    if override:
        return override
    ext = os.path.splitext(path)[1].lower()
    if ext in _EXT_TO_FMT:
        return _EXT_TO_FMT[ext]
    raise SystemExit(
        f"qc_to_molden: cannot auto-detect format from extension '{ext}'.\n"
        f"  Pass --format explicitly.  For Gaussian/ORCA/Q-Chem OUTPUT logs\n"
        f"  (.log/.out) use --format gaussianlog | orcalog | qchemlog.\n"
        f"  For GAMESS use --format gamess.")


def convert_via_iodata(inp: str, out: str, fmt: str, verbose: bool) -> None:
    try:
        import iodata
    except ImportError:
        raise SystemExit(
            "qc_to_molden: the `iodata` package is required.\n"
            "  Install it with:  pip install qc-iodata\n"
            "  (on LRZ: load a python module first, then pip install --user qc-iodata)")

    if verbose:
        print(f"[qc_to_molden] loading '{inp}' as iodata format '{fmt}' ...")
    mol = iodata.load_one(inp, fmt=fmt)

    if verbose:
        _print_summary(mol)

    # Guard: the preprocessing needs MO coefficients.  Some formats (pure
    # geometry / density-only files) won't have them -- fail clearly here
    # rather than producing a molden the C++ side then rejects.
    if mol.mo is None or mol.mo.coeffs is None:
        raise SystemExit(
            f"qc_to_molden: '{inp}' has no molecular-orbital coefficients.\n"
            f"  The static-exchange pipeline needs occupied MOs.  Re-run the\n"
            f"  QC calculation with full orbital output (Gaussian: 'pop=full'\n"
            f"  + GFINPUT; Molpro: 'put,molden'; PySCF: dump_scf).")

    iodata.dump_one(mol, out, fmt="molden")
    if verbose:
        print(f"[qc_to_molden] wrote canonical molden -> '{out}'")


def convert_pyscf_chk(inp: str, out: str, verbose: bool) -> None:
    try:
        import pyscf
        from pyscf import scf
        from pyscf.tools import molden as pyscf_molden
        from pyscf import lib
    except ImportError:
        raise SystemExit(
            "qc_to_molden: reading a PySCF .chk needs the `pyscf` package.\n"
            "  Install it with:  pip install pyscf\n"
            "  OR (simpler): in your PySCF script call\n"
            "      from pyscf.tools import molden\n"
            "      molden.dump_scf(mf, 'out.molden')\n"
            "  and feed out.molden directly to preprocess_molden.")

    if verbose:
        print(f"[qc_to_molden] loading PySCF checkpoint '{inp}' ...")
    # Rebuild the Mole object stored in the checkpoint.
    mol = lib.chkfile.load_mol(inp)
    scf_dict = lib.chkfile.load(inp, "scf")
    if scf_dict is None:
        raise SystemExit(
            f"qc_to_molden: '{inp}' has no 'scf' group.  Is this a PySCF SCF\n"
            f"  checkpoint?  (Post-HF .chk files store the reference SCF under\n"
            f"  a different key -- dump the molden from your PySCF script.)")
    mo_coeff  = scf_dict["mo_coeff"]
    mo_energy = scf_dict["mo_energy"]
    mo_occ    = scf_dict["mo_occ"]

    # PySCF's molden writer emits spec-compliant molden (no-CS, molden
    # m-order, unit-norm) -- the same convention the C++ reader expects.
    if verbose:
        print(f"[qc_to_molden] dumping spec-compliant molden via "
              f"pyscf.tools.molden ...")
    pyscf_molden.from_mo(mol, out, mo_coeff,
                         ene=mo_energy, occ=mo_occ)
    if verbose:
        print(f"[qc_to_molden] wrote canonical molden -> '{out}'")


def _print_summary(mol) -> None:
    print("[qc_to_molden] === parsed molecule ===")
    print(f"    n_atom : {mol.natom}")
    print(f"    atnums : {list(mol.atnums)}")
    try:
        print(f"    nbf    : {mol.mo.coeffs.shape[0]}")
        print(f"    n_mo   : {mol.mo.coeffs.shape[1]}")
        print(f"    mo kind: {mol.mo.kind}")
        print(f"    nelec  : {mol.nelec}")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="QC output file (any supported format)")
    ap.add_argument("output", help="canonical molden output path")
    ap.add_argument("--format", default=None,
                    help="override format auto-detection (see header for the list)")
    ap.add_argument("--verbose", action="store_true",
                    help="print the parsed molecule summary")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"qc_to_molden: input file not found: '{args.input}'")

    fmt = detect_format(args.input, args.format)
    print(f"[qc_to_molden] input='{args.input}'  format='{fmt}'  "
          f"output='{args.output}'")

    if fmt == "pyscf-chk":
        convert_pyscf_chk(args.input, args.output, args.verbose)
    else:
        convert_via_iodata(args.input, args.output, fmt, args.verbose)

    # Belt-and-braces: confirm the file is non-empty and has the molden magic.
    if not os.path.isfile(args.output) or os.path.getsize(args.output) == 0:
        raise SystemExit("qc_to_molden: output molden was not written or is empty.")
    with open(args.output, "r") as f:
        head = f.read(64)
    if "[Molden Format]" not in head and "[Atoms]" not in head:
        raise SystemExit(
            "qc_to_molden: output does not look like a molden file "
            "(missing [Molden Format] / [Atoms] header).")
    print("[qc_to_molden] done.  Feed the output to preprocess_molden; the "
          "SCE self-check there is your second safety net.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
