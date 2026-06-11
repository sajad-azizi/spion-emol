#!/usr/bin/env python3
"""
merge_cc_dipole_h5.py
=====================
Combine the per-κ HDF5 files written by `cc_dipole_mpi` into a single
`cc_dipole.h5` matching the layout that `phase_a_assembler` and
`two_photon_delay.py` expect from `cc_dipole_driver`:

    root attrs        : Nr, dr, N_psi, l_cont, n_keep_lo, n_keep_hi
    group pair_kKKKK_nNNNN/
        attrs         : ik_kappa, ik_nu, E_kappa, E_nu
        datasets      : cc_raw_len_x, cc_raw_len_y, cc_raw_len_z

Per-κ source files are produced by the MPI driver as
    <in-dir>/cc_dipole_kKKKK.h5
each containing the same root attrs + the subset of pair_* groups for
that one κ.  The merge is a pure metadata + dataset COPY: no math.

Usage
-----
    python3 merge_cc_dipole_h5.py \\
        --in-dir $WORK/cc_dipole_mpi \\
        --output $WORK/cc_dipole.h5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def merge(in_dir: Path, output: Path) -> int:
    files = sorted(in_dir.glob("cc_dipole_k*.h5"))
    if not files:
        print(f"ERROR: no cc_dipole_k*.h5 files in {in_dir}", file=sys.stderr)
        return 1
    print(f"  merging {len(files)} per-κ files from {in_dir}")
    if output.exists():
        output.unlink()
    out = h5py.File(output, "w")
    root_attrs_done = False
    n_pairs = 0
    for f_path in files:
        with h5py.File(f_path, "r") as src:
            # Root attrs: copy once, then verify identical on all subsequent files.
            for name, val in src.attrs.items():
                if not root_attrs_done:
                    out.attrs[name] = val
                else:
                    ref = out.attrs[name]
                    if isinstance(val, np.ndarray):
                        if not np.array_equal(val, ref):
                            print(f"WARN: root attr {name} mismatch in {f_path}",
                                  file=sys.stderr)
                    elif val != ref:
                        print(f"WARN: root attr {name} = {val} in {f_path} "
                              f"vs {ref} in first file", file=sys.stderr)
            root_attrs_done = True
            # Copy each pair_* group verbatim.
            for key in src.keys():
                if not key.startswith("pair_"):
                    continue
                if key in out:
                    print(f"WARN: duplicate pair {key} in {f_path}; skipping",
                          file=sys.stderr)
                    continue
                src.copy(key, out, name=key)
                n_pairs += 1
        print(f"    {f_path.name}: cumulative pairs = {n_pairs}")
    out.close()
    print(f"  wrote {output}  ({n_pairs} pair groups)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, type=Path,
        help="Directory containing cc_dipole_kKKKK.h5 files from cc_dipole_mpi.")
    ap.add_argument("--output", required=True, type=Path,
        help="Single merged cc_dipole.h5 to create (overwritten if exists).")
    args = ap.parse_args()
    return merge(args.in_dir, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
