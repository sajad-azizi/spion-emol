#!/usr/bin/env python3
"""
gather_dipoles.py
-----------------
Read a scattering scan dir produced by the C++ code (contains manifest.h5
and per-energy ikNNNN.h5 files) and write per-channel .dat files, one file
per (gauge, polarization, channel).

Output file format (unchanged from version_0 postprocessing, kept so the
cross-section script is a drop-in consumer):

    dipole_len_homo_{x,y,z}_{mu}.dat
    dipole_vel_homo_{x,y,z}_{mu}.dat

Columns: k_elec  E_kin  omega  Re(D_mu)  Im(D_mu)  |D_mu|^2  arg(D_mu)

All energies in atomic units.

What has changed vs. version_0:

  * No factor-2 correction. The C++ code writes D_reduced (orthogonalized)
    directly into /dipole/.../D_ortho_re|im. That IS the corrected amplitude.
  * No A1g odd-l symmetry filter. It was molecule-specific to Oh systems
    and should live elsewhere if needed.
  * Input is the HDF5 scan dir (not .dat files). Much less fragile.
  * Output goes into $WORK/gathered_<scan_tag>/ by default.

Usage:
    python gather_dipoles.py <scan_dir> [--output-dir DIR] [--raw]

If <scan_dir> is relative and $WORK is set, the script first looks in
$WORK/<scan_dir>. If --output-dir is omitted, output lands in
$WORK/gathered_<basename of scan_dir>/ or ./gathered_<basename>/ if
$WORK is unset.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np


# -------------------- angular index helpers --------------------
def idx_to_lm(mu: int) -> Tuple[int, int]:
    """Real-Y convention: mu = l*l + l + m."""
    l = int(np.floor(np.sqrt(mu)))
    while (l + 1) * (l + 1) <= mu:
        l += 1
    m = mu - l * l - l
    return l, m


# -------------------- HDF5 readers --------------------
def read_manifest(scan_dir: Path) -> dict:
    path = scan_dir / "manifest.h5"
    if not path.exists():
        raise FileNotFoundError(f"no manifest.h5 in {scan_dir}")
    out = {}
    with h5py.File(path, "r") as f:
        g = f["/grid"]
        out["r_min"]           = float(g.attrs["r_min"])
        out["dr"]              = float(g.attrs["dr"])
        out["N_grid"]          = int(g.attrs["N_grid"])
        out["l_max_continuum"] = int(g.attrs["l_max_continuum"])
        out["E_HOMO"]          = float(g.attrs["E_HOMO"])
        k = f["/kgrid"]
        out["dk"]     = float(k.attrs["dk"])
        out["ik_min"] = int(k.attrs["ik_min"])
        out["ik_max"] = int(k.attrs["ik_max"])
        if "/occ" in f:
            out["occ_energies"]     = f["/occ/energies"][:]
            out["occ_spin_factors"] = f["/occ/spin_factors"][:]
        if "/run" in f:
            out["molecule_name"] = f["/run"].attrs["molecule_name"].decode("ascii", "replace") \
                if isinstance(f["/run"].attrs["molecule_name"], (bytes, bytearray)) \
                else str(f["/run"].attrs["molecule_name"])
    return out


def enumerate_ik_files(scan_dir: Path) -> List[Tuple[int, Path]]:
    out = []
    for p in scan_dir.iterdir():
        n = p.name
        if n.startswith("ik") and n.endswith(".h5") and p.is_file():
            try:
                ik = int(n[2:-3])
            except ValueError:
                continue
            out.append((ik, p))
    out.sort(key=lambda t: t[0])
    return out


def read_one_energy(path: Path, use_raw: bool) -> Dict[str, np.ndarray]:
    """Return {"ik","k","E","omega", gauge×pol -> complex_vec}."""
    r = {}
    with h5py.File(path, "r") as f:
        m = f["/meta"]
        r["ik"]    = int(m.attrs["ik"])
        r["k"]     = float(m.attrs["k"])
        r["E"]     = float(m.attrs["E"])
        r["omega"] = float(m.attrs["omega"])
        for gauge in ("length", "velocity"):
            for pol in ("x", "y", "z"):
                g = f[f"/dipole/{gauge}/{pol}"]
                re_name = "D_raw_re"   if use_raw else "D_ortho_re"
                im_name = "D_raw_im"   if use_raw else "D_ortho_im"
                d = g[re_name][:] + 1j * g[im_name][:]
                r[(gauge, pol)] = d
    return r


# -------------------- gathering --------------------
def resolve_scan_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_absolute() and p.exists():
        return p
    # Try as given (relative to cwd).
    if p.exists():
        return p.resolve()
    # Try under $WORK.
    work = os.environ.get("WORK")
    if work:
        q = Path(work) / arg
        if q.exists():
            return q
    raise FileNotFoundError(f"scan dir not found: {arg}")


def resolve_output_dir(cli_out: str | None, scan_dir: Path) -> Path:
    if cli_out:
        p = Path(cli_out)
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()
    work = os.environ.get("WORK")
    base = Path(work) if work else Path.cwd()
    out = base / f"gathered_{scan_dir.name}"
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def gather(scan_dir: Path, output_dir: Path, use_raw: bool) -> None:
    meta = read_manifest(scan_dir)
    files = enumerate_ik_files(scan_dir)
    if not files:
        raise RuntimeError(f"no ikNNNN.h5 files under {scan_dir}")

    # First pass: grid + channel count from first file.
    first_ik, first_path = files[0]
    first = read_one_energy(first_path, use_raw)
    n_channels = len(first[("length", "x")])

    # Collect (k, E, omega, D[gauge,pol]) arrays across all ik.
    n_e = len(files)
    k_arr     = np.zeros(n_e)
    E_arr     = np.zeros(n_e)
    omega_arr = np.zeros(n_e)
    # Allocate D buffers: shape (n_e, n_channels) per (gauge, pol).
    buf: Dict[Tuple[str, str], np.ndarray] = {
        (g, p): np.zeros((n_e, n_channels), dtype=complex)
        for g in ("length", "velocity") for p in ("x", "y", "z")
    }

    for row, (ik, path) in enumerate(files):
        rec = read_one_energy(path, use_raw)
        if len(rec[("length", "x")]) != n_channels:
            raise RuntimeError(f"channel count differs between ik files "
                               f"(ik={ik}: {len(rec[('length','x')])} vs {n_channels})")
        k_arr[row]     = rec["k"]
        E_arr[row]     = rec["E"]
        omega_arr[row] = rec["omega"]
        for key in buf:
            buf[key][row, :] = rec[key]

    print(f"  read {n_e} energy points; {n_channels} channels per energy")
    print(f"  k range: [{k_arr.min():.4f}, {k_arr.max():.4f}] au")
    print(f"  E range: [{E_arr.min():.6f}, {E_arr.max():.6f}] au")
    print(f"  omega  : [{omega_arr.min():.6f}, {omega_arr.max():.6f}] au")

    # Write one .dat per (gauge, pol, mu).
    prefix_of = {"length": "dipole_len_homo", "velocity": "dipole_vel_homo"}
    header_base = (
        f"scan_dir: {scan_dir}\n"
        f"dk = {meta['dk']:.12g} au   E_HOMO = {meta['E_HOMO']:.6f} au   "
        f"Ip = {-meta['E_HOMO']:.6f} au   dipole = {'RAW (pre-ortho)' if use_raw else 'ORTHO'}\n"
        "Columns: k_elec[au]  E_kin[au]  omega[au]  Re(D)  Im(D)  |D|^2  arg(D)"
    )

    n_written = 0
    for (gauge, pol), D in buf.items():
        prefix = prefix_of[gauge]
        for mu in range(n_channels):
            l, m = idx_to_lm(mu)
            col_re  = D[:, mu].real
            col_im  = D[:, mu].imag
            col_abs2 = (np.abs(D[:, mu])) ** 2
            col_arg = np.angle(D[:, mu])
            out = np.column_stack([k_arr, E_arr, omega_arr,
                                   col_re, col_im, col_abs2, col_arg])
            path = output_dir / f"{prefix}_{pol}_{mu}.dat"
            hdr = header_base + f"\nmu = {mu}  (l={l}, m={m:+d})"
            np.savetxt(path, out, header=hdr, fmt="%.14e")
            n_written += 1

    print(f"  wrote {n_written} .dat files to {output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gather per-channel dipole matrix elements from a scan dir into .dat files.")
    ap.add_argument("scan_dir", help="Path to dipole_<hash>_<scan>/ (absolute or relative; "
                                     "if relative and $WORK is set, $WORK is tried too).")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write .dat files. Default: "
                         "$WORK/gathered_<scan_dir>/ or ./gathered_<scan_dir>/ if $WORK is unset.")
    ap.add_argument("--raw", action="store_true",
                    help="Use D_raw (pre-orthogonalization) instead of D_ortho. Diagnostic only.")
    args = ap.parse_args()

    scan_dir = resolve_scan_dir(args.scan_dir)
    output_dir = resolve_output_dir(args.output_dir, scan_dir)
    print("=" * 72)
    print("gather_dipoles.py")
    print("=" * 72)
    print(f"  scan_dir  : {scan_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  source    : {'D_raw (PRE-ortho)' if args.raw else 'D_ortho (orthogonalized)'}")
    print()

    gather(scan_dir, output_dir, args.raw)
    print("Done.")


if __name__ == "__main__":
    main()
