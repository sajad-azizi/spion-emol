#!/usr/bin/env python3
"""
phase_a_assembler.py
--------------------
Consolidate the Phase-A matrix-element inputs that Phase C will consume
into a SINGLE HDF5 file: ``two_photon_me_<scan_id>.h5``.

What goes in:

  1. cc_dipole.h5            -- continuum-continuum radial dipole matrices
                                in the back-prop basis, per (κ, ν) pair,
                                per polarisation.  Produced by
                                ``cc_dipole_driver``.
  2. <scan_dir>/manifest.h5  -- grid + kgrid + run metadata for the
                                scattering scan that produced the saved
                                ψ from which (1) was computed.
  3. <scan_dir>/ikNNNN.h5    -- per-ik scattering outputs:
                                A, B, b_overlap, d_raw (b-c radial
                                dipole in back-prop basis, both gauges
                                and x/y/z), D_raw, D_ortho (in-state
                                basis, both gauges, x/y/z), d_correction.

This script does NOT recompute anything.  It re-organises existing
results into the layout described below so Phase C has a single,
self-contained input.

CONSOLIDATED HDF5 LAYOUT (output)
=================================
Top-level attrs:
    E_HOMO         = ε_i  (initial-state energy, e.g. anion SOMO)
    N_grid         = Nr
    dr, r_min
    l_cont         = continuum angular cutoff
    N_psi          = (l_cont + 1)² = # channels
    dk, ik_min, ik_max, n_ik
    n_keep_lo, n_keep_hi  (radial window covered by ψ on disk)
    molecule, git_hash, iso_date_utc, scan_id

Datasets:
    /channels/l_mu (N_psi,)   -- l of each channel μ = l² + l + m
    /channels/m_mu (N_psi,)   -- m of each channel μ

Per-ik (one group per ik in [ik_min, ik_max]):
    /per_ik/ik<NNNN>/
        attrs: ik, k, E_kin, omega, fit_residual_rel, K_symmetry_err
        /A                (N_psi, N_psi)   asymptotic-fit A matrix
        /B                (N_psi, N_psi)
        /b_overlap        (N_psi, n_occ)
        /d_raw_len_q      (N_psi,)         REAL  q ∈ {x, y, z}
        /d_raw_vel_q      (N_psi,)         REAL
        /D_raw_len_q_re   (N_psi,)         COMPLEX (re/im pair)
        /D_raw_len_q_im   (N_psi,)
        /D_ortho_len_q_re (N_psi,)
        /D_ortho_len_q_im (N_psi,)
        /D_raw_vel_q_re/im, /D_ortho_vel_q_re/im
        /d_correction_<gauge>_q  (n_occ,)

Per-pair (one group per unique (κ, ν) pair in cc_dipole.h5):
    /pairs/pair_kNNNN_nMMMM/
        attrs: ik_kappa, ik_nu, E_kappa, E_nu
        /cc_raw_len_q     (N_psi, N_psi)   q ∈ {x, y, z}

Phase C entry point: ``compute_T.py`` opens this file and iterates over
the energy grid using ``/per_ik/`` for the b-c side and ``/pairs/`` for
the c-c side.

Usage
=====
::

    python phase_a_assembler.py \
        --cc-h5    /path/to/cc_dipole.h5 \
        --scan-dir /path/to/dipole_<hash>_<scan_id> \
        --out      /path/to/two_photon_me_<scan_id>.h5
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np

# Currently produced by cc_dipole_driver: length gauge only.
_GAUGES = ("len",)
_POLS   = ("x", "y", "z")


def _idx_to_lm(N_psi: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (l_mu, m_mu) tables for μ = 0 .. N_psi-1 with μ = l² + l + m."""
    l_mu = np.empty(N_psi, dtype=np.int32)
    m_mu = np.empty(N_psi, dtype=np.int32)
    for mu in range(N_psi):
        l = int(np.floor(np.sqrt(mu)))
        while (l + 1) ** 2 <= mu:
            l += 1
        l_mu[mu] = l
        m_mu[mu] = mu - l * l - l
    return l_mu, m_mu


def _read_scan_dir(scan_dir: Path) -> Dict:
    """Pull manifest + every ik*.h5 in the scan dir into a dict."""
    manifest_path = scan_dir / "manifest.h5"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.h5 in {scan_dir}")
    out = {}
    with h5py.File(manifest_path, "r") as fm:
        out["E_HOMO"]   = float(fm["/grid"].attrs["E_HOMO"])
        out["N_grid"]   = int  (fm["/grid"].attrs["N_grid"])
        out["dr"]       = float(fm["/grid"].attrs["dr"])
        out["r_min"]    = float(fm["/grid"].attrs["r_min"])
        out["l_cont"]   = int  (fm["/grid"].attrs["l_max_continuum"])
        out["dk"]       = float(fm["/kgrid"].attrs["dk"])
        out["ik_min"]   = int  (fm["/kgrid"].attrs["ik_min"])
        out["ik_max"]   = int  (fm["/kgrid"].attrs["ik_max"])
        if "/run" in fm:
            for k in ("molecule_name", "git_hash", "iso_date_utc"):
                if k in fm["/run"].attrs:
                    v = fm["/run"].attrs[k]
                    out[k] = v.decode() if isinstance(v, bytes) else str(v)
    out["ik_files"] = sorted(scan_dir.glob("ik*.h5"))
    out["scan_dir"] = scan_dir
    return out


def _read_cc(cc_h5_path: Path) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """Read cc_dipole_driver output -> {(ik_κ, ik_ν): {pol: cc_raw_len}}."""
    out: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    pair_re = re.compile(r"pair_k(\d+)_n(\d+)")
    with h5py.File(cc_h5_path, "r") as f:
        for name in f.keys():
            m = pair_re.match(name)
            if not m:
                continue
            ik_kappa = int(m.group(1))
            ik_nu    = int(m.group(2))
            g = f[name]
            pair: Dict[str, np.ndarray] = {}
            for pol in _POLS:
                ds = f"cc_raw_len_{pol}"
                if ds in g:
                    pair[pol] = g[ds][()]
            if not pair:
                raise RuntimeError(
                    f"{name} has no cc_raw_len_* datasets")
            # cc_dipole_driver stores E_kappa/E_nu/ik_kappa/ik_nu as
            # group ATTRIBUTES (H5Acreate2), not datasets.
            pair["E_kappa"] = float(g.attrs["E_kappa"])
            pair["E_nu"]    = float(g.attrs["E_nu"])
            out[(ik_kappa, ik_nu)] = pair
    return out


def _copy_per_ik(fin: h5py.File, fout: h5py.File, ik: int):
    """Copy A, B, b_overlap, d_raw, D_raw, D_ortho, d_correction from
    a per-ik scattering HDF5 into /per_ik/ik<NNNN>/ in the output."""
    g_name = f"/per_ik/ik{ik:04d}"
    if g_name in fout:
        del fout[g_name]
    g = fout.create_group(g_name)
    # /meta attrs
    meta = fin["/meta"].attrs
    for k in ("ik", "k", "E", "omega", "fit_residual_rel", "K_symmetry_err"):
        if k in meta:
            g.attrs[k] = float(meta[k]) if k != "ik" else int(meta[k])
    # /A, /B, /b_overlap
    for k in ("A", "B", "b_overlap"):
        if k in fin:
            g.create_dataset(k, data=fin[k][()])
    # /dipole/{length,velocity}/{x,y,z}/...
    gauge_short = {"length": "len", "velocity": "vel"}
    for gauge_full, gs in gauge_short.items():
        if f"/dipole/{gauge_full}" not in fin:
            continue
        for pol in _POLS:
            base = f"/dipole/{gauge_full}/{pol}"
            if base not in fin:
                continue
            sub = fin[base]
            # REAL d_raw (back-prop-basis b-c matrix element)
            if "d_raw" in sub:
                g.create_dataset(f"d_raw_{gs}_{pol}", data=sub["d_raw"][()])
            # COMPLEX D_raw (in-state basis, no ortho correction)
            for tag in ("D_raw", "D_ortho"):
                if f"{tag}_re" in sub and f"{tag}_im" in sub:
                    g.create_dataset(f"{tag}_{gs}_{pol}_re",
                                     data=sub[f"{tag}_re"][()])
                    g.create_dataset(f"{tag}_{gs}_{pol}_im",
                                     data=sub[f"{tag}_im"][()])
            if "d_correction" in sub:
                g.create_dataset(f"d_correction_{gs}_{pol}",
                                 data=sub["d_correction"][()])


def assemble(cc_h5: Path, scan_dir: Path, out: Path) -> None:
    scan = _read_scan_dir(scan_dir)
    cc   = _read_cc(cc_h5)

    # Cross-check: every ik referenced by cc.h5 must have a matching ik*.h5
    # in the scan dir.  Phase C cannot recover from a missing per-ik file
    # because it needs A, B, d_raw at that energy.
    all_ik_in_cc = sorted({ik for pair in cc for ik in pair})
    have = {int(re.search(r"ik(\d+)", p.name).group(1)) for p in scan["ik_files"]}
    missing = [ik for ik in all_ik_in_cc if ik not in have]
    if missing:
        raise RuntimeError(
            f"cc_dipole.h5 references ik(s) {missing} but {scan_dir} has no "
            f"matching ik*.h5 file(s).  Re-run scattering at those ik or "
            f"prune the cc_dipole.h5 input.")

    N_psi = None
    n_occ = None
    n_keep_lo = None
    n_keep_hi = None

    print(f"[phase A] writing {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        # ---- Top-level metadata ----
        for k in ("E_HOMO", "N_grid", "dr", "r_min", "l_cont",
                  "dk", "ik_min", "ik_max",
                  "molecule_name", "git_hash", "iso_date_utc"):
            if k in scan:
                f.attrs[k] = scan[k]
        f.attrs["scan_id"] = scan_dir.name

        # ---- Per-ik groups ----
        n_ik = 0
        for ik_path in scan["ik_files"]:
            ik = int(re.search(r"ik(\d+)", ik_path.name).group(1))
            with h5py.File(ik_path, "r") as fin:
                _copy_per_ik(fin, f, ik)
                # Cache N_psi / n_occ from the first file
                if N_psi is None and "A" in fin:
                    N_psi = int(fin["A"].shape[0])
                if n_occ is None and "b_overlap" in fin:
                    n_occ = int(fin["b_overlap"].shape[1])
            n_ik += 1
        f.attrs["n_ik"]  = n_ik
        f.attrs["N_psi"] = N_psi if N_psi is not None else -1
        f.attrs["n_occ"] = n_occ if n_occ is not None else -1
        # n_keep_lo/_hi do not live in the per-ik h5 -- they are encoded
        # in the BackPropagator's psi-checkpoint manifest, which the
        # cc_dipole_driver already validated.  We pull them from the
        # cc_dipole_driver output if present.  Stored as top-level
        # ATTRIBUTES by cc_dipole_driver (not datasets).
        with h5py.File(cc_h5, "r") as fcc:
            for k in ("n_keep_lo", "n_keep_hi", "Nr", "dr",
                      "N_psi", "l_cont"):
                if k in fcc.attrs:
                    v = fcc.attrs[k]
                    f.attrs[f"cc_h5_{k}"] = v

        # ---- Channel (l, m) tables ----
        if N_psi is not None and N_psi > 0:
            l_mu, m_mu = _idx_to_lm(N_psi)
            ch = f.create_group("/channels")
            ch.create_dataset("l_mu", data=l_mu)
            ch.create_dataset("m_mu", data=m_mu)

        # ---- Per (κ, ν) pair groups ----
        g_pairs = f.create_group("/pairs")
        for (ik_k, ik_n), pair in sorted(cc.items()):
            gname = f"pair_k{ik_k:04d}_n{ik_n:04d}"
            gp = g_pairs.create_group(gname)
            gp.attrs["ik_kappa"] = ik_k
            gp.attrs["ik_nu"]    = ik_n
            gp.attrs["E_kappa"]  = float(pair["E_kappa"])
            gp.attrs["E_nu"]     = float(pair["E_nu"])
            for pol in _POLS:
                if pol in pair:
                    gp.create_dataset(f"cc_raw_len_{pol}",
                                       data=pair[pol])

        print(f"[phase A]   {n_ik} per-ik groups, "
              f"{len(cc)} (κ,ν) pairs, N_psi={N_psi}, n_occ={n_occ}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cc-h5", required=True, type=Path,
                    help="cc_dipole_driver output HDF5")
    ap.add_argument("--scan-dir", required=True, type=Path,
                    help="scattering dipole_<hash>_<scan_id> dir with "
                         "manifest.h5 + ik*.h5")
    ap.add_argument("--out", required=True, type=Path,
                    help="output consolidated HDF5 path")
    args = ap.parse_args()
    if not args.cc_h5.exists():
        sys.exit(f"--cc-h5 not found: {args.cc_h5}")
    if not args.scan_dir.exists():
        sys.exit(f"--scan-dir not found: {args.scan_dir}")
    assemble(args.cc_h5, args.scan_dir, args.out)


if __name__ == "__main__":
    main()
