#!/usr/bin/env python3
r"""
gather_dipoles_merge.py
=======================
Merge SEVERAL scattering scan dirs — possibly with different continuum
l_max and/or different ik windows — into ONE gathered per-channel .dat
set that cross_section_delay.py and gauge_panels.py can consume directly.

Motivating case
---------------
A production run is often split to save compute: a high-l_max scan over
the high-energy points and a cheaper low-l_max scan over the low-energy
points, e.g.

    scan A:  ik = 40..400,  l_count = 100   (n_ch = 101² = 10201)
    scan B:  ik = 15..39,   l_count =  80   (n_ch =  81² =  6561)

gather_dipoles.py handles ONE scan dir and requires a uniform channel
count, so it cannot ingest both at once.  This tool reads each scan's
HDF5 directly (reusing gather_dipoles' readers), then merges by k.

How the merge stays accurate
----------------------------
  * Channel union + zero-pad.  The combined set has the LARGEST l_max
    (here 100).  For a scan with fewer channels (B), the missing high-l
    channels are ZERO-PADDED over that scan's energies.  This is
    physically correct — at low k the centrifugal barrier makes l>80
    amplitudes negligible — and it preserves scan A's true high-l
    content at high k (truncating everything to l≤80 would throw that
    away).
  * Ip consistency.  Refuses to merge scans whose Ip = −E_HOMO differ
    (their ω columns would be incompatible → silently wrong σ).
  * k identity / overlap.  Dedupes points with identical k, keeping the
    higher-l one, and reports any overlap.
  * Seam diagnostic — THE key guard.  At each energy where the channel
    count jumps (the A/B boundary), it reports the fraction of |d|² that
    lives in the extra high-l channels.  If that is tiny (≪1%), the
    zero-padding is safe and the τ_W = ∂_E arg D derivative across the
    seam is trustworthy; if not, l_max was too low at the seam and you
    need overlap / higher l there.

The merge core (`assemble`) is pure (no I/O) and unit-tested via
--selftest, so the accuracy-critical logic is verifiable without HDF5.

Usage
-----
    # merge two scans (order irrelevant; sorted by k automatically)
    python gather_dipoles_merge.py SCAN_B SCAN_A --output-dir gathered_merged

    # optional ik window applied per scan (e.g. only ik 15..100)
    python gather_dipoles_merge.py SCAN_B SCAN_A --output-dir OUT \
        --ik-min 15 --ik-max 100

    # validate the merge core, then exit
    python gather_dipoles_merge.py --selftest

Then feed OUT to the unchanged postprocessing:
    python cross_section_delay.py OUT --xaxis k
    python gauge_panels.py        OUT --xaxis k --eta-mode fixed
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Reuse the tested HDF5 readers / index helpers from gather_dipoles.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gather_dipoles import (  # noqa: E402
    enumerate_ik_files,
    idx_to_lm,
    read_manifest,
    read_one_energy,
    resolve_scan_dir,
)

_GAUGE_POL = [(g, p) for g in ("length", "velocity") for p in ("x", "y", "z")]


# ---------------------------------------------------------------------------
# Pure merge core (no I/O) — unit-tested by _selftest().
# ---------------------------------------------------------------------------
def assemble(records: List[dict]) -> dict:
    """Dedupe by k (keep the larger channel set), sort by k, take the
    channel union, zero-pad the short records.

    Each record: dict with keys ik, k, E, omega, n_ch, and
    (gauge, pol) -> complex vector of length n_ch.

    Returns a dict with the assembled k/E/omega arrays, the padded
    buffers buf[(gauge,pol)] of shape (n_e, union), the per-row channel
    counts, the seam row indices, and any overlapping ik.
    """
    by_k: Dict[int, dict] = {}
    overlaps: List[int] = []
    for r in records:
        key = int(round(r["k"] / 1e-12))            # k identity to ~1e-12 a.u.
        if key in by_k:
            overlaps.append(r["ik"])
            if r["n_ch"] > by_k[key]["n_ch"]:       # prefer the higher-l point
                by_k[key] = r
        else:
            by_k[key] = r
    recs = sorted(by_k.values(), key=lambda r: r["k"])

    n_e = len(recs)
    union = max(r["n_ch"] for r in recs)
    k_arr = np.array([r["k"] for r in recs])
    E_arr = np.array([r["E"] for r in recs])
    omega_arr = np.array([r["omega"] for r in recs])
    nch_row = np.array([r["n_ch"] for r in recs])

    buf: Dict[Tuple[str, str], np.ndarray] = {
        key: np.zeros((n_e, union), dtype=complex) for key in _GAUGE_POL
    }
    for row, r in enumerate(recs):
        for key in _GAUGE_POL:
            v = r[key]
            buf[key][row, :v.shape[0]] = v          # remaining columns stay 0

    seams = [i for i in range(1, n_e) if nch_row[i] != nch_row[i - 1]]
    return {
        "recs": recs, "k": k_arr, "E": E_arr, "omega": omega_arr,
        "buf": buf, "union": union, "nch_row": nch_row,
        "seams": seams, "overlaps": sorted(set(overlaps)),
    }


# ---------------------------------------------------------------------------
# I/O glue.
# ---------------------------------------------------------------------------
def collect_records(scan_dirs: List[Path], use_raw: bool,
                    ik_min, ik_max) -> Tuple[List[dict], List[dict]]:
    """Read manifests + every selected energy file across all scans."""
    metas = [read_manifest(sd) for sd in scan_dirs]

    ips = [-m["E_HOMO"] for m in metas]
    if max(ips) - min(ips) > 1e-6:
        raise RuntimeError(
            "scan dirs disagree on Ip = -E_HOMO (different preprocessing); "
            f"merging would be physically invalid. Ip values: {ips}")

    print("  scans to merge:")
    for sd, m in zip(scan_dirs, metas):
        print(f"    {sd.name}: l_max={m['l_max_continuum']} "
              f"(n_ch={(m['l_max_continuum']+1)**2}), dk={m['dk']:.6g}, "
              f"ik[{m['ik_min']},{m['ik_max']}]")

    records: List[dict] = []
    for sd in scan_dirs:
        for ik, path in enumerate_ik_files(sd):
            if ik_min is not None and ik < ik_min:
                continue
            if ik_max is not None and ik > ik_max:
                continue
            rec = read_one_energy(path, use_raw)
            rec["n_ch"] = len(rec[("length", "x")])
            records.append(rec)
    if not records:
        raise RuntimeError("no energy points selected (check --ik-min/--ik-max).")
    return records, metas


def report_seams(A: dict) -> None:
    """Print the strength fraction in the zero-padded high-l channels at
    each channel-count seam (length gauge, summed over x/y/z)."""
    if not A["seams"]:
        print("  (no channel-count seam: all scans share the same l_max)")
        return
    print("  seam diagnostics — fraction of |d|^2 in the EXTRA high-l "
          "channels at the seam (want ≪ 1e-2):")
    for i in A["seams"]:
        lo = int(min(A["nch_row"][i], A["nch_row"][i - 1]))
        tot = extra = 0.0
        for p in ("x", "y", "z"):
            col = A["buf"][("length", p)][i]
            tot += float(np.sum(np.abs(col) ** 2))
            extra += float(np.sum(np.abs(col[lo:]) ** 2))
        frac = extra / tot if tot > 0 else 0.0
        lmax_lo = int(round(np.sqrt(lo))) - 1
        flag = "OK" if frac < 1e-2 else "** CHECK l-CONVERGENCE AT SEAM **"
        print(f"    k={A['k'][i]:.4f} au (ik={A['recs'][i]['ik']}): "
              f"l>{lmax_lo} carries {frac:.2e} of the strength   [{flag}]")


def write_merged(A: dict, output_dir: Path, scan_dirs, ip,
                 use_raw, ik_min, ik_max) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    union = A["union"]
    union_lmax = int(round(np.sqrt(union))) - 1
    names = ", ".join(sd.name for sd in scan_dirs)
    header_base = (
        f"MERGED from {len(scan_dirs)} scan(s): {names}\n"
        f"ik window: [{ik_min if ik_min is not None else 'min'}, "
        f"{ik_max if ik_max is not None else 'max'}]   "
        f"union l_max={union_lmax} (n_ch={union})   Ip={ip:.6f} au   "
        f"dipole={'RAW (pre-ortho)' if use_raw else 'ORTHO'}\n"
        "NOTE: channels above a scan's l_max are ZERO-PADDED for that "
        "scan's energies (see seam diagnostics in the merge log).\n"
        "Columns: k_elec[au]  E_kin[au]  omega[au]  Re(D)  Im(D)  |D|^2  arg(D)"
    )
    prefix_of = {"length": "dipole_len_homo", "velocity": "dipole_vel_homo"}
    k_arr, E_arr, omega_arr = A["k"], A["E"], A["omega"]
    n_written = 0
    for (gauge, pol), D in A["buf"].items():
        prefix = prefix_of[gauge]
        for mu in range(union):
            l, m = idx_to_lm(mu)
            out = np.column_stack([k_arr, E_arr, omega_arr,
                                   D[:, mu].real, D[:, mu].imag,
                                   np.abs(D[:, mu]) ** 2, np.angle(D[:, mu])])
            path = output_dir / f"{prefix}_{pol}_{mu}.dat"
            hdr = header_base + f"\nmu = {mu}  (l={l}, m={m:+d})"
            np.savetxt(path, out, header=hdr, fmt="%.14e")
            n_written += 1
    print(f"  wrote {n_written} .dat files to {output_dir}")


def merge(scan_dirs: List[Path], output_dir: Path, use_raw: bool,
          ik_min, ik_max) -> None:
    records, metas = collect_records(scan_dirs, use_raw, ik_min, ik_max)
    ip = float(np.mean([-m["E_HOMO"] for m in metas]))
    A = assemble(records)
    union_lmax = int(round(np.sqrt(A["union"]))) - 1
    print(f"  merged: {len(A['recs'])} energy points, union l_max={union_lmax} "
          f"(n_ch={A['union']}), k=[{A['k'][0]:.4f}, {A['k'][-1]:.4f}] au")
    if A["overlaps"]:
        print(f"  WARNING: overlapping ik across scans {A['overlaps']} — "
              f"kept the higher-l point at each.")
    report_seams(A)
    write_merged(A, output_dir, scan_dirs, ip, use_raw, ik_min, ik_max)


# ---------------------------------------------------------------------------
# Unit test of the merge core.
# ---------------------------------------------------------------------------
def _selftest() -> int:
    print("=== gather_dipoles_merge.assemble self-test ===")

    def mk(ik, k, n_ch, fill):
        r = {"ik": ik, "k": k, "E": 0.5 * k * k, "omega": 0.5 * k * k + 0.1,
             "n_ch": n_ch}
        for key in _GAUGE_POL:
            r[key] = np.full(n_ch, fill + 0j)
        return r

    # low-l (n_ch=4, l=1) at small k; high-l (n_ch=9, l=2) at large k;
    # a duplicate k=0.40 with FEWER channels (must be dropped); out of order.
    records = [
        mk(40, 0.40, 9, 5.0),
        mk(15, 0.15, 4, 1.0),
        mk(41, 0.41, 9, 6.0),
        mk(16, 0.16, 4, 2.0),
        mk(40, 0.40, 4, 99.0),    # duplicate k, fewer channels -> dropped
    ]
    A = assemble(records)
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok ' if cond else 'FAIL'}] {msg}")

    check(A["union"] == 9, f"union channels == 9 (got {A['union']})")
    check(list(A["k"]) == sorted(A["k"]), "energies sorted by k")
    check(len(A["recs"]) == 4, f"duplicate k dropped -> 4 rows (got {len(A['recs'])})")
    row040 = int(np.argmin(np.abs(A["k"] - 0.40)))
    check(abs(A["buf"][("length", "x")][row040, 0] - 5.0) < 1e-12,
          "overlap resolved in favour of the higher-l point")
    check(40 in A["overlaps"], "overlap at ik=40 reported")
    row015 = int(np.argmin(np.abs(A["k"] - 0.15)))
    check(np.allclose(A["buf"][("length", "x")][row015, 4:], 0.0),
          "high-l channels zero-padded at low k")
    check(abs(A["buf"][("length", "x")][row015, 0] - 1.0) < 1e-12,
          "low-l channels preserved at low k")
    check(A["seams"] == [2], f"one seam at row 2 (got {A['seams']})")

    # padding must not leak into the seam strength fraction when the extra
    # channels are genuinely populated on the high-l side.
    i = A["seams"][0]
    lo = int(min(A["nch_row"][i], A["nch_row"][i - 1]))
    col = A["buf"][("length", "x")][i]
    frac = float(np.sum(np.abs(col[lo:]) ** 2) / np.sum(np.abs(col) ** 2))
    check(abs(frac - (9 - 4) / 9) < 1e-12,
          f"seam fraction = (9-4)/9 for uniform fill (got {frac:.4f})")

    print("=== self-test", "PASSED ===" if ok else "FAILED ===")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge multiple scan dirs (different l_max / ik windows) "
                    "into one gathered .dat set.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", nargs="*",
                    help="Two or more dipole_<hash>_<scan>/ dirs to merge "
                         "(absolute/relative; $WORK tried for relative paths).")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write the merged .dat files (REQUIRED).")
    ap.add_argument("--raw", action="store_true",
                    help="Use D_raw (pre-orthogonalization) instead of D_ortho.")
    ap.add_argument("--ik-min", type=int, default=None,
                    help="Keep only energies with ik >= this (per-scan ik).")
    ap.add_argument("--ik-max", type=int, default=None,
                    help="Keep only energies with ik <= this (per-scan ik).")
    ap.add_argument("--selftest", action="store_true",
                    help="Run the merge-core unit test and exit.")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if len(args.scan_dir) < 1:
        ap.error("give at least one scan_dir (two+ to merge), or --selftest.")
    if args.output_dir is None:
        ap.error("--output-dir is required.")

    scan_dirs = [resolve_scan_dir(s) for s in args.scan_dir]
    output_dir = Path(args.output_dir).resolve()

    print("=" * 72)
    print("gather_dipoles_merge.py")
    print("=" * 72)
    print(f"  output_dir: {output_dir}")
    print(f"  source    : {'D_raw (PRE-ortho)' if args.raw else 'D_ortho (orthogonalized)'}")
    merge(scan_dirs, output_dir, args.raw, args.ik_min, args.ik_max)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
