#!/usr/bin/env python3
"""
inspect_somo.py -- diagnose spatial symmetry of an anion's SOMO.

For each molden file on the command line, this tool
  1. parses the Gaussian basis + MO coefficients (reusing the evaluator in
     gen_reference.py),
  2. identifies the highest-energy alpha MO with occupation > 0 (the SOMO),
  3. evaluates |psi_SOMO(r)|^2 along the three Cartesian axes through the
     origin (positive AND negative directions),
  4. integrates |psi_SOMO|^2 inside three narrow cones along the x, y, z
     axes and reports the ratios.

Output:
  - stdout: per-file table of axial integrated weights and their ratios,
  - a PNG with one overlaid panel per molden file showing the three line
    profiles |psi_SOMO|^2 along x, y, z.

Usage:
  python tools/inspect_somo.py <molden1> [<molden2> ...] [--out fig.png]
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the Python molden evaluator in gen_reference.py.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_reference import (               # noqa: E402
    _parse_molden_shells,
    _build_molden_aos,
    _mo_coeffs_from_molden,
)


def pick_alpha_somo(mos, nbf):
    """Return (index, mo) of highest-energy alpha MO with occ > 0."""
    best = None
    for i, mo in enumerate(mos):
        if mo["spin"] != "alpha":
            continue
        if mo["occ"] <= 0.0:
            continue
        if best is None or mo["ene"] > best[1]["ene"]:
            best = (i, mo)
    if best is None:
        raise RuntimeError("no occupied alpha MO found")
    return best


def evaluate_psi_on_line(aos, C, center, axis, r_vals):
    """Evaluate psi(r) along a line: position = center + r * axis_unit_vec."""
    axis_vec = np.zeros(3); axis_vec[axis] = 1.0
    R = center + np.outer(r_vals, axis_vec)
    phi = np.empty((R.shape[0], len(aos)))
    for m, ao in enumerate(aos):
        phi[:, m] = ao.eval(R)
    return phi @ C


def cone_weight(aos, C, center, axis, r_max, n_r=40, n_theta=16, n_phi=24, half_angle_rad=0.3):
    """Integrate |psi|^2 inside a narrow cone about `axis` (0=x,1=y,2=z),
    from r=0 to r=r_max. Fully vectorized: builds all cone points at once
    and evaluates every AO on the full grid in a single numpy pass.
    """
    r = np.linspace(1e-3, r_max, n_r)
    dr = r[1] - r[0]
    theta = np.linspace(0.0, half_angle_rad, n_theta)
    dtheta = theta[1] - theta[0] if n_theta > 1 else half_angle_rad
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = phi[1] - phi[0]

    R3, T3, P3 = np.meshgrid(r, theta, phi, indexing="ij")
    e = np.zeros(3); e[axis] = 1.0
    other = [i for i in range(3) if i != axis]
    t1 = np.zeros(3); t1[other[0]] = 1.0
    t2 = np.zeros(3); t2[other[1]] = 1.0
    cos_t = np.cos(T3); sin_t = np.sin(T3)
    cos_p = np.cos(P3); sin_p = np.sin(P3)

    dx = cos_t * e[0] + sin_t * (cos_p * t1[0] + sin_p * t2[0])
    dy = cos_t * e[1] + sin_t * (cos_p * t1[1] + sin_p * t2[1])
    dz = cos_t * e[2] + sin_t * (cos_p * t1[2] + sin_p * t2[2])

    pts = np.stack([
        (center[0] + R3 * dx).ravel(),
        (center[1] + R3 * dy).ravel(),
        (center[2] + R3 * dz).ravel(),
    ], axis=1)   # (N, 3),  N = n_r * n_theta * n_phi

    # Vectorized evaluation: outer loop is over AOs (small), inner is over N.
    phi_mat = np.empty((pts.shape[0], len(aos)))
    for m, ao in enumerate(aos):
        phi_mat[:, m] = ao.eval(pts)
    psi = phi_mat @ C

    weight = (R3 * R3 * sin_t).ravel() * (dr * dtheta * dphi)
    return float(np.sum(psi * psi * weight))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moldens", nargs="+", help="one or more molden files")
    ap.add_argument("--out", default="somo_inspect.png", help="output PNG path")
    ap.add_argument("--r-max", type=float, default=6.0, help="r_max for line scans (Bohr)")
    ap.add_argument("--n-pts", type=int, default=401, help="points per axis")
    ap.add_argument("--center", default=None,
                    help="comma-separated x,y,z origin in Bohr (default: geometric mean)")
    args = ap.parse_args()

    fig, axes = plt.subplots(len(args.moldens), 1,
                             figsize=(9, 3.5 * len(args.moldens)),
                             sharex=True, squeeze=False)

    for i_file, path in enumerate(args.moldens):
        print(f"\n=== {path} ===")
        with open(path) as f:
            lines = f.readlines()
        atoms_parsed, shells, _sph = _parse_molden_shells(lines)
        # nbf = sum over shells of (2l+1) for pure or (l+1)(l+2)/2 for cart
        nbf = 0
        for sh in shells:
            L = sh["l"]
            if sh.get("pure", True) or L <= 1:
                nbf += (1 if L == 0 else 2 * L + 1)
            else:
                nbf += (L + 1) * (L + 2) // 2
        aos = _build_molden_aos(shells)
        mos = _mo_coeffs_from_molden(lines, nbf)
        idx, somo = pick_alpha_somo(mos, nbf)

        # Pick center: geometric mean of atoms unless overridden.
        if args.center:
            center = np.array([float(s) for s in args.center.split(",")])
        else:
            center = np.mean(np.array([a[1] for a in atoms_parsed]), axis=0)

        print(f"  n_atom={len(atoms_parsed)}  n_shells={len(shells)}  nbf={nbf}"
              f"  n_mos={len(mos)}  center={center}")
        print(f"  SOMO: alpha index {idx}  energy {somo['ene']:.6f} Ha  occ {somo['occ']}")

        C = somo["C"]
        r = np.linspace(-args.r_max, args.r_max, args.n_pts)
        psi_x = evaluate_psi_on_line(aos, C, center, 0, r)
        psi_y = evaluate_psi_on_line(aos, C, center, 1, r)
        psi_z = evaluate_psi_on_line(aos, C, center, 2, r)

        # Integrated cone weights along each axis (for quantitative compare).
        # Narrow cone (half-angle = 0.25 rad ~= 14 deg), out to r_max.
        w_x = cone_weight(aos, C, center, 0, args.r_max)
        w_y = cone_weight(aos, C, center, 1, args.r_max)
        w_z = cone_weight(aos, C, center, 2, args.r_max)
        print(f"  cone-integrated |psi|^2 (half-angle 14°):")
        print(f"     along x:  {w_x:.6e}")
        print(f"     along y:  {w_y:.6e}")
        print(f"     along z:  {w_z:.6e}")
        ratio_max = max(w_x, w_y, w_z) / max(min(w_x, w_y, w_z), 1e-30)
        print(f"  max/min ratio across axes: {ratio_max:.3f}    "
              f"(O_h-symmetric SOMO should be 1.000)")

        ax = axes[i_file, 0]
        ax.plot(r, psi_x ** 2, label=f"along x  (cone weight = {w_x:.3e})", color="tab:red")
        ax.plot(r, psi_y ** 2, label=f"along y  (cone weight = {w_y:.3e})", color="tab:green")
        ax.plot(r, psi_z ** 2, label=f"along z  (cone weight = {w_z:.3e})", color="tab:blue")
        ax.set_ylabel(r"$|\psi_{\mathrm{SOMO}}(\vec r)|^2$")
        ax.set_xlabel(r"$r$ along axis  (Bohr)")
        title = os.path.basename(path)
        ax.set_title(f"{title}    (SOMO idx={idx}, E={somo['ene']:.4f} Ha, "
                     f"asym = {ratio_max:.2f})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\n--> wrote {args.out}")


if __name__ == "__main__":
    main()
