#!/usr/bin/env python3
"""
diagnose_beta_zero.py
---------------------
Diagnostic for β(E) ≡ 0 across the whole scan.  Drops 4 possible causes:

  C1.  All D channels are zero (or below 1e-30) -- σ_tot fallback path.
  C2.  D_x ≡ D_y ≡ D_z within numerical noise -- gives β = N_1/σ − 1 = 0
       exactly (analytic).  Common when the dipole writer mixed up the
       per-pol output.
  C3.  D_x, D_y, D_z are pairwise scalar multiples of one ANOTHER (i.e.,
       same angular shape, scaled).  Also makes β = 0.
  C4.  All three are different but happen to satisfy N_1 = σ_tot
       (genuine symmetry zero -- this is what real O_h cubic systems
       can give for certain HOMO symmetries).

Usage:
    python3 diagnose_beta_zero.py <gathered_dipole_dir>
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np


def load_pol(input_dir: Path, pol: str):
    """Return k array + complex D[n_e, n_ch] for one polarization."""
    files = sorted(glob.glob(str(input_dir / f"dipole_len_homo_{pol}_*.dat")),
                   key=lambda p: int(re.search(r"_(\d+)\.dat$", p).group(1)))
    if not files:
        raise FileNotFoundError(f"no dipole_len_homo_{pol}_*.dat in {input_dir}")
    first = np.loadtxt(files[0])
    if first.ndim == 1:
        first = first.reshape(1, -1)
    n_e = len(first)
    k = first[:, 0]
    D = np.zeros((n_e, len(files)), dtype=complex)
    mus = []
    for j, f in enumerate(files):
        mu = int(re.search(r"_(\d+)\.dat$", f).group(1))
        mus.append(mu)
        a = np.loadtxt(f)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        D[:, j] = a[:, 3] + 1j * a[:, 4]
    return k, np.array(mus), D


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="gathered_dipole_* dir")
    args = ap.parse_args()

    d = Path(args.input_dir)
    if not d.is_dir():
        print(f"ERROR: {d} is not a directory", file=sys.stderr); sys.exit(1)

    print(f"=== reading dipoles from {d} ===")
    k, mu_x, D_x = load_pol(d, "x")
    _, mu_y, D_y = load_pol(d, "y")
    _, mu_z, D_z = load_pol(d, "z")
    if not (np.array_equal(mu_x, mu_y) and np.array_equal(mu_y, mu_z)):
        print(f"WARN: channel sets differ between pols")
    n_e, n_ch = D_x.shape
    print(f"  energies: {n_e}   channels: {n_ch}\n")

    # C1: are the dipoles all zero?
    max_x = np.max(np.abs(D_x)); max_y = np.max(np.abs(D_y)); max_z = np.max(np.abs(D_z))
    print(f"C1  global |D|max:  x={max_x:.3e}   y={max_y:.3e}   z={max_z:.3e}")
    if max(max_x, max_y, max_z) < 1e-15:
        print("    => ALL dipoles below 1e-15 -- the writer wrote zeros."); print()

    # C2: is D_x ≡ D_y ≡ D_z?
    diff_xy = np.max(np.abs(D_x - D_y))
    diff_yz = np.max(np.abs(D_y - D_z))
    diff_xz = np.max(np.abs(D_x - D_z))
    print(f"C2  identical-pol test:   max|D_x−D_y|={diff_xy:.3e}   "
          f"max|D_y−D_z|={diff_yz:.3e}   max|D_x−D_z|={diff_xz:.3e}")
    if max(diff_xy, diff_yz, diff_xz) < 1e-10 * max(max_x, max_y, max_z):
        print("    => D_x ≡ D_y ≡ D_z (or numerically equal).")
        print("       This makes β = 0 *exactly* -- by the formula N_1=σ_tot.")
        print("       Likely the dipole writer is OVERWRITING all three pols")
        print("       with the same data (bug in scattering, NOT a physics result).")
    print()

    # C3: is D_y a scalar multiple of D_x?  (etc.)
    def best_scalar(A, B):
        if np.max(np.abs(A)) == 0 or np.max(np.abs(B)) == 0: return None, None
        c = np.vdot(A.flat, B.flat) / np.vdot(A.flat, A.flat)
        res = np.max(np.abs(B - c*A)) / max(np.max(np.abs(B)), 1e-300)
        return c, res
    cyx, ryx = best_scalar(D_x, D_y); czx, rzx = best_scalar(D_x, D_z)
    if cyx is not None:
        def fmt_c(z): return f"({z.real:+.4e} {z.imag:+.4e}j)"
        print(f"C3  best-fit  D_y ≈ {fmt_c(cyx)} · D_x   residual = {ryx:.3e}")
        print(f"    best-fit  D_z ≈ {fmt_c(czx)} · D_x   residual = {rzx:.3e}")
        if ryx < 1e-6 and rzx < 1e-6:
            print("    => D_y and D_z are SCALAR MULTIPLES of D_x.")
            print("       Same angular shape, different magnitudes/phases.")
            print("       β = N_1/σ_tot − 1 is still 0 in this case.")
    print()

    # C4: per-energy ⟨P_2⟩ via channel/Cartesian integral (mirror the
    # cross_section_delay.py formula exactly so result is the canonical β).
    print("C4  per-energy diagnostic (energies 0, n/4, n/2, 3n/4, n-1)")
    print(f"    {'ik_idx':>7s} {'k':>8s} {'σ_x':>10s} {'σ_y':>10s} {'σ_z':>10s}"
          f" {'N_1':>10s} {'σ_tot':>10s} {'β':>8s}")
    # NOTE: do NOT use scipy.special.sph_harm here -- on python ≤ 3.9 /
    # scipy < 1.15 it overflows the factorial normaliser at high (l, |m|),
    # returning NaN at l ≳ 70.  That silently makes σ_tot = NaN, which in
    # turn makes `if sigma_tot > 1e-30` False, and compute_beta returns
    # 0.0 across the whole scan.  This was the original bug here.
    #
    # Instead, use a normalised-Legendre 3-term recurrence -- stable to
    # l > 1000, no factorials computed explicitly.
    def _Ptilde(l, m, theta):
        am = abs(m)
        if am > l:
            return np.zeros_like(theta)
        x = np.cos(theta); s = np.sin(theta)
        P_mm = np.full_like(x, 1.0 / (2.0 * np.sqrt(np.pi)))
        for k in range(1, am + 1):
            P_mm = -np.sqrt((2.0 * k + 1.0) / (2.0 * k)) * s * P_mm
        if l == am:
            return P_mm
        P_p2 = P_mm
        P_p1 = np.sqrt(2.0 * am + 3.0) * x * P_mm
        if l == am + 1:
            return P_p1
        for ll in range(am + 2, l + 1):
            a = np.sqrt((4.0 * ll * ll - 1.0) / (ll * ll - am * am))
            b = np.sqrt((2.0 * ll + 1.0) / (2.0 * ll - 3.0)
                        * ((ll - 1.0) * (ll - 1.0) - am * am)
                        / (ll * ll - am * am))
            P_cur = a * x * P_p1 - b * P_p2
            P_p2 = P_p1; P_p1 = P_cur
        return P_p1

    # 40×80 angular grid like cross_section_delay.compute_beta
    n_theta, n_phi = 40, 80
    x_, w_ = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(x_); phi = np.arange(n_phi)*2*np.pi/n_phi
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    dphi = 2*np.pi/n_phi
    weights = w_[:,None]*dphi
    nx = np.sin(TH)*np.cos(PH); ny = np.sin(TH)*np.sin(PH); nz = np.cos(TH)

    def real_Ylm(l, m, theta, phi):
        am = abs(m)
        P_lm = _Ptilde(l, am, theta)
        if m == 0:
            return P_lm
        elif m > 0:
            return np.sqrt(2.0) * ((-1) ** m) * P_lm * np.cos(m * phi)
        else:
            return np.sqrt(2.0) * ((-1) ** am) * P_lm * np.sin(am * phi)

    def lm_of(mu):
        l = int(np.floor(np.sqrt(mu)))
        while (l+1)*(l+1) <= mu: l += 1
        return l, mu - l*l - l

    Y_table = np.empty((n_ch, n_theta, n_phi))
    for idx, mu in enumerate(mu_x):
        l, m = lm_of(mu)
        Y_table[idx] = real_Ylm(l, m, TH, PH)

    sample_idx = np.unique([0, n_e//4, n_e//2, 3*n_e//4, n_e-1])
    for ie in sample_idx:
        Dm_x = np.tensordot(D_x[ie], Y_table, axes=([0],[0]))
        Dm_y = np.tensordot(D_y[ie], Y_table, axes=([0],[0]))
        Dm_z = np.tensordot(D_z[ie], Y_table, axes=([0],[0]))
        sig_x = np.sum(np.abs(Dm_x)**2 * weights)
        sig_y = np.sum(np.abs(Dm_y)**2 * weights)
        sig_z = np.sum(np.abs(Dm_z)**2 * weights)
        sigma_tot = (sig_x + sig_y + sig_z)/3
        Ddk = Dm_x*nx + Dm_y*ny + Dm_z*nz
        N1 = np.sum(np.abs(Ddk)**2 * weights)
        beta = (N1/sigma_tot - 1) if sigma_tot>1e-30 else 0.0
        print(f"    {ie:>7d} {k[ie]:>8.4f} {sig_x:>10.3e} {sig_y:>10.3e} "
              f"{sig_z:>10.3e} {N1:>10.3e} {sigma_tot:>10.3e} {beta:>8.4f}")
    print()

    # If sig_x ≈ sig_y ≈ sig_z but β ≠ 0 here, the formula works.
    # If β = 0 to all digits AND sig_x = sig_y = sig_z AND N_1 = σ_tot, that's
    # the algebraic-zero pattern from D_x = D_y = D_z or scalar-multiple case.

    print("=== conclusion guidance ===")
    if max(max_x, max_y, max_z) < 1e-15:
        print("  All dipoles below 1e-15: SCATTERING DIDN'T WRITE THEM.")
        print("  Check that --dipole-only finished and gather_dipoles.py ran")
        print("  on the post-dipole-only output dir, not an empty one.")
    elif max(diff_xy, diff_yz, diff_xz) < 1e-10 * max(max_x, max_y, max_z):
        print("  D_x ≡ D_y ≡ D_z: BUG IN THE DIPOLE WRITER OR GATHER STEP.")
        print("  Check scattering output: per-energy ikNNNN.h5 should have")
        print("  distinct values of d_raw_len_x, d_raw_len_y, d_raw_len_z.")
        print("  If they ARE distinct in ik*.h5, the bug is in gather_dipoles.py")
        print("  (probably reading the same dataset under different filenames).")
    else:
        print("  D_x, D_y, D_z are distinct in the dat files.  β = 0 then")
        print("  arises only if N_1 = σ_tot by a specific channel-symmetry")
        print("  coincidence, which IS possible for O_h with certain HOMO")
        print("  representations -- but unusual.  Run the per-energy table")
        print("  above to see whether N_1 ≈ σ_tot at all energies (genuine")
        print("  zero) or only at some (numerical β_zero average).")


if __name__ == "__main__":
    main()
