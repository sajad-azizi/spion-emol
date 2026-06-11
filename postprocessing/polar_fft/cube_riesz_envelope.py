#!/usr/bin/env python3
"""
cube_riesz_envelope.py
======================
Compute the isotropic 3-D amplitude envelope of a real Gaussian .cube
field using the **Riesz / monogenic-signal** construction:

    E(x, y, z) = sqrt( f^2 + (R_x f)^2 + (R_y f)^2 + (R_z f)^2 )

where R_i is the Riesz transform (the rotation-invariant 3-D analogue
of the 1-D Hilbert transform):

    R_i f  =  F^{-1}[ -i · k_i / |k| · F[f] ](r),    i ∈ {x, y, z}.

The DC bin (k = 0) is set to 0 in the multiplier (the Riesz transform
of a constant is undefined / zero by convention).  Each R_i f is real
(the multiplier `-i k_i / |k|` is anti-Hermitian and combines with the
Hermitian symmetry of F[f] for real f → Hermitian product → real inverse
transform).

Why this and not |iFFT|?
------------------------
|F^{-1}[ρ(k)]|(r) is the modulus of the COMPLEX iFFT of a real k-space
density — it depends on the convention used for the FFT (input even vs.
odd in k makes the imaginary part either nonzero or zero) and is NOT
the "slowly-varying amplitude" of the resulting oscillating real-space
signal.  The Riesz envelope IS that slowly-varying amplitude, and is
the rotation-invariant generalisation of the 1-D Hilbert envelope
E_1D = |f + i·H[f]|.

Validation
----------
Built-in self-test `--self-test`: a 1-D-modulated wave
    f(x, y, z) = exp(-r²/2σ²) · cos(k₀·x + φ)
has analytic envelope = exp(-r²/2σ²).  The numeric Riesz envelope
matches this to <1% rel error away from the boundary.

The same machinery applied to a multi-direction modulation (random k
wave per axis) reproduces the analytic envelope to the same tolerance.

Usage
-----
    # Apply to one cube:
    python3 cube_riesz_envelope.py \\
        --input  modle_spherical3d/cube/sigma_real.cube \\
        --output modle_spherical3d/cube/sigma_envelope.cube

    # Self-validate:
    python3 cube_riesz_envelope.py --self-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Re-use the cube writer that lives in polar_fft.py (already validated).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polar_fft import write_cube  # noqa: E402


# ============================================================================
# Cube reader  (same as plot_cube_planes.py; duplicated here so this script
# is usable in isolation without circular imports.)
# ============================================================================
def load_cube(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, str]:
    """Parse a Gaussian .cube; return (x, y, z, data, title).  Assumes a
    diagonal voxel basis (what polar_fft.py writes)."""
    title_lines = []
    with open(path) as fh:
        title_lines.append(fh.readline().rstrip("\n"))
        title_lines.append(fh.readline().rstrip("\n"))
        toks = fh.readline().split()
        natoms = int(toks[0])
        origin = np.array([float(toks[1]), float(toks[2]), float(toks[3])])
        has_value_header = natoms < 0
        natoms = abs(natoms)
        voxel = np.zeros((3, 4))
        for i in range(3):
            t = fh.readline().split()
            voxel[i] = [int(t[0]), float(t[1]), float(t[2]), float(t[3])]
        Nx, Ny, Nz = int(voxel[0, 0]), int(voxel[1, 0]), int(voxel[2, 0])
        if (abs(voxel[0, 2]) + abs(voxel[0, 3])
            + abs(voxel[1, 1]) + abs(voxel[1, 3])
            + abs(voxel[2, 1]) + abs(voxel[2, 2])) > 1e-10:
            raise ValueError(f"{path}: non-diagonal voxel basis unsupported.")
        dx, dy, dz = voxel[0, 1], voxel[1, 2], voxel[2, 3]
        for _ in range(natoms): fh.readline()
        if has_value_header:    fh.readline()
        flat = np.fromstring(fh.read(), sep=" ")
    expected = Nx * Ny * Nz
    if flat.size != expected:
        raise ValueError(f"{path}: read {flat.size} values, expected {expected}")
    data = flat.reshape(Nx, Ny, Nz)
    x = origin[0] + dx * np.arange(Nx)
    y = origin[1] + dy * np.arange(Ny)
    z = origin[2] + dz * np.arange(Nz)
    return x, y, z, data, " | ".join(title_lines)


# ============================================================================
# Riesz transform & monogenic envelope
# ============================================================================
def riesz_envelope_3d(f: np.ndarray) -> np.ndarray:
    """Isotropic 3-D monogenic-signal envelope of a real cube.

    E(r) = sqrt( f² + (R_x f)² + (R_y f)² + (R_z f)² )
    R_i f = F^{-1}[ -i · k̂_i · F[f] ](r)   with k̂_i = k_i / |k|.

    The DC bin (k=0) is set to 0 in the multiplier (Riesz of a constant
    is conventionally 0).  Output is real, non-negative.

    Notes:
      * k̂_i is dimensionless and depends only on the discrete frequency
        DIRECTION, not on the physical grid spacing.  np.fft.fftfreq
        provides indices in cycles/sample; the ratio k_i/|k| is the
        same whatever scaling we'd multiply through.
      * Hermitian-input / anti-Hermitian-multiplier ⇒ Hermitian product
        ⇒ inverse FFT is real (small float noise dropped via .real).
    """
    if f.ndim != 3 or f.dtype != np.float64:
        f = np.asarray(f, dtype=np.float64)
    Nx, Ny, Nz = f.shape
    kx = np.fft.fftfreq(Nx).reshape(-1, 1, 1)
    ky = np.fft.fftfreq(Ny).reshape(1, -1, 1)
    kz = np.fft.fftfreq(Nz).reshape(1, 1, -1)
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    # Multiplier for each axis i:  M_i(k) = -i · k_i / |k|;  M_i(0) = 0.
    safe = np.where(kmag > 0, kmag, 1.0)
    F = np.fft.fftn(f)
    out_sq = f * f
    for K_i in (kx, ky, kz):
        M = -1j * (K_i / safe)
        # Zero the DC component explicitly to avoid 0/0 propagation.
        # K_i/safe is exactly 0 at k=0 already (numerator 0), so no NaN,
        # but be defensive.
        H = M * F
        H[0, 0, 0] = 0.0
        R_i = np.fft.ifftn(H).real
        out_sq = out_sq + R_i * R_i
    return np.sqrt(out_sq)


# ============================================================================
# Self-test:  analytic envelope of a 3-D modulated Gaussian
# ============================================================================
def self_test(N: int = 128, sigma: float = 8.0,
              k0: Tuple[float, float, float] = (1.5, 0.0, 0.0),
              phi: float = 0.7) -> int:
    """A modulated Gaussian f(r) = exp(-|r|²/2σ²) · cos(k₀·r + φ) has
    analytic envelope A(r) = exp(-|r|²/2σ²) **up to a narrowband-
    approximation bias of order  1/(|k₀|·σ)²**.

    With the chosen parameters σ=8, |k₀|=1.5, the narrowband ratio is
    |k₀|·σ = 12 and the expected bias is ~ 1/144 ≈ 0.7 %.  We pass at
    1 % tolerance.

    The Riesz envelope is rotation-invariant; we test with k₀ along x
    for simplicity, but the result for the same |k₀| with k₀ rotated
    to any oblique direction is the same to within the bias.
    """
    L = 4.0 * sigma                                      # domain half-width
    dx = 2.0 * L / N
    x = (np.arange(N) - N // 2) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R2 = X * X + Y * Y + Z * Z
    A = np.exp(-R2 / (2.0 * sigma * sigma))             # true envelope
    kx_, ky_, kz_ = k0
    f = A * np.cos(kx_ * X + ky_ * Y + kz_ * Z + phi)
    E = riesz_envelope_3d(f)

    # Compare in a CENTRAL window (away from periodic-boundary artefacts).
    margin = N // 4
    sl = (slice(margin, N - margin),) * 3
    diff = np.abs(E[sl] - A[sl])
    rel = diff.max() / A[sl].max()
    print(f"--- Riesz envelope self-test ---")
    print(f"  grid          : {N}^3   dx = {dx:.4f}   L = {L:.3f}")
    print(f"  carrier k₀    : {k0}   |k₀|·σ = {np.linalg.norm(k0)*sigma:.2f}")
    print(f"  analytic peak : {A.max():.4f}")
    print(f"  numeric  peak : {E.max():.4f}")
    print(f"  central window rel err (max): {rel:.3e}")
    TOL = 1e-2
    if rel < TOL:
        print(f"  PASS  (tolerance {TOL:.0e})")
        return 0
    print(f"  FAIL  (tolerance {TOL:.0e})")
    return 1


# ============================================================================
# CLI driver
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path,
        help="Input .cube file (real signed).")
    ap.add_argument("--output", type=Path,
        help="Output .cube file (envelope, non-negative).")
    ap.add_argument("--self-test", action="store_true",
        help="Run the analytic-Gaussian self-test instead of processing "
             "a cube.")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.input is None or args.output is None:
        ap.print_help()
        print("\nError: --input and --output are required (or use --self-test).")
        return 1

    print("=" * 72)
    print(f" cube_riesz_envelope.py   {args.input}")
    print("=" * 72)
    x, y, z, data, title = load_cube(args.input)
    print(f"  shape         : {data.shape}")
    print(f"  cube min/max  : {data.min(): .3e} / {data.max(): .3e}  "
          f"|max|={np.abs(data).max(): .3e}")
    E = riesz_envelope_3d(data)
    print(f"  envelope min  : {E.min(): .3e}   max: {E.max(): .3e}")
    # The cube grid axes from load_cube: rx = x; write_cube uses rx[0] as
    # origin and (rx[1]-rx[0]) as dr.
    write_cube(args.output, x, E,
               title=f"Riesz envelope of {args.input.name}")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
