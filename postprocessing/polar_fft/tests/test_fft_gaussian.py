#!/usr/bin/env python3
"""
test_fft_gaussian.py
====================
Validate `polar_fft.fft3_centered` against the closed-form analytic
inverse Fourier transform of a 3-D Gaussian.

Convention used by polar_fft.fft3_centered:
    F(r) = (1/(2π)^3) ∫ f(k) e^{+i k·r} d³k
    F(r_m) ≈ (1/dr)^3 · numpy_ifftn_with_shifts[m]      (eq. *)

For f(k) = exp(-|k|² / (2 σ²)) the 1-D inverse FT gives
    ∫ e^{-k²/(2σ²)} e^{+ikr} dk = σ √(2π) e^{-r² σ²/2}
so the 3-D version (separable) is
    F(r) = (σ / √(2π))^3 · e^{-|r|² σ² / 2}.

This test:
  1. Builds f(k) on a centered cube of side N (no polar pipeline; this
     isolates the FFT centering + normalisation alone).
  2. Calls fft3_centered, applies the physical factor (1/dr)^3.
  3. Compares to the analytic F(r) on the matching r-cube.
  4. Asserts max relative error < 1 % across the cube (the dominant
     residual is the truncation of the Gaussian at finite k_max).

PASS criterion: rel_err < 1e-2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Tests sit in postprocessing/polar_fft/tests/; the module is one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polar_fft import fft3_centered  # noqa: E402


def main() -> int:
    sigma_k = 1.0        # Gaussian width in k-space (a.u.)

    # 1. Centered cube in k-space.
    N = 64
    k_max = 6.0 * sigma_k                # 6σ → truncation error e^{-18} ≈ 1.5e-8
    dk = 2.0 * k_max / N
    kx = (np.arange(N) - N // 2) * dk
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K2 = KX * KX + KY * KY + KZ * KZ
    f_k = np.exp(-K2 / (2.0 * sigma_k * sigma_k))

    # 2. Centered inverse FFT.
    rx, f_r_raw = fft3_centered(kx, f_k)
    dr = rx[1] - rx[0]
    f_r_num = f_r_raw.real * (1.0 / dr) ** 3      # physical scale per eq. (*)

    # 3. Analytic continuous IFT on the same r-grid.
    RX, RY, RZ = np.meshgrid(rx, rx, rx, indexing="ij")
    R2 = RX * RX + RY * RY + RZ * RZ
    f_r_ana = (sigma_k / np.sqrt(2.0 * np.pi)) ** 3 \
              * np.exp(-R2 * sigma_k * sigma_k / 2.0)

    # 4. Compare.
    diff = np.abs(f_r_num - f_r_ana)
    rel  = diff / np.abs(f_r_ana).max()
    print("--- FFT self-test (analytic 3-D Gaussian) ---")
    print(f"  cube         : N = {N}    k_max = {k_max}    dk = {dk:.4e}")
    print(f"  r-cube       : r_max = {N/2 * dr:.3f} a.u.   dr = {dr:.4e}")
    print(f"  analytic F(0)= {(sigma_k / np.sqrt(2.0 * np.pi)) ** 3:.6e}")
    print(f"  numeric  F(0)= {f_r_num[N//2, N//2, N//2]:.6e}")
    print(f"  max |F_num - F_ana| / max|F_ana| = {rel.max():.3e}")
    print(f"  mean rel err over grid           = {rel.mean():.3e}")
    print(f"  imag part RMS / |F_ana|.max()    = "
          f"{np.sqrt(np.mean(f_r_raw.imag ** 2)) * (1.0 / dr) ** 3 / np.abs(f_r_ana).max():.3e}")

    TOL = 1.0e-2
    if rel.max() < TOL:
        print(f"  PASS  (tolerance {TOL:.0e})")
        return 0
    print(f"  FAIL  (tolerance {TOL:.0e})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
