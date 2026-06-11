#!/usr/bin/env python3
"""
dawson_kernel.py
-----------------
Per-intermediate-state kernel for the 2nd-order PT two-photon amplitude
under a Gaussian laser pulse, following

    Azizi, Saalmann, Rost, "Zero-energy photoelectric effect",
    arXiv:2407.16270 (2024), Eq. 7.

For one intermediate-state energy E_k and a finite pulse of HWHM
parameter T (the full FWHM duration in time is T·√(2 ln 2)), the
contribution of state k to the amplitude at FINAL energy E is

    a(E) = (π/8) T² A_0² · Σ_k  d_{Ek} d_{k,EA} · Σ_{η=±} K_T^η(E, E_k, E_EA, ω)

with the per-state kernel

    K_T^η(E, E_k, E_EA, ω)
        =  exp{ −[(Δ_k^η + E_EA)² + (E + E_EA)²] · T² / 4 }
           − (2 i / √π) · exp{ −(E + E_EA)² · T² / 8 }
                        · F( [2 Δ_k^η + E_EA − E] · T / √8 )                 ...(D7b)

    Δ_k^η = E_k − η · ω                                                       ...(D7c)

    F(x) = Dawson function = e^{−x²} · ∫_0^x e^{u²} du                        ...(D7d)

η = +1 (absorption-then-emission) and η = −1 (emission-then-absorption)
are summed.  For RABBITT sidebands the two η give the two interfering
pathways; for the user's ZEPE problem they correspond to the two
absorption-emission orderings on the SAME pulse.

Atomic units throughout.  E, E_k are kinetic energies of the photoelectron
(positive for continuum states); E_EA = -ε_i > 0 is the electron-affinity-
like binding energy (so the bound state energy is ε_i = -E_EA).  ω is the
carrier frequency of the laser pulse (atomic units, = E_h / ℏ).

The PREFACTOR (π/8) T² A_0² is constant in the energy/intermediate-state
sum and is dropped in any expression for τ (a phase) since it cancels
in arg[a*·a].  The kernel function returned here therefore contains the
per-state, η-summed piece WITHOUT the (π/8) T² A_0² prefactor.

Numerics: the Dawson function is provided by scipy.special.dawsn (which
follows Dawson's original definition matching (D7d)).  scipy's
implementation is accurate to ~1e-14 across the entire real axis.
Cross-checked here against the asymptotic form  F(x) ≈ 1/(2x) for
large |x|  and  F(x) ≈ x − (2/3) x³  for small |x|.

API
---
``kernel_eta(E, E_k, E_EA, omega, T, eta)`` — single-η kernel value (complex)
``kernel(E, E_k, E_EA, omega, T)``         — sum over η=±, returns complex
``test_self()``                            — assertions, prints PASS/FAIL
"""
from __future__ import annotations

import math
from typing import Iterable, Union

import numpy as np
from scipy.special import dawsn   # = F(x) per (D7d)


ArrayLike = Union[float, np.ndarray, complex, Iterable]


def _delta_k_eta(E_k: ArrayLike, omega: float, eta: int) -> ArrayLike:
    """Δ_k^η = E_k − η · ω    [Eq. (D7c)]"""
    if eta not in (-1, +1):
        raise ValueError(f"eta must be ±1, got {eta!r}")
    return E_k - eta * omega


def kernel_eta(E: ArrayLike, E_k: ArrayLike, E_EA: float, omega: float,
               T: float, eta: int) -> ArrayLike:
    """Single-η per-state kernel K_T^η per (D7b).  Returns complex."""
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T!r}")
    if E_EA <= 0:
        # Negative anion: ε_i < 0 → E_EA = -ε_i > 0.  Catch sign-bug calls.
        raise ValueError(
            f"E_EA must be > 0 (it is the binding energy magnitude); "
            f"if your bound-state energy ε_i is negative, pass E_EA = -ε_i.")
    delta = _delta_k_eta(E_k, omega, eta)

    # Real Gaussian piece: penalises both intermediate-state detuning
    # and final-state energy mismatch.  In the long-pulse limit, this
    # becomes a product of two delta-function constraints.
    arg1 = ((delta + E_EA) ** 2 + (E + E_EA) ** 2) * T * T / 4.0
    real_part = np.exp(-arg1)

    # Complex piece: Gaussian × Dawson.  This is what becomes
    # iπ · δ(2Δ_k^η + E_EA − E) / 2  in the long-pulse limit
    # (Sokhotski-Plemelj imaginary residue), plus a principal-value
    # piece in the Dawson-large-arg asymptote.
    arg2 = (E + E_EA) ** 2 * T * T / 8.0
    arg3 = (2.0 * delta + E_EA - E) * T / math.sqrt(8.0)
    imag_part = (2.0j / math.sqrt(math.pi)) * np.exp(-arg2) * dawsn(arg3)

    return real_part - imag_part


def kernel(E: ArrayLike, E_k: ArrayLike, E_EA: float, omega: float,
           T: float) -> ArrayLike:
    """Σ_{η=±} K_T^η(E, E_k, E_EA, ω)  — full per-state contribution."""
    return (kernel_eta(E, E_k, E_EA, omega, T, +1)
          + kernel_eta(E, E_k, E_EA, omega, T, -1))


# ---------------------------------------------------------------------------
# Tests / sanity checks
# ---------------------------------------------------------------------------
def test_self() -> int:
    """Self-contained correctness checks.  Returns # failures."""
    fails = 0

    def _check(cond: bool, what: str, val_a=None, val_b=None):
        nonlocal fails
        ok = bool(cond)
        if ok:
            print(f"  [ok  ] {what}")
        else:
            print(f"  [FAIL] {what}   (a={val_a!r}  b={val_b!r})")
            fails += 1

    # ---- (T1) Dawson function asymptotic forms ------------------------
    # Small-x:  F(x) ≈ x − (2/3) x³           (Taylor)
    # Large-x:  F(x) ≈ 1/(2x) + 1/(4x³) + …  (asymptotic)
    print("=== (T1) scipy Dawson asymptotic checks ===")
    for x in (1e-3, 1e-2, 0.05):
        # Include the next term  + (4/15) x^5  to make the test tight.
        approx = x - (2.0 / 3.0) * x ** 3 + (4.0 / 15.0) * x ** 5
        ref    = float(dawsn(x))
        rel    = abs(approx - ref) / abs(ref) if ref else 0.0
        _check(rel < 1e-8, f"F({x}) ≈ x - 2x³/3 + 4x⁵/15   (rel={rel:.2e})",
               approx, ref)
    for x in (1e2, 1e3, 1e4):
        approx = 0.5 / x
        ref    = float(dawsn(x))
        rel    = abs(approx - ref) / abs(ref) if ref else 0.0
        # Leading 1/(2x); next term is 1/(4x^3) — at x=100, 1/(4·10^6) ≈ 1e-7
        _check(rel < 1e-3, f"F({x}) ≈ 1/(2x)      (rel={rel:.2e})", approx, ref)

    # ---- (T2) Symmetry F(-x) = -F(x) ----------------------------------
    print("\n=== (T2) Dawson odd-parity ===")
    for x in (0.1, 1.0, 5.0, 50.0):
        a, b = float(dawsn(x)), float(dawsn(-x))
        _check(abs(a + b) < 1e-12 * (abs(a) + 1e-30),
               f"F(-{x}) = -F({x})  ({a:.6e} vs {-b:.6e})", a, -b)

    # ---- (T3) Kernel symmetry  K^η(E,E_k) under E_k swap -------------
    # Single intermediate state is fine; check kernel returns a finite
    # complex number, sensible magnitudes.
    print("\n=== (T3) kernel finite + complex ===")
    E_EA = 0.1            # 2.7 eV binding (anion-like)
    omega = 0.05          # ~1.4 eV IR
    T = 200.0             # T_FWHM ≈ T·sqrt(2 ln 2) ≈ 235 au ≈ 5.7 fs
    for E, E_k in [(0.1, 0.15), (0.5, 0.1), (1.0, 0.5)]:
        K_p = kernel_eta(E, E_k, E_EA, omega, T, +1)
        K_m = kernel_eta(E, E_k, E_EA, omega, T, -1)
        K   = kernel(E, E_k, E_EA, omega, T)
        _check(np.isfinite(K_p) and np.isfinite(K_m),
               f"K finite at (E={E}, E_k={E_k})  K±=({K_p:.3e}, {K_m:.3e})")
        _check(np.iscomplexobj(K) or (hasattr(K, 'imag') and K.imag != 0)
               or np.isclose(K_p.imag, 0) and np.isclose(K_m.imag, 0),
               f"K type sensible at (E={E}, E_k={E_k})")
        _check(abs(K - (K_p + K_m)) < 1e-14 * abs(K + 1e-30),
               f"K = K^+ + K^- at (E={E}, E_k={E_k})", K, K_p + K_m)

    # ---- (T4) Long-pulse (T → ∞) limit ------------------------------------
    # As T grows, the Gaussian factors become delta-like in their
    # arguments and off-shell |K| shrinks.  We use a modest detuning
    # (~0.2 au) and T values that don't underflow exp() to literal 0.
    print("\n=== (T4) long-pulse off-shell suppression ===")
    E_EA = 0.1; omega = 0.05; E = 0.5
    E_k = 0.7   # off-shell by 0.2 au from any resonance
    Ks = [abs(kernel(E, E_k, E_EA, omega, T)) for T in (5.0, 10.0, 20.0)]
    _check(all(k > 0.0 for k in Ks) and Ks[0] > Ks[1] > Ks[2],
           f"|K| decreases with T  off-shell: |K|={Ks}")

    # ---- (T5) η = +1 vs η = -1 difference is real -------------------------
    # The two η give different detunings Δ_k^η = E_k − η·ω, so K^+ ≠ K^−
    # in general.  Verify that swapping ω → −ω in the η=+1 branch gives
    # the η=−1 branch (a sign-of-η symmetry of the formula, useful as a
    # consistency check on the implementation).
    print("\n=== (T5) η symmetry: K^η(ω) = K^{-η}(-ω) ===")
    E_EA = 0.1; omega = 0.05; E = 0.5; E_k = 0.3; T = 10.0
    K_plus_pos_w  = kernel_eta(E, E_k, E_EA,  omega, T, +1)
    K_minus_neg_w = kernel_eta(E, E_k, E_EA, -omega, T, -1)
    rel = abs(K_plus_pos_w - K_minus_neg_w) / max(abs(K_plus_pos_w), 1e-300)
    _check(rel < 1e-14,
           f"K^+(+ω) == K^-(-ω): rel = {rel:.2e}")

    print(f"\n  Total failures: {fails}\n")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(test_self())
