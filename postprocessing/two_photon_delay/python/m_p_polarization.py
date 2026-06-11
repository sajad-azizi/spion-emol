"""
m_p_polarization.py
-------------------
Cartesian → spherical-tensor conversion for the molecular-frame
polarisation direction of the dipole matrix elements.

Convention (Condon-Shortley spherical tensor T^(1)_m, derivation §(D2)):

    T^(1)_(-1) = ( +x − i·y) / √2
    T^(1)_( 0) =    z
    T^(1)_(+1) = (−x − i·y) / √2

Inverting gives the linear map applied to *matrix elements*:

    M^{m_p = -1} = ( M^x − i·M^y) / √2          (D2d)
    M^{m_p =  0} =   M^z                         (D2e)
    M^{m_p = +1} = (−M^x − i·M^y) / √2          (D2f)

Phase A stores M^q for q ∈ {x, y, z}.  These are complex in general
(because the c-c dipole in the in-state basis is complex), but the
LINEAR combinations above are unambiguous.

This module is intentionally thin -- it just makes the (D2d-f) formula
a named function so the (D6) angular assembly is easy to audit.
"""
from __future__ import annotations

import numpy as np


SQRT_HALF = 1.0 / np.sqrt(2.0)


def cart_to_spherical(Mx, My, Mz):
    """Given matrix-element triples M^x, M^y, M^z (any common shape;
    scalar, vector, or N-D array), return the spherical-tensor triple
    (M^{-1}, M^{0}, M^{+1}) per (D2d-f).

    Inputs can be real or complex; outputs are always complex.
    """
    Mx = np.asarray(Mx)
    My = np.asarray(My)
    Mz = np.asarray(Mz)
    if not (Mx.shape == My.shape == Mz.shape):
        raise ValueError(
            f"shape mismatch: Mx={Mx.shape}  My={My.shape}  Mz={Mz.shape}")
    Mx = Mx.astype(np.complex128, copy=False)
    My = My.astype(np.complex128, copy=False)
    Mz = Mz.astype(np.complex128, copy=False)
    M_m1 = SQRT_HALF * (Mx - 1j * My)
    M_0  = Mz
    M_p1 = SQRT_HALF * (-Mx - 1j * My)
    return M_m1, M_0, M_p1


def spherical_to_cart(M_m1, M_0, M_p1):
    """Inverse: given (M^{-1}, M^{0}, M^{+1}), recover (M^x, M^y, M^z).

    Solving (D2d-f):
        M^x =  (M^{-1} - M^{+1}) / √2
        M^y =  i·(M^{-1} + M^{+1}) / √2
        M^z =   M^{0}
    """
    M_m1 = np.asarray(M_m1).astype(np.complex128, copy=False)
    M_0  = np.asarray(M_0 ).astype(np.complex128, copy=False)
    M_p1 = np.asarray(M_p1).astype(np.complex128, copy=False)
    Mx =       SQRT_HALF * (M_m1 - M_p1)
    My = 1j *  SQRT_HALF * (M_m1 + M_p1)
    Mz = M_0
    return Mx, My, Mz


def stack_polarisations(M_dict, order=("x", "y", "z")):
    """Convenience: pull M^x, M^y, M^z out of a dict (keys 'x','y','z')
    and call cart_to_spherical.  Returns a dict
        {-1: M_m1, 0: M_0, +1: M_p1}.
    """
    Mx, My, Mz = (M_dict[order[0]], M_dict[order[1]], M_dict[order[2]])
    M_m1, M_0, M_p1 = cart_to_spherical(Mx, My, Mz)
    return {-1: M_m1, 0: M_0, +1: M_p1}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _self_test() -> int:
    """Sanity checks for (D2d-f) and its inverse."""
    fails = 0
    def _check(cond: bool, what: str):
        nonlocal fails
        if cond:
            print(f"  [ok  ] {what}")
        else:
            print(f"  [FAIL] {what}")
            fails += 1

    rng = np.random.default_rng(1)

    print("=== (T1) round-trip scalar ===")
    Mx, My, Mz = (rng.normal() + 1j * rng.normal(),
                  rng.normal() + 1j * rng.normal(),
                  rng.normal() + 1j * rng.normal())
    M_m1, M_0, M_p1 = cart_to_spherical(Mx, My, Mz)
    Mx2, My2, Mz2 = spherical_to_cart(M_m1, M_0, M_p1)
    err = max(abs(Mx - Mx2), abs(My - My2), abs(Mz - Mz2))
    _check(err < 1e-15, f"max |cart - inverse|_complex = {err:.2e}")

    print("\n=== (T2) round-trip array ===")
    shp = (4, 5, 3)
    Mx = rng.normal(size=shp) + 1j * rng.normal(size=shp)
    My = rng.normal(size=shp) + 1j * rng.normal(size=shp)
    Mz = rng.normal(size=shp) + 1j * rng.normal(size=shp)
    M_m1, M_0, M_p1 = cart_to_spherical(Mx, My, Mz)
    Mx2, My2, Mz2 = spherical_to_cart(M_m1, M_0, M_p1)
    err = max(np.max(np.abs(Mx - Mx2)),
              np.max(np.abs(My - My2)),
              np.max(np.abs(Mz - Mz2)))
    _check(err < 1e-15, f"array round-trip max err = {err:.2e}")

    print("\n=== (T3) explicit values ===")
    # M^x = 1, M^y = 0, M^z = 0  → M^{-1} = 1/√2,  M^0 = 0,  M^{+1} = -1/√2.
    M_m1, M_0, M_p1 = cart_to_spherical(1.0, 0.0, 0.0)
    _check(abs(M_m1 -  SQRT_HALF) < 1e-15, f"x-only: M^(-1) = +1/√2  got {M_m1!r}")
    _check(abs(M_0 ) < 1e-15,              f"x-only: M^(0)  = 0      got {M_0!r}")
    _check(abs(M_p1 - (-SQRT_HALF)) < 1e-15,
                                            f"x-only: M^(+1) = -1/√2  got {M_p1!r}")

    # M^y = 1 → M^{-1} = -i/√2, M^0 = 0, M^{+1} = -i/√2.
    M_m1, M_0, M_p1 = cart_to_spherical(0.0, 1.0, 0.0)
    _check(abs(M_m1 - (-1j * SQRT_HALF)) < 1e-15,
                                            f"y-only: M^(-1) = -i/√2  got {M_m1!r}")
    _check(abs(M_0 ) < 1e-15,              f"y-only: M^(0)  = 0      got {M_0!r}")
    _check(abs(M_p1 - (-1j * SQRT_HALF)) < 1e-15,
                                            f"y-only: M^(+1) = -i/√2  got {M_p1!r}")

    # M^z = 1 → M^{-1} = 0, M^0 = 1, M^{+1} = 0.
    M_m1, M_0, M_p1 = cart_to_spherical(0.0, 0.0, 1.0)
    _check(abs(M_m1) < 1e-15,              f"z-only: M^(-1) = 0  got {M_m1!r}")
    _check(abs(M_0 - 1.0) < 1e-15,          f"z-only: M^(0)  = 1  got {M_0!r}")
    _check(abs(M_p1) < 1e-15,              f"z-only: M^(+1) = 0  got {M_p1!r}")

    print("\n=== (T4) unitarity of the 3×3 transform ===")
    # The map (Mx, My, Mz)^T → (M^{-1}, M^0, M^{+1})^T is implemented by a
    # 3×3 matrix V; check V V† = I (so it preserves Σ_q |M^q|²).
    V = np.array([
        [ SQRT_HALF, -1j * SQRT_HALF, 0.0],
        [       0.0,             0.0, 1.0],
        [-SQRT_HALF, -1j * SQRT_HALF, 0.0],
    ], dtype=np.complex128)
    err = np.max(np.abs(V @ V.conj().T - np.eye(3)))
    _check(err < 1e-15, f"||V V† - I||_inf = {err:.2e}")

    # And consistency with our function:
    rng2 = np.random.default_rng(7)
    Mq = rng2.normal(size=3) + 1j * rng2.normal(size=3)
    via_V = V @ Mq
    M_m1, M_0, M_p1 = cart_to_spherical(*Mq)
    err = max(abs(via_V[0] - M_m1), abs(via_V[1] - M_0), abs(via_V[2] - M_p1))
    _check(err < 1e-15, f"function matches 3x3 V matrix  (max err = {err:.2e})")

    print(f"\n  Total failures: {fails}\n")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
