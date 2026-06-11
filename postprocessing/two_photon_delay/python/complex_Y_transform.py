"""
complex_Y_transform.py
----------------------
Block-diagonal unitary U that converts coefficients / matrix elements
between our REAL-Y_lm basis (used everywhere in static_exchangeHF) and
the COMPLEX-Y_lm basis used by BW17 angular machinery (Wigner-D, etc).

Convention -- IDENTICAL to ``src/angular/Gaunt.hpp::U_real_to_complex``
(lines 89-109):

    Y^C_(l, mC) = Σ_mR  U(l, mR, mC) · Y^R_(l, mR)

with the per-l unitary block

    U(l, 0, 0)         =  1
    U(l, +k, -k)       =  1/√2          (mR=+k, mC=-k)
    U(l, +k, +k)       =  (-1)^k / √2   (mR=+k, mC=+k)
    U(l, -k, -k)       =  +i/√2         (mR=-k, mC=-k)
    U(l, -k, +k)       = -i·(-1)^k / √2 (mR=-k, mC=+k)
    all other          =  0

For a state Ψ = Σ_(l,mR) c^R · Y^R = Σ_(l,mC) c^C · Y^C the relation is

    c^R = U  · c^C            (so c^R[mR] = Σ_mC U(l,mR,mC) c^C[mC])
    c^C = U† · c^R

For a matrix element  ⟨Ψ′|O|Ψ⟩  (operator O does NOT depend on basis):

    O^R = U  O^C  U†            (real basis matrix from complex)
    O^C = U†  O^R  U            (complex basis matrix from real)

Both forms are useful: the m-resolved RABBITT amplitude (D6) wants to
multiply  Y^C_(LM_C)(k̂) · U(L, M_R, M_C)  in the same expression as our
REAL-basis M_in[(L, M_R), μ_ν].  See ``docs/phase_c_derivation.md``
§(D4) and §(D6).

Verification
------------
``_self_test()`` checks:
1.  Per-l U is unitary (U U† = I and U† U = I).
2.  Round-trip on a random vector: c^R == U U† c^R within machine eps.
3.  Action on a known case: Y^R_(1, 0) == Y^C_(1, 0)  (mR=mC=0 row).
4.  Reality recovery: complex-basis representation of a real-valued
    angular function is reproduced from c^C = U† c^R back via U.
5.  Direct match against C++ U_real_to_complex via an inline list of
    selected (l, mR, mC) → value triples taken from the C++ formula.
"""
from __future__ import annotations

import numpy as np


SQRT_HALF = 1.0 / np.sqrt(2.0)


def _phase(m: int) -> float:
    """(-1)^m for integer m -- exactly matches phase() in Gaunt.hpp."""
    return -1.0 if (m & 1) else 1.0


# ---------------------------------------------------------------------------
# Per-l block
# ---------------------------------------------------------------------------
def U_block(l: int) -> np.ndarray:
    """Return the (2l+1)×(2l+1) complex unitary U(l) with rows indexed by
    mR ∈ [-l..l] and columns indexed by mC ∈ [-l..l].  Identical to the
    C++ U_real_to_complex(l, mR, mC).

    Index mapping:   row i = mR + l,    col j = mC + l.
    """
    dim = 2 * l + 1
    U = np.zeros((dim, dim), dtype=np.complex128)
    for mR in range(-l, l + 1):
        i = mR + l
        if mR == 0:
            j = 0 + l
            U[i, j] = 1.0
            continue
        if mR > 0:
            k = mR
            j_neg = -k + l
            j_pos = +k + l
            U[i, j_neg] = SQRT_HALF
            U[i, j_pos] = SQRT_HALF * _phase(k)
        else:
            p = -mR
            j_neg = -p + l
            j_pos = +p + l
            U[i, j_neg] = +1j * SQRT_HALF
            U[i, j_pos] = -1j * SQRT_HALF * _phase(p)
    return U


# ---------------------------------------------------------------------------
# Full block-diagonal U up to Lmax
# ---------------------------------------------------------------------------
def U_full(Lmax: int) -> np.ndarray:
    """Block-diagonal U built from U_block(l) for l = 0..Lmax.  Size N×N
    with N = (Lmax+1)²; channel index μ = l² + l + m matches our packing
    convention (Gaunt.hpp::lm_to_idx)."""
    N = (Lmax + 1) ** 2
    U = np.zeros((N, N), dtype=np.complex128)
    off = 0
    for l in range(Lmax + 1):
        dim = 2 * l + 1
        U[off:off + dim, off:off + dim] = U_block(l)
        off += dim
    return U


# ---------------------------------------------------------------------------
# Coefficient & matrix transforms
# ---------------------------------------------------------------------------
def cR_to_cC(cR: np.ndarray, Lmax: int | None = None) -> np.ndarray:
    """Coefficient transform  c^C = U† c^R."""
    if Lmax is None:
        N = cR.shape[0]
        Lmax = int(round(np.sqrt(N))) - 1
        if (Lmax + 1) ** 2 != N:
            raise ValueError(f"cR length {N} is not (Lmax+1)² for any integer Lmax")
    U = U_full(Lmax)
    return U.conj().T @ cR


def cC_to_cR(cC: np.ndarray, Lmax: int | None = None) -> np.ndarray:
    """Coefficient transform  c^R = U c^C."""
    if Lmax is None:
        N = cC.shape[0]
        Lmax = int(round(np.sqrt(N))) - 1
        if (Lmax + 1) ** 2 != N:
            raise ValueError(f"cC length {N} is not (Lmax+1)² for any integer Lmax")
    U = U_full(Lmax)
    return U @ cC


def OR_to_OC(OR: np.ndarray, Lmax: int | None = None) -> np.ndarray:
    """Matrix transform  O^C = U† O^R U  (operator from real to complex)."""
    if Lmax is None:
        N = OR.shape[0]
        Lmax = int(round(np.sqrt(N))) - 1
        if (Lmax + 1) ** 2 != N:
            raise ValueError(f"OR size {N} is not (Lmax+1)² for any integer Lmax")
    U = U_full(Lmax)
    return U.conj().T @ OR @ U


def OC_to_OR(OC: np.ndarray, Lmax: int | None = None) -> np.ndarray:
    """Matrix transform  O^R = U O^C U†  (operator from complex to real)."""
    if Lmax is None:
        N = OC.shape[0]
        Lmax = int(round(np.sqrt(N))) - 1
        if (Lmax + 1) ** 2 != N:
            raise ValueError(f"OC size {N} is not (Lmax+1)² for any integer Lmax")
    U = U_full(Lmax)
    return U @ OC @ U.conj().T


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _self_test() -> int:
    """Suite of correctness checks.  Returns # failures."""
    fails = 0

    def _check(cond: bool, what: str):
        nonlocal fails
        if cond:
            print(f"  [ok  ] {what}")
        else:
            print(f"  [FAIL] {what}")
            fails += 1

    print("=== (T1) per-l U unitary ===")
    for l in range(7):
        U = U_block(l)
        UU = U @ U.conj().T
        I = np.eye(2 * l + 1)
        err = np.max(np.abs(UU - I))
        _check(err < 1e-14, f"l={l}: ||U U† - I||_inf = {err:.2e}")
        UU = U.conj().T @ U
        err = np.max(np.abs(UU - I))
        _check(err < 1e-14, f"l={l}: ||U† U - I||_inf = {err:.2e}")

    print("\n=== (T2) full block-diagonal U unitary up to Lmax=5 ===")
    Lmax = 5
    U = U_full(Lmax)
    err = np.max(np.abs(U @ U.conj().T - np.eye((Lmax + 1) ** 2)))
    _check(err < 1e-14, f"||U U† - I||_inf = {err:.2e}")

    print("\n=== (T3) coefficient round-trip ===")
    rng = np.random.default_rng(0)
    cR = rng.normal(size=(Lmax + 1) ** 2)
    cC = cR_to_cC(cR.astype(np.complex128))
    cR2 = cC_to_cR(cC)
    err = np.max(np.abs(cR - cR2))
    _check(err < 1e-14, f"||c^R - U U† c^R||_inf = {err:.2e}")

    print("\n=== (T4) matrix round-trip ===")
    OR = rng.normal(size=((Lmax + 1) ** 2, (Lmax + 1) ** 2))
    OR = (OR + OR.T) / 2.0   # symmetric real -> Hermitian after transform
    OC = OR_to_OC(OR.astype(np.complex128))
    OR2 = OC_to_OR(OC)
    err = np.max(np.abs(OR - OR2))
    _check(err < 1e-13, f"||O^R - U U† O^R U U†||_inf = {err:.2e}")

    # Hermiticity of OC if OR was symmetric real
    err_h = np.max(np.abs(OC - OC.conj().T))
    _check(err_h < 1e-13, f"symmetric real O^R → Hermitian O^C  (||O^C - O^C†|| = {err_h:.2e})")

    print("\n=== (T5) explicit values vs C++ U_real_to_complex ===")
    # (l, mR, mC, expected) — directly read off C++ formula:
    expected = [
        (0,  0,  0,  1.0 + 0.0j),
        (1,  0,  0,  1.0 + 0.0j),
        (1, +1, -1,  SQRT_HALF + 0.0j),
        (1, +1, +1, -SQRT_HALF + 0.0j),        # (-1)^1 / sqrt(2)
        (1, -1, -1,  0.0 + SQRT_HALF * 1j),
        (1, -1, +1,  0.0 - (-SQRT_HALF) * 1j), # -i·(-1)^1 / sqrt(2) = +i/sqrt(2)
        (2, +2, -2,  SQRT_HALF + 0.0j),
        (2, +2, +2,  SQRT_HALF + 0.0j),        # (-1)^2 / sqrt(2) = +1/sqrt(2)
        (2, -2, -2,  0.0 + SQRT_HALF * 1j),
        (2, -2, +2,  0.0 - SQRT_HALF * 1j),    # -i·(-1)^2 / sqrt(2) = -i/sqrt(2)
        (3, +1, +1, -SQRT_HALF + 0.0j),
        (3, -1, +1,  0.0 - (-SQRT_HALF) * 1j), # +i/sqrt(2)
    ]
    for l, mR, mC, exp in expected:
        Ub = U_block(l)
        got = Ub[mR + l, mC + l]
        err = abs(got - exp)
        _check(err < 1e-15,
               f"U({l},{mR},{mC})={got!r:>40s}  (expected {exp!r}, err={err:.2e})"
               .replace("U_block", "U"))

    # (T6) Parity identity for the complex Y_lm.  Since Y^R is real, the
    # relation Y^C_{l,-m} = (-1)^m · (Y^C_{l,m})* implies
    #     U(l, mR, -mC) = (-1)^mC · U*(l, mR, mC)
    # (acting on the column / complex-m index only).
    print("\n=== (T6) Y^C parity identity (m_C -> -m_C) ===")
    max_err = 0.0
    for l in range(5):
        Ub = U_block(l)
        for mR in range(-l, l + 1):
            for mC in range(-l, l + 1):
                lhs = Ub[mR + l, -mC + l]
                rhs = _phase(mC) * np.conj(Ub[mR + l, mC + l])
                err = abs(lhs - rhs)
                if err > max_err:
                    max_err = err
    _check(max_err < 1e-15,
           f"max |U(l, mR, -mC) - (-1)^mC U*(l, mR, mC)| = {max_err:.2e}")

    print(f"\n  Total failures: {fails}\n")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
