"""
tau_2homega.py
--------------
Sideband delay extraction per derivation §(D8):

    τ_2hω(2q; k̂, R̂_γ)  =  (1/2ω) · arg [ M_<*  ·  M_> ]
                       = − (1/2ω) · arg [ M_<   ·  M_>* ]

SIGN: re-derived from user §3 expansion 2026-05-16.  Original derivation
§4 had  τ = +(1/2ω) arg(M_<·M_>*), which differs from §3's |a|²
expansion by an overall sign (the user explicitly flagged the §4 sign as
low-confidence, "2/10").  Cross-checked against BW17 Eq. 21 / Dahlström
RABBITT form (τ_2hω ∝ arg M_> − arg M_<), which matches the version
above.

with the two-path amplitudes M_<, M_> coming from
``compute_M.compute_M_one_path`` evaluated at the appropriate sideband
energy and two harmonic orderings:

* M_< : absorb Ω_{2q-1},  absorb IR     (sideband 2q via lower harmonic)
* M_> : absorb Ω_{2q+1},  emit IR       (sideband 2q via upper harmonic)

Three entry points:

1. ``tau_pointwise(M_less, M_greater, omega_IR)``
        scalar / elementwise.  Returns τ.

2. ``tau_angle_averaged(M_less, M_greater, omega_IR, weights)``
        BW17-style angle average:
            τ_avg = (1/2ω) · arg [ <M_< M_>*>_weighted ]
        This is the correct quantity for an experiment that does NOT
        resolve k̂ / orientation, NOT the average of pointwise τ.

3. ``tau_cc_anion(L, eps_kappa, omega_IR)``
        For atomic anions and short-range potentials the Coulomb
        continuum-continuum delay τ_cc vanishes (no 1/r tail → no
        Coulomb-log).  Returns 0 with a documented justification.

τ has the **same physical sign convention** as the Wigner-Smith /
Eisenbud-Wigner delay used elsewhere in the project (see
``cross_section_delay.py``): positive τ means the photoelectron leaves
*later* than the equivalent free-particle process.

Notes on ARG / branch cuts
--------------------------
``np.angle`` returns the principal value in (-π, π], so τ ∈
(-π/(2ω), π/(2ω)].  For a 1.55 eV IR, that bound is ≈ ±0.673 fs = 673 as
-- already much larger than any physically interesting RABBITT delay.
We do NOT unwrap by default; the caller should unwrap explicitly if
scanning along a continuous parameter (energy, geometry...) and crossing
the branch cut is physically meaningful.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np


ArrayLike = Union[complex, np.ndarray]


# ---------------------------------------------------------------------------
# Pointwise delay
# ---------------------------------------------------------------------------
def tau_pointwise(
    M_less: ArrayLike,
    M_greater: ArrayLike,
    omega_IR: float,
) -> ArrayLike:
    """τ(k̂, R̂_γ) = (1/2ω) arg[ M_<* · M_> ] = (1/2ω) [arg M_> − arg M_<].

    Inputs broadcast: scalar or array-like.  Output dtype is float.
    """
    if omega_IR <= 0:
        raise ValueError(f"omega_IR must be > 0, got {omega_IR!r}")
    M_less    = np.asarray(M_less,    dtype=np.complex128)
    M_greater = np.asarray(M_greater, dtype=np.complex128)
    return np.angle(np.conj(M_less) * M_greater) / (2.0 * omega_IR)


# ---------------------------------------------------------------------------
# Angle-averaged delay (BW17-style coherent average)
# ---------------------------------------------------------------------------
def tau_angle_averaged(
    M_less: np.ndarray,
    M_greater: np.ndarray,
    omega_IR: float,
    weights: Optional[np.ndarray] = None,
) -> float:
    """τ_avg = (1/2ω) arg [ Σ_i w_i (M_<)_i* · (M_>)_i ].

    This is NOT the average of pointwise τ -- it is the argument of the
    coherently-averaged interference factor, which is the quantity that
    appears in an unresolved-detection sideband oscillation
    (S_(2q)(τ) ∝ cos[2ωτ + arg<M_< M_>*>]).

    Parameters
    ----------
    M_less, M_greater : same-shape complex arrays; the (k̂, R̂_γ) axes
                        are flattened internally.
    weights            : same-shape float array, optional.  Uniform
                        weights are used if omitted (caller should pass
                        the angular-grid integration weights from
                        ``compute_M.angular_grid``).
    """
    M_less    = np.asarray(M_less,    dtype=np.complex128).ravel()
    M_greater = np.asarray(M_greater, dtype=np.complex128).ravel()
    if M_less.shape != M_greater.shape:
        raise ValueError(
            f"M_less {M_less.shape} != M_greater {M_greater.shape}")
    if weights is None:
        weights = np.ones_like(M_less, dtype=float) / M_less.size
    else:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.shape != M_less.shape:
            raise ValueError(
                f"weights {weights.shape} != amplitude shape {M_less.shape}")
    interference = np.sum(weights * np.conj(M_less) * M_greater)
    return float(np.angle(interference) / (2.0 * omega_IR))


# ---------------------------------------------------------------------------
# τ_cc for short-range continuum (anion case)
# ---------------------------------------------------------------------------
def tau_cc_anion(L: int, eps_kappa: float, omega_IR: float) -> float:
    """Continuum-continuum delay for a short-range potential.

    BW17 Eq. 24 expresses τ_cc via the asymptotic Coulomb phase of the
    intermediate / final continuum.  For an ANION (and for our C₈F₈⁻
    case) the photoelectron sees a short-range potential at large r --
    NO Coulomb tail, NO log-phase log(2kr)/k accumulation.  The
    centrifugal phase shift exists but is the SAME for the < and > paths
    (only the kinetic energy differs by 2ω, and we work in the limit
    where the centrifugal contribution is smooth in energy).

    For now: return 0.0 with a documented justification.  We can refine
    when comparing against the Azizi 2024 atomic benchmark (Phase D).

    Parameters L, eps_kappa, omega_IR are accepted for forward-
    compatibility with a future Coulomb / centrifugal correction.
    """
    _ = (L, eps_kappa, omega_IR)
    return 0.0


def tau_mol(
    M_less: ArrayLike,
    M_greater: ArrayLike,
    omega_IR: float,
    L: int = 0,
    eps_kappa: float = 0.0,
) -> ArrayLike:
    """τ_mol = τ_2hω − τ_cc  (BW17 Eq. 25)."""
    return tau_pointwise(M_less, M_greater, omega_IR) \
         - tau_cc_anion(L, eps_kappa, omega_IR)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _self_test() -> int:
    fails = 0
    def _check(cond: bool, what: str):
        nonlocal fails
        if cond:
            print(f"  [ok  ] {what}")
        else:
            print(f"  [FAIL] {what}")
            fails += 1

    omega_IR = 0.057  # 1.55 eV ≈ 0.057 au

    print("=== (T1) τ_pointwise scalar ===")
    # M_< = e^{i·0.3}, M_> = e^{i·0.7}  →  M_<* M_> = e^{i(0.7-0.3)} = e^{+i·0.4}
    # τ = +0.4 / (2ω) = +3.5 au^-1
    M_less    = np.exp(1j * 0.3)
    M_greater = np.exp(1j * 0.7)
    tau = tau_pointwise(M_less, M_greater, omega_IR)
    ref = +0.4 / (2.0 * omega_IR)
    _check(abs(tau - ref) < 1e-14, f"τ = (φ_> - φ_<) / (2ω): got {tau:.6g}, ref {ref:.6g}")

    print("\n=== (T2) τ_pointwise array ===")
    n = 10
    rng = np.random.default_rng(4)
    phi_less    = rng.uniform(-np.pi, np.pi, n)
    phi_greater = rng.uniform(-np.pi, np.pi, n)
    M_less    = np.exp(1j * phi_less)
    M_greater = np.exp(1j * phi_greater)
    tau = tau_pointwise(M_less, M_greater, omega_IR)
    # tau · 2ω should equal phi_greater - phi_less (mod 2π principal).
    diff = phi_greater - phi_less
    diff_pv = np.mod(diff + np.pi, 2 * np.pi) - np.pi
    diff_pv[diff_pv == -np.pi] = np.pi
    ref = diff_pv / (2 * omega_IR)
    err = np.max(np.abs(tau - ref))
    _check(err < 1e-12, f"τ vs principal-value phase diff: max err = {err:.2e}")

    print("\n=== (T3) tau_angle_averaged: coherent vs incoherent ===")
    # Two angles: at θ=0 give M_< M_>* = e^{+i·0.5}, at θ=π give M_< M_>* = e^{-i·0.5}
    # Pointwise τ averages would be 0 — but the AVERAGE of M_< M_>* is real, so arg = 0
    # → τ_avg = 0.  Same here because the two cancel.
    M_l = np.array([1.0,           np.exp(-1j * 0.5)])  # arbitrary
    M_g = np.array([np.exp(-1j*0.5), 1.0])              # M_<*M_>* = (+0.5, -0.5)
    tau_avg = tau_angle_averaged(M_l, M_g, omega_IR)
    _check(abs(tau_avg) < 1e-13, f"symmetric cancel: τ_avg = {tau_avg:.2e}")

    # Now bias toward the first sample: weights are (3, 1).
    tau_avg = tau_angle_averaged(M_l, M_g, omega_IR, weights=np.array([3.0, 1.0]))
    # interference = Σ w · M_<* M_> with this M_<* M_> = (e^{-i 0.5}, e^{+i 0.5})
    ref_sum = 3 * np.conj(M_l[0]) * M_g[0] + 1 * np.conj(M_l[1]) * M_g[1]
    ref = float(np.angle(ref_sum) / (2 * omega_IR))
    _check(abs(tau_avg - ref) < 1e-14, f"weighted avg matches direct arg: {tau_avg:.6g} vs {ref:.6g}")

    print("\n=== (T4) tau_cc_anion returns 0 ===")
    _check(tau_cc_anion(0, 1.0, 0.057) == 0.0, "tau_cc_anion(L=0) = 0")
    _check(tau_cc_anion(5, 1.0, 0.057) == 0.0, "tau_cc_anion(L=5) = 0")

    print("\n=== (T5) tau_mol = tau_2hω - tau_cc ===")
    M_l = np.exp(1j * 0.4)
    M_g = np.exp(1j * 0.1)
    expected = (0.1 - 0.4) / (2 * omega_IR)   # arg(M_>) - arg(M_<)
    got = tau_mol(M_l, M_g, omega_IR)
    _check(abs(got - expected) < 1e-14, f"tau_mol = {got:.6g}, ref {expected:.6g}")

    print(f"\n  Total failures: {fails}\n")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
