"""
dawson_kernel_rabbitt.py
------------------------
Closed-form two-pulse 2nd-order PT kernel for RABBITT (XUV + IR),
derived in
``docs/a test derivation/rabbitt_two_pulse_dawson_derivation.md``
and validated to machine precision against adaptive quadrature in
``docs/a test derivation/test_rabbitt_two_pulse_dawson_kernel_output.txt``
(max rel err 5.25e-16, max |erf − Dawson| 9.30e-17 across 5 cases).

Math (formulas K1-K4 of the derivation):

    K_ba(α, β; τ_a, τ_b, T_a, T_b)
        = ∫dt e^{-(t-τ_b)²/T_b²} e^{iβt}
          · ∫_{-∞}^{t} dt' e^{-(t'-τ_a)²/T_a²} e^{iαt'}
        = (π T_a T_b / 2) · exp[-α² T_a²/4 - β² T_b²/4 + iα τ_a + iβ τ_b]
          · (1 + erf z)
    z   = (τ_b - τ_a + i(β T_b² - α T_a²)/2) / √(T_a² + T_b²)
    1 + erf z  =  1 + (2i/√π) e^{-z²} F(-iz)         (equivalent Dawson form)

RABBITT mapping (derivation §3): for sideband 2q,

    α_<  =  E_k + E_EA - Ω_{2q-1},      β_<  =  E - E_k - ω
    α_>  =  E_k + E_EA - Ω_{2q+1},      β_>  =  E - E_k + ω

where E_EA = -ε_i > 0 (binding-energy magnitude) and the XUV step is
always absorption.  The two paths interfere in the sideband.

UNITS.  The math is unit-agnostic: every detuning has dimensions of
energy and every T has dimensions of inverse energy (= time × 1/ℏ).  We
keep ATOMIC UNITS throughout the rest of the postproc pipeline; the
user's test script in ``docs/a test derivation/`` uses (eV, fs) and
converts via ħ = 0.6582119514 eV·fs.  Both work, you just have to be
consistent.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import erf, dawsn, wofz


HBAR_EV_FS = 0.6582119514     # ħ in eV·fs (helper for unit conversion)


# ---------------------------------------------------------------------------
# Core kernel
# ---------------------------------------------------------------------------
def _z(alpha: float, beta: float, T_a: float, T_b: float,
       tau_a: float, tau_b: float) -> complex:
    S = T_a * T_a + T_b * T_b
    return (tau_b - tau_a + 0.5j * (beta * T_b * T_b - alpha * T_a * T_a)) \
            / np.sqrt(S)


def _prefactor(alpha: float, beta: float, T_a: float, T_b: float,
               tau_a: float, tau_b: float) -> complex:
    pref = np.pi * T_a * T_b / 2.0
    expo = np.exp(
        -((alpha * T_a) ** 2) / 4.0
        - ((beta  * T_b) ** 2) / 4.0
        + 1j * (alpha * tau_a + beta * tau_b)
    )
    return pref * expo


def kernel_erf(alpha: float, beta: float, T_a: float, T_b: float,
               tau_a: float = 0.0, tau_b: float = 0.0) -> complex:
    """Two-pulse Gaussian kernel  K = (π T_a T_b/2)·exp[...]·(1 + erf z).

    Numerically STABLE form via the Faddeeva function
    ``scipy.special.wofz`` w(z) = e^{−z²}·erfc(−iz):

        1 + erf(z)  =  2  −  e^{−z²} · w(iz)

    The bare ``erf(z)`` for complex z with large |Im z| blows up as
    e^{Im(z)²}; combining e^{−z²} with the Gaussian prefactor analytically
    cancels that explosion *before* evaluating any transcendental — see
    derivation note below.  Computed by:

        K  =  2 · pref  −  pref_eff · w(iz)

    where pref_eff = (πT_aT_b/2) · exp(combined_exponent), and the
    combined exponent is the algebraic sum  (−α²T_a²/4 − β²T_b²/4 +
    iα τ_a + iβ τ_b) + (−z²).  Algebra: setting R=τ_b−τ_a, I=½(βT_b²−αT_a²),
    S=T_a²+T_b²:

        real_part = −T_a²T_b²·(α+β)²/(4 S) − R²/S
        imag_part = α τ_a + β τ_b − 2 R I / S

    Both bounded; no transcendental overflow.

    Reduces to the naive erf form to machine precision when the naive
    form is finite (checked against ``kernel_quad`` in ``_self_test``).
    """
    # original prefactor (Gaussian-decaying):
    pref = _prefactor(alpha, beta, T_a, T_b, tau_a, tau_b)

    # pref · exp(−z²) — algebraically combined so no transcendental
    # produces a huge intermediate.
    S = T_a * T_a + T_b * T_b
    R = tau_b - tau_a
    I = 0.5 * (beta * T_b * T_b - alpha * T_a * T_a)
    real_combined = -T_a * T_a * T_b * T_b * (alpha + beta) ** 2 / (4.0 * S) \
                    - R * R / S
    imag_combined = alpha * tau_a + beta * tau_b - 2.0 * R * I / S
    pref_eff = (np.pi * T_a * T_b / 2.0) * np.exp(real_combined + 1j * imag_combined)

    z  = _z(alpha, beta, T_a, T_b, tau_a, tau_b)
    iz = 1j * z
    return 2.0 * pref - pref_eff * wofz(iz)


def kernel_dawson(alpha: float, beta: float, T_a: float, T_b: float,
                  tau_a: float = 0.0, tau_b: float = 0.0) -> complex:
    """Equivalent kernel via the complex-Dawson identity 1+erf z = 1 +
    (2i/√π) e^{-z²} F(-iz).  Useful as an independent cross-check."""
    z = _z(alpha, beta, T_a, T_b, tau_a, tau_b)
    return _prefactor(alpha, beta, T_a, T_b, tau_a, tau_b) \
         * (1.0 + 2j / np.sqrt(np.pi) * np.exp(-z * z) * dawsn(-1j * z))


def kernel_quad(alpha: float, beta: float, T_a: float, T_b: float,
                tau_a: float = 0.0, tau_b: float = 0.0) -> complex:
    """Adaptive-quadrature reference: do the inner integral analytically,
    the outer integral numerically.  Independent of kernel_erf/dawson,
    so this is what we test against.  Slow (~ms per call) -- diagnostic
    use only."""
    from scipy.integrate import quad
    inner_pref = np.sqrt(np.pi) * T_a / 2.0
    inner_pref *= np.exp(-((alpha * T_a) ** 2) / 4.0 + 1j * alpha * tau_a)
    def inner(t):
        return inner_pref * (1.0 + erf((t - tau_a) / T_a - 0.5j * alpha * T_a))
    def integrand(t):
        return np.exp(-((t - tau_b) / T_b) ** 2 + 1j * beta * t) * inner(t)
    lo = min(tau_a, tau_b) - 12 * max(T_a, T_b)
    hi = max(tau_a, tau_b) + 12 * max(T_a, T_b)
    re = quad(lambda x: np.real(integrand(x)), lo, hi,
              epsabs=1e-11, epsrel=1e-11, limit=300)[0]
    im = quad(lambda x: np.imag(integrand(x)), lo, hi,
              epsabs=1e-11, epsrel=1e-11, limit=300)[0]
    return re + 1j * im


# ---------------------------------------------------------------------------
# RABBITT pathway helpers
# ---------------------------------------------------------------------------
def rabbitt_detunings(E: float, E_k: float, E_EA: float,
                      omega_IR: float, Omega_XUV: float,
                      ir_action: str) -> Tuple[float, float]:
    """Return (α, β) detunings for one RABBITT pathway through state k.

    Parameters
    ----------
    E         : final photoelectron kinetic energy
    E_k       : intermediate kinetic energy
    E_EA      : initial-state binding energy magnitude (= −ε_i > 0)
    omega_IR  : IR carrier frequency
    Omega_XUV : XUV harmonic carrier frequency for this pathway
    ir_action : 'absorb' (→ s_IR=+1) or 'emit' (→ s_IR=-1)

    Standard RABBITT (XUV always absorbed):
        '<' path:  Ω = (2q-1)·ω,  ir_action='absorb'
        '>' path:  Ω = (2q+1)·ω,  ir_action='emit'
    """
    if ir_action == "absorb":
        s_IR = +1.0
    elif ir_action == "emit":
        s_IR = -1.0
    else:
        raise ValueError(
            f"ir_action must be 'absorb' or 'emit', got {ir_action!r}")
    alpha = E_k + E_EA - Omega_XUV      # XUV always absorbed: s_XUV = +1
    beta  = E   - E_k   - s_IR * omega_IR
    return alpha, beta


# ---------------------------------------------------------------------------
# Self-test entry point (re-runs the 5-case validation from the doc dir)
# ---------------------------------------------------------------------------
def _self_test() -> int:
    """Cross-check kernel_erf vs kernel_dawson vs kernel_quad.  Same
    parameter set as docs/a test derivation/test_rabbitt....py; expected
    rel err ~ machine epsilon."""
    T_a = (0.35 / HBAR_EV_FS) / np.sqrt(2 * np.log(2))   # T_X_FWHM = 0.35 fs
    T_b = (5.0  / HBAR_EV_FS) / np.sqrt(2 * np.log(2))   # T_IR_FWHM = 5 fs
    tau_b = 0.45 / HBAR_EV_FS
    cases = [
        (0.12,  -0.05, T_a, T_b, 0.0,   tau_b),
        (-0.08,  0.13, T_a, T_b, 0.0,   tau_b),
        (0.00,   0.00, T_a, T_b, 0.0,   0.0),
        (0.35,  -0.42, T_a, T_a, 0.0,   0.0),    # Eq. (7)-like same-pulse
        (1.00,  -0.70, T_a, 0.7 * T_a,
                              -0.2 / HBAR_EV_FS, 0.8 / HBAR_EV_FS),
    ]
    print("=== dawson_kernel_rabbitt self-test (erf vs dawson vs quad) ===")
    print(f"T_a={T_a:.6g} (~T_X_FWHM=0.35 fs)   "
          f"T_b={T_b:.6g} (~T_IR_FWHM=5 fs)\n")
    fails = 0
    for i, case in enumerate(cases, 1):
        k_e = kernel_erf(*case)
        k_d = kernel_dawson(*case)
        k_q = kernel_quad(*case)
        rel_ed = abs(k_e - k_d) / max(abs(k_e), 1e-300)
        rel_eq = abs(k_e - k_q) / max(abs(k_e), 1e-300)
        ok = (rel_ed < 1e-13) and (rel_eq < 1e-12)
        tag = "ok  " if ok else "FAIL"
        print(f"  [{tag}] case {i}  rel|erf-dawson|={rel_ed:.2e}  "
              f"rel|erf-quad|={rel_eq:.2e}")
        if not ok:
            fails += 1
    print(f"\n  Total failures: {fails}\n")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
