"""
Numerical test for the two-pulse RABBITT Dawson kernel.

The code compares:
  1. closed-form erf kernel,
  2. equivalent complex-Dawson kernel,
  3. direct adaptive quadrature of the time-ordered integral.

Units: energies in eV, times in eV^{-1}; convert fs to eV^{-1} by t[eV^-1]=t[fs]/hbar_eV_fs.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import erf, dawsn

HBAR_EV_FS = 0.6582119514


def kernel_erf(alpha, beta, T_a, T_b, tau_a, tau_b):
    """Closed two-pulse Gaussian kernel in erf form."""
    S = T_a**2 + T_b**2
    z = (tau_b - tau_a + 0.5j * (beta * T_b**2 - alpha * T_a**2)) / np.sqrt(S)
    pref = np.pi * T_a * T_b / 2
    expo = np.exp(
        -((alpha * T_a) ** 2) / 4
        -((beta * T_b) ** 2) / 4
        + 1j * (alpha * tau_a + beta * tau_b)
    )
    return pref * expo * (1 + erf(z))


def kernel_dawson(alpha, beta, T_a, T_b, tau_a, tau_b):
    """Same kernel in complex Dawson form."""
    S = T_a**2 + T_b**2
    z = (tau_b - tau_a + 0.5j * (beta * T_b**2 - alpha * T_a**2)) / np.sqrt(S)
    pref = np.pi * T_a * T_b / 2
    expo = np.exp(
        -((alpha * T_a) ** 2) / 4
        -((beta * T_b) ** 2) / 4
        + 1j * (alpha * tau_a + beta * tau_b)
    )
    return pref * expo * (1 + 2j / np.sqrt(np.pi) * np.exp(-z * z) * dawsn(-1j * z))


def kernel_quad(alpha, beta, T_a, T_b, tau_a, tau_b):
    """Adaptive quadrature of the original ordered integral.

    The inner integral is evaluated analytically; the outer integral is done
    numerically. This is independent enough to test the closed two-pulse kernel.
    """
    inner_pref = np.sqrt(np.pi) * T_a / 2
    inner_pref *= np.exp(-((alpha * T_a) ** 2) / 4 + 1j * alpha * tau_a)

    def inner(t):
        arg = (t - tau_a) / T_a - 0.5j * alpha * T_a
        return inner_pref * (1 + erf(arg))

    def integrand(t):
        return np.exp(-((t - tau_b) / T_b) ** 2 + 1j * beta * t) * inner(t)

    lo = min(tau_a, tau_b) - 12 * max(T_a, T_b)
    hi = max(tau_a, tau_b) + 12 * max(T_a, T_b)

    re = quad(lambda x: np.real(integrand(x)), lo, hi, epsabs=1e-11, epsrel=1e-11, limit=300)[0]
    im = quad(lambda x: np.imag(integrand(x)), lo, hi, epsabs=1e-11, epsrel=1e-11, limit=300)[0]
    return re + 1j * im


def main():
    # RABBITT-like finite-pulse parameters.
    T_fwhm_x_fs = 0.35
    T_fwhm_ir_fs = 5.0
    tau_fs = 0.45

    T_a = (T_fwhm_x_fs / HBAR_EV_FS) / np.sqrt(2 * np.log(2))
    T_b = (T_fwhm_ir_fs / HBAR_EV_FS) / np.sqrt(2 * np.log(2))
    tau_a = 0.0
    tau_b = tau_fs / HBAR_EV_FS

    cases = [
        # alpha, beta, T_a, T_b, tau_a, tau_b
        (0.12, -0.05, T_a, T_b, tau_a, tau_b),
        (-0.08, 0.13, T_a, T_b, tau_a, tau_b),
        (0.00, 0.00, T_a, T_b, tau_a, 0.0),
        # Eq. (7)-like same-pulse zero-delay case.
        (0.35, -0.42, T_a, T_a, 0.0, 0.0),
        # Unequal durations and nonzero relative timing.
        (1.00, -0.70, T_a, 0.7 * T_a, -0.2 / HBAR_EV_FS, 0.8 / HBAR_EV_FS),
    ]

    max_rel_quad = 0.0
    max_abs_quad = 0.0
    max_rel_dawson = 0.0

    print(f"T_a = {T_a:.16g} eV^-1, T_b = {T_b:.16g} eV^-1, tau_b = {tau_b:.16g} eV^-1")
    for i, case in enumerate(cases, 1):
        k_erf = kernel_erf(*case)
        k_daw = kernel_dawson(*case)
        k_num = kernel_quad(*case)

        rel_quad = abs(k_erf - k_num) / max(abs(k_erf), 1e-300)
        abs_quad = abs(k_erf - k_num)
        rel_dawson = abs(k_erf - k_daw) / max(abs(k_erf), 1e-300)

        max_rel_quad = max(max_rel_quad, rel_quad)
        max_abs_quad = max(max_abs_quad, abs_quad)
        max_rel_dawson = max(max_rel_dawson, rel_dawson)

        print(f"\ncase {i}")
        print(f"  erf      = {k_erf:.17g}")
        print(f"  dawson   = {k_daw:.17g}")
        print(f"  quadrat. = {k_num:.17g}")
        print(f"  rel[erf-quad]    = {rel_quad:.3e}")
        print(f"  abs[erf-quad]    = {abs_quad:.3e}")
        print(f"  rel[erf-dawson]  = {rel_dawson:.3e}")

    print("\nsummary")
    print(f"  max relative error, analytic vs quadrature = {max_rel_quad:.3e}")
    print(f"  max absolute error, analytic vs quadrature = {max_abs_quad:.3e}")
    print(f"  max relative error, erf vs Dawson          = {max_rel_dawson:.3e}")


if __name__ == "__main__":
    main()
