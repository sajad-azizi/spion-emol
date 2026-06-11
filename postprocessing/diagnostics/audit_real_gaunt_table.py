#!/usr/bin/env python3
"""
Compute the real-Y Gaunt coefficient
    G^R(l1, m1; l2, m2; l3, m3) = int Y^R_{l1,m1} Y^R_{l2,m2} Y^R_{l3,m3} dOmega
by direct numerical angular integration on a Lebedev-like grid (Gauss-
Legendre x trapezoid in phi).  Then for the dipole case (l2, m2) = (1, q),
enumerate every (l_mu, m_mu, l_nu, m_nu) with l_mu, l_nu <= L_max and:
  (a) print the nonzero values G^R(l_mu, m_mu; 1, q; l_nu, m_nu)
  (b) check whether the current scattering code's gate
        m_mu == m_nu + q   (and |l_mu - l_nu| == 1)
      accepts each nonzero one.
"""
import numpy as np
from scipy.special import lpmv
from math import factorial, sqrt, pi


def real_Y(l, m, theta, phi):
    """Real spherical harmonic Y^R_{l,m}(theta, phi).
    Convention: q-map x -> +1, y -> -1, z -> 0; Condon-Shortley dropped
    in the radial Legendre."""
    am = abs(m)
    # Normalised assoc Legendre P_l^|m| (no Condon-Shortley phase here;
    # we put it in via the matching overall sign convention).
    ct = np.cos(theta)
    P = lpmv(am, l, ct)             # scipy: includes (-1)^m (Condon-Shortley)
    # Strip CS phase to match our convention:
    P = P * ((-1) ** am)
    norm = sqrt((2*l + 1) / (4*pi) * factorial(l - am) / factorial(l + am))
    if m == 0:
        return norm * P
    if m > 0:
        return sqrt(2.0) * norm * P * np.cos(m * phi)
    return sqrt(2.0) * norm * P * np.sin(am * phi)


def real_gaunt(l1, m1, l2, m2, l3, m3, n_th=80, n_ph=160):
    """Numerical integration of the triple product on a (theta, phi) grid."""
    # Gauss-Legendre over cos(theta) ∈ [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_th)
    th = np.arccos(x)
    sin_th = np.sin(th)
    # Trapezoid in phi
    ph = np.linspace(0, 2*pi, n_ph, endpoint=False)
    dph = 2*pi / n_ph

    Y1 = real_Y(l1, m1, th[:, None], ph[None, :])
    Y2 = real_Y(l2, m2, th[:, None], ph[None, :])
    Y3 = real_Y(l3, m3, th[:, None], ph[None, :])
    # int dOmega = sum_th sum_ph Y1*Y2*Y3 * (w[th] / sin_th[th]) * dph * sin_th[th]
    # but in cos-theta basis the GL weight already absorbs sin(theta) dtheta:
    #   int f(cos th) dcos th = sum w * f
    # so:
    return float(np.sum(Y1 * Y2 * Y3 * (w * sin_th)[:, None] * dph
                        / sin_th[:, None]))   # cancel: w * sin / sin = w


def main():
    L = 3   # check up to f-waves -- enough to catch any pathological zeros

    print("Real-Y Gaunt G^R(l_mu, m_mu; 1, q; l_nu, m_nu)")
    print("Gating in code (DipoleMatrixElement.cpp:205):")
    print("  if (mmu != mnu + q)  continue;")
    print("Below: every nonzero G^R, with whether the gate KEEPS it.\n")

    for q in (-1, 0, +1):
        print(f"=== q = {q:+d}  ({'y' if q==-1 else 'z' if q==0 else 'x'} polarization) ===")
        kept_total   = 0.0
        dropped_total = 0.0
        n_kept = 0
        n_dropped = 0
        for l_mu in range(L + 1):
            for m_mu in range(-l_mu, l_mu + 1):
                for l_nu in range(L + 1):
                    if abs(l_mu - l_nu) != 1: continue
                    for m_nu in range(-l_nu, l_nu + 1):
                        G = real_gaunt(l_mu, m_mu, 1, q, l_nu, m_nu)
                        if abs(G) < 1e-10:
                            continue
                        gated_in = (m_mu == m_nu + q)
                        tag = "KEPT" if gated_in else "DROPPED"
                        marker = " " if gated_in else "*"
                        print(f"{marker} G^R(l_mu={l_mu}, m_mu={m_mu:+d}; 1,{q:+d};"
                              f" l_nu={l_nu}, m_nu={m_nu:+d}) = {G:+.6f}  [{tag}]")
                        if gated_in:
                            kept_total += abs(G)
                            n_kept += 1
                        else:
                            dropped_total += abs(G)
                            n_dropped += 1
        print(f"   summary q={q:+d}: KEPT  {n_kept} couplings, sum|G| = {kept_total:.4f}")
        print(f"            DROPPED {n_dropped} couplings, sum|G| = {dropped_total:.4f}\n")


if __name__ == "__main__":
    main()
