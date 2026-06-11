// Legendre.hpp — plain and normalized associated Legendre functions via
// stable upward recurrence. Replaces std::legendre / std::sph_legendre
// which AppleClang's libc++ does not ship.
//
// Two quantities are provided:
//
// 1. legendre_P(n, x):  P_n(x), the degree-n Legendre polynomial.
//       P_0(x) = 1, P_1(x) = x,
//       n P_n(x) = (2n-1) x P_{n-1}(x) - (n-1) P_{n-2}(x).
//
// 2. sph_legendre_noCS(l, m, theta):  the non-Condon-Shortley normalized
//    associated Legendre function S_{l,m}(theta), defined so that
//
//       S_{l,m}(theta) = sqrt( (2l+1)/(4 pi) * (l-m)!/(l+m)! ) * P_l^m(cos theta)
//
//    with P_l^m the no-CS associated Legendre polynomial
//    (i.e. P_l^m(x) = (1-x^2)^{m/2} d^m P_l(x)/dx^m; positive for x near 1).
//
//    Equivalently: std::sph_legendre(l, m, theta) = (-1)^m * S_{l,m}(theta).
//
//    Recurrence, standard and numerically stable (Holmes & Featherstone 2002):
//       S_{0,0}              = 1 / sqrt(4 pi)
//       S_{m,m}(theta)       = sqrt( (2m+1)/(2m) ) * sin(theta) * S_{m-1,m-1}(theta)
//       S_{m+1,m}(theta)     = sqrt( 2m+3 ) * cos(theta) * S_{m,m}(theta)
//       S_{l,m}(theta) for l > m+1:
//           a = sqrt( (4 l^2 - 1) / (l^2 - m^2) )
//           b = sqrt( ((l-1)^2 - m^2) / (4 (l-1)^2 - 1) )
//           S_{l,m}  = a * ( cos(theta) * S_{l-1,m}  -  b * S_{l-2,m} )
//
// Tests verify S_{0,0}=1/sqrt(4pi), S_{1,0}(theta)=sqrt(3/(4pi)) cos(theta),
// S_{1,1}(theta)=sqrt(3/(8pi)) sin(theta) (positive!), S_{2,2}=sqrt(15/(32pi)) sin^2(theta).

#pragma once

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace preproc::angular {

// P_n(x) via three-term recurrence. Stable for x in [-1, 1] and all n.
inline double legendre_P(int n, double x) {
    if (n < 0) throw std::runtime_error("legendre_P: n < 0");
    if (n == 0) return 1.0;
    if (n == 1) return x;
    double Pkm2 = 1.0;     // P_{k-2}
    double Pkm1 = x;       // P_{k-1}
    double Pk   = 0.0;
    for (int k = 2; k <= n; ++k) {
        Pk   = ((2.0 * k - 1.0) * x * Pkm1 - (k - 1.0) * Pkm2) / k;
        Pkm2 = Pkm1;
        Pkm1 = Pk;
    }
    return Pkm1;
}

// d/dx P_n(x) via the identity
//     (1 - x^2) P_n'(x) = n * ( P_{n-1}(x) - x P_n(x) )
// valid for |x| < 1. Caller must not pass x = +-1.
inline double legendre_Pderiv(int n, double x) {
    if (n == 0) return 0.0;
    const double Pn   = legendre_P(n, x);
    const double Pnm1 = legendre_P(n - 1, x);
    const double denom = 1.0 - x * x;
    return n * (Pnm1 - x * Pn) / denom;
}

// Non-Condon-Shortley normalized spherical associated Legendre S_{l,m}(theta)
// for 0 <= m <= l. Returns 0 if m > l.
inline double sph_legendre_noCS(int l, int m, double theta) {
    if (l < 0 || m < 0)  return 0.0;
    if (m > l)           return 0.0;
    const double s = std::sin(theta);
    const double c = std::cos(theta);

    // S_{0,0} = 1 / sqrt(4 pi)
    double S_mm = 1.0 / std::sqrt(4.0 * M_PI);
    // Build the diagonal up to S_{m,m}
    for (int k = 1; k <= m; ++k) {
        S_mm *= std::sqrt((2.0 * k + 1.0) / (2.0 * k)) * s;
    }
    if (l == m) return S_mm;

    // One step up: S_{m+1, m} = sqrt(2m + 3) * cos(theta) * S_{m,m}
    double S_lm1 = S_mm;                                  // S_{l-1, m}, starts at l-1 = m
    double S_l   = std::sqrt(2.0 * m + 3.0) * c * S_mm;   // S_{m+1, m}
    if (l == m + 1) return S_l;

    // Ascend: for k from m+2 up to l
    //   a = sqrt((4 k^2 - 1) / (k^2 - m^2))
    //   b = sqrt(((k-1)^2 - m^2) / (4 (k-1)^2 - 1))
    //   S_k = a * (cos(theta) * S_{k-1} - b * S_{k-2})
    double S_lm2 = S_lm1;     // S_{k-2}  (= S_{m, m} initially, then shifts)
    S_lm1        = S_l;       // S_{k-1}  (= S_{m+1, m})
    for (int k = m + 2; k <= l; ++k) {
        const double a = std::sqrt((4.0 * k * k - 1.0) / (static_cast<double>(k) * k - m * m));
        const double b = std::sqrt((static_cast<double>(k - 1) * (k - 1) - m * m)
                                 / (4.0 * (k - 1.0) * (k - 1.0) - 1.0));
        const double S_k = a * (c * S_lm1 - b * S_lm2);
        S_lm2 = S_lm1;
        S_lm1 = S_k;
    }
    return S_lm1;
}

}  // namespace preproc::angular
