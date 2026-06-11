// Ylm.hpp — real spherical harmonics matching version_0's convention.
//
// Convention (must match version_0/src/Potentials.cpp:1119-1136 exactly):
//
//     Y^R_{l, 0}(theta, phi) =  std::sph_legendre(l, 0, theta)
//     Y^R_{l,+m}(theta, phi) =  sqrt(2) * (-1)^m
//                               * std::sph_legendre(l, m, theta)
//                               * cos(m * phi)                           (m > 0)
//     Y^R_{l,-m}(theta, phi) =  sqrt(2) * (-1)^m
//                               * std::sph_legendre(l, m, theta)
//                               * sin(m * phi)                           (m > 0)
//
// std::sph_legendre(l, m, theta) is C++17's normalized spherical associated
// Legendre function for m >= 0: it includes the Condon-Shortley phase
// (-1)^m and normalization sqrt((2l+1)(l-m)!/(4pi(l+m)!)).
//
// Because the (-1)^m in std::sph_legendre cancels with the (-1)^m prefactor,
// the net effect is that Y^R_{l,m} has NO overall Condon-Shortley phase when
// expressed in terms of the bare associated Legendre P_l^m(cos theta) (the
// one without CS). This is the standard "physics real spherical harmonics".
//
// Sanity checks (enforced by tests):
//   - Y^R_{0,0} = 1 / sqrt(4*pi)
//   - Y^R_{1,+1} = sqrt(3/(4*pi)) * sin(theta) * cos(phi)     (= p_x / r)
//   - Y^R_{1, 0} = sqrt(3/(4*pi)) * cos(theta)                (= p_z / r)
//   - Y^R_{1,-1} = sqrt(3/(4*pi)) * sin(theta) * sin(phi)     (= p_y / r)
//   - Orthonormality: int Y^R_{lm} Y^R_{l'm'} dOmega = delta_{ll'} delta_{mm'}

#pragma once

#include "angular/Legendre.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>

namespace preproc::angular {

// Packing (l, m) -> linear index, matching version_0: l*l + l + m.
inline int lm_index(int l, int m) { return l * l + l + m; }

// Inverse: index -> (l, m).
inline void idx_to_lm(int idx, int& l, int& m) {
    l = static_cast<int>(std::floor(std::sqrt(static_cast<double>(idx))));
    if ((l + 1) * (l + 1) <= idx) ++l;   // guard floating-point boundary
    m = idx - l * l - l;
}

// Total number of (l, m) channels for angular cutoff Lmax: (Lmax+1)^2.
inline int n_channels(int Lmax) { return (Lmax + 1) * (Lmax + 1); }

// Evaluate a single real spherical harmonic Y^R_{l,m}(theta, phi).
// theta in [0, pi], phi in [0, 2*pi). Returns 0 if |m| > l.
inline double real_Ylm(int l, int m, double theta, double phi) {
    if (l < 0 || std::abs(m) > l) return 0.0;
    const int am = std::abs(m);
    // S_{l,|m|}: no-CS normalized spherical associated Legendre.
    const double S = sph_legendre_noCS(l, am, theta);
    if (m == 0)      return S;
    if (m >  0)      return std::sqrt(2.0) * S * std::cos( m * phi);
    /* m < 0 */      return std::sqrt(2.0) * S * std::sin(am * phi);
}

}  // namespace preproc::angular
