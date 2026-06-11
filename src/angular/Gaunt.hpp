// Gaunt.hpp -- real-spherical-harmonic Gaunt coefficients.
//
// Port of version_0/src/gaunt_coefficients.hpp, trimmed to what we actually
// use downstream. The version_0 file carried complex-Gaunt helpers as
// reference implementations; here we keep only what the production
// Potentials / Dipole code calls. The math, conventions and identities are
// UNCHANGED from version_0 -- verified bit-for-bit by the test suite.
//
// Definitions and identities (real Y^R convention matching
// preprocessing/src/angular/Ylm.hpp):
//
//     Y^R_{l, 0}   =  Y_{l,0}
//     Y^R_{l,+m}   =  (1/sqrt(2)) [ (-1)^m Y_{l,m} + Y_{l,-m} ]     (m > 0)
//     Y^R_{l,-m}   =  (i/sqrt(2)) [ Y_{l,-m} - (-1)^m Y_{l,m} ]     (m > 0)
//
// Real Gaunt:  G^R(l1 m1; l2 m2; l3 m3) = integral Y^R_{l1,m1} Y^R_{l2,m2} Y^R_{l3,m3} dOmega.
// For real harmonics (Y^R)* = Y^R so no conjugation in the integrand.
//
// Complex helper (no conjugation convention):
//     gaunt_complex_no_conj = integral Y_{l1,m1} Y_{l2,m2} Y_{l3,m3} dOmega
// with selection rule m1 + m2 + m3 = 0. Formula in terms of 3j symbols
// (GSL's gsl_sf_coupling_3j uses doubled arguments):
//
//     G^C = sqrt((2l1+1)(2l2+1)(2l3+1)/(4 pi))
//           * (l1 l2 l3; 0 0 0) * (l1 l2 l3; m1 m2 m3).
//
// Real Gaunt is obtained by transforming the three real-Ylm legs to the
// complex basis and summing over complex (m1, m2, m3) with m1+m2+m3 = 0.
// The result is guaranteed real (imaginary part is numerical noise).

#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <gsl/gsl_sf_coupling.h>

namespace scatt::angular {

// -----------------------------------------------------------------------------
// Index conversion -- SAME convention as preprocessing/src/angular/Ylm.hpp.
//   idx = l*l + l + m                        (m in [-l, l])
//   (l, m) pairs in order (0,0),(1,-1),(1,0),(1,1),(2,-2)...
// -----------------------------------------------------------------------------
inline int lm_to_idx(int l, int m) { return l * l + l + m; }

inline void idx_to_lm(int idx, int& l, int& m) {
    l = static_cast<int>(std::sqrt(static_cast<double>(idx)));
    if ((l + 1) * (l + 1) <= idx) ++l;   // guard against floor rounding
    m = idx - l * l - l;
}

// Total # of (l,m) channels for L = 0..Lmax.
inline int n_channels(int Lmax) { return (Lmax + 1) * (Lmax + 1); }

// -----------------------------------------------------------------------------
// Phase (-1)^m, integer argument.
// -----------------------------------------------------------------------------
inline double phase(int m) { return (m & 1) ? -1.0 : 1.0; }

// -----------------------------------------------------------------------------
// Complex Gaunt -- NO conjugation: int Y Y Y. Selection rule m1+m2+m3=0.
// -----------------------------------------------------------------------------
inline double gaunt_complex_no_conj(int l1, int m1, int l2, int m2, int l3, int m3) {
    if (m1 + m2 + m3 != 0)                 return 0.0;
    if (std::abs(l1 - l2) > l3)            return 0.0;
    if (l1 + l2 < l3)                      return 0.0;
    if ((l1 + l2 + l3) & 1)                return 0.0;  // parity

    const double w3j_000 = gsl_sf_coupling_3j(2*l1, 2*l2, 2*l3, 0, 0, 0);
    const double w3j_mmm = gsl_sf_coupling_3j(2*l1, 2*l2, 2*l3, 2*m1, 2*m2, 2*m3);
    // Bit-level finite check.  std::isfinite is unsafe under -ffinite-math-only
    // (the SYCL build uses -ffast-math) -- the optimiser may fold it to true.
    // We inspect the IEEE-754 exponent bits directly: NaN and ±Inf both have
    // exponent == 0x7FF, so a non-finite value is exactly that pattern.
    auto is_finite_bits = [](double x) noexcept {
        std::uint64_t bits;
        std::memcpy(&bits, &x, sizeof(bits));
        return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
    };
    if (!is_finite_bits(w3j_000) || !is_finite_bits(w3j_mmm)) return 0.0;

    const double prefactor = std::sqrt((2.0*l1 + 1.0) * (2.0*l2 + 1.0) * (2.0*l3 + 1.0)
                                       / (4.0 * M_PI));
    return prefactor * w3j_000 * w3j_mmm;
}

// -----------------------------------------------------------------------------
// Transformation matrix U(l, mR, mC): Y^complex_mC = sum_mR U Y^real_mR.
// Wikipedia/standard convention (matches version_0/gaunt_coefficients.hpp).
// -----------------------------------------------------------------------------
inline std::complex<double> U_real_to_complex(int /*l*/, int mR, int mC) {
    constexpr double inv_sqrt2 = M_SQRT1_2;

    if (mR == 0)
        return (mC == 0) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);

    if (mR > 0) {
        if (mC == -mR) return {inv_sqrt2, 0.0};
        if (mC ==  mR) return {inv_sqrt2 * phase(mR), 0.0};
        return {0.0, 0.0};
    }
    // mR < 0
    const int p = -mR;
    if (mC == -p) return {0.0,  inv_sqrt2};
    if (mC ==  p) return {0.0, -inv_sqrt2 * phase(p)};
    return {0.0, 0.0};
}

// -----------------------------------------------------------------------------
// Real Gaunt: G^R(l1 mR1; l2 mR2; l3 mR3) = int Y^R Y^R Y^R dOmega.
// -----------------------------------------------------------------------------
inline double gaunt_real(int l1, int mR1, int l2, int mR2, int l3, int mR3) {
    if (std::abs(l1 - l2) > l3 || l1 + l2 < l3) return 0.0;
    if ((l1 + l2 + l3) & 1)                      return 0.0;

    std::complex<double> sum(0.0, 0.0);
    for (int m1 = -l1; m1 <= l1; ++m1) {
        const auto U1 = U_real_to_complex(l1, mR1, m1);
        if (U1.real() == 0.0 && U1.imag() == 0.0) continue;
        for (int m2 = -l2; m2 <= l2; ++m2) {
            const auto U2 = U_real_to_complex(l2, mR2, m2);
            if (U2.real() == 0.0 && U2.imag() == 0.0) continue;
            const int m3 = -(m1 + m2);
            if (m3 < -l3 || m3 > l3) continue;
            const auto U3 = U_real_to_complex(l3, mR3, m3);
            if (U3.real() == 0.0 && U3.imag() == 0.0) continue;
            const double Gc = gaunt_complex_no_conj(l1, m1, l2, m2, l3, m3);
            if (Gc == 0.0) continue;
            sum += U1 * U2 * U3 * Gc;
        }
    }
    return sum.real();
}

// -----------------------------------------------------------------------------
// Evaluate a single real spherical harmonic Y^R_{l,m}(theta, phi). This is
// the EXACT same convention used in preprocessing/src/angular/Ylm.hpp and
// in version_0/src/Potentials.cpp::compute_real_Ylm. The choice matters for
// the V_en multipole expansion below (atom-centered angular factors).
// -----------------------------------------------------------------------------
inline double real_Ylm(int l, int m, double theta, double phi);    // forward decl

// Inline implementation requires sph_legendre_noCS (no-CS normalized P).
// We supply a private routine that mirrors
// preprocessing/src/angular/Legendre.hpp::sph_legendre_noCS.
namespace detail {
inline double sph_legendre_noCS(int l, int m, double theta) {
    if (l < 0 || m < 0 || m > l) return 0.0;
    const double s = std::sin(theta), c = std::cos(theta);
    double S_mm = 1.0 / std::sqrt(4.0 * M_PI);
    for (int k = 1; k <= m; ++k)
        S_mm *= std::sqrt((2.0 * k + 1.0) / (2.0 * k)) * s;
    if (l == m) return S_mm;
    double S_prev = S_mm;
    double S_cur  = std::sqrt(2.0 * m + 3.0) * c * S_mm;
    if (l == m + 1) return S_cur;
    double S_prev_prev = S_prev;
    S_prev = S_cur;
    for (int k = m + 2; k <= l; ++k) {
        const double a = std::sqrt((4.0 * k * k - 1.0) / (static_cast<double>(k) * k - m * m));
        const double b = std::sqrt((static_cast<double>(k - 1) * (k - 1) - m * m)
                                   / (4.0 * (k - 1.0) * (k - 1.0) - 1.0));
        const double S_k = a * (c * S_prev - b * S_prev_prev);
        S_prev_prev = S_prev;
        S_prev = S_k;
    }
    return S_prev;
}
}  // namespace detail

inline double real_Ylm(int l, int m, double theta, double phi) {
    if (l < 0 || std::abs(m) > l) return 0.0;
    const int am = std::abs(m);
    const double S = detail::sph_legendre_noCS(l, am, theta);
    if (m == 0) return S;
    if (m > 0)  return std::sqrt(2.0) * S * std::cos(m * phi);
    // m < 0
    return std::sqrt(2.0) * S * std::sin(am * phi);
}

}  // namespace scatt::angular
