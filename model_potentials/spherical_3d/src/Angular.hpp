// Angular.hpp -- real spherical harmonics Y^R_{l,m}(theta, phi) and the
// real-Gaunt coefficient G^R(l1 m1; l2 m2; l3 m3) needed for the
// dipole matrix element.  Conventions match
// static_exchangeHF/src/angular/Gaunt.hpp byte for byte:
//
//   Y^R_{l, 0}(θ,φ)   = Y_{l,0}(θ,φ)             (real)
//   Y^R_{l,+m}(θ,φ)   = √2 · N_l|m| · P_l^|m|(cosθ) · cos(m φ)     (m > 0)
//   Y^R_{l,-m}(θ,φ)   = √2 · N_l|m| · P_l^|m|(cosθ) · sin(|m| φ)   (m > 0)
//
//   q-map for the dipole:   x ↔ q=+1,  y ↔ q=-1,  z ↔ q=0.
//
// Real Gaunt is computed by transforming the three real-Y legs to the
// complex basis and summing complex Gaunt G^C with selection rule
// m1+m2+m3=0; the result is real (imaginary part is roundoff).
#pragma once

#include "Common.hpp"

#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>

namespace ang3d_init {
struct DisableGslAbort {
    DisableGslAbort() { gsl_set_error_handler_off(); }
};
inline DisableGslAbort _disable_gsl_abort;
}  // namespace ang3d_init

namespace ang3d {

// (l, m) <-> packed index, m in [-l, +l].
inline int  lm_to_idx(int l, int m) { return l * l + l + m; }
inline void idx_to_lm(int idx, int& l, int& m) {
    l = static_cast<int>(std::sqrt(static_cast<double>(idx)));
    if ((l + 1) * (l + 1) <= idx) ++l;
    m = idx - l * l - l;
}
inline int  n_channels(int l_max) { return (l_max + 1) * (l_max + 1); }

inline double phase(int m) { return (m & 1) ? -1.0 : 1.0; }

// --- Normalized associated Legendre P_l^m(cosθ) without the
//     Condon-Shortley phase, divided by √(4π) so that it's the spherical
//     harmonic radial factor (m >= 0 only here). ---
inline double sph_legendre_noCS(int l, int m, double theta) {
    if (l < 0 || m < 0 || m > l) return 0.0;
    const double s = std::sin(theta);
    const double c = std::cos(theta);
    double S_mm = 1.0 / std::sqrt(4.0 * M_PI);
    for (int k = 1; k <= m; ++k) {
        S_mm *= std::sqrt((2.0 * k + 1.0) / (2.0 * k)) * s;
    }
    if (l == m) return S_mm;
    double S_prev = S_mm;
    double S_cur  = std::sqrt(2.0 * m + 3.0) * c * S_mm;
    if (l == m + 1) return S_cur;
    double S_pp = S_prev;
    S_prev = S_cur;
    for (int k = m + 2; k <= l; ++k) {
        const double a = std::sqrt(
            (4.0 * k * k - 1.0) /
            (static_cast<double>(k) * k - m * m));
        const double b = std::sqrt(
            (static_cast<double>(k - 1) * (k - 1) - m * m) /
            (4.0 * (k - 1.0) * (k - 1.0) - 1.0));
        const double S_k = a * (c * S_prev - b * S_pp);
        S_pp   = S_prev;
        S_prev = S_k;
    }
    return S_prev;
}

// --- Real Y^R_{l,m}(theta, phi). ---
inline double real_Ylm(int l, int m, double theta, double phi) {
    if (l < 0 || std::abs(m) > l) return 0.0;
    const int    am = std::abs(m);
    const double S  = sph_legendre_noCS(l, am, theta);
    if (m == 0) return S;
    if (m > 0)  return std::sqrt(2.0) * S * std::cos(m  * phi);
    /*m < 0*/   return std::sqrt(2.0) * S * std::sin(am * phi);
}

// --- Complex Gaunt int Y Y Y dΩ (no conjugation). ---
inline double gaunt_complex_no_conj(int l1, int m1, int l2, int m2,
                                    int l3, int m3) {
    if (m1 + m2 + m3 != 0)        return 0.0;
    if (std::abs(l1 - l2) > l3)   return 0.0;
    if (l1 + l2 < l3)             return 0.0;
    if ((l1 + l2 + l3) & 1)       return 0.0;

    const double w3j_000 =
        gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, 0, 0, 0);
    const double w3j_mmm =
        gsl_sf_coupling_3j(2 * l1, 2 * l2, 2 * l3, 2 * m1, 2 * m2, 2 * m3);
    if (!std::isfinite(w3j_000) || !std::isfinite(w3j_mmm)) return 0.0;

    const double prefactor = std::sqrt(
        (2.0 * l1 + 1.0) * (2.0 * l2 + 1.0) * (2.0 * l3 + 1.0) /
        (4.0 * M_PI));
    return prefactor * w3j_000 * w3j_mmm;
}

// --- Transformation U(l, mR, mC):  Y^complex_{l,mC} = sum_{mR} U Y^R_{l,mR}.
//     Same convention as Gaunt.hpp. ---
inline dcompx U_real_to_complex(int /*l*/, int mR, int mC) {
    constexpr double inv_sqrt2 = 0.7071067811865476;
    if (mR == 0)
        return (mC == 0) ? dcompx(1.0, 0.0) : dcompx(0.0, 0.0);
    if (mR > 0) {
        if (mC == -mR) return {inv_sqrt2, 0.0};
        if (mC ==  mR) return {inv_sqrt2 * phase(mR), 0.0};
        return {0.0, 0.0};
    }
    const int p = -mR;
    if (mC == -p) return {0.0,  inv_sqrt2};
    if (mC ==  p) return {0.0, -inv_sqrt2 * phase(p)};
    return {0.0, 0.0};
}

// --- Real Gaunt G^R(l1 mR1; l2 mR2; l3 mR3). ---
inline double gaunt_real(int l1, int mR1, int l2, int mR2,
                         int l3, int mR3) {
    if (std::abs(l1 - l2) > l3 || l1 + l2 < l3) return 0.0;
    if ((l1 + l2 + l3) & 1)                     return 0.0;

    dcompx sum(0.0, 0.0);
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

// --- Gauss-Legendre nodes/weights on [-1, 1] (Newton on Legendre). ---
inline void gauss_legendre(int N, std::vector<double>& x,
                           std::vector<double>& w) {
    x.assign(N, 0.0);
    w.assign(N, 0.0);
    constexpr double eps = 3e-16;
    const int m = (N + 1) / 2;
    for (int i = 1; i <= m; ++i) {
        double z  = std::cos(M_PI * (i - 0.25) / (N + 0.5));
        double z1 = 0.0;
        double pp = 0.0;
        for (int it = 0; it < 100; ++it) {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 1; j <= N; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            pp = N * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z  = z1 - p1 / pp;
            if (std::fabs(z - z1) < eps) break;
        }
        x[i - 1] = -z;
        x[N - i] =  z;
        w[i - 1] = 2.0 / ((1.0 - z * z) * pp * pp);
        w[N - i] = w[i - 1];
    }
}

// --- Spherical Bessel functions via GSL (libc++ AppleClang lacks
//     std::sph_bessel / std::sph_neumann). ---
inline double sph_jl(int l, double x) {
    if (l < 0)    return 0.0;
    if (x <= 0.0) return (l == 0) ? 1.0 : 0.0;
    gsl_sf_result r;
    if (gsl_sf_bessel_jl_e(l, x, &r) == GSL_SUCCESS) return r.val;
    return 0.0;
}
inline double sph_yl(int l, double x) {
    if (l < 0)    return 0.0;
    if (x <= 0.0) return -std::numeric_limits<double>::infinity();
    gsl_sf_result r;
    if (gsl_sf_bessel_yl_e(l, x, &r) == GSL_SUCCESS) return r.val;
    return 0.0;
}

// --- 3D Riccati-Bessel functions:  S_l(x) = x j_l(x),  C_l(x) = -x y_l(x).
//     Asymptotic forms:  S_l(x) -> sin(x - lπ/2),  C_l(x) -> cos(x - lπ/2).
//     With these, K = B A^{-1} for chi_l(r) ~ A·S_l(kr) + B·C_l(kr). ---
inline double riccati_S(int l, double x) {
    if (l < 0) return 0.0;
    return x * sph_jl(l, x);
}
inline double riccati_C(int l, double x) {
    if (l < 0) return 0.0;
    return -x * sph_yl(l, x);
}

// --- Spherical Bessel zeros via Newton iteration; only needed for
//     selecting a fitting window past the centrifugal barrier. ---
inline double spherical_bessel_zero(int l, int n) {
    // McMahon-type asymptotic seed: x_{l,n} ≈ (n + l/2) π for large n.
    if (n < 1) n = 1;
    double x      = (n + 0.5 * l) * M_PI;
    const double x_floor = 0.5 + 0.5 * l;     // never go below the first zero
    for (int it = 0; it < 100; ++it) {
        if (x < x_floor) x = x_floor;          // GSL domain guard
        const double f  = sph_jl(l, x);
        const double fp = (l == 0)
            ? -sph_jl(1, x)
            :  sph_jl(l - 1, x) - (l + 1.0) / x * sph_jl(l, x);
        if (std::fabs(fp) < 1e-30) break;
        double dx = f / fp;
        // Damp wild Newton steps that overshoot to negative or huge values.
        if (std::fabs(dx) > 0.5 * x) dx = (dx > 0 ? 1.0 : -1.0) * 0.5 * x;
        x -= dx;
        if (x < x_floor) x = x_floor;
        if (std::fabs(dx) < 1e-13) break;
    }
    return x;
}

}  // namespace ang3d
