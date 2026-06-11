// Primitive.hpp — Cartesian Gaussian primitive utilities.
//
// A Cartesian primitive Gaussian centered at A with angular function
// x^i y^j z^k (i+j+k = L) has the form
//
//     g(r; alpha, i,j,k, A) = N * (x-Ax)^i (y-Ay)^j (z-Az)^k * exp(-alpha |r-A|^2)
//
// with primitive normalization
//
//     N(alpha,i,j,k) = (2 alpha / pi)^{3/4}
//                      * sqrt( (8 alpha)^{i+j+k} * i! j! k! /
//                              ( (2i)! (2j)! (2k)! ) )
//
// equivalently
//
//     N = (2 alpha / pi)^{3/4}
//         * (4 alpha)^{L/2} / sqrt( (2i-1)!!(2j-1)!!(2k-1)!! )
//
// so that int |g|^2 d^3r = 1.
//
// For two primitives centered at the SAME point with the SAME angular
// function (i,j,k) but different exponents alpha, beta, the overlap is
//
//     <g_alpha | g_beta> = ( 2 * sqrt(alpha*beta) / (alpha+beta) )^{L + 3/2}
//
// independent of (i,j,k) within the shell. We use this to renormalize
// contractions (see Shell::renormalize_contraction).
//
// Units: atomic units throughout. Positions in Bohr.

#pragma once

#include <cmath>
#include <cstdint>

namespace preproc::basis {

// Double factorial (2n-1)!! with the convention (-1)!! = 1, 1!! = 1, 3!! = 3 ...
inline double dfact_odd(int n) {
    // returns (2n-1)!! for n >= 0; (2*0-1)!! = (-1)!! = 1
    double r = 1.0;
    for (int k = 1; k <= n; ++k) r *= (2.0 * k - 1.0);
    return r;
}

// Primitive Cartesian-Gaussian normalization constant.
inline double primitive_norm(double alpha, int i, int j, int k) {
    const double L = static_cast<double>(i + j + k);
    const double prefactor = std::pow(2.0 * alpha / M_PI, 0.75);
    const double num = std::pow(4.0 * alpha, L * 0.5);
    const double den = std::sqrt(dfact_odd(i) * dfact_odd(j) * dfact_odd(k));
    return prefactor * num / den;
}

// Evaluate a single normalized Cartesian primitive at offset dx = r - A.
// dx2 = dx*dx + dy*dy + dz*dz (we pass it in so callers can reuse).
inline double primitive_value(double alpha, int i, int j, int k,
                              double dx, double dy, double dz, double dx2) {
    const double N = primitive_norm(alpha, i, j, k);
    const double xpow = (i == 0) ? 1.0 : (i == 1 ? dx : std::pow(dx, i));
    const double ypow = (j == 0) ? 1.0 : (j == 1 ? dy : std::pow(dy, j));
    const double zpow = (k == 0) ? 1.0 : (k == 1 ? dz : std::pow(dz, k));
    return N * xpow * ypow * zpow * std::exp(-alpha * dx2);
}

// Overlap of two primitives with the SAME angular function (i,j,k) at the
// same center, different exponents alpha, beta.  Needed for contraction
// renormalization. L = i+j+k.
inline double primitive_overlap_same_shell(double alpha, double beta, int L) {
    const double p = 2.0 * std::sqrt(alpha * beta) / (alpha + beta);
    return std::pow(p, static_cast<double>(L) + 1.5);
}

}  // namespace preproc::basis
