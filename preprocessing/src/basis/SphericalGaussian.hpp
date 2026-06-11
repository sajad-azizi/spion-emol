// SphericalGaussian.hpp — normalization constants and overlap formulas for
// spherical real-SH Gaussian primitives.
//
// A spherical Gaussian primitive with angular momentum l (and a specific m
// in [−l, l]) and exponent alpha is
//
//     g^sph_{l,m}(r; alpha, A) = N_sph(alpha, l) * R_{l,m}(r - A) * exp(-alpha |r-A|^2)
//
// where R_{l,m}(r) = r^l Y^R_{l,m}(theta, phi)  (regular solid harmonic),
// with Y^R the real spherical harmonic from angular/Ylm.hpp.
//
// Unit normalization  int |g^sph|^2 d^3r = 1  requires
//
//     N_sph(alpha, l)^2
//         = 2^{l+2} * (2 alpha)^{l + 3/2}
//           / ( sqrt(pi) * (2l+1)!! )
//
// Derivation:  < g^sph | g^sph > = N_sph^2 * int R^2 e^{-2 alpha r^2} d^3 r
//                                = N_sph^2 * [1] * int_0^inf r^{2l+2} e^{-2 alpha r^2} dr
// where the angular integral is 1 because int |Y^R_{l,m}|^2 dOmega = 1.
// The radial integral is (1/2) * Gamma((2l+3)/2) / (2 alpha)^{l+3/2}
// with Gamma((2l+3)/2) = sqrt(pi) * (2l+1)!! / 2^{l+1}.
//
// Same-shell same-center overlap (two exponents alpha, beta, same l,m):
//
//     < g^sph_{l,m,alpha} | g^sph_{l,m,beta} >
//         = ( 2 sqrt(alpha * beta) / (alpha + beta) )^{l + 3/2}.
//
// (identical closed form to the Cartesian same-shell overlap in
// basis/Primitive.hpp, because that formula is driven by L = i+j+k = l and
// the angular integrals are unit-normalized on both sides). So contraction
// renormalization uses the SAME helper `primitive_overlap_same_shell` as
// the s/p path -- we don't need to duplicate that.

#pragma once

#include "basis/Primitive.hpp"

#include <cmath>

namespace preproc::basis {

// Double factorial (2l+1)!!. For l=0: 1. For l=1: 3. For l=2: 15. Etc.
inline double dfact_pos_odd(int l) {
    double r = 1.0;
    for (int k = 1; k <= l; ++k) r *= (2.0 * k + 1.0);
    return r;
}

// N_sph(alpha, l) — unit-norm constant for the spherical primitive.
// Stable form:
//   N_sph^2 = 2^{2l+7/2} * alpha^{l+3/2} / ( sqrt(pi) * (2l+1)!! )
inline double spherical_primitive_norm(double alpha, int l) {
    // exponent of 2 is 2l + 7/2, NOT (2l + 7)/2.
    const double pow2      = std::pow(2.0, 2.0 * l + 1.5 + 2.0);   // 2^{2l+7/2}
    const double pow_alpha = std::pow(alpha, static_cast<double>(l) + 1.5);
    return std::sqrt(pow2 * pow_alpha / (std::sqrt(M_PI) * dfact_pos_odd(l)));
}

}  // namespace preproc::basis
