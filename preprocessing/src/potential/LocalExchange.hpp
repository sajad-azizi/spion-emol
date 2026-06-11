// LocalExchange.hpp — local exchange potential (KS-LDA / Slater Xa).
//
// This is NOT the Hartree-Fock exchange operator (which is non-local and
// lives in the main scattering code). It is an approximate local
// potential sometimes useful as a starting point or cross-check:
//
//     V_x^{Slater}(r)  =  -( 3 / pi )^{1/3}  *  rho(r)^{1/3}          (alpha = 1)
//     V_x^{KS-LDA}(r)  =  -( 3/2 ) * ( 3 rho(r) / pi )^{1/3}          (alpha = 2/3)
//
// Both are pointwise local density functionals. We build them as a regular
// 3D function F(r_vec) = f(rho(r_vec)) and SCE-project in the same way we
// do with V_en, V_H, or rho itself.
//
// The closed-shell total density rho is passed as a callable; the returned
// value is the "alpha spin + beta spin" total density. For the expressions
// above that is exactly what enters.

#pragma once

#include "sce/SCE.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <functional>

namespace preproc::potential {

enum class LDAAlpha { Slater_1_0, KohnSham_2_3 };

inline preproc::sce::F3D make_V_x_local(
    const preproc::sce::F3D& rho_of_r,
    LDAAlpha which)
{
    const double cbrt3_over_pi = std::cbrt(3.0 / M_PI);  // (3/pi)^{1/3}
    return [rho_of_r, cbrt3_over_pi, which](const Eigen::Vector3d& r) {
        const double rho = std::max(0.0, rho_of_r(r));
        const double rho_cbrt = std::cbrt(rho);
        switch (which) {
            case LDAAlpha::Slater_1_0:
                return -cbrt3_over_pi * rho_cbrt;
            case LDAAlpha::KohnSham_2_3:
            default:
                return -1.5 * cbrt3_over_pi * rho_cbrt;
        }
    };
}

}  // namespace preproc::potential
