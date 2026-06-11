// Synthesize.hpp — evaluate an SCE-expanded function F at arbitrary 3D points.
//
// Given F^R_{l,m}(r_k) on a uniform radial grid (Nlm rows, Nr cols) and an
// origin, reconstruct
//     F(r_vec) = sum_{l,m}  F^R_{l,m}(r)  Y^R_{l,m}(theta, phi)
//
// where (r, theta, phi) are spherical coordinates of (r_vec - origin).
//
// Radial interpolation: 4-point centered Lagrange (cubic). For a uniform
// grid this is O(h^4) locally, and F_{l,m}(r) is smooth (no cusps for V_H,
// smooth densities, and well-behaved orbitals), so this is well-matched.
//
// For points where r < r_min or r > r_max, returns 0 (caller chooses its
// own far-field extrapolation).

#pragma once

#include "angular/Ylm.hpp"
#include "sce/RadialGrid.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace preproc::sce {

// 4-point Lagrange interpolation of F(r) given F on a uniform grid.
// Returns F(r_star) using the 4 nearest grid points.
inline double lagrange4_uniform(const double* __restrict__ Fcol_stride_Nlm,
                                int Nlm, int Nr, double rmin, double dr,
                                int channel, double r_star)
{
    // Find the 4 points surrounding r_star. Use the stencil {k-1, k, k+1, k+2}
    // where k = floor((r_star - rmin)/dr), clamped so the stencil is in-range.
    double x = (r_star - rmin) / dr;
    int k = static_cast<int>(std::floor(x));
    if (k < 1) k = 1;
    if (k > Nr - 3) k = Nr - 3;
    // Local coordinate s = (r_star - r_k) / dr; s in [-1, Nr-1-k] typically in [0,1]
    const double s = (r_star - (rmin + k * dr)) / dr;
    // Lagrange basis L_i(s) for nodes at (-1, 0, 1, 2)
    //   L_{-1}(s) = -s(s-1)(s-2)/6
    //   L_0(s)    =  (s+1)(s-1)(s-2)/2
    //   L_1(s)    = -(s+1)s(s-2)/2
    //   L_2(s)    =  (s+1)s(s-1)/6
    const double Lm1 = -s * (s - 1.0) * (s - 2.0) / 6.0;
    const double L0  =  (s + 1.0) * (s - 1.0) * (s - 2.0) * 0.5;
    const double L1  = -(s + 1.0) * s * (s - 2.0) * 0.5;
    const double L2  =  (s + 1.0) * s * (s - 1.0) / 6.0;
    // Fcol stride is Nlm (column-major matrix with Nlm rows).
    const double f_m1 = Fcol_stride_Nlm[(static_cast<size_t>(k - 1)) * Nlm + channel];
    const double f_0  = Fcol_stride_Nlm[(static_cast<size_t>(k    )) * Nlm + channel];
    const double f_1  = Fcol_stride_Nlm[(static_cast<size_t>(k + 1)) * Nlm + channel];
    const double f_2  = Fcol_stride_Nlm[(static_cast<size_t>(k + 2)) * Nlm + channel];
    return Lm1 * f_m1 + L0 * f_0 + L1 * f_1 + L2 * f_2;
}

// Synthesize F at an arbitrary 3D point P (absolute coordinates).
// Uses a Lmax given explicitly (truncation cutoff); must satisfy
// (Lmax+1)^2 <= Flm.rows().
inline double synthesize_at(const Eigen::MatrixXd& Flm,
                            const RadialGrid& r_grid,
                            const Eigen::Vector3d& origin,
                            int Lmax,
                            const Eigen::Vector3d& P)
{
    const Eigen::Vector3d d = P - origin;
    const double r = d.norm();
    if (r < r_grid.rmin - 1e-14 || r > r_grid.rmin + (r_grid.N - 1) * r_grid.dr + 1e-14)
        return 0.0;
    double theta = 0.0, phi = 0.0;
    if (r > 1e-14) {
        theta = std::acos(std::clamp(d.z() / r, -1.0, 1.0));
        phi   = std::atan2(d.y(), d.x());
    }

    const int Nlm = angular::n_channels(Lmax);
    const int Nr  = r_grid.N;
    const double* Fdata = Flm.data();   // column-major with stride Nlm_rows
    const int Nlm_rows  = static_cast<int>(Flm.rows());

    // Sum F^R_{l,m}(r) * Y^R_{l,m}(theta, phi). We call real_Ylm per channel
    // for clarity; could be optimized with an lm recurrence if this becomes
    // a hot path.
    double s = 0.0;
    for (int l = 0; l <= Lmax; ++l) {
        for (int m = -l; m <= l; ++m) {
            const int ch = l * l + l + m;
            const double F_r = lagrange4_uniform(Fdata, Nlm_rows, Nr,
                                                 r_grid.rmin, r_grid.dr,
                                                 ch, r);
            s += F_r * angular::real_Ylm(l, m, theta, phi);
        }
    }
    (void)Nlm;
    return s;
}

}  // namespace preproc::sce
