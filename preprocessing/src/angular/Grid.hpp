// Grid.hpp — angular grid and precomputed Y^R table for SCE.
//
// Grid choice (mandated by the user; Lebedev is forbidden for this project):
//     n_theta = Lmax + 1  Gauss-Legendre nodes in cos(theta)    -> exact for P_{2(Lmax+1)-1}
//     n_phi   = 2*(Lmax + 1)  equi-spaced nodes in phi          -> exact for any Fourier
//                                                                 component m with |m| <= Lmax
//
// Integration formula:
//     int_S F(theta, phi) dOmega
//       = int_0^{2pi} dphi int_{-1}^{1} F(acos(x), phi) dx
//       ~= (2*pi / n_phi) * sum_j sum_i w_i * F(theta_i, phi_j).
//
// Exactness: the tensor-product rule is exact for a product P(cos theta) * T(phi)
// with degree(P) <= 2*n_theta - 1 = 2*Lmax + 1 and T any trig polynomial of order
// <= n_phi / 2 - 1 = Lmax (strictly: trapezoidal is exact for any sum of
// e^{i m phi} with |m| <= n_phi - 1, but aliasing makes only |m| <= n_phi/2 safe).
//
// For a bandlimited function Y^R_{l,m} Y^R_{l',m'} with l, l' <= Lmax, the product
// has angular degree <= 2*Lmax in (theta, phi), so this grid yields
// bit-exact orthonormality up to machine epsilon.

#pragma once

#include "angular/GaussLegendre.hpp"
#include "angular/Ylm.hpp"

#include <Eigen/Dense>
#include <vector>

namespace preproc::angular {

struct AngGrid {
    int Lmax;
    int nTheta;          // = Lmax + 1
    int nPhi;            // = 2 * (Lmax + 1)
    GLNodes gl;          // theta-quadrature: gl.x = cos(theta_i), gl.w = weights, gl.theta
    std::vector<double> phi;     // phi_j = 2*pi*j/nPhi, j = 0..nPhi-1

    // Y^R[l*l + l + m][i_theta * nPhi + j_phi]
    //   Table of Y^R_{l,m}(theta_i, phi_j) for all channels (Lmax+1)^2.
    // Use eval_from_table() for dot-product projections.
    std::vector<double> Yreal;

    // Build basic grid (theta, phi nodes + weights). O(Lmax) memory, always safe.
    static AngGrid build_basic(int Lmax) {
        AngGrid g;
        g.Lmax   = Lmax;
        g.nTheta = Lmax + 1;
        g.nPhi   = 2 * (Lmax + 1);
        g.gl     = GLNodes::build(g.nTheta);
        g.phi.resize(g.nPhi);
        for (int j = 0; j < g.nPhi; ++j) g.phi[j] = 2.0 * M_PI * j / g.nPhi;
        return g;
    }

    // Build grid AND precompute the full Y^R table. Memory:
    //     (Lmax+1)^2 * nTheta * nPhi  doubles
    // That is 45 MB at Lmax=40, ~4 GB at Lmax=100, ~130 GB at Lmax=300.
    // Use only for orthonormality/round-trip tests at small Lmax. The SCE
    // projector does NOT need this table; it uses STable (see sce/SCE.hpp).
    static AngGrid build(int Lmax) {
        AngGrid g = build_basic(Lmax);
        const int Nlm = n_channels(Lmax);
        g.Yreal.assign(static_cast<size_t>(Nlm) * g.nTheta * g.nPhi, 0.0);
        for (int l = 0; l <= Lmax; ++l) {
            for (int m = -l; m <= l; ++m) {
                const int k = lm_index(l, m);
                for (int i = 0; i < g.nTheta; ++i) {
                    for (int j = 0; j < g.nPhi; ++j) {
                        const double y = real_Ylm(l, m, g.gl.theta[i], g.phi[j]);
                        g.Yreal[static_cast<size_t>(k) * g.nTheta * g.nPhi
                              + static_cast<size_t>(i) * g.nPhi + j] = y;
                    }
                }
            }
        }
        return g;
    }

    // Index into Yreal.
    inline double Y(int l, int m, int i_theta, int j_phi) const {
        const int k = lm_index(l, m);
        return Yreal[static_cast<size_t>(k) * nTheta * nPhi
                   + static_cast<size_t>(i_theta) * nPhi + j_phi];
    }

    // (theta, phi) position at grid index (i, j).
    inline double theta_i(int i) const { return gl.theta[i]; }
    inline double phi_j  (int j) const { return phi[j]; }

    // Direction vector (sin t cos p, sin t sin p, cos t) on the unit sphere.
    inline Eigen::Vector3d dir(int i, int j) const {
        const double t = gl.theta[i], p = phi[j];
        const double s = std::sin(t);
        return {s * std::cos(p), s * std::sin(p), std::cos(t)};
    }

    // Integrate a function sampled on the grid: F[i*nPhi + j] -> int F dOmega.
    // dOmega weight at (i, j) is gl.w[i] * (2 pi / nPhi).
    inline double integrate(const std::vector<double>& F) const {
        const double dphi = 2.0 * M_PI / nPhi;
        double s = 0.0;
        for (int i = 0; i < nTheta; ++i) {
            const double wi = gl.w[i];
            double row = 0.0;
            for (int j = 0; j < nPhi; ++j) row += F[static_cast<size_t>(i) * nPhi + j];
            s += wi * row;
        }
        return s * dphi;
    }
};

}  // namespace preproc::angular
