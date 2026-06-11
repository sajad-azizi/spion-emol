// GaussLegendre.hpp — nodes and weights for Gauss-Legendre quadrature on [-1, 1].
//
// For integration in cos(theta):
//     int_{-1}^{1} f(x) dx ~= sum_i w_i * f(x_i)
// is exact for polynomials f of degree <= 2n-1.
//
// Setting x = cos(theta) with dx = -sin(theta) d(theta), we can evaluate
//     int_0^{pi} g(theta) sin(theta) d(theta) = int_{-1}^{1} g(acos(x)) dx
//                                            ~= sum_i w_i * g(acos(x_i)).
// The sin(theta) factor is absorbed into dx; callers should NOT multiply by
// sin(theta) again.
//
// Nodes computed by Newton-Raphson on P_n(x), starting from the classical
// Tricomi approximation x_i ~ cos(pi*(i+0.75)/(n+0.5)). Converges in <10
// iterations to ~1e-15.

#pragma once

#include "angular/Legendre.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace preproc::angular {

struct GLNodes {
    int n;
    std::vector<double> x;    // nodes x_i  (sorted ascending)
    std::vector<double> w;    // weights w_i
    std::vector<double> theta;  // acos(x_i), i.e. theta in (0, pi)

    // Build Gauss-Legendre nodes with n points (n >= 1).
    static GLNodes build(int n) {
        if (n < 1) throw std::runtime_error("GLNodes: n must be >= 1");
        GLNodes r;
        r.n = n;
        r.x.assign(n, 0.0);
        r.w.assign(n, 0.0);
        r.theta.assign(n, 0.0);

        const int m = (n + 1) / 2;       // by symmetry we compute half, then mirror
        for (int i = 0; i < m; ++i) {
            // Initial guess (Tricomi)
            double x = std::cos(M_PI * (i + 0.75) / (n + 0.5));
            // Newton on P_n(x) = 0 with derivative n * (P_{n-1}(x) - x*P_n(x)) / (1 - x^2)
            double xn = x;
            for (int it = 0; it < 1000; ++it) {
                const double Pn   = legendre_P(n,     x);
                const double Pnm1 = legendre_P(n - 1, x);
                const double denom = 1.0 - x * x;
                if (denom == 0.0) throw std::runtime_error("GL: denom zero (x=+-1)");
                const double dPn = n * (Pnm1 - x * Pn) / denom;
                xn = x - Pn / dPn;
                if (std::abs(xn - x) < 1e-16) { x = xn; break; }
                x = xn;
            }
            // Weight: w_i = 2 / ((1 - x_i^2) * [P_n'(x_i)]^2)
            const double Pn   = legendre_P(n,     x);
            const double Pnm1 = legendre_P(n - 1, x);
            const double dPn  = n * (Pnm1 - x * Pn) / (1.0 - x * x);
            const double w    = 2.0 / ((1.0 - x * x) * dPn * dPn);

            // Store in ascending order: negative root first, positive last.
            r.x[i]           = -x;
            r.x[n - 1 - i]   =  x;
            r.w[i]           =  w;
            r.w[n - 1 - i]   =  w;
        }
        if (n & 1) {
            // Odd n: the middle node is exactly 0.
            const int mid = n / 2;
            r.x[mid] = 0.0;
            // Recompute its weight directly for safety.
            const double Pnm1 = legendre_P(n - 1, 0.0);
            // For x=0: dP_n(0) = n * (P_{n-1}(0) - 0) / 1 = n * P_{n-1}(0).
            const double dPn = n * Pnm1;
            r.w[mid] = 2.0 / (dPn * dPn);
        }
        for (int i = 0; i < n; ++i) {
            r.theta[i] = std::acos(r.x[i]);
        }
        return r;
    }
};

}  // namespace preproc::angular
