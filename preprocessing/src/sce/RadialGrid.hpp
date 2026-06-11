// RadialGrid.hpp — uniform radial grid used by the SCE.
//
// Grid points: r_k = r_min + k * dr, for k = 0 .. N-1.
// (Version_0 uses r_min = 0, dr = 0.01, N = 10001 for r_max = 100.)
//
// Provides the composite Simpson weight vector s_k such that
//     int_{r_min}^{r_max} f(r) dr  ~=  sum_k s_k * f(r_k)
// which is exact for piecewise cubic f on panels {2i, 2i+1, 2i+2}.
// Requires N >= 3 and odd; if N is even we fall back to trapezoidal on the
// last panel so the formula still integrates the full range correctly.
//
// For any polynomial of degree <= 3 and spacing h = dr, Simpson's
// rule has error O(h^5 * f^(4)). Plenty for our use (dr = 0.01, smooth
// radial functions).

#pragma once

#include <cassert>
#include <stdexcept>
#include <vector>

namespace preproc::sce {

struct RadialGrid {
    double rmin = 0.0;
    double dr   = 0.01;
    int    N    = 0;
    std::vector<double> r;     // r_k
    std::vector<double> swt;   // Simpson weight s_k (absolute, i.e. includes dr factor)

    static RadialGrid build(double rmin, double dr, int N) {
        if (N < 3)  throw std::runtime_error("RadialGrid: N must be >= 3");
        if (dr <= 0) throw std::runtime_error("RadialGrid: dr must be > 0");
        RadialGrid g;
        g.rmin = rmin; g.dr = dr; g.N = N;
        g.r.assign(N, 0.0);
        g.swt.assign(N, 0.0);
        for (int k = 0; k < N; ++k) g.r[k] = rmin + k * dr;

        // Composite Simpson on an odd number of points (N-1 even intervals).
        // For N even, apply Simpson on first N-1 points and trapezoid on the last interval.
        const bool N_odd = (N & 1);
        const int  N_simp = N_odd ? N : (N - 1);   // # points covered by Simpson
        for (int k = 0; k < N_simp; ++k) {
            if (k == 0 || k == N_simp - 1) g.swt[k] = dr / 3.0;
            else if (k & 1)                g.swt[k] = 4.0 * dr / 3.0;
            else                           g.swt[k] = 2.0 * dr / 3.0;
        }
        if (!N_odd) {
            // Trapezoid over the last interval [r_{N-2}, r_{N-1}].
            g.swt[N - 2] += 0.5 * dr;
            g.swt[N - 1] += 0.5 * dr;
        }
        return g;
    }

    double integrate(const std::vector<double>& f) const {
        double s = 0.0;
        for (int k = 0; k < N; ++k) s += swt[k] * f[k];
        return s;
    }

    // Cumulative integral  I[k] = int_{r_0}^{r_k} f(r) dr  for k = 0..N-1.
    //
    // Uses a 3-point Lagrange-fit between r_k and r_{k+1} with the forward
    // stencil (f[k], f[k+1], f[k+2]):
    //
    //     I[k+1] - I[k] = (h/12) * ( 5 f[k] + 8 f[k+1] - f[k+2] )
    //
    // which integrates the interpolating quadratic exactly and is O(h^5)
    // locally, O(h^4) globally. Summing two consecutive forward steps
    // reproduces Simpson's rule exactly, so this is "cumulative Simpson"
    // at every grid point including odd indices, not just even ones.
    //
    // At the very last interval (k = N-2) we switch to the backward stencil
    // (f[N-3], f[N-2], f[N-1]) which has the same order of accuracy:
    //
    //     I[N-1] - I[N-2] = (h/12) * (-f[N-3] + 8 f[N-2] + 5 f[N-1])
    //
    // Requires N >= 3.
    //
    // Input:  f.size() == N
    // Output: I.size() == N, with I[0] = 0.
    void cumulative_integrate(const std::vector<double>& f,
                              std::vector<double>& I) const {
        if (static_cast<int>(f.size()) != N)
            throw std::runtime_error("cumulative_integrate: size mismatch");
        if (N < 3)
            throw std::runtime_error("cumulative_integrate: need N >= 3");
        I.assign(N, 0.0);
        const double h12 = dr / 12.0;
        for (int k = 0; k < N - 1; ++k) {
            if (k + 2 < N) {
                I[k + 1] = I[k] + h12 * (5.0 * f[k] + 8.0 * f[k + 1] - f[k + 2]);
            } else {
                // k == N - 2: backward stencil
                I[k + 1] = I[k] + h12 * (-f[k - 1] + 8.0 * f[k] + 5.0 * f[k + 1]);
            }
        }
    }
};

}  // namespace preproc::sce
