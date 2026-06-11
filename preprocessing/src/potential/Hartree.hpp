// Hartree.hpp — Hartree electron-electron potential V_H from a density
// represented by its SCE coefficients rho_{lm}(r), via the radial Poisson
// equation (path A, production).
//
// Derivation. Coulomb integral plus Laplace expansion of 1/|r - r'|:
//
//     V_H(r_vec) = int rho(r'_vec) / |r - r'| d^3 r'
//                = sum_{l,m}  (4 pi / (2l+1)) Y^R_{l,m}(n_hat)
//                             * int_0^inf  r'^2 dr'
//                                   (r_<^l / r_>^{l+1}) * rho_{l,m}(r').
//
// Splitting the r' integral at r:
//
//     V_H^R_{l,m}(r) = (4 pi / (2l+1))
//                      * [ (1/r^{l+1}) * int_0^r  rho_{l,m}(r') r'^{l+2} dr'
//                        +      r^l   * int_r^inf rho_{l,m}(r') r'^{1-l} dr' ]
//
// For smooth rho, both integrands are smooth:
//   - The first is r'^{l+2} * rho_{l,m}(r')  ~ r'^{2l+2}   at r'->0  (since
//     rho_{l,m} ~ r'^l near the origin by angular symmetry).
//   - The second is r'^{1-l} * rho_{l,m}(r') ~ r'         at r'->0
//     (same reason).  Formally 0^{1-l} is singular for l >= 2; we set
//     the integrand to 0 at r' = 0 since the product limit is 0.
//
// Implementation:
//   - For each channel (l, m), build the two integrands at all N_r grid
//     points.
//   - Build running cumulative integrals I_in(r) = int_0^r ... dr' and
//     I_out(r) = int_r^{Rmax} ... dr'  using the O(h^4) cumulative integrator
//     from RadialGrid.
//   - Combine into V_H^R_{l,m}(r). For r = 0, V_H(0) = (4 pi / (2l+1)) *
//     [0 + 0]  is undefined for l >= 1 (Coulomb near-nucleus behavior at
//     origin); but rho_{l,m}(0) = 0 for l >= 1 so the limit is finite and
//     we set V_H(0) = 0 for l >= 1; for l = 0 we use V_H,00(0) =
//     4 pi * int_0^inf rho_{00}(r') r' dr'.
//
// OpenMP: parallel over channels (Lmax+1)^2. Thread-local scratch vectors
// avoid any shared state.
//
// Cost: O(N_lm * N_r). At Lmax=300, N_r=10000: 9e8 ops total. Milliseconds
// parallel.

#pragma once

#include "angular/Ylm.hpp"
#include "sce/RadialGrid.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#if defined(PREPROC_HAS_OPENMP) && PREPROC_HAS_OPENMP
#include <omp.h>
#endif

namespace preproc::potential {

// Build V_H from the SCE coefficients of rho.
//
//   rho_lm  :  (Nlm, Nr) matrix of rho^R_{l,m}(r_k)
//   r_grid  :  radial grid (must match the columns of rho_lm)
//   Lmax    :  must satisfy (Lmax+1)^2 == rho_lm.rows()
//
// Returns V_H^R_{l,m}(r_k) with the same shape. Channels with l > Lmax are
// ignored (if Lmax provided is smaller than rho_lm.rows() would imply).
inline Eigen::MatrixXd build_V_H(const Eigen::MatrixXd& rho_lm,
                                 const sce::RadialGrid& r_grid,
                                 int Lmax,
                                 bool verbose = false) {
    const int Nr  = r_grid.N;
    const int Nlm_expect = angular::n_channels(Lmax);
    if (rho_lm.cols() != Nr)
        throw std::runtime_error("build_V_H: rho_lm cols != N_r");
    if (rho_lm.rows() < Nlm_expect)
        throw std::runtime_error("build_V_H: rho_lm rows < (Lmax+1)^2");

    const int Nlm = Nlm_expect;
    Eigen::MatrixXd V_H(Nlm, Nr);
    V_H.setZero();

    // ----------------------------------------------------------------
    // Numerical safety: clamp the largest l we'll process to whatever
    // keeps std::pow(r, l) and std::pow(r, 1 - l) inside double range.
    //
    // ln(DBL_MAX) ≈ 709.7;  use 690 with margin.  Then:
    //     std::pow(r_max,    l) overflows when  l    *  ln(r_max)    > 690
    //     std::pow(r_min, 1-l) overflows when (l-1) *  ln(1/r_min)   > 690
    // For the production C8F8 grid (r_max = 100, r_min = dr = 0.01):
    //     ln(100)  = 4.605  ->  l_safe_outer = 690 / 4.605 ≈ 149
    //     ln(100)  = 4.605  ->  l_safe_inner = 690 / 4.605 ≈ 149
    // i.e. the formula breaks for l > ~149 regardless of how small ρ
    // is.  Beyond that, computed (inf × tiny) gives inf/NaN that then
    // contaminates J = 0.5*<ρ|V_H>.
    //
    // Note this is a numerical-stability limit, not a physical one: a
    // real molecular density has zero content past l ~ 30–50, so V_H
    // for these high-l channels is simply zero, which is exactly what
    // we get by skipping them (V_H is zero-initialized).
    //
    // The proper long-term fix is to solve the radial Poisson ODE
    // d²(rV)/dr² - l(l+1)/r²·(rV) = -4π·r·ρ_lm directly via Numerov,
    // which has no r^l / r^{1-l} factors.  TODO once we hit a case
    // where Lmax_safe is too restrictive.
    int l_safe = Lmax;
    {
        const double LOG_DBL_MAX_SAFE = 690.0;
        const double r_max_grid = r_grid.r[Nr - 1];
        // First strictly-positive grid point (r_min may be 0 by design).
        double r_min_pos = 0.0;
        for (int k = 0; k < Nr; ++k) {
            if (r_grid.r[k] > 0.0) { r_min_pos = r_grid.r[k]; break; }
        }
        if (r_max_grid > 1.0) {
            l_safe = std::min(
                l_safe,
                static_cast<int>(LOG_DBL_MAX_SAFE / std::log(r_max_grid)));
        }
        if (r_min_pos > 0.0 && r_min_pos < 1.0) {
            const int l_inner =
                1 + static_cast<int>(LOG_DBL_MAX_SAFE / std::log(1.0 / r_min_pos));
            l_safe = std::min(l_safe, l_inner);
        }
        if (l_safe < Lmax) {
            std::cerr << "[V_H] numerical safety: capping V_H computation at "
                      << "l_safe=" << l_safe
                      << "  (Lmax=" << Lmax
                      << ", r_max=" << r_max_grid
                      << ", r_min_pos=" << r_min_pos << ")\n";
            std::cerr << "[V_H]   channels with l > l_safe stay 0 (their physical "
                         "content is below machine precision anyway for any real "
                         "molecule).\n";
        }
    }

    #if defined(PREPROC_HAS_OPENMP) && PREPROC_HAS_OPENMP
    #pragma omp parallel
    #endif
    {
        std::vector<double> f_in (Nr);
        std::vector<double> f_out(Nr);
        std::vector<double> I_in (Nr);
        std::vector<double> I_out_left(Nr);   // int_0^r f_out dr'

        #if defined(PREPROC_HAS_OPENMP) && PREPROC_HAS_OPENMP
        #pragma omp for schedule(static)
        #endif
        for (int ch = 0; ch < Nlm; ++ch) {
            int l, m;
            angular::idx_to_lm(ch, l, m);

            // (1) Numerical-overflow guard: above l_safe (computed above),
            // std::pow(r, l) and/or std::pow(r, 1-l) overflow IEEE doubles
            // for at least one grid point.  inf · tiny = inf or NaN, which
            // contaminates V_H and downstream <rho|V_H>.  V_H stays 0 here,
            // which is the correct value for any real molecular density
            // (their content past l ~ 30 is below machine precision).
            if (l > l_safe) continue;

            // (2) Empty-channel guard: channels whose ρ_lm is purely
            // numerical noise from the SCE projection (typical at l > ~30
            // for real molecules) contribute nothing physical.  Skip them:
            // V_H[ch] stays 0.  Threshold 1e-12 is well below the
            // SCE-projection residual for occupied orbitals.
            const double rho_max_ch = rho_lm.row(ch).cwiseAbs().maxCoeff();
            if (rho_max_ch < 1e-12) continue;

            const double pref = 4.0 * M_PI / (2.0 * l + 1.0);

            // --- Build the two integrands ---
            for (int k = 0; k < Nr; ++k) {
                const double r  = r_grid.r[k];
                const double rho_k = rho_lm(ch, k);
                // Inner: rho * r^{l+2}  -- always finite.
                // Use std::pow for clarity; for small l we could specialize.
                const double rp_l_plus_2 = std::pow(r, l + 2);
                f_in[k] = rho_k * rp_l_plus_2;

                // Outer: rho * r^{1-l}
                //   l == 0:  rho * r      -> finite, 0 at r=0
                //   l == 1:  rho          -> finite, 0 at r=0 (rho_{1m} ~ r)
                //   l >= 2:  rho / r^{l-1} -> limit is 0 at r=0 (rho ~ r^l)
                if (l == 0) {
                    f_out[k] = rho_k * r;
                } else if (r <= 0.0) {
                    f_out[k] = 0.0;              // physical limit
                } else {
                    f_out[k] = rho_k * std::pow(r, 1 - l);
                }
            }

            // --- Cumulative integrals ---
            r_grid.cumulative_integrate(f_in , I_in);
            r_grid.cumulative_integrate(f_out, I_out_left);
            const double I_out_total = I_out_left[Nr - 1];

            // --- Combine ---
            // Defensive: even within l_safe we can hit a non-finite if the
            // grid is unusual (e.g. r exactly at the overflow boundary).
            // Sweep any inf/NaN to 0 and tally so we can warn after the
            // loop -- silently zeroing is risky, ignoring is worse.
            int n_nonfinite = 0;
            for (int k = 0; k < Nr; ++k) {
                const double r = r_grid.r[k];
                const double I_out_from_r = I_out_total - I_out_left[k];
                double term1;
                if (r <= 0.0) {
                    // Limit: I_in(r) ~ rho(0) r^{2l+3} / (2l+3), divided by r^{l+1}
                    // gives ~ rho(0) r^{l+2} / (2l+3) -> 0. So safe to set 0 at r=0.
                    term1 = 0.0;
                } else {
                    term1 = I_in[k] / std::pow(r, l + 1);
                }
                const double term2 = std::pow(r, l) * I_out_from_r;
                double v = pref * (term1 + term2);
                if (!std::isfinite(v)) {
                    v = 0.0;
                    ++n_nonfinite;
                }
                V_H(ch, k) = v;
            }
            if (n_nonfinite > 0) {
                #if defined(PREPROC_HAS_OPENMP) && PREPROC_HAS_OPENMP
                #pragma omp critical(v_h_warn)
                #endif
                std::cerr << "[V_H] channel l=" << l << ", m=" << m
                          << ": " << n_nonfinite << "/" << Nr
                          << " grid points produced inf/NaN (zeroed). "
                          << "Lower Lmax_sce or raise rho_threshold.\n";
            }
            // Special case l = 0, r = 0: V_H,00(0) = 4 pi * int_0^inf rho_{00}(r') r' dr'
            if (l == 0) {
                V_H(ch, 0) = pref * I_out_total;
            }
        }
    }

    if (verbose) {
        std::cerr << "[V_H] built Nlm=" << Nlm << " Nr=" << Nr << "\n";
    }
    return V_H;
}

}  // namespace preproc::potential
