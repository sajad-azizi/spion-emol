// AsymptoticAmplitudes.hpp -- extract (A, B) amplitudes from the
// back-propagated ψ in the asymptotic region, then derive K and S.
//
// For each (μ, j) the scattering state has the outer form (PDF eq. 22):
//
//     ψ_{μj}(r → ∞)  =  A_{μj} · ĵ_{ℓ_μ}(k r)  +  B_{μj} · ŷ_{ℓ_μ}(k r)
//
// with Riccati–Bessel ĵ_ℓ(x) = x · j_ℓ(x) and Riccati–Neumann ŷ_ℓ(x) =
// x · y_ℓ(x). We recover A_{μj} and B_{μj} by a 2-parameter least-squares
// fit to ψ_{μj}(r_n) over a user-chosen asymptotic window [n_start, n_end].
// Fitting many points (default ~200) averages out Numerov's local finite-h
// error and is more robust than a two-point match.
//
// Derived quantities:
//
//   K  =  B · A^{-1}                  real, symmetric in exact arithmetic
//   S  =  (A + i B) · (A − i B)^{-1}  complex, unitary
//   δ  =  atan(eigenvalues of K)      eigenphases
//
// S is evaluated in the (A, B) form rather than (I + iK)(I − iK)^{-1} so
// that (a) we never need to invert K explicitly (fine at bound-state E
// where K diverges), and (b) the expression makes physical sense even
// when A is not exactly I (the regular-normalisation ansatz only assumes
// A is invertible, not equal to identity).
//
// WHY THIS OVER KMatrixExtractor (Step 5)?
//
//   - Step 5's K came from a two-point match at the very outer edge. It
//     is sensitive to finite-h Numerov error at those particular grid
//     points and requires a specific R-matrix index convention.
//   - The A/B fit uses ~200 points of ψ data, LSQ-averaging the finite-h
//     wiggle. It also gives A explicitly, which the dipole matrix
//     element will need.
//   - Matches version_0's `calculate_A_B_matrices_ofPsi` approach so we
//     can regression-compare.

#pragma once

#include "scatt/BackPropagator.hpp"
#include "scatt/SolverParams.hpp"

#include <Eigen/Dense>

#include <complex>
#include <string>
#include <vector>

namespace scatt {

struct AmplitudeResult {
    Eigen::MatrixXd     A;              // (N_ψ × N_ψ) amplitude of ĵ  (real)
    Eigen::MatrixXd     B;              // (N_ψ × N_ψ) amplitude of ŷ  (real)
    Eigen::MatrixXd     K;              // K = B · A^{-1}, symmetric
    Eigen::MatrixXcd    S;              // S = (A + iB)(A − iB)^{-1}, unitary
    std::vector<double> eigenphases;    // atan(eig K)

    // Diagnostics.
    double fit_residual_max  = 0.0;     // max |ψ - A·ĵ - B·ŷ| over (μ, j, n)
    double fit_residual_rel  = 0.0;     // same, divided by max |ψ| in window
    double A_minus_I_max     = 0.0;     // ‖A − I‖_∞  (should be ~0 for regular norm)
    double K_symmetry_err    = 0.0;     // ‖K − Kᵀ‖_∞
    double S_unitarity_err   = 0.0;     // ‖S†S − I‖_∞

    int    n_fit_start       = -1;
    int    n_fit_end         = -1;
    double r_fit_start       = 0.0;
    double r_fit_end         = 0.0;
};

class AsymptoticAmplitudes {
public:
    struct Config {
        // Asymptotic window in grid units. Negative = auto:
        //   auto start = max(n_grid − 200, floor(0.9·n_grid))
        //   auto end   = n_grid − 1 − n_exclude_outer
        int    n_fit_start      = -1;
        int    n_fit_end        = -1;
        int    n_exclude_outer  = 3;    // drop the last few grid points
        bool   verbose          = true;

        // OPT-IN closed-channel substitution.  When set, ℓ values whose
        // 2×2 normal matrix is catastrophically singular (k·r_max ≲ ℓ —
        // the channel is exponentially evanescent across the entire fit
        // window) are MARKED CLOSED rather than aborting the run:
        //
        //     A_{μν} := δ_{μν},    B_{μν} := 0     for all μ with l_μ = ℓ_closed
        //
        // The downstream (A − iB) is identity on the closed block, so
        // (A − iB)⁻¹ stays well-conditioned and the K-matrix vanishes
        // on those channels.  This is mathematically equivalent to
        // truncating l_cont at floor(k·r_max) for the final
        // photoionization observables — verifiable by running a
        // converged l_cont calculation with a smaller box and
        // comparing.
        //
        // DEFAULT: false.  The user MUST set this knowingly, with a
        // convergence justification in hand (typical workflow: run at
        // l_cont = L_safe ≤ k·r_max where everything is open and the
        // cross section is converged; then trust closed-channel = 0
        // for any ℓ > L_safe).
        bool   allow_closed_channels = false;
    };

    AsymptoticAmplitudes(const SolverParams& sp, BackPropagator& bp);

    // The main entry point. Requires bp to have been run() with a keep
    // range covering the fit window.
    AmplitudeResult extract(const Config& cfg);
    AmplitudeResult extract() { return extract(Config{}); }

private:
    const SolverParams& sp_;
    BackPropagator&     bp_;
    std::vector<int>    l_psi_;      // ℓ for each μ channel

    void build_channel_info_();

    static double riccati_j_(int l, double x);
    static double riccati_y_(int l, double x);
};

}  // namespace scatt
