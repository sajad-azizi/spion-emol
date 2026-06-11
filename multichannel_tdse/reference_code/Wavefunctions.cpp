// Wavefunctions.cpp
//
// Bound-state reconstruction for the multichannel radial Schrodinger
// equation. Adapted from the 2D polar code; the logic is identical
// because the reconstruction is purely algebraic (ratio matrices and
// their inverses), independent of spatial dimension.
//
#include "Wavefunctions.hpp"

Wavefunctions::Wavefunctions(Equations* equations, Parameters* parameters)
    : equations_(equations), parameters_(parameters)
{
    N_grid_ = parameters_->N_grid;
    N_ch_   = parameters_->N_ch_keep > 0 ? parameters_->N_ch_keep : 0;

    // If N_ch_keep is unset, derive from saved Rinv_vector
    if (N_ch_ == 0 && !equations_->Rinv_vector.empty()) {
        N_ch_ = static_cast<int>(equations_->Rinv_vector[0].rows());
    }

    dr_ = parameters_->dr;

    eigfunc.resize(N_grid_);
    for (int k = 0; k < N_grid_; ++k) {
        eigfunc[k] = Eigen::VectorXcd::Zero(N_ch_);
    }
}

// ============================================================
// Bound-state reconstruction
// ============================================================
//
// At the matching point, find the null eigenvector of (R_m^{-1} - R_{m+1}).
// That's the direction in channel space that satisfies both the forward
// and backward boundary conditions. All other directions would produce
// a discontinuity in u or u' at the match point.
//
// Then sweep backward and forward, propagating the amplitude vector:
//   f_{i}    = R_i * f_{i+1}   (backward)   (we have R_i^{-1} stored, so f_{i-1} = R_{i-1}^{-1} f_i)
//   u_i      = W_i^{-1} * f_i
//
void Wavefunctions::calculate_eigenfunction(double Energy, int i_match)
{
    Eigen::MatrixXcd Rm, Rmp1;

    // Re-propagate and save (true)
    equations_->propagateForward(Energy, i_match, Rm, true);
    equations_->propagateBackward(Energy, i_match, Rmp1, true);

    // At matching: (R_m^{-1} - R_{m+1}) f_match = 0 has a null eigenvector
    // The 2D code uses Rmp1.inverse() - Rm, which is the same equation
    // written the other way.
    Eigen::MatrixXcd diff = Rmp1.inverse() - Rm;

    // Find eigenvector of the smallest (by magnitude) eigenvalue
    es_.compute(diff);
    int jm = 0;
    double vmin = std::abs(es_.eigenvalues().real()(0));
    for (int m = 1; m < N_ch_; ++m) {
        double v = std::abs(es_.eigenvalues().real()(m));
        if (v < vmin) { vmin = v; jm = m; }
    }

    Eigen::VectorXcd fn_match = es_.eigenvectors().col(jm);

    // Backward sweep: i_match - 1 down to 1
    Eigen::VectorXcd fn, fn_next = fn_match;
    for (int k = i_match - 1; k > 0; --k) {
        fn = equations_->Rinv_vector[k] * fn_next;
        eigfunc[k] = equations_->Winv_vector[k] * fn;
        fn_next = fn;
    }

    // Forward sweep: i_match + 1 up to N_grid - 2
    Eigen::VectorXcd fn_prev = fn_match;
    for (int k = i_match + 1; k < N_grid_ - 1; ++k) {
        fn = equations_->Rinv_vector[k] * fn_prev;
        eigfunc[k] = equations_->Winv_vector[k] * fn;
        fn_prev = fn;
    }

    // At the matching point itself
    eigfunc[i_match] = equations_->Winv_vector[i_match] * fn_match;
}

// ============================================================
// Continuum-state reconstruction (box-quantized)
// ============================================================
//
// Correct procedure for continuum box states (E > 0, wavefunction 
// propagating in at least one channel):
//
//   1. Forward-propagate from origin to N_grid - 1 (save=true).
//      This populates Rinv_vector[k] for all k, which we need for the 
//      subsequent back-propagation.
//   2. Diagonalize R_{N-1} itself. Its smallest-magnitude eigenvalue is 
//      (to numerical precision) zero by construction at a converged box 
//      eigenvalue; the corresponding eigenvector is the channel-space 
//      direction along which the wavefunction vanishes at the wall.
//   3. Back-propagate the amplitude vector from N-1 inward:
//        f_{k-1} = R_{k-1}^{-1} f_k,    u_k = W_k^{-1} f_k
//   4. At the origin, u_0 = 0 (regular boundary condition); this is
//      enforced automatically by the recursion.
//
// This is the appropriate algorithm for open-channel continuum states 
// at E > 0 with Dirichlet BC at r = L. It avoids the inward-sweep-
// from-the-wall matching of the bound-state procedure, which is 
// ambiguous when one channel is propagating at the wall (any linear 
// combination satisfying u(L) = 0 is mathematically allowed, but the 
// SAME linear combination at different E_n produces eigenfunctions 
// with different short-range structure; the current bisection + 
// matching algorithm selects this linear combination by an 
// eigendecomposition that can flip between two nearby eigenvalues 
// at different E_n, producing the bimodal-matrix-element pattern we 
// observed in v4 output).
//
// The backward-propagated wavefunction is NOT yet normalized on exit.
// Call Normalization() afterward to impose sum_a |u_a(r)|^2 dr = 1.
//
void Wavefunctions::calculate_eigenfunction_continuum(double Energy)
{
    // Forward sweep to the box wall, saving R^{-1} and W^{-1} at every step
    Eigen::MatrixXcd Rm;
    equations_->propagateForward(Energy, N_grid_ - 1, Rm, /*save=*/true);

    // Diagonalize R_{N-1}. In a 2-channel problem with one open channel
    // and one closed (evanescent) channel at E > 0:
    //   - The closed channel is exponentially suppressed at r = L by 
    //     e^{-kappa L}, so R_{N-1} has one perpetually-small eigenvalue 
    //     in the closed-channel direction, regardless of whether E is a 
    //     box eigenvalue.
    //   - The open channel only has a small eigenvalue of R_{N-1} at
    //     exact box-quantization energies.
    // 
    // Bisection targets the OPEN-channel null condition (that's what 
    // OutwardNodeCounting detects — node in the oscillating open-channel 
    // component). So the physically meaningful eigenvector at a converged 
    // box eigenvalue is the OPEN-CHANNEL-DOMINANT one, NOT the smallest-
    // eigenvalue one.
    //
    // Selection rule: pick the eigenvector whose channel-0 (open) 
    // component has the largest |magnitude|. Channel 0 is, by Potentials 
    // construction, the lowest-threshold (always-open) channel for any 
    // physically accessible energy.
    //
    es_.compute(Rm);
    int jm = 0;
    double best = std::abs(es_.eigenvectors().col(0)(0));
    for (int m = 1; m < N_ch_; ++m) {
        double w = std::abs(es_.eigenvectors().col(m)(0));
        if (w > best) { best = w; jm = m; }
    }

    // Initial amplitude vector at the wall
    Eigen::VectorXcd fn_wall = es_.eigenvectors().col(jm);

    // Back-propagate: k = N_grid - 1 down to 0
    //   f_{k-1} = R_{k-1}^{-1} f_k
    //   u_k     = W_k^{-1} f_k
    Eigen::VectorXcd fn = fn_wall;
    eigfunc[N_grid_ - 1] = equations_->Winv_vector[N_grid_ - 1] * fn;
    for (int k = N_grid_ - 2; k > 0; --k) {
        fn = equations_->Rinv_vector[k] * fn;
        eigfunc[k] = equations_->Winv_vector[k] * fn;
    }
    // At the origin: u(0) = 0 (regular boundary condition).
    eigfunc[0] = Eigen::VectorXcd::Zero(N_ch_);
}

// ============================================================
// Normalization to unit integral
// ============================================================
void Wavefunctions::Normalization()
{
    double sum = 0.0;
    for (int ir = 0; ir < N_grid_; ++ir) {
        for (int a = 0; a < N_ch_; ++a) {
            double mag = std::abs(eigfunc[ir](a));
            sum += mag * mag;
        }
    }
    sum *= dr_;

    if (sum > 0.0) {
        double norm = std::sqrt(sum);
        for (int ir = 0; ir < N_grid_; ++ir) {
            eigfunc[ir] /= norm;
        }
    }
}

// ============================================================
// Channel population
// ============================================================
double Wavefunctions::channel_population(int alpha) const
{
    double sum = 0.0;
    for (int ir = 0; ir < N_grid_; ++ir) {
        double mag = std::abs(eigfunc[ir](alpha));
        sum += mag * mag;
    }
    return sum * dr_;
}
