// Wavefunctions.hpp
//
// Wave-function reconstruction for the multichannel problem.
//
// This first pass handles only the BOUND-STATE case, which is all that 
// is needed for the halo verification test. Scattering-state 
// reconstruction (including the asymptotic fit to obtain the A/B matrices,
// the K matrix, and the S matrix) is added in a second pass.
//
// The bound-state reconstruction follows the B.R. Johnson procedure:
//
// 1. Propagate forward from origin to i_match and backward from box edge
//    to i_match + 1, saving the ratio matrices R_i^{-1} and W_i^{-1} 
//    along the way.
// 2. At the matching point, find the null eigenvector of (R_m^{-1} - R_{m+1}).
//    This gives the linear combination of channels at the matching point.
// 3. Sweep backward from i_match - 1 to 0 and forward from i_match + 1 to
//    N_grid - 1, propagating the amplitude vector via
//       f_{i-1} = R_i^{-1} f_i
//       u_i     = W_i^{-1} f_i
// 4. Normalize: ∫₀^L Σ_α |u_α(r)|^2 dr = 1
//
// The result is stored in eigfunc, one Eigen::VectorXcd per grid point.
// Component α of eigfunc[ir] is u_α(r_ir) = r_ir * R_α(r_ir).
//
#pragma once

#include "Common.hpp"
#include "Parameters.hpp"
#include "Equations.hpp"

class Wavefunctions {
public:
    Wavefunctions(Equations* equations, Parameters* parameters);

    // Reconstruct the bound-state wave function at the given energy and
    // matching index. Result is stored in eigfunc.
    // 
    // Use this for states that are truly bound in ALL channels (exponentially
    // decaying at r -> L). Requires both inward and outward sweeps, matched
    // at i_match.
    void calculate_eigenfunction(double Energy, int i_match);

    // Reconstruct the CONTINUUM box state at the given energy.
    //
    // For a state in the open-channel continuum (box-quantized by the wall
    // condition u(L) = 0), the correct procedure is:
    //   1. Propagate outward from origin to N_grid - 1 (save=true).
    //   2. Diagonalize the final ratio matrix R_{N-1}. The eigenvector
    //      corresponding to the smallest (most-nearly-zero) eigenvalue
    //      is the direction in channel space that the box wall kills
    //      (i.e., u_{N-1} = R_{N-1} u_{N-2} = 0 along that direction).
    //   3. Back-propagate the amplitude vector from N-1 inward via
    //        f_{k-1} = R_{k-1}^{-1} f_k
    //        u_k     = W_k^{-1} f_k
    //
    // This avoids the inward-sweep-from-the-wall procedure of the bound-
    // state case, which numerically misbehaves for an open channel at E>0
    // because the open-channel wavefunction is PROPAGATING (not decaying)
    // at r = L, and marching it backward from a Dirichlet wall over 10^6
    // grid points gives an ambiguous linear combination whose short-range
    // structure depends on numerical noise.
    //
    // The smallest eigenvalue of R_{N-1} is what OutwardNodeCounting/
    // bisection already targets when locating box eigenvalues, so by
    // construction this eigenvector is physically meaningful at every
    // converged bisection energy.
    void calculate_eigenfunction_continuum(double Energy);

    // Normalize eigfunc to unit integral: ∫ Σ_α |u_α|^2 dr = 1.
    void Normalization();

    // Integrate |u_α(r)|^2 dr for a given channel α. Returns the population.
    double channel_population(int alpha) const;

    // Storage: eigfunc[ir] is a channels-vector (complex)
    std::vector<Eigen::VectorXcd> eigfunc;

private:
    Equations*  equations_;
    Parameters* parameters_;

    int N_grid_;
    int N_ch_;
    double dr_;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es_;
};
