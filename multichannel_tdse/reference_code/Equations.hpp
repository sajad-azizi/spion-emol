// Equations.hpp
//
// Renormalized Numerov propagator for the multichannel radial
// Schrodinger equation:
//
//   -(1/2μ) u''_α + Σ_β V_{αβ}(r) u_β = E u_α,    α = 0, ..., N_ch - 1
//
// Adapted from the 2D polar version. The only substantive change is
// in proper_initialization_R: in 3D s-wave there is NO centrifugal
// -1/(4r^2) term (that was a 2D artifact), so we do not need the 
// analytic Bessel-function correction at the origin. The Renormalized
// Numerov ratio matrix can be initialized from zero (corresponding to
// u(r) ∝ r for r -> 0, the regular solution).
//
// Node-counting uses the B.R. Johnson theorem: a node is registered 
// whenever W has a positive eigenvalue AND R has a negative eigenvalue
// at the same grid point. This logic is unchanged from the 2D code.
//
#pragma once

#include "Common.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"

class Equations {
public:
    Equations(Potentials* potentials, Parameters* parameters);

    // Propagate the ratio matrix from the origin outward to i = i_match,
    // writing the result to resRm. If save=true, store R^{-1} and W^{-1}
    // along the way for subsequent wave-function reconstruction.
    void propagateForward(double Energy, int i_match, Eigen::MatrixXcd& resRm, bool save);

    // Propagate backward from i = N_grid - 2 down to i = i_match + 1.
    void propagateBackward(double Energy, int i_match, Eigen::MatrixXcd& resRmp1, bool save);

    // Initialize the ratio matrix for the first (p-1) grid points using the
    // regular 3D s-wave solution u_α(r) ∝ r near the origin.
    void proper_initialization_R(double Energy, Eigen::MatrixXcd& resRinv);

    // Count nodes from outward propagation. Returns (n_nodes, r_last_node).
    std::pair<int, double> OutwardNodeCounting(double Energy);

    // Stored propagation data, used by Wavefunctions
    std::vector<Eigen::MatrixXcd> Rinv_vector;         // forward-swept Rinv
    std::vector<Eigen::MatrixXcd> Rinv_vector_back;    // backward-swept Rinv
    std::vector<Eigen::MatrixXcd> Winv_vector;         // W^{-1} at every grid point

private:
    Potentials* potentials_;
    Parameters* parameters_;

    int N_grid_;
    int N_ch_;
    double dr_;
    double mu_;   // reduced mass (from Parameters)
    int p_;       // number of steps used for initialization near origin

    // Workspace
    Eigen::MatrixXcd In_;      // identity
    Eigen::MatrixXcd Wmat_;
    Eigen::MatrixXcd U_;
    Eigen::MatrixXcd R_;
    Eigen::MatrixXcd Rinv_;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es_;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es1_;
};
