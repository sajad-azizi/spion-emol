// ClosedChannelInhom.hpp
//
// Finite-difference inhomogeneous solver for the closed-channel problem:
//
//   (E_int - H_{closed}) X(r) = S(r)
//
// with regular BC at r=0 (X=0) and decaying/Robin BC at r=L
// (X' = -kappa X where kappa_alpha = sqrt(2mu*(E_{th,alpha} - E_int))).
//
// Block-tridiagonal via the multi-channel three-point second-difference
// discretization; solved with block-Thomas in O(N_grid * N_ch^3).
//
// This directly computes the full closed-channel Green's function
// application, G_closed(E_int) * S, including BOTH the bound-state pole
// structure AND the continuum spectral contribution. It is therefore
// the correct replacement for a bound-state-only spectral sum when
// computing virtual closed-channel amplitudes.
//
// The source S(r) is a channel-vector field on the same radial grid
// as the Potentials object.

#pragma once

#include "Common.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"
#include <vector>

class ClosedChannelInhom {
public:
    // Build the solver for a specific closed block `pot` at intermediate
    // energy `E_int`. The energy E_int must be below all channel
    // thresholds of `pot` (strictly closed block).
    ClosedChannelInhom(const Potentials* pot, const Parameters* params,
                        double E_int);

    // Solve (E_int - H) X = S where S is specified as N_grid vectors of
    // length N_ch. Returns X as vector of N_ch-dim vectors on the same
    // grid.
    std::vector<Eigen::VectorXd>
    solve(const std::vector<Eigen::VectorXd>& S) const;

    // Compact two-channel version of the same solve. The source is evaluated
    // on demand at grid index i, avoiding a grid-sized dynamic VectorXd source.
    // This is algebraically the same finite-difference system as solve(), but
    // stores fixed-size 2x2/2-vector blocks contiguously.
    std::vector<Eigen::Vector2d>
    solve_2ch(const std::function<Eigen::Vector2d(int)>& source) const;

    // Relative residual of the same finite-difference system solved above:
    //   ||(E_int - H)X - S|| / ||S||
    // Uses the same origin and exponential-tail boundary convention as solve().
    double relative_residual(const std::vector<Eigen::VectorXd>& X,
                             const std::vector<Eigen::VectorXd>& S) const;

    double relative_residual_2ch(
        const std::vector<Eigen::Vector2d>& X,
        const std::function<Eigen::Vector2d(int)>& source) const;

    // Accessors
    double E_int() const { return E_int_; }
    const Eigen::VectorXd& kappa_out() const { return kappa_out_; }
    int N_grid() const { return N_grid_; }
    int N_ch()   const { return N_ch_; }
    double dr()  const { return dr_; }

private:
    const Potentials* pot_;
    const Parameters* params_;
    double E_int_;
    int N_grid_;
    int N_ch_;
    double dr_;
    double mu_;
    Eigen::VectorXd kappa_out_;   // decay wavenumbers per channel
};
