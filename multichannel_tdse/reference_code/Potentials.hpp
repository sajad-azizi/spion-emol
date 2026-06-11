// Potentials.hpp
//
// Analytic square-well potential matrix for the multichannel Feshbach
// problem. Replaces the Fourier-integrated V(r, phi) construction of
// the 2D polar code with a direct analytic formula:
//
//   V_{α β}(r) = [ -V_bar δ_{αβ} + ΔV <α | s1.s2 | β> ] θ(r0 - r)
//              +  E_α^th δ_{αβ}
//
// with:
//   V_bar = (3 V_T + V_S) / 4    (central, direct)
//   ΔV    = V_S - V_T            (exchange, spin-dependent)
//
// E_α^th is the two-body threshold of channel α (sum of single-atom
// Zeeman energies at the prescribed magnetic field).
//
// NO angular integration. NO centrifugal term (s-wave, l = 0).
// NO Fourier decomposition.
//
// Storage mirrors the 2D code: a std::vector<Eigen::MatrixXcd> indexed
// by radial grid point. This lets Equations.cpp use the same
// Renormalized Numerov routine without modification.
//
#pragma once

#include "Common.hpp"
#include "Parameters.hpp"
#include "SpinAlgebra.hpp"

class Potentials {
public:
    // Build the potential matrix for the specified M_F block.
    //
    // The caller passes in:
    //   - parameters: grid + well depths + threshold block
    //   - spin:       atomic-physics layer at the requested B field
    //
    // The constructor:
    //   1) Enumerates channels in the requested M_F block
    //   2) Truncates to N_ch_keep channels (lowest threshold first)
    //   3) Computes <α|s1.s2|β> via SpinAlgebra
    //   4) Stores the thresholds E_α^th (in atomic units, referenced to the
    //      lowest threshold of the SAME block)
    //   5) Optionally allocates pot_component[ir] and fills with V(r_ir)
    //
    // IMPORTANT: For a single-block calculation (e.g. M_F=-4), the
    // thresholds are referenced to the lowest threshold of that block,
    // so the open-channel threshold is at E = 0. For a multi-block
    // scattering calculation the caller must supply a common reference
    // energy (this will be added later as a setReference method).
    //
    // Constructor with default (self-referenced) threshold: the lowest
    // channel of this block has threshold 0.
    Potentials(Parameters* parameters, SpinAlgebra* spin);

    // Constructor with explicit global reference energy (in MHz, the 
    // absolute single-atom+Zeeman sum for the reference state). All
    // thresholds are then expressed relative to this reference, so a
    // state at E = 0 (in a.u., as seen by the bound/scattering solvers)
    // corresponds to the physical energy E_global_ref_MHz in absolute
    // terms. Use this when you need M_F = -4, -3, -5 blocks that all
    // share a common energy origin (e.g. the open-channel threshold of
    // the M_F = -4 entrance block).
    Potentials(Parameters* parameters, SpinAlgebra* spin, double E_global_ref_MHz);

    // Resulting potential matrix at each radial grid point (channels x channels).
    // Same format as the 2D code's Potentials::pot_component. This is only
    // populated when Parameters::store_potential_grid is true.
    std::vector<Eigen::MatrixXcd> pot_component;

    // Accessors
    int num_channels() const { return (int)channels_.size(); }
    const std::vector<TwoBodyChannel>& channels() const { return channels_; }
    const Eigen::MatrixXd& s1s2() const { return s1s2_mat_; }
    const Eigen::VectorXd& thresholds() const { return thresholds_au_; }
    double V_bar() const { return V_bar_; }
    double dV()    const { return dV_; }
    double r0()    const { return r0_; }
    Eigen::MatrixXd  real_matrix_at_index(int ir) const;
    Eigen::MatrixXcd matrix_at_index(int ir) const;
    Eigen::Matrix2d  real_matrix2_at_index(int ir) const;

private:
    Parameters* parameters_;
    SpinAlgebra* spin_;

    // Channels kept in this block
    std::vector<TwoBodyChannel> channels_;
    // s1.s2 matrix in the kept channels (N x N, real)
    Eigen::MatrixXd s1s2_mat_;
    // Thresholds in atomic units, referenced to the lowest kept channel
    Eigen::VectorXd thresholds_au_;

    // Physical parameters cached from parameters_
    double V_bar_;
    double dV_;
    double r0_;
    int    N_grid_;
    int    N_ch_;
    double dr_;
    double smooth_width_;
    bool   store_grid_;
};
