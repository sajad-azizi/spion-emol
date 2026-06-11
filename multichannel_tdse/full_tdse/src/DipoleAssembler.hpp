// DipoleAssembler.hpp -- ⟨α | η^(+) | β⟩ matrix elements between
// adjacent M_F-block eigenstate sets.
//
// In the recipe TDSE, the time-dependent perturbation is
//
//   V_RF(t) = (Ω_R/2) χ(t) [ e^{-iωt} η^(+) + e^{+iωt} η^(-) ]
//
// where η^(+) = Ŝ_{1,+} + Ŝ_{2,+} raises total M_F by 1.  In the
// interaction picture, the Taylor matrix-vector propagator needs
// matrix elements of η^(+) between eigenstates of H_0:
//
//   d^(+)_{αβ} = ⟨α^{M_F+1} | η^(+) | β^{M_F}⟩
//             = ∫_0^L dr  u_α^{M_F+1}(r)^T  η^(+)_{f,i}(M_F)  u_β^{M_F}(r)
//
// where η^(+)_{f,i}(M_F) is the (channel, channel) σ⁺ vertex coming
// from Rb85Spin::sigma_plus_block(M_F): an N_high × N_low matrix.
//
// Both BlockEigenstates must share the SAME radial grid (N_grid, dr);
// it is an error to assemble across mismatched grids.
#pragma once

#include "BlockEigenstates.hpp"

#include <Eigen/Dense>

namespace mc_tdse {

class Rb85Spin;       // forward decl, includes only Rb85Spin.hpp on demand

// d^(+)_{αβ} for α ∈ MF+1 (row), β ∈ MF (col).
// out shape = (high.n_states(), low.n_states())
Eigen::MatrixXcd assemble_d_plus(const BlockEigenstates& low,    // M_F
                                 const BlockEigenstates& high,   // M_F + 1
                                 const Rb85Spin& spin);

}  // namespace mc_tdse
