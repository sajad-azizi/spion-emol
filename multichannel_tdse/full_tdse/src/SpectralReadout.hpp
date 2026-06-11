// SpectralReadout.hpp -- dP/dE for each M_F block (Stage 5).
//
// After the TDSE pulse, every box-quantized eigenstate carries a
// post-pulse amplitude b_α(T).  A discrete probability mass |b_α|² is
// associated to its eigenenergy E_α.  The recipe's physical observable
// is dP/dE smoothed by a Gaussian kernel of width δE comparable to the
// local level spacing:
//
//   dP/dE^{(M_F)}(E) = Σ_{α ∈ M_F} |b_α(T)|² · g_{δE}(E - E_α)
//
//   g_δ(x) = (1/√(2π) δ) exp(-x²/(2δ²))
//
// Recipe's main outputs:
//   * dP/dE^{(-4)}(E) -- ZEPE channel.  Bound state delta function at
//     E_h plus the ZEPE peak just above threshold.
//   * dP/dE^{(-3)}(E) -- 1γ channel.  Peak around E_b + ω above threshold.
//   * dP/dE^{(-2)}(E) -- 2γ channel.  Peak around 2ω - E_b.
//
// The integrated area under each dP/dE peak is the corresponding
// recipe target (P_ZEPE, P_1γ, P_2γ).
#pragma once

#include "PooledBasis.hpp"

#include <Eigen/Dense>
#include <vector>

namespace mc_tdse {

struct SpectralOptions {
    // Smoothing width δE in atomic units.  Typical: 1-2× the local level
    // spacing (so the discrete spectrum becomes a smooth continuous curve).
    double delta_E_au = 0.0;

    // Energy grid (atomic units, recipe origin) on which dP/dE is sampled.
    // Common usage: pick a window per block that covers the physical peaks.
    std::vector<double> E_grid_au;
};

// Returns dP/dE for the block at index k_block (= position in pb.block_MFs).
// Output is a vector of length opt.E_grid_au.size().  Units: 1/Hartree
// (so ∫ dE · dP/dE ≈ Σ |b_α|² when the grid covers all in-block α).
std::vector<double> dPdE_block(const PooledBasis& pb,
                               const Eigen::VectorXcd& b_T,
                               int k_block,
                               const SpectralOptions& opt);

}  // namespace mc_tdse
