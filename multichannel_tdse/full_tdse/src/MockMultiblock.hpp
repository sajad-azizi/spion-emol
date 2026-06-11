// MockMultiblock.hpp -- builder for the 4-block ZEPE toy system.
//
// Four M_F blocks: {-5, -4, -3, -2}.  Each block holds n_per_block
// states with hand-set energies that mimic the recipe's level
// structure (units are dimensionless / "natural"; Stage 4 swaps in
// real ⁸⁵Rb numbers in MHz/kHz).
//
// d^(+) (the σ⁺ raising vertex) is sparse: nonzero only between
// adjacent M_F blocks (M_F → M_F+1).  Magnitudes are O(1) random.
//
// We also expose a `zero_coupling(mf_low, mf_high)` that selectively
// turns off one block-pair coupling -- needed by Test 6 to isolate
// the ae / ea / aa pathways.
#pragma once

#include "Common.hpp"

#include <random>
#include <vector>

namespace mc_tdse {

struct MultiblockSystem {
    Eigen::VectorXd  E;           // (N) all energies
    Eigen::MatrixXcd d_plus;      // (N x N) σ⁺ vertex
    std::vector<int> mf_label;    // (N) M_F of each state
    std::vector<int> mf_values;   // sorted unique M_F values
    int              n_per_block = 0;

    int n_states() const { return static_cast<int>(E.size()); }

    // Population in M_F block.
    double block_population(const Eigen::VectorXcd& b, int mf) const {
        double s = 0.0;
        for (int i = 0; i < n_states(); ++i) {
            if (mf_label[i] == mf) s += std::norm(b(i));
        }
        return s;
    }

    // Indices belonging to a given M_F.
    std::vector<int> indices_in(int mf) const {
        std::vector<int> out;
        for (int i = 0; i < n_states(); ++i) if (mf_label[i] == mf) out.push_back(i);
        return out;
    }

    // Zero out σ⁺ coupling FROM mf_low TO mf_high (= mf_low + 1).
    // Affects both d_plus[high, low] (raise) and the implicit d^- = d_plus^†
    // (lowering) used by TDSEHamiltonian.
    void zero_coupling(int mf_low, int mf_high) {
        if (mf_high != mf_low + 1) {
            throw std::runtime_error(
                "zero_coupling: mf_high must equal mf_low + 1");
        }
        for (int f = 0; f < n_states(); ++f) {
            if (mf_label[f] != mf_high) continue;
            for (int i = 0; i < n_states(); ++i) {
                if (mf_label[i] != mf_low) continue;
                d_plus(f, i) = dcompx(0.0, 0.0);
            }
        }
    }
};

// Build the toy system.
//   n_per_block : how many states per M_F block.
//   omega_Z     : Zeeman shift between adjacent M_F blocks (default 0.75).
//   delta_m5    : extra detuning of the M_F=-5 block above what Zeeman
//                 alone would give -- mimics the GHz-scale detuning
//                 of the recipe (default 2.0 in dimensionless units).
//   coupling    : random scale for d^(+) entries (default 0.5).
//   seed        : RNG seed for reproducibility.
MultiblockSystem make_zepe_toy(int    n_per_block = 4,
                               double omega_Z     = 0.75,
                               double delta_m5    = 2.0,
                               double coupling    = 0.5,
                               unsigned long long seed = 12345);

}  // namespace mc_tdse
