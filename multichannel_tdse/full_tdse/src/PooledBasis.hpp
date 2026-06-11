// PooledBasis.hpp -- aggregation of per-M_F BlockEigenstates into a
// single contiguous basis indexed α = 0 .. N_total - 1, with the
// dipole vertex matrices d^(+)(MF→MF+1) precomputed.
//
// The pooled-basis index α encodes (M_F, n_state).  Block boundaries
// are stored so the TDSE driver can walk α-indices, find which block
// each state is in, and apply σ⁺/σ⁻ couplings between adjacent blocks
// via the precomputed (Na × Nb) dipole submatrices.
//
// Energies E_α are referenced to the recipe origin (M_F=-4 entrance
// threshold).  d^(+)_pair[k] couples blocks (block_MFs[k], block_MFs[k]+1).
#pragma once

#include "BlockEigenstates.hpp"

#include <Eigen/Dense>
#include <map>
#include <vector>

namespace mc_tdse {

class Rb85Spin;

struct PooledBasis {
    // Block list, in order of ascending M_F.
    std::vector<int>              block_MFs;        // e.g. {-5, -4, -3, -2}
    std::vector<BlockEigenstates> blocks;           // blocks[k] for block_MFs[k]
    // Pooled lab-frame energies (recipe origin).
    std::vector<double>           E_au;             // size = N_total
    // Per-block thresholds (recipe origin), used for the rotating-frame
    // transformation that absorbs the Zeeman ladder ω_carrier into ω.
    std::vector<double>           E_th_au;          // length = blocks.size()
    // Block-relative energies E_α - E_th(of_block[α]).  These are what
    // the recipe TDSE phase factor uses when ω is the small DETUNING
    // from the Zeeman-resonant carrier (the recipe's "ω = 8 E_b").
    std::vector<double>           E_au_block_rel;   // size = N_total
    std::vector<int>              of_block;         // of_block[α] = k
    std::vector<int>              n_in_block;       // n_in_block[α] = n
    std::vector<int>              block_offset;     // alpha-base of block k
    int N_total = 0;

    // d_plus_pair[k] = ⟨α^{block[k+1]} | η^(+) | β^{block[k]}⟩, shape
    //   (blocks[k+1].n_states(), blocks[k].n_states())
    // Defined when block_MFs[k+1] == block_MFs[k] + 1; otherwise zero.
    std::vector<Eigen::MatrixXcd> d_plus_pair;

    // Convenience: index into d_plus_pair given the lower M_F.
    // Returns -1 if not adjacent.
    int pair_index_of_low_MF(int MF_low) const {
        for (size_t k = 0; k + 1 < block_MFs.size(); ++k)
            if (block_MFs[k] == MF_low && block_MFs[k + 1] == MF_low + 1)
                return static_cast<int>(k);
        return -1;
    }
};

// Build the pooled basis given per-block options.  Each block is built
// with possibly DIFFERENT options (e.g. M_F=-5 needs a much wider
// E_max_kHz_above_threshold than the open blocks).  All block grids
// (N_grid, dr) MUST match -- the dipole assembler integrates wavefunctions
// on a common grid.
PooledBasis build_pooled_basis(const std::vector<int>& block_MFs,
                                const std::vector<BlockBuildOptions>& per_block_opts,
                                const Rb85Spin& spin);

}  // namespace mc_tdse
