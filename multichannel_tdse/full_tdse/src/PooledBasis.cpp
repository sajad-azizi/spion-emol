#include "PooledBasis.hpp"
#include "DipoleAssembler.hpp"
#include "Rb85Spin.hpp"
#include "Common.hpp"

#include <stdexcept>

namespace mc_tdse {

PooledBasis build_pooled_basis(const std::vector<int>& block_MFs,
                                const std::vector<BlockBuildOptions>& per_block_opts,
                                const Rb85Spin& spin) {
    if (block_MFs.size() != per_block_opts.size())
        throw std::runtime_error("build_pooled_basis: block list and options length mismatch");
    if (block_MFs.empty())
        throw std::runtime_error("build_pooled_basis: empty block list");

    PooledBasis pb;
    pb.block_MFs = block_MFs;
    pb.blocks.reserve(block_MFs.size());
    for (size_t k = 0; k < block_MFs.size(); ++k) {
        pb.blocks.push_back(build_block_eigenstates(block_MFs[k], per_block_opts[k]));
    }
    // Verify common grid.
    const int    N_grid = pb.blocks[0].N_grid;
    const double dr     = pb.blocks[0].dr;
    for (size_t k = 1; k < pb.blocks.size(); ++k) {
        if (pb.blocks[k].N_grid != N_grid || pb.blocks[k].dr != dr)
            throw std::runtime_error("build_pooled_basis: blocks must share radial grid");
    }
    // Per-block threshold (recipe origin) for the rotating-frame
    // phases.  The threshold is the block's lowest channel asymptotic
    // energy, taken from Rb85Spin (recipe origin).
    pb.E_th_au.reserve(pb.blocks.size());
    for (size_t k = 0; k < pb.blocks.size(); ++k) {
        auto chs = spin.channels(pb.block_MFs[k]);
        if (chs.empty())
            throw std::runtime_error("build_pooled_basis: empty channel list for M_F");
        pb.E_th_au.push_back(AU::MHz_to_au(chs.front().E_th_MHz));
    }
    // Pool indices.
    pb.block_offset.push_back(0);
    for (size_t k = 0; k < pb.blocks.size(); ++k) {
        const int Nk = pb.blocks[k].n_states();
        for (int n = 0; n < Nk; ++n) {
            pb.E_au.push_back(pb.blocks[k].E_au[n]);
            pb.E_au_block_rel.push_back(pb.blocks[k].E_au[n] - pb.E_th_au[k]);
            pb.of_block.push_back(static_cast<int>(k));
            pb.n_in_block.push_back(n);
        }
        pb.block_offset.push_back(pb.block_offset.back() + Nk);
    }
    pb.N_total = pb.block_offset.back();

    // Build d_plus for each adjacent pair (k, k+1) when block_MFs match.
    pb.d_plus_pair.reserve(pb.blocks.size() > 0 ? pb.blocks.size() - 1 : 0);
    for (size_t k = 0; k + 1 < pb.blocks.size(); ++k) {
        if (pb.block_MFs[k + 1] == pb.block_MFs[k] + 1) {
            pb.d_plus_pair.push_back(
                assemble_d_plus(pb.blocks[k], pb.blocks[k + 1], spin));
        } else {
            pb.d_plus_pair.push_back(
                Eigen::MatrixXcd::Zero(pb.blocks[k + 1].n_states(),
                                       pb.blocks[k].n_states()));
        }
    }
    return pb;
}

}  // namespace mc_tdse
