#include "SpectralReadout.hpp"

#include <cmath>
#include <stdexcept>

namespace mc_tdse {

std::vector<double> dPdE_block(const PooledBasis& pb,
                               const Eigen::VectorXcd& b_T,
                               int k_block,
                               const SpectralOptions& opt) {
    if (k_block < 0 || k_block >= static_cast<int>(pb.blocks.size()))
        throw std::runtime_error("dPdE_block: k_block out of range");
    if (opt.delta_E_au <= 0.0)
        throw std::runtime_error("dPdE_block: delta_E_au must be > 0");
    if (b_T.size() != pb.N_total)
        throw std::runtime_error("dPdE_block: b_T size != N_total");

    const int    N_E   = static_cast<int>(opt.E_grid_au.size());
    const double sigma = opt.delta_E_au;
    const double inv_norm = 1.0 / (std::sqrt(2.0 * M_PI) * sigma);
    const double inv_2s2  = 1.0 / (2.0 * sigma * sigma);

    std::vector<double> y(N_E, 0.0);
    const int o = pb.block_offset[k_block];
    const int n = pb.blocks[k_block].n_states();
    for (int i = 0; i < n; ++i) {
        const int alpha = o + i;
        const double w   = std::norm(b_T(alpha));
        if (w == 0.0) continue;
        const double Ea  = pb.E_au[alpha];
        for (int j = 0; j < N_E; ++j) {
            const double x = opt.E_grid_au[j] - Ea;
            y[j] += w * inv_norm * std::exp(-x * x * inv_2s2);
        }
    }
    return y;
}

}  // namespace mc_tdse
