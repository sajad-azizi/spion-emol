#include "MockMultiblock.hpp"

namespace mc_tdse {

MultiblockSystem make_zepe_toy(int    n_per_block,
                               double omega_Z,
                               double delta_m5,
                               double coupling,
                               unsigned long long seed)
{
    MultiblockSystem sys;
    sys.n_per_block = n_per_block;
    sys.mf_values   = {-5, -4, -3, -2};   // sorted ascending

    const int N = static_cast<int>(sys.mf_values.size()) * n_per_block;
    sys.E.resize(N);
    sys.mf_label.resize(N);

    // Energies relative to entrance threshold (M_F=-4 ground = E_h).
    // Within each block we place:
    //   * one near-threshold "halo-like" bound state at  E_block + (-1.0)
    //     i.e. shift inside the block.
    //   * a few above-threshold "continuum-like" states stepping by 0.05.
    //
    // The block thresholds:
    //     M_F = -2  ->   2·omega_Z       (above entrance, two raises)
    //     M_F = -3  ->     omega_Z       (one raise above entrance)
    //     M_F = -4  ->        0          (entrance threshold)
    //     M_F = -5  ->   -omega_Z + delta_m5    (one lower + extra detuning)
    //
    // delta_m5 puts M_F=-5 *above* entrance by a large amount (recipe
    // says ~2.68 GHz).  In our dimensionless units delta_m5=2.0 gives
    // a clear "virtual far-detuned" feel.

    auto block_threshold = [&](int mf) -> double {
        if (mf == -2) return  2.0 * omega_Z;
        if (mf == -3) return  1.0 * omega_Z;
        if (mf == -4) return  0.0;
        if (mf == -5) return -1.0 * omega_Z + delta_m5;
        throw std::runtime_error("unknown mf");
    };

    int idx = 0;
    for (int mf : sys.mf_values) {
        const double thr = block_threshold(mf);
        // First state: a near-threshold "halo-like" bound state slightly
        // BELOW the threshold (E < thr).  Used as initial state for M_F=-4.
        sys.E(idx)        = thr - 1.0;
        sys.mf_label[idx] = mf;
        ++idx;
        // Remaining states: small positive offsets above the threshold.
        for (int j = 1; j < n_per_block; ++j) {
            sys.E(idx)        = thr + 0.05 * j;
            sys.mf_label[idx] = mf;
            ++idx;
        }
    }

    // d^(+) σ⁺ vertex: only between adjacent M_F blocks.  Random complex
    // entries with magnitude scale `coupling`.
    sys.d_plus = Eigen::MatrixXcd::Zero(N, N);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> G(0.0, 1.0);
    for (int f = 0; f < N; ++f) {
        for (int i = 0; i < N; ++i) {
            const int dmf = sys.mf_label[f] - sys.mf_label[i];
            if (dmf == 1) {
                // σ⁺ raises by 1: d^(+)_{fi} ≠ 0.
                sys.d_plus(f, i) = coupling * dcompx(G(rng), G(rng));
            }
        }
    }
    return sys;
}

}  // namespace mc_tdse
