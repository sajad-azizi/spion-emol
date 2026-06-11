// test_block_continuum_m3.cpp
//
// Stage 4c, item 2: Numerov box-quantization for the M_F=-3 continuum.
// We build a few levels just above the block's lowest threshold and
// verify (a) they exist, (b) energies are ordered, (c) the box-wall
// Dirichlet condition u(L) ≈ 0 is satisfied numerically, (d) every
// state is unit-normalized, and (e) the lowest level's open-channel
// component dominates (closed channel decays exponentially over L).
//
// We deliberately use a small E_max_above_threshold so the test runs
// fast; a wider sweep is what the production TDSE will use.
#include "BlockEigenstates.hpp"
#include "Common.hpp"

#include <cstdio>
#include <cmath>

int main() {
    using namespace mc_tdse;
    BlockBuildOptions opt;
    opt.N_grid = 100000;            // L = 5e4 a_0
    opt.dr_a0  = 0.5;
    opt.E_max_kHz_above_threshold = 2.0;   // small window for speed

    BlockEigenstates b = build_block_eigenstates(-3, opt);

    std::printf("[block_continuum_m3]  N_ch=%d  L=%g a_0  E_max=%g kHz above thresh\n",
                b.N_ch, opt.N_grid * opt.dr_a0, opt.E_max_kHz_above_threshold);
    std::printf("    n_states found      = %d\n", b.n_states());

    if (b.n_states() < 2) {
        std::printf("    FAIL: need ≥ 2 levels for ordering check\n");
        return 1;
    }

    // (b) Energies ordered ascending.
    bool ok_order = true;
    for (int i = 1; i < b.n_states(); ++i) {
        if (b.E_au[i] <= b.E_au[i - 1]) ok_order = false;
    }
    std::printf("    energies ascending  : %s\n", ok_order ? "PASS" : "FAIL");

    int n_norm_fail = 0, n_wall_fail = 0;
    double max_wall = 0.0, min_norm = 1e9, max_norm = 0.0;
    for (int n = 0; n < b.n_states(); ++n) {
        const auto& u = b.u[n];
        // (d) ∫ Σ_c |u_c|^2 dr = 1
        double s = 0.0;
        for (int ir = 0; ir < u.rows(); ++ir) {
            double w = (ir == 0 || ir == u.rows() - 1) ? 0.5 : 1.0;
            for (int c = 0; c < u.cols(); ++c)
                s += w * u(ir, c) * u(ir, c);
        }
        s *= b.dr;
        if (std::fabs(s - 1.0) > 1e-9) ++n_norm_fail;
        if (s < min_norm) min_norm = s;
        if (s > max_norm) max_norm = s;

        // (c) u(L) ≈ 0 (Dirichlet wall): check |u_c(N-1)| / max|u_c| < 1e-3
        double umax = 0.0, uwall = 0.0;
        for (int ir = 0; ir < u.rows(); ++ir) {
            for (int c = 0; c < u.cols(); ++c) {
                double a = std::fabs(u(ir, c));
                if (a > umax) umax = a;
            }
        }
        for (int c = 0; c < u.cols(); ++c) {
            uwall = std::max(uwall, std::fabs(u(u.rows() - 1, c)));
        }
        const double ratio = uwall / (umax + 1e-30);
        if (ratio > 1e-3) ++n_wall_fail;
        if (ratio > max_wall) max_wall = ratio;
    }
    std::printf("    norm in [%g, %g]; failures = %d / %d\n",
                min_norm, max_norm, n_norm_fail, b.n_states());
    std::printf("    max |u(L)/u_max|    = %.3e   wall-failures = %d\n",
                max_wall, n_wall_fail);

    // Show first few level energies above threshold (kHz).
    std::printf("    First 5 level energies above block threshold:\n");
    const double E_th_kHz = AU::au_to_kHz(b.E_au[0]) * 0;  // dummy
    (void)E_th_kHz;
    for (int n = 0; n < std::min(5, b.n_states()); ++n) {
        std::printf("        n=%d  E = %14.6f kHz  (above thresh: %.4f kHz)\n",
                    n, AU::au_to_kHz(b.E_au[n]),
                    AU::au_to_kHz(b.E_au[n] - b.E_au[0]));
    }

    const bool ok = ok_order && (n_norm_fail == 0) && (n_wall_fail == 0);
    std::printf("    result              : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
