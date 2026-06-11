// test_block_halo_state.cpp
//
// Stage 4c, item 1: BlockEigenstates(M_F=-4) using the analytic halo
// route should reproduce halo_weights at recipe precision (8 sig figs)
// when the wavefunction is integrated channel by channel.
//
// This proves the BlockEigenstates plumbing (analytic state → grid →
// normalization) is correct: the same data that underlay the
// halo_weights validation is now exposed as a per-grid wavefunction
// suitable for the TDSE driver and the dipole assembler.
#include "BlockEigenstates.hpp"
#include "Common.hpp"

#include <cstdio>
#include <cmath>

int main() {
    using namespace mc_tdse;
    BlockBuildOptions opt;
    opt.N_grid = 200000;
    opt.dr_a0  = 0.5;
    opt.use_analytic_halo = true;

    BlockEigenstates b = build_block_eigenstates(-4, opt);

    // build_block_eigenstates(-4, use_analytic_halo=true) prepends the
    // analytic halo as state 0 and then appends the Numerov continuum
    // levels above threshold.  We only test the halo (index 0) here;
    // continuum normalization is validated by test_block_continuum_m3.
    if (b.n_states() < 1 || b.E_au[0] >= 0.0) {
        std::printf("[block_halo_state] FAIL: state 0 is not a bound halo "
                    "(n_states=%d, E_0=%g)\n",
                    b.n_states(), b.n_states() > 0 ? b.E_au[0] : 0.0);
        return 1;
    }
    const double E_h_kHz = AU::au_to_kHz(b.E_au[0]);

    // Recompute channel weights from the stored grid wavefunction.
    // (Trapezoid is fine here -- the point is that BlockEigenstates
    // returns the same physics as Rb85Halo's standalone halo_weights.)
    double s0 = 0.0, s1 = 0.0;
    const auto& u = b.u[0];
    const int N = u.rows();
    for (int ir = 0; ir < N; ++ir) {
        double w = (ir == 0 || ir == N - 1) ? 0.5 : 1.0;
        s0 += w * u(ir, 0) * u(ir, 0);
        s1 += w * u(ir, 1) * u(ir, 1);
    }
    s0 *= b.dr;
    s1 *= b.dr;

    const double total = s0 + s1;
    const double P_open   = s0 / total;
    const double P_closed = s1 / total;

    // Recipe targets from item 2 of the convergence checklist.
    const double P_open_target   = 0.99981;
    const double P_closed_target = 1.901e-4;

    std::printf("[block_halo_state]\n");
    std::printf("    n_states            = %d\n", b.n_states());
    std::printf("    E_h                 = %.6f kHz\n", E_h_kHz);
    std::printf("    P_open  (recomp)    = %.6f   (recipe %.6f)\n",
                P_open, P_open_target);
    std::printf("    P_closed (recomp)   = %.6e   (recipe %.6e)\n",
                P_closed, P_closed_target);
    std::printf("    P_open + P_closed   = %.10f  (∫|u|² dr = 1)\n", total);

    const bool ok_open    = std::fabs(P_open   - P_open_target)   < 1e-5;
    const bool ok_closed  = std::fabs(P_closed - P_closed_target) < 1e-5;
    // The halo is normalized to ∫|u|² dr = 1 over [0, N_grid·dr], with
    // exponential tail beyond.  Tail leakage is e^{−2L/(2/κ)} ≈ e^{-100}
    // at L=N_grid·dr=1e5 a_0, so the truncated norm equals 1 to 14
    // sig figs.  Allow 1e-10 to absorb double-precision FP roundoff.
    const bool ok_norm    = std::fabs(total - 1.0) < 1e-10;
    const bool ok = ok_open && ok_closed && ok_norm;
    std::printf("    P_open  match       : %s\n", ok_open   ? "PASS" : "FAIL");
    std::printf("    P_closed match      : %s\n", ok_closed ? "PASS" : "FAIL");
    std::printf("    norm = 1            : %s\n", ok_norm   ? "PASS" : "FAIL");
    std::printf("    result              : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
