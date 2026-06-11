// test_numerov_stack.cpp -- Numerov-stack validator (correlation form).
//
// Why this test, and not a Numerov-vs-analytic eigenvalue match?
//
// (i)  The HALO is not Numerov-resolvable on any tractable grid: the
//      binding fraction is 1 in 10^6 of the well depth, so node-count
//      and matching-condition bisection both fail (smooth plateau, no
//      sign change in any reasonable bracket).  We therefore use the
//      analytic square-well bound state for the halo throughout.
//
// (ii) Even DEEPER analytic eigenvalues can be O(MHz) shifted by
//      Numerov O(h^4) discretization at dr=0.5 a_0, since the radial
//      wavefunction has a kink at r_0 (the square well's edge) and
//      Numerov's local truncation depends on smoothness.  That makes
//      Numerov-vs-analytic eigenvalue comparison unreliable as a
//      stack-correctness witness.
//
// What IS independent of those discretization shifts:
//   At every analytic eigenvalue E_n, the discretized Numerov spectrum
//   has a state SOMEWHERE near E_n.  In particular,
//     nc_Numerov(E_n + δ) − nc_Numerov(E_n − δ) ≥ 1
//   for any δ that brackets E_n's discretized counterpart but no
//   neighbouring analytic state.  We pick δ = (E_{n+1} − E_n)/3, half
//   the gap to the nearest analytic neighbour, which is large enough
//   to absorb O(MHz) discretization shifts but small enough to
//   straddle exactly one analytic state.
//
// This catches every "wired wrong" failure mode (potential matrix,
// reduced mass, propagator coefficients, near-origin initialization)
// without being fooled by O(h^4) eigenvalue shifts.
#include "Common.hpp"
#include "Rb85Halo.hpp"

#include <cstdio>
#include <algorithm>

int main() {
    using namespace mc_tdse;
    Rb85HaloOptions opt;
    // dr=0.1 a_0; O(h^4)·V_T ≈ 1.5e-10 Ha = 100 Hz; should shift even
    // the shallowest deep states (~ -50 MHz) by under a kHz, well
    // within the smallest gap/3 (~17 MHz).
    // L = N·dr = 5e4 a_0 (~25/κ for halo, ample for any bound state).
    opt.N_grid_numerov = 500000;
    opt.dr_numerov     = 0.1;
    opt.p_init_numerov = 3;

    // All analytic bound states of M_F=-4 below threshold, excluding
    // the halo (Numerov can't resolve it).  Well depth ≈ 9.7 GHz, so
    // bracket [-9.5 GHz, -100 kHz] captures the ~20 deep eigenvalues.
    const double E_lo_kHz = -9.5e6;     // -9.5 GHz (just inside V_T)
    const double E_hi_kHz = -1.0e2;     // -100 kHz  (halo at -10 kHz EXCLUDED)
    auto E_states = mf4_bound_states_analytic(opt, E_lo_kHz, E_hi_kHz);

    std::printf("[numerov_stack]  %zu analytic eigenvalues in [%g, %g] kHz\n",
                E_states.size(), E_lo_kHz, E_hi_kHz);
    if (E_states.size() < 2) {
        std::printf("    Not enough states to pick δ; widen bracket.\n");
        return 1;
    }
    // Sort ascending (most negative first → least negative)
    std::sort(E_states.begin(), E_states.end());

    int n_pass = 0;
    int n_fail = 0;
    for (size_t i = 0; i < E_states.size(); ++i) {
        const double E_n = E_states[i];
        // δ = 1/3 of the smaller gap to neighbours (or to bracket edges).
        double gap_lo = (i == 0) ? std::abs(E_n - AU::kHz_to_au(E_lo_kHz))
                                 : (E_n - E_states[i - 1]);
        double gap_hi = (i + 1 == E_states.size())
                        ? std::abs(AU::kHz_to_au(E_hi_kHz) - E_n)
                        : (E_states[i + 1] - E_n);
        const double delta = std::min(gap_lo, gap_hi) / 3.0;

        const int nc_below = mf4_node_count_numerov(E_n - delta, opt);
        const int nc_above = mf4_node_count_numerov(E_n + delta, opt);
        const int dnc      = nc_above - nc_below;
        const bool ok      = (dnc >= 1);

        if (i < 3 || i + 3 >= E_states.size() || !ok) {
            std::printf("    n=%2zu  E_n = %14.4f kHz  δ = %9.4f kHz  "
                        "nc(±δ) = (%2d,%2d)  Δnc = %d  %s\n",
                        i, AU::au_to_kHz(E_n), AU::au_to_kHz(delta),
                        nc_below, nc_above, dnc, ok ? "ok" : "FAIL");
        }
        if (ok) ++n_pass; else ++n_fail;
    }
    std::printf("    pass / fail       : %d / %d\n", n_pass, n_fail);
    std::printf("    result            : %s\n",
                n_fail == 0 ? "PASS" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}
