// test_halo_weights.cpp -- recipe item 2 of the Full-TDSE convergence
// checklist:
//
//   Halo weights:  P_open = 0.99981, P_closed = 1.901e-4
//
// These are the integrals ⟨u_α|u_α⟩ for the two M_F=-4 channels of
// the analytic halo state, normalized so that P_open + P_closed = 1.
//
// We integrate on a grid extending many halo lengths (1/κ ≈ 2050 a_0)
// past r_0 so the closed-channel exponential tail is captured
// cleanly.  Tolerance ±1e-5 (recipe gives 5 significant figures).
#include "Common.hpp"
#include "Rb85Halo.hpp"

#include <cstdio>

int main() {
    using namespace mc_tdse;
    Rb85HaloOptions opt;
    // 1/κ ≈ 2050 a_0 (recipe).  Use L = 50/κ ~ 100 000 a_0 → very
    // small residual from cutting the tail (e^{-50} ~ 1e-22).
    opt.N_grid_numerov = 200000;
    opt.dr_numerov     = 0.5;          // L = 100 000 a_0

    const HaloWeights w = halo_weights_analytic(opt);
    const double P_open_target   = 0.99981;
    const double P_closed_target = 1.901e-4;

    const double err_open   = std::fabs(w.P_open   - P_open_target);
    const double err_closed = std::fabs(w.P_closed - P_closed_target);

    std::printf("[halo_weights_analytic]\n");
    std::printf("    P_open    = %.6f      (recipe %.6f)   |Δ| = %.3e\n",
                w.P_open,   P_open_target,   err_open);
    std::printf("    P_closed  = %.6e      (recipe %.6e)   |Δ| = %.3e\n",
                w.P_closed, P_closed_target, err_closed);
    std::printf("    sum       = %.10f\n", w.P_open + w.P_closed);

    const bool ok_open   = err_open   < 1e-5;
    const bool ok_closed = err_closed < 1e-5;
    const bool ok_sum    = std::fabs(w.P_open + w.P_closed - 1.0) < 1e-12;
    const bool ok = ok_open && ok_closed && ok_sum;
    std::printf("    P_open match    : %s\n", ok_open   ? "PASS" : "FAIL");
    std::printf("    P_closed match  : %s\n", ok_closed ? "PASS" : "FAIL");
    std::printf("    sum = 1         : %s\n", ok_sum    ? "PASS" : "FAIL");
    std::printf("    result          : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
