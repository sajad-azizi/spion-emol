// test_strong_rabi_isolated_pair.cpp -- inside the multichannel system,
// build a d^(+) that couples ONLY one specific pair of states (one in
// M_F=-4, one in M_F=-3), at the resonant carrier ω = E_e - E_g.  All
// other off-block entries set to zero.  The exact RWA answer is a
// Rabi flop: |b_e|² = sin²(θ/2),  θ = Ω_R · ∫χ dt, with the rest of
// the system staying *exactly* at zero population.
//
// This is a "leak test" -- if the stepper accidentally couples to
// other states (matrix-build bug, bad indexing, etc.) those states
// would pick up amplitude.  At strong field, π-pulse area, even tiny
// stray couplings would show.
#include "Common.hpp"
#include "MockMultiblock.hpp"
#include "Pulse.hpp"
#include "TDSEDriver.hpp"
#include "TDSEHamiltonian.hpp"

#include <cstdio>

int main() {
    using namespace mc_tdse;

    auto sys = make_zepe_toy(4, 0.75, 2.0, 0.5, 11111);

    // Pick the M_F=-4 halo (idx i_g) and one M_F=-3 state (idx i_e).
    const int i_g = sys.indices_in(-4).front();
    const int i_e = sys.indices_in(-3).back();

    // Wipe d^(+); install a single non-zero entry: σ⁺ raises i_g -> i_e.
    Eigen::MatrixXcd dp = Eigen::MatrixXcd::Zero(sys.n_states(), sys.n_states());
    dp(i_e, i_g) = dcompx(1.0, 0.0);

    // Carrier RESONANT: ω = E_e - E_g.  No other transition is resonant.
    const double E_g = sys.E(i_g);
    const double E_e = sys.E(i_e);
    const double omega = E_e - E_g;

    // Strong field, π-pulse area (θ = π).  ∫χ_Gauss = √(2π)·τ → θ = Ω_R·√(2π)·τ.
    const double tau = 4.0;
    const double t_c = 12.0;
    const double T   = 24.0;
    const double Omega_R = M_PI / (std::sqrt(2.0 * M_PI) * tau);   // gives θ ≈ π

    TDSEHamiltonian H(sys.E, dp, omega, Omega_R,
                      make_gaussian(tau, t_c));

    Eigen::VectorXcd b0 = Eigen::VectorXcd::Zero(sys.n_states());
    b0(i_g) = 1.0;

    TDSEDriverConfig cfg;
    cfg.dt = 2.0 * M_PI / (60.0 * omega);
    auto res = propagate(H, b0, 0.0, T, cfg);

    const double pop_g = std::norm(res.b_final(i_g));
    const double pop_e = std::norm(res.b_final(i_e));
    double pop_other_max = 0.0;
    for (int j = 0; j < sys.n_states(); ++j) {
        if (j == i_g || j == i_e) continue;
        const double p = std::norm(res.b_final(j));
        if (p > pop_other_max) pop_other_max = p;
    }
    double pop_other_sum = 0.0;
    for (int j = 0; j < sys.n_states(); ++j) {
        if (j == i_g || j == i_e) continue;
        pop_other_sum += std::norm(res.b_final(j));
    }
    // Analytic π-pulse: pop_e ≈ 1, pop_g ≈ 0.  Counter-rotating term
    // gives ~ (Ω_R/(2ω))² shift; here Ω_R ≈ 0.31, ω ≈ ω_Z ≈ 0.75,
    // so correction ~ 4%.  We test pop_e > 0.9.
    std::printf("[isolated_rabi]  i_g=%d  i_e=%d  ω=%.4f  Ω_R=%.4f  τ=%.1f\n",
                i_g, i_e, omega, Omega_R, tau);
    std::printf("    pulse area θ = Ω_R √(2π) τ = %.4f rad (target π)\n",
                Omega_R * std::sqrt(2.0 * M_PI) * tau);
    std::printf("    pop_g (start state)         = %.6e\n", pop_g);
    std::printf("    pop_e (target state)        = %.6e\n", pop_e);
    std::printf("    max pop in any other state  = %.3e   (must be 0)\n", pop_other_max);
    std::printf("    sum pop other states        = %.3e   (must be 0)\n", pop_other_sum);

    const bool ok_flop = pop_e > 0.9 && pop_g < 0.1;
    const bool ok_leak = pop_other_max < 1e-15;
    const bool ok = ok_flop && ok_leak;
    std::printf("    π-flop : %s\n", ok_flop ? "PASS" : "FAIL");
    std::printf("    no leak: %s\n", ok_leak ? "PASS" : "FAIL");
    std::printf("    result : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
