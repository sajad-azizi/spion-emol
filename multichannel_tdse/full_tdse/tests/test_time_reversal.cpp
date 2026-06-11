// test_time_reversal.cpp -- the centerpiece test.
//
// For a Hermitian H(t), the midpoint Taylor stepper is exactly unitary
// per step, AND its inverse is exp(+i H_mid Δt) = exp(-i H_mid·(-Δt)).
// So
//
//     propagate_reverse( propagate(b₀) ) == b₀     to roundoff
//
// independent of:
//   * field strength (works for arbitrarily strong Ω_R, π-pulses, etc.)
//   * pulse shape
//   * multilevel structure
//   * Hamiltonian time-dependence
//
// This is the strongest single check we can construct -- any sign /
// phase / accumulator / asymmetry bug in the stepper or hamiltonian
// builder shows up as a non-zero round-trip error.
#include "Common.hpp"
#include "MockMultiblock.hpp"
#include "Pulse.hpp"
#include "TDSEDriver.hpp"
#include "TDSEHamiltonian.hpp"

#include <cstdio>

int main() {
    using namespace mc_tdse;

    auto sys = make_zepe_toy(/*n_per_block=*/4,
                             /*omega_Z=*/0.75,
                             /*delta_m5=*/2.0,
                             /*coupling=*/0.5,
                             /*seed=*/97531);

    // Strong field, full multichannel, both σ⁺ and σ⁻ active.
    const double omega   = 0.75;
    const double Omega_R = 1.5;        // very strong; multi-π pulse area
    const double tau     = 8.0, t_c = 20.0, T = 40.0;

    TDSEHamiltonian H(sys.E, sys.d_plus, omega, Omega_R,
                      make_gaussian(tau, t_c));

    Eigen::VectorXcd b0 = Eigen::VectorXcd::Zero(sys.n_states());
    b0(sys.indices_in(-4).front()) = 1.0;
    const Eigen::VectorXcd b0_save = b0;

    TDSEDriverConfig cfg;
    cfg.dt = 2.0 * M_PI / (40.0 * omega);

    // Forward
    auto fwd = propagate(H, b0, 0.0, T, cfg);
    const double P_after_fwd = (fwd.b_final - b0_save).norm();

    // Backward
    auto bwd = propagate_reverse(H, fwd.b_final, 0.0, T, cfg);
    const double round_trip_err = (bwd.b_final - b0_save).norm();

    std::printf("[time_reversal]\n");
    std::printf("    Ω_R=%.2f (strong, multi-π)   T=%.1f   steps=%.0f\n",
                Omega_R, T, T / cfg.dt);
    std::printf("    ‖b(T) - b₀‖ after forward-only       = %.3e\n", P_after_fwd);
    std::printf("    ‖propagate_reverse(propagate(b₀)) - b₀‖\n");
    std::printf("    round-trip error                     = %.3e\n", round_trip_err);

    const bool ok = (round_trip_err < 1e-11) && (P_after_fwd > 0.1);
    std::printf("    result: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
