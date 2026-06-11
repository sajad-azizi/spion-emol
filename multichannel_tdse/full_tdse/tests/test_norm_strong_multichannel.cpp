// test_norm_strong_multichannel.cpp -- unitarity at strong field, full
// multichannel.  Norm should stay at 1 to roundoff regardless of how
// hard we drive.  This complements test_norm_conservation_time_dep
// which used a small random 4-level toy.
#include "Common.hpp"
#include "MockMultiblock.hpp"
#include "Pulse.hpp"
#include "TDSEDriver.hpp"
#include "TDSEHamiltonian.hpp"

#include <cstdio>

int main() {
    using namespace mc_tdse;

    auto sys = make_zepe_toy(4, 0.75, 2.0, 0.5, 13579);

    const double omega   = 0.75;
    const double Omega_R = 2.0;       // very strong driving
    const double tau     = 6.0, t_c = 15.0, T = 30.0;

    TDSEHamiltonian H(sys.E, sys.d_plus, omega, Omega_R,
                      make_gaussian(tau, t_c));

    Eigen::VectorXcd b0 = Eigen::VectorXcd::Zero(sys.n_states());
    b0(sys.indices_in(-4).front()) = 1.0;

    TDSEDriverConfig cfg;
    cfg.dt = 2.0 * M_PI / (30.0 * omega);
    cfg.record_trace = true;
    cfg.trace_stride = 5;

    auto res = propagate(H, b0, 0.0, T, cfg);

    double max_drift = 0.0;
    for (const auto& row : res.trace) {
        const double d = std::fabs(row.norm - 1.0);
        if (d > max_drift) max_drift = d;
    }
    const double final_drift = std::fabs(res.b_final.norm() - 1.0);

    std::printf("[norm_strong_multichannel]   Ω_R=%.1f   N=%d   T=%.1f\n",
                Omega_R, sys.n_states(), T);
    std::printf("    max drift during run        = %.3e\n", max_drift);
    std::printf("    final |‖b‖ - 1|             = %.3e\n", final_drift);

    const bool ok = (max_drift < 1e-12) && (final_drift < 1e-12);
    std::printf("    result: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
