// test_dt_convergence_multichannel.cpp -- Δt² midpoint convergence on
// the full 4-block ZEPE toy.  Reference is a very-fine-Δt run; halving
// Δt should cut error by ~4.
#include "Common.hpp"
#include "MockMultiblock.hpp"
#include "Pulse.hpp"
#include "TDSEDriver.hpp"
#include "TDSEHamiltonian.hpp"

#include <cstdio>
#include <vector>

int main() {
    using namespace mc_tdse;

    auto sys = make_zepe_toy(4, 0.75, 2.0, 0.5, 24680);
    const double omega   = 0.7;       // OFF-resonant (so commutators non-trivial)
    const double Omega_R = 0.5;
    const double tau     = 8.0, t_c = 20.0, T = 40.0;

    TDSEHamiltonian H(sys.E, sys.d_plus, omega, Omega_R,
                      make_gaussian(tau, t_c));
    Eigen::VectorXcd b0 = Eigen::VectorXcd::Zero(sys.n_states());
    b0(sys.indices_in(-4).front()) = 1.0;

    auto run = [&](double dt) {
        TDSEDriverConfig cfg; cfg.dt = dt;
        return propagate(H, b0, 0.0, T, cfg).b_final;
    };

    // Reference: VERY fine.
    const double dt_ref = 1.0e-3;
    Eigen::VectorXcd b_ref = run(dt_ref);

    const std::vector<double> dts = {0.5, 0.25, 0.125, 0.0625, 0.03125};
    std::printf("[dt_convergence_multichannel]  N=%d  T=%.1f  ref_dt=%.4f\n",
                sys.n_states(), T, dt_ref);
    std::printf("    dt           ‖b - b_ref‖_2          ratio\n");
    std::vector<double> errs;
    for (double dt : dts) {
        Eigen::VectorXcd b = run(dt);
        const double err = (b - b_ref).norm();
        std::printf("    %-10.5f   %.3e", dt, err);
        if (!errs.empty()) {
            const double r = errs.back() / std::max(err, 1e-300);
            std::printf("    %.3f", r);
        }
        std::printf("\n");
        errs.push_back(err);
    }
    const double r_last1 = errs[errs.size()-3] / errs[errs.size()-2];
    const double r_last2 = errs[errs.size()-2] / errs[errs.size()-1];
    std::printf("    last two ratios = %.3f, %.3f (target ~4)\n", r_last1, r_last2);
    const bool ok = (r_last1 > 3.0 && r_last1 < 5.0) &&
                    (r_last2 > 3.0 && r_last2 < 5.0);
    std::printf("    result: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
