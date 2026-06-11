// test_power_scaling_weak.cpp -- weak-field corner check (recipe item 8).
//
// Vary Ω_R over a decade in the perturbative regime; populations
// should obey:
//
//   * P^(-3) ∝ Ω_R²        (one-photon)
//   * P_aa^(-2) ∝ Ω_R⁴     (absorb-absorb)
//   * P_ae+ea^(-4-final)  ∝ Ω_R⁴   (ZEPE -- aggregate, both pathways)
//
// Slopes are extracted from log-log fits.  Targets: 2.0 and 4.0 within
// ±0.10.  This is NOT the primary test of the propagator -- it's a
// sanity check that the multichannel coupling structure is wired up
// correctly.
#include "Common.hpp"
#include "MockMultiblock.hpp"
#include "Pulse.hpp"
#include "TDSEDriver.hpp"
#include "TDSEHamiltonian.hpp"

#include <cstdio>
#include <vector>

namespace {

// Linear regression of (log y) on (log x); returns the slope.
double log_slope(const std::vector<double>& xs,
                 const std::vector<double>& ys)
{
    const int n = static_cast<int>(xs.size());
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < n; ++i) {
        const double lx = std::log(xs[i]);
        const double ly = std::log(ys[i]);
        sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly;
    }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

}  // namespace

int main() {
    using namespace mc_tdse;

    auto sys_template = make_zepe_toy(4, 0.75, 2.0, 0.5, 24681);

    const int N = sys_template.n_states();
    const int i_h = sys_template.indices_in(-4).front();   // halo

    const double omega = 0.75;
    const double tau   = 30.0, t_c = 80.0, T = 160.0;

    const std::vector<double> ORs = {1e-5, 3e-5, 1e-4, 3e-4, 1e-3};
    std::vector<double> P3, P2, P4final;

    for (double OR : ORs) {
        TDSEHamiltonian H(sys_template.E, sys_template.d_plus, omega, OR,
                          make_gaussian(tau, t_c));
        Eigen::VectorXcd b0 = Eigen::VectorXcd::Zero(N);
        b0(i_h) = 1.0;
        TDSEDriverConfig cfg; cfg.dt = 2.0 * M_PI / (40.0 * omega);
        auto res = propagate(H, b0, 0.0, T, cfg);

        // P^(-3) total.
        P3.push_back(sys_template.block_population(res.b_final, -3));
        // P^(-2) total (= aa background; nothing else exits in -2 in this
        //               model since no σ⁻ vertex is supplied).
        P2.push_back(sys_template.block_population(res.b_final, -2));
        // P in M_F=-4 final states (excluding the halo itself).
        double p4 = 0.0;
        for (int idx : sys_template.indices_in(-4)) {
            if (idx == i_h) continue;
            p4 += std::norm(res.b_final(idx));
        }
        P4final.push_back(p4);
    }

    const double s3  = log_slope(ORs, P3);
    const double s2  = log_slope(ORs, P2);
    const double s4f = log_slope(ORs, P4final);

    std::printf("[power_scaling_weak]\n");
    std::printf("    Ω_R           P^(-3)             P^(-2)             P^(-4 ZEPE final)\n");
    for (std::size_t i = 0; i < ORs.size(); ++i) {
        std::printf("    %.0e        %.3e          %.3e          %.3e\n",
                    ORs[i], P3[i], P2[i], P4final[i]);
    }
    std::printf("\n    log-log slopes:\n");
    std::printf("       slope(P^(-3))         = %.4f   (target 2)\n", s3);
    std::printf("       slope(P^(-2))         = %.4f   (target 4)\n", s2);
    std::printf("       slope(P^(-4) ZEPE)    = %.4f   (target 4)\n", s4f);

    const bool ok3  = std::fabs(s3 - 2.0) < 0.1;
    const bool ok2  = std::fabs(s2 - 4.0) < 0.1;
    const bool ok4f = std::fabs(s4f- 4.0) < 0.1;
    const bool ok = ok3 && ok2 && ok4f;
    std::printf("    1-photon (slope 2) : %s\n", ok3  ? "PASS" : "FAIL");
    std::printf("    aa       (slope 4) : %s\n", ok2  ? "PASS" : "FAIL");
    std::printf("    ZEPE     (slope 4) : %s\n", ok4f ? "PASS" : "FAIL");
    std::printf("    result : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
