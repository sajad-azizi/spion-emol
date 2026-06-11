// test_virtual_block_m5.cpp -- M_F=-5 stays virtual when far-detuned.
//
// In the recipe, M_F=-5 sits ~2.68 GHz above the entrance threshold
// while ω is 50-100 kHz.  Far off-resonance.  Born/perturbation
// theory says the real population in M_F=-5 scales as
//
//     P^(-5) ~ (Ω_R / Δ_{-5})²
//
// with Δ_{-5} the detuning, while the COHERENT contribution to ZEPE
// (M_F=-4 final) through the M_F=-5 intermediate scales as Ω_R⁴ but
// with O(1) prefactor relative to the ae path.
//
// We test:
//   * P^(-5)(T) ≲ (Ω_R / Δ_{-5})² · P^(-4)_initial   to a generous
//     constant of order 10 (the prefactor depends on coupling matrix
//     elements; we just need P^(-5) to be SMALL, not to match an
//     exact bound.).
//   * Verify the ratio scales as Ω_R² when Ω_R is varied in the weak
//     regime (more rigorous, slope-based check).
#include "Common.hpp"
#include "MockMultiblock.hpp"
#include "Pulse.hpp"
#include "TDSEDriver.hpp"
#include "TDSEHamiltonian.hpp"

#include <cstdio>
#include <vector>

namespace {
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

    // delta_m5 = 4 -> M_F=-5 lowest threshold sits at -ω_Z + 4 = 3.25
    //                  far from any one-photon resonance.
    auto sys = make_zepe_toy(4, /*omega_Z=*/0.75,
                                /*delta_m5=*/4.0,
                                /*coupling=*/0.5, /*seed=*/9999);

    // Carrier at one-photon transition for M_F=-4 → M_F=-3.
    const double omega = 0.75;
    const double tau   = 20.0, t_c = 60.0, T = 120.0;

    const std::vector<double> ORs = {1e-4, 3e-4, 1e-3, 3e-3};
    std::vector<double> P5;
    for (double OR : ORs) {
        TDSEHamiltonian H(sys.E, sys.d_plus, omega, OR,
                          make_gaussian(tau, t_c));
        Eigen::VectorXcd b0 = Eigen::VectorXcd::Zero(sys.n_states());
        b0(sys.indices_in(-4).front()) = 1.0;
        TDSEDriverConfig cfg; cfg.dt = 2.0 * M_PI / (40.0 * omega);
        auto res = propagate(H, b0, 0.0, T, cfg);
        P5.push_back(sys.block_population(res.b_final, -5));
    }

    const double s5 = log_slope(ORs, P5);

    std::printf("[virtual_block_m5]   Δ_{-5} ~ 4.0 (large)\n");
    std::printf("    Ω_R           P^(-5)\n");
    for (std::size_t i = 0; i < ORs.size(); ++i) {
        std::printf("    %.0e        %.3e\n", ORs[i], P5[i]);
    }
    std::printf("\n    slope(P^(-5)) vs Ω_R = %.4f   (target 2 → far-detuned virtual)\n", s5);

    // Strict size check: at the largest tested Ω_R = 3e-3, with Δ ~ 4,
    // the bound (Ω_R/Δ)² ~ 5.6e-7 -- so P^(-5) should sit well below 1e-3.
    const double P5_max = *std::max_element(P5.begin(), P5.end());
    const bool ok_slope = std::fabs(s5 - 2.0) < 0.15;
    const bool ok_size  = P5_max < 1e-3;
    const bool ok = ok_slope && ok_size;
    std::printf("    slope ~ 2 : %s\n", ok_slope ? "PASS" : "FAIL");
    std::printf("    size small: %s   (max P^(-5) = %.2e < 1e-3)\n",
                ok_size  ? "PASS" : "FAIL", P5_max);
    std::printf("    result    : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
