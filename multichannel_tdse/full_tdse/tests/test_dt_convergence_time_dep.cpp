// test_dt_convergence_time_dep.cpp -- midpoint freezing gives global
// O(Δt²) error for time-dependent H WHEN the commutator [H(t₁),H(t₂)]
// is non-zero.  At resonance in a 2-level system, M(t) = (Ω_R/2)·χ(t)·σ_x
// commutes with itself at all times, so the midpoint propagator is
// exact up to roundoff -- not a useful Δt² test.
//
// Here we use:
//   * 3-level system with non-degenerate E_i (genuine multilevel coupling)
//   * Off-resonant carrier (commutators non-trivial)
//   * Random complex d^(+)
// and verify err(Δt/2)/err(Δt) → 4 as Δt→0.
#include "Common.hpp"
#include "Pulse.hpp"
#include "TDSEDriver.hpp"
#include "TDSEHamiltonian.hpp"

#include <cstdio>
#include <random>
#include <vector>

int main() {
    using namespace mc_tdse;

    constexpr int N = 3;
    const double omega   = 0.7;       // OFF-resonant, no degenerate transitions
    const double Omega_R = 0.05;      // moderate driving
    const double tau     = 30.0, t_c = 80.0, T = 160.0;

    Eigen::VectorXd E(N);
    E << 0.0, 1.0, 1.7;               // non-degenerate, no resonance with ω

    // Random complex d^(+) (M(t) construction Hermitizes via d^- = (d^+)†).
    std::mt19937_64 rng(31415);
    std::normal_distribution<double> G(0.0, 1.0);
    Eigen::MatrixXcd dp(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            dp(i, j) = dcompx(G(rng), G(rng));

    Eigen::VectorXcd b0(N);
    b0 << dcompx(1.0, 0.0), dcompx(0.0, 0.0), dcompx(0.0, 0.0);

    auto run = [&](double dt) {
        TDSEHamiltonian H(E, dp, omega, Omega_R, make_gaussian(tau, t_c));
        TDSEDriverConfig cfg; cfg.dt = dt;
        return propagate(H, b0, 0.0, T, cfg).b_final;
    };

    // Reference: very fine dt.  We need a lot finer than the test grid
    // so that "err = ‖b - b_ref‖" is dominated by the test grid's Δt².
    const double dt_ref = 0.001;
    Eigen::VectorXcd b_ref = run(dt_ref);
    std::printf("[Δt convergence]  3-level off-resonant test\n");
    std::printf("    reference dt = %.4f   |b_ref(0)| = %.4e   |b_ref(1)| = %.4e\n",
                dt_ref, std::abs(b_ref(0)), std::abs(b_ref(1)));

    // Halve Δt successively and watch error against b_ref.
    const std::vector<double> dts = {0.5, 0.25, 0.125, 0.0625, 0.03125};
    std::printf("    dt           ‖b - b_ref‖_2          ratio (target ~4)\n");
    std::vector<double> errs;
    for (double dt : dts) {
        Eigen::VectorXcd b = run(dt);
        const double err = (b - b_ref).norm();
        std::printf("    %-10.5f   %.3e", dt, err);
        if (!errs.empty()) {
            const double ratio = errs.back() / std::max(err, 1e-300);
            std::printf("    %.3f", ratio);
        }
        std::printf("\n");
        errs.push_back(err);
    }
    // Expected: each halving of dt → error / 4 (Δt² midpoint).  We
    // require the LAST two ratios to land in [3.0, 5.0].
    const double r1 = errs[errs.size() - 3] / errs[errs.size() - 2];
    const double r2 = errs[errs.size() - 2] / errs[errs.size() - 1];
    std::printf("    last two ratios = %.3f, %.3f (target ~4)\n", r1, r2);
    const bool ok = (r1 > 3.0 && r1 < 5.0) && (r2 > 3.0 && r2 < 5.0);
    std::printf("    result: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
