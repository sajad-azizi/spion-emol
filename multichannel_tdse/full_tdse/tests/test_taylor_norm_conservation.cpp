// test_taylor_norm_conservation.cpp
//
// For ANY hermitian H (real or complex), exp(-iHt) is unitary, hence
// ‖b(t)‖₂ = ‖b(0)‖₂.  The Taylor approximation is unitary only in the
// limit K → ∞, but for our adaptive stop (eps_rel = 1e-14) the
// per-step deviation is below ε_machine and the total deviation after
// many steps stays in the 1e-13 range.
//
// We construct a random hermitian H of moderate size (N=20), random
// state, and propagate ~10 000 steps with a step that puts ‖HΔt‖ in
// the 0.5..1 range (the "hard" regime for Taylor convergence; we want
// to confirm we don't bleed norm there).
#include "Common.hpp"
#include "TaylorStepper.hpp"

#include <cstdio>
#include <random>

int main() {
    using namespace mc_tdse;

    constexpr int    N       = 20;
    constexpr int    n_steps = 10000;
    constexpr double dt      = 0.1;

    std::mt19937_64 rng(123456);
    std::normal_distribution<double> G(0.0, 1.0);

    // Random complex matrix M; H = (M + M^H)/2 is hermitian.
    Eigen::MatrixXcd M(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            M(i, j) = dcompx(G(rng), G(rng));
    Eigen::MatrixXcd H = 0.5 * (M + M.adjoint());

    // Scale H so that ‖HΔt‖ ≈ 0.7 (mild but nontrivial per-step rotation).
    const double Hnorm = H.operatorNorm();
    H *= 0.7 / (Hnorm * dt);

    // Random initial state, normalized.
    Eigen::VectorXcd b(N);
    for (int i = 0; i < N; ++i) b(i) = dcompx(G(rng), G(rng));
    b /= b.norm();
    const double norm0 = b.norm();

    TaylorStats stats;
    double max_drift = 0.0;
    int    max_K     = 0;
    for (int s = 0; s < n_steps; ++s) {
        b = taylor_step_const_H(H, b, dt, {}, &stats);
        if (stats.last_K > max_K) max_K = stats.last_K;
        const double drift = std::abs(b.norm() - norm0);
        if (drift > max_drift) max_drift = drift;
    }
    const double final_drift = std::abs(b.norm() - norm0);

    std::printf("[norm_conservation]  N=%d  dt=%.3f  steps=%d\n", N, dt, n_steps);
    std::printf("                     ‖HΔt‖ ≈ 0.7   max Taylor K = %d\n", max_K);
    std::printf("                     max drift during run = %.3e\n", max_drift);
    std::printf("                     final |‖b‖ - 1|      = %.3e\n", final_drift);

    // Tolerance: ε_machine × √N × n_steps × tiny per-step floor.
    const bool ok = (max_drift < 5e-11) && (final_drift < 5e-11);
    std::printf("                     result : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
