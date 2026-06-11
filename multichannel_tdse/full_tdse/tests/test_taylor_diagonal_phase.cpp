// test_taylor_diagonal_phase.cpp
//
// For a DIAGONAL Hamiltonian H = diag(E_1, ..., E_N), the exact propagator
// over Δt is U = diag(e^{-i E_1 Δt}, ..., e^{-i E_N Δt}).  Each component
// of the state |c⟩ should evolve by an independent phase; magnitudes are
// preserved EXACTLY.
//
// The Taylor stepper with K large enough should reproduce this to ~1e-14
// per component.  This is the strictest analytic test of the stepper:
// any sign / coefficient bug shows up immediately.
#include "Common.hpp"
#include "TaylorStepper.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>

int main() {
    using namespace mc_tdse;

    constexpr int     N    = 12;
    constexpr double  dt   = 0.7;
    constexpr int     n_steps = 50;

    // Eigenvalues spanning a few orders of magnitude.
    Eigen::VectorXd E(N);
    for (int i = 0; i < N; ++i) E(i) = 0.05 * (i + 1) * (1.0 + 0.001 * i * i);
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(N, N);
    for (int i = 0; i < N; ++i) H(i, i) = E(i);

    // Random initial state.
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    Eigen::VectorXcd b0(N);
    for (int i = 0; i < N; ++i) b0(i) = dcompx(U(rng), U(rng));
    b0 /= b0.norm();

    Eigen::VectorXcd b = b0;
    TaylorStats stats;
    for (int s = 0; s < n_steps; ++s) {
        b = taylor_step_const_H(H, b, dt, {}, &stats);
    }
    const double T = dt * n_steps;

    // Analytic: c_i(T) = c_i(0) · e^{-i E_i T}
    Eigen::VectorXcd b_exact(N);
    for (int i = 0; i < N; ++i) {
        b_exact(i) = b0(i) * std::exp(-I_unit * E(i) * T);
    }

    const double err = (b - b_exact).cwiseAbs().maxCoeff();
    const double mag_err = std::abs(b.norm() - 1.0);
    std::printf("[diag] N=%d  dt=%.3f  steps=%d  T=%.3f\n", N, dt, n_steps, T);
    std::printf("       last K used = %d   last residual = %.2e\n",
                stats.last_K, stats.last_res);
    std::printf("       max |b - b_exact|       = %.3e\n", err);
    std::printf("       |‖b‖ - 1|               = %.3e\n", mag_err);

    const bool ok = (err < 1e-12) && (mag_err < 1e-12);
    std::printf("       result : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
