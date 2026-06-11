// test_taylor_2level_rabi.cpp
//
// Two-level Rabi problem:  H = (Ω/2) σ_x  =  (Ω/2) [[0,1],[1,0]]
//
// Initial state |0⟩ = (1, 0)^T.  The exact solution is
//
//     |ψ(t)⟩ = cos(Ω t / 2) |0⟩ - i sin(Ω t / 2) |1⟩
//
// Probability to be in |1⟩ at time t:  P_1(t) = sin²(Ω t / 2).
//
// We propagate with a sequence of Taylor steps and compare to the
// analytic formula at the final time T = N · Δt.
//
// Pass if the L_∞ amplitude error stays below 1e-12 over many Rabi
// periods.
#include "Common.hpp"
#include "TaylorStepper.hpp"

#include <cstdio>

int main() {
    using namespace mc_tdse;

    // Choose Ω so that one full Rabi period 2π/Ω fits ~25 steps of dt.
    const double Omega = 1.0;
    const double dt    = 2.0 * M_PI / (Omega * 25.0);
    const int    n_steps = 200;                 // 8 full Rabi periods

    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(2, 2);
    H(0, 1) = 0.5 * Omega;
    H(1, 0) = 0.5 * Omega;

    Eigen::VectorXcd b(2);
    b << dcompx(1.0, 0.0), dcompx(0.0, 0.0);

    TaylorStats stats;
    double max_err = 0.0;
    int    max_K   = 0;
    for (int s = 1; s <= n_steps; ++s) {
        b = taylor_step_const_H(H, b, dt, {}, &stats);
        if (stats.last_K > max_K) max_K = stats.last_K;
        const double t = dt * s;
        const dcompx c0 = std::cos(0.5 * Omega * t);
        const dcompx c1 = -I_unit * std::sin(0.5 * Omega * t);
        const double e0 = std::abs(b(0) - c0);
        const double e1 = std::abs(b(1) - c1);
        if (e0 > max_err) max_err = e0;
        if (e1 > max_err) max_err = e1;
    }
    const double mag_err = std::abs(b.norm() - 1.0);

    std::printf("[2-level Rabi]  Omega=%.3f  dt=%.4f (~%.1f steps/period)  steps=%d  T=%.3f\n",
                Omega, dt, 2.0 * M_PI / (Omega * dt), n_steps, n_steps * dt);
    std::printf("                max Taylor K used = %d\n", max_K);
    std::printf("                max |b - b_exact| over all steps = %.3e\n", max_err);
    std::printf("                |‖b(T)‖ - 1|                     = %.3e\n", mag_err);

    const bool ok = (max_err < 1e-12) && (mag_err < 1e-12);
    std::printf("                result : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
