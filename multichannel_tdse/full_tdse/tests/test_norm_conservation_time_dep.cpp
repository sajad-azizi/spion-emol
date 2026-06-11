// test_norm_conservation_time_dep.cpp -- Hermiticity of M(t) at every
// instant guarantees a unitary midpoint-frozen propagator each step,
// so ‖b‖ should stay at 1 throughout.
//
// Setup: 4-level system with random hermitian d^(+), Gaussian envelope.
// We propagate over many pulse durations (covering the rising edge,
// peak, and falling edge of the envelope) and track norm drift.
#include "Common.hpp"
#include "Pulse.hpp"
#include "TDSEDriver.hpp"
#include "TDSEHamiltonian.hpp"

#include <cstdio>
#include <random>

int main() {
    using namespace mc_tdse;

    constexpr int N = 4;
    Eigen::VectorXd E(N);
    E << 0.0, 0.7, 1.3, 2.1;        // arbitrary energies

    // Random complex d^(+) (full matrix; the time-dep H makes it
    // Hermitian via M(t) construction even if d^(+) itself is not).
    std::mt19937_64 rng(7);
    std::normal_distribution<double> G(0.0, 1.0);
    Eigen::MatrixXcd dp(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            dp(i, j) = dcompx(G(rng), G(rng));

    const double omega   = 0.5;
    const double Omega_R = 0.05;          // not too small (real driving)
    const double tau     = 50.0, t_c = 250.0, T = 500.0;

    TDSEHamiltonian H(E, dp, omega, Omega_R, make_gaussian(tau, t_c));

    Eigen::VectorXcd b0(N);
    for (int i = 0; i < N; ++i) b0(i) = dcompx(G(rng), G(rng));
    b0 /= b0.norm();
    const double norm0 = b0.norm();

    TDSEDriverConfig cfg;
    cfg.dt = 2.0 * M_PI / (30.0 * omega);
    cfg.record_trace = true;
    cfg.trace_stride = 50;
    auto res = propagate(H, b0, 0.0, T, cfg);

    double max_drift = 0.0;
    for (const auto& row : res.trace) {
        const double drift = std::fabs(row.norm - norm0);
        if (drift > max_drift) max_drift = drift;
    }
    const double final_drift = std::fabs(res.b_final.norm() - norm0);

    std::printf("[norm_conservation, time-dep]  N=%d  dt=%.4f  T=%.1f  steps=%zu\n",
                N, cfg.dt, T, res.trace.size() * cfg.trace_stride);
    std::printf("    max norm drift during run = %.3e\n", max_drift);
    std::printf("    final |‖b‖ - 1|           = %.3e\n", final_drift);

    const bool ok = (max_drift < 1e-12) && (final_drift < 1e-12);
    std::printf("    result: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
