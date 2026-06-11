// test_first_order_pt.cpp -- resonant Rabi-area test.
//
// Two-level system (g, e) with E_g = 0, E_e = ω₀.  σ⁺ vertex
//   d^(+)_{eg} = 1,   all other d's zero.
// Carrier ω = ω₀ (resonant).  Pulse envelope χ(t).
//
// In the rotating-wave approximation (RWA, valid when ω · τ ≫ 1) the
// resonant 2-level problem with envelope χ(t) reduces to a constant-
// detuning Bloch problem with effective Rabi frequency Ω_R · χ(t).
// The closed-form solution is
//
//     |b_e(T)|² = sin²(θ/2),    θ = Ω_R · ∫_0^T χ(t) dt
//
// (valid for ANY pulse area, not just perturbative).  This is a
// stricter check than 1st-order PT because it covers the full
// non-linear regime up to π pulses.
//
// For Gaussian χ(t) = exp(-(t-t_c)²/(2τ²)) on a window much wider than τ:
//     ∫χ(t) dt = √(2π) · τ
// For sin²(πt/T_pulse) on [0, T_pulse]:
//     ∫χ(t) dt = T_pulse / 2
//
// We propagate the full TDSE (no RWA; both σ⁺ and σ⁻ terms kept).
// The counter-rotating contribution (Bloch-Siegert) is suppressed by
// (Ω_R/(2ω))²; for our parameters that's ~1e-7 relative.
#include "Common.hpp"
#include "Pulse.hpp"
#include "TDSEDriver.hpp"
#include "TDSEHamiltonian.hpp"

#include <cstdio>

int main() {
    using namespace mc_tdse;
    int n_fail = 0;

    const double E_g    = 0.0;
    const double E_e    = 1.0;            // ω₀ in atomic units
    const double omega  = 1.0;            // resonant
    const double Omega_R= 1.0e-3;         // weak field

    // d^(+) absorbs (raises): only (e ← g) entry non-zero.
    Eigen::MatrixXcd dp(2, 2);
    dp.setZero();
    dp(1, 0) = 1.0;            // d^(+)_{eg} = 1

    Eigen::VectorXd E(2);
    E << E_g, E_e;

    // ---- Gaussian envelope --------------------------------------
    std::printf("[Gaussian]  ω₀=%.3f  Ω_R=%.0e  τ=200  t_c=1000  T=2000\n",
                E_e - E_g, Omega_R);
    {
        const double tau = 200.0;
        const double t_c = 1000.0;
        const double T   = 2000.0;
        // Δt: 30 steps per carrier period (recipe rule of thumb).
        const double dt  = 2.0 * M_PI / (30.0 * omega);
        TDSEHamiltonian H(E, dp, omega, Omega_R, make_gaussian(tau, t_c));

        Eigen::VectorXcd b0(2);
        b0 << dcompx(1.0, 0.0), dcompx(0.0, 0.0);
        TDSEDriverConfig cfg; cfg.dt = dt;
        auto res = propagate(H, b0, 0.0, T, cfg);
        const double pop_e_num = std::norm(res.b_final(1));

        // RWA Rabi-area answer:  |b_e|² = sin²(Ω_R · √(2π) τ / 2)
        const double theta = Omega_R * std::sqrt(2.0 * M_PI) * tau;
        const double pop_e_rwa = std::pow(std::sin(0.5 * theta), 2);
        const double rel = std::fabs(pop_e_num - pop_e_rwa) / pop_e_rwa;
        std::printf("    pulse area θ = Ω_R √(2π) τ = %.4f rad\n", theta);
        std::printf("    P_e (TDSE)  = %.6e\n", pop_e_num);
        std::printf("    P_e (RWA)   = %.6e\n", pop_e_rwa);
        std::printf("    rel err     = %.3e\n", rel);
        // Tolerance: dominated by Bloch-Siegert (counter-rotating)
        //   correction ~ (Ω_R/(4ω))² ~ 6e-8, plus midpoint Δt²
        //   truncation.  5e-4 covers both at 30 steps/period.
        const bool ok = (rel < 5e-4);
        std::printf("    result      : %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++n_fail;
    }

    // ---- sin² envelope ------------------------------------------
    std::printf("\n[sin²]   ω₀=%.3f  Ω_R=%.0e  T_pulse=1500  t_start=0\n",
                E_e - E_g, Omega_R);
    {
        const double T_pulse = 1500.0;
        const double t_start = 0.0;
        const double T       = T_pulse;
        const double dt      = 2.0 * M_PI / (30.0 * omega);
        TDSEHamiltonian H(E, dp, omega, Omega_R,
                          make_sin_squared(T_pulse, t_start));

        Eigen::VectorXcd b0(2);
        b0 << dcompx(1.0, 0.0), dcompx(0.0, 0.0);
        TDSEDriverConfig cfg; cfg.dt = dt;
        auto res = propagate(H, b0, 0.0, T, cfg);
        const double pop_e_num = std::norm(res.b_final(1));

        // RWA Rabi-area answer:  ∫χ = T_pulse/2 ⇒ θ = Ω_R T_pulse / 2.
        const double theta = Omega_R * T_pulse / 2.0;
        const double pop_e_rwa = std::pow(std::sin(0.5 * theta), 2);
        const double rel = std::fabs(pop_e_num - pop_e_rwa) / pop_e_rwa;
        std::printf("    pulse area θ = Ω_R T/2     = %.4f rad\n", theta);
        std::printf("    P_e (TDSE)  = %.6e\n", pop_e_num);
        std::printf("    P_e (RWA)   = %.6e\n", pop_e_rwa);
        std::printf("    rel err     = %.3e\n", rel);
        const bool ok = (rel < 5e-4);
        std::printf("    result      : %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++n_fail;
    }

    std::printf("\nTotal failures: %d\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
