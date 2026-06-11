// TaylorStepper.hpp -- one-step matrix-exponential propagator
// computed by truncated Taylor series with matrix-VECTOR products only.
//
// For a TIME-INDEPENDENT Hamiltonian H (complex N×N hermitian), the
// exact propagator over Δt is
//
//     U(Δt) = exp(-i H Δt)         (atomic units, ℏ = 1)
//
// We approximate
//
//     U(Δt) b ≈ Σ_{k=0}^{K} (1/k!) (-i Δt)^k H^k b
//
// by the iteration
//
//     y_0   = b                (first term)
//     y_{k} = (-i Δt / k) · H · y_{k-1}        (matrix-vector product)
//     out  += y_k
//     stop when ‖y_k‖ / ‖y_0‖ < eps_rel  OR k == K_max.
//
// The ONLY heavy operation per Taylor term is a single matrix-vector
// multiplication; we never form H·H or H·U.  This is the form your
// project explicitly requested: "matrix-vector, not matrix-matrix".
//
// For TIME-DEPENDENT H(t), use `step_midpoint` which evaluates H at
// t + Δt/2 and freezes it for the step.  The midpoint freezing is
// 2nd-order accurate in Δt; combined with a Taylor truncation order
// K (~ 12 by default), the per-step error is
//
//     ‖U_exact b - U_approx b‖ ≤ C₁ Δt^{K+1}  +  C₂ Δt^3  · ‖[H',H]‖
//
// where the second term is the midpoint-freezing error.  K is chosen
// so the first term sits below the second on the working Δt.
//
// The stepper is templated on the matrix type so it works with
// Eigen::MatrixXcd (dense) or any Eigen-compatible sparse type that
// supports `M * v` returning a VectorXcd.  H is passed by *const ref*
// or as a callable returning H at given t.
#pragma once

#include "Common.hpp"

#include <functional>

namespace mc_tdse {

struct TaylorOptions {
    // Maximum Taylor order tried.  K=20 is comfortable head-room for
    // ‖HΔt‖ up to ~5; we usually stop much earlier via eps_rel.
    int    K_max  = 30;
    // Stop when ‖y_k‖_2 / ‖y_0‖_2 falls below this fraction.
    double eps_rel = 1.0e-14;
};

struct TaylorStats {
    int    last_K   = 0;       // Taylor terms actually used in last step
    double last_res = 0.0;     // ‖y_K‖/‖y_0‖ at termination
};

// Time-INDEPENDENT step.  Returns b(t+Δt) ≈ exp(-i H Δt) · b.
// H is applied via H * v (Eigen-compatible matrix expression).
inline Eigen::VectorXcd
taylor_step_const_H(const Eigen::MatrixXcd& H,
                    const Eigen::VectorXcd& b,
                    double                  dt,
                    const TaylorOptions&    opt   = {},
                    TaylorStats*            stats = nullptr)
{
    Eigen::VectorXcd out = b;
    Eigen::VectorXcd y   = b;
    const double y0_norm = b.norm();
    const double tol     = opt.eps_rel * std::max(y0_norm, 1.0e-300);

    int  k_used = 0;
    double last_res = 0.0;
    for (int k = 1; k <= opt.K_max; ++k) {
        // y_k = (-i Δt / k) · H · y_{k-1}
        const dcompx coef = -I_unit * dt / static_cast<double>(k);
        y = coef * (H * y);            // ONE matrix-vector product
        out += y;
        const double yn = y.norm();
        last_res = (y0_norm > 0) ? yn / y0_norm : yn;
        k_used = k;
        if (yn < tol) break;
    }
    if (stats) { stats->last_K = k_used; stats->last_res = last_res; }
    return out;
}

// Time-DEPENDENT step.  H_at_t is a callable: (double t) -> MatrixXcd.
// We evaluate H at t + Δt/2 and freeze it for the step (midpoint rule,
// 2nd-order accurate globally).  The resulting Taylor expansion is the
// same as the time-INDEPENDENT version using H_mid.
inline Eigen::VectorXcd
taylor_step_midpoint(const std::function<Eigen::MatrixXcd(double)>& H_at_t,
                     const Eigen::VectorXcd& b,
                     double                  t,
                     double                  dt,
                     const TaylorOptions&    opt   = {},
                     TaylorStats*            stats = nullptr)
{
    Eigen::MatrixXcd H_mid = H_at_t(t + 0.5 * dt);
    return taylor_step_const_H(H_mid, b, dt, opt, stats);
}

}  // namespace mc_tdse
