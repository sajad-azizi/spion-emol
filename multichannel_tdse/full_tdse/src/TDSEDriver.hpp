// TDSEDriver.hpp -- propagation loop using midpoint Taylor.
//
// Algorithm:
//   t = t0
//   b = b_initial (size N, complex)
//   while t + dt <= t1:
//       M_mid = ham.at(t + dt/2)              ← midpoint freezing (2nd-order)
//       b     = exp(-i M_mid dt) b            ← Taylor expansion, matrix-vector
//       t    += dt
//   return b
//
// Returns the final amplitude vector AND optionally a trajectory of
// per-step diagnostics (norm, t, populations).
//
// All matrix-vector products live inside taylor_step_const_H.  No
// matrix-matrix products are formed.
#pragma once

#include "Common.hpp"
#include "TDSEHamiltonian.hpp"
#include "TaylorStepper.hpp"

#include <vector>

namespace mc_tdse {

struct TDSETraceRow {
    double t;
    double norm;       // ‖b‖₂  (= 1 for unitary propagation)
    int    K_used;     // Taylor order used in this step
    double max_pop;    // largest |b_i|² in this step (sanity)
};

struct TDSEDriverResult {
    Eigen::VectorXcd b_final;
    std::vector<TDSETraceRow> trace;          // optional, written if cfg.record_trace
};

struct TDSEDriverConfig {
    double dt              = 0.0;             // time step
    bool   record_trace    = false;
    int    trace_stride    = 1;
    TaylorOptions taylor;                     // K_max, eps_rel
};

// Compute a fixed integer step count from the requested span, so
// forward and reverse propagators traverse IDENTICAL midpoints
// (essential for time-reversal exactness).
inline int n_steps_from_span(double t_start, double t_end, double dt) {
    const double span = t_end - t_start;
    if (span <= 0.0 || dt <= 0.0) return 0;
    return static_cast<int>(std::round(span / dt));
}

inline TDSEDriverResult
propagate(const TDSEHamiltonian&  ham,
          const Eigen::VectorXcd& b0,
          double                  t_start,
          double                  t_end,
          const TDSEDriverConfig& cfg)
{
    if (cfg.dt <= 0.0)
        throw std::runtime_error("propagate: dt must be > 0");
    if (t_end < t_start)
        throw std::runtime_error("propagate: t_end < t_start");

    Eigen::VectorXcd b = b0;
    TDSEDriverResult res;
    const int n_steps = n_steps_from_span(t_start, t_end, cfg.dt);
    if (cfg.record_trace) res.trace.reserve(static_cast<std::size_t>(n_steps) + 1);

    TaylorStats stats;
    auto record = [&](int step) {
        if (!cfg.record_trace) return;
        if (step % cfg.trace_stride != 0) return;
        TDSETraceRow row;
        row.t      = t_start + step * cfg.dt;
        row.norm   = b.norm();
        row.K_used = stats.last_K;
        row.max_pop= b.cwiseAbs2().maxCoeff();
        res.trace.push_back(row);
    };
    record(0);

    // Step `step` maps b at t_start + step*dt → b at t_start + (step+1)*dt
    // using midpoint t_start + (step + 0.5)*dt.  Integer indexing
    // guarantees no FP creep in the midpoint times.
    for (int step = 0; step < n_steps; ++step) {
        const double t_mid = t_start + (step + 0.5) * cfg.dt;
        Eigen::MatrixXcd M_mid = ham.at(t_mid);
        b = taylor_step_const_H(M_mid, b, cfg.dt, cfg.taylor, &stats);
        record(step + 1);
    }

    res.b_final = b;
    return res;
}

// Reverse propagation: invert each forward step EXACTLY by stepping
// with -Δt at the SAME midpoint H(t-Δt/2) the forward step used.
// For a Hermitian H_mid the inverse of exp(-i H_mid Δt) is
// exp(+i H_mid Δt) = exp(-i H_mid · (-Δt)), so this is the analytic
// inverse, not just an approximation -- the propagate_reverse(propagate(b))
// composition equals identity to roundoff.
inline TDSEDriverResult
propagate_reverse(const TDSEHamiltonian&  ham,
                  const Eigen::VectorXcd& b_T,
                  double                  t_start,
                  double                  t_end,
                  const TDSEDriverConfig& cfg)
{
    if (cfg.dt <= 0.0)
        throw std::runtime_error("propagate_reverse: dt must be > 0");
    if (t_end < t_start)
        throw std::runtime_error("propagate_reverse: t_end < t_start");

    Eigen::VectorXcd b = b_T;
    TDSEDriverResult res;
    const int n_steps = n_steps_from_span(t_start, t_end, cfg.dt);
    if (cfg.record_trace) res.trace.reserve(static_cast<std::size_t>(n_steps) + 1);

    TaylorStats stats;
    auto record = [&](int step) {
        if (!cfg.record_trace) return;
        if (step % cfg.trace_stride != 0) return;
        TDSETraceRow row;
        row.t      = t_start + step * cfg.dt;
        row.norm   = b.norm();
        row.K_used = stats.last_K;
        row.max_pop= b.cwiseAbs2().maxCoeff();
        res.trace.push_back(row);
    };
    record(n_steps);

    // Walk steps in reverse: step `step` (1..n_steps) was the forward
    // mapping t_start+(step-1)*dt → t_start+step*dt with midpoint
    // t_start+(step-0.5)*dt.  We undo it with -dt at the SAME midpoint.
    for (int step = n_steps; step >= 1; --step) {
        const double t_mid = t_start + (step - 0.5) * cfg.dt;
        Eigen::MatrixXcd M_mid = ham.at(t_mid);
        b = taylor_step_const_H(M_mid, b, -cfg.dt, cfg.taylor, &stats);
        record(step - 1);
    }

    res.b_final = b;
    return res;
}

}  // namespace mc_tdse
