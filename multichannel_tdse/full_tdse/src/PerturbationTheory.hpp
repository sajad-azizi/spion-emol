// PerturbationTheory.hpp -- closed-form 1st- and 2nd-order time-
// dependent perturbation theory amplitudes for a Gaussian RWA pulse,
// in the same pooled-eigenstate basis used by the TDSE.
//
// We solve, for the interaction-picture amplitudes b_α(t) with halo
// initial state |i⟩ (b_i(0) = 1, all else 0):
//
//   i ḃ_α(t) = Σ_β H_I,αβ(t) b_β(t)
//   H_I,αβ(t) = (Ω_R/2) χ(t) e^{i(E^rel_α - E^rel_β)t} ·
//        [ e^{-iωt} d⁺_{αβ}    (σ⁺ absorption: M_F → M_F+1)
//        + e^{+iωt} d⁻_{αβ}    (σ⁻ emission:   M_F → M_F-1) ]
//
// PT recursion in interaction picture:
//   b^{(0)}_α  = δ_{α,i}
//   b^{(1)}_α(T) = -i ∫ dt H_I,α,i(t)
//   b^{(2)}_α(T) = -i ∫ dt Σ_β H_I,αβ(t) b^{(1)}_β(t)
//
// For χ(t) = exp(-(t-t_c)²/2τ²) the integrals close in terms of the
// Dawson function F(z) = e^{-z²} ∫_0^z e^{u²} du   (real z):
//
//   1st order, halo --σ^P--> α  (P = ±1):
//     b^{(1)}_α = -i (Ω_R/2) d^P_{α,i} √(2π) τ
//                  · exp(-(Δ^P_α)² τ²/2 + i Δ^P_α t_c)
//     Δ^P_α = E^rel_α - E^rel_i - P·ω
//
//   2nd order, halo --σ^P--> k --σ^Q--> β:
//     b^{(2)}_β = -(Ω_R/2)² Σ_k d^Q_{β,k} d^P_{k,i} I_PQ(β,k)
//     I_PQ(β,k) = π τ² · {
//          exp(-(Δ_in² + Δ_out²) τ²/2)
//        + (2i/√π) exp(-Δ_total² τ²/4) F((Δ_out − Δ_in) τ /2) } · e^{i Δ_total t_c}
//     Δ_in   = E^rel_k - E^rel_i  - P·ω
//     Δ_out  = E^rel_β - E^rel_k  - Q·ω
//     Δ_total = Δ_in + Δ_out = E^rel_β - E^rel_i - (P+Q)·ω
//
// (Derivation: nested Gaussian-times-phase integrals; the Dawson
// piece accounts for time ordering, the pure-Gaussian piece is the
// factored / off-shell limit.  Identical in form to Eq. (7) of
// Azizi/Saalmann/Rost 2024 [arXiv:2407.16270] after rotating-frame
// transformation to RWA: each pathway here has fixed (P, Q) instead
// of the paper's η = ± sum over linear-polarization sign choices.)
//
// Predicted population per pooled state α at t = T:
//   M_F(α) = M_F(i)      → |1 + b^{(2)}_α|²   (initial-block dressing)
//   |M_F(α) - M_F(i)| = 1 → |b^{(1)}_α|²
//   |M_F(α) - M_F(i)| = 2 → |b^{(2)}_α|²
//
// Scope: Gaussian pulse only (caller provides τ, t_c).  Other shapes
// can be added by exposing a numeric integrator path.

#pragma once

#include "PooledBasis.hpp"
#include "PooledTDSE.hpp"

#include <Eigen/Dense>

namespace mc_tdse::pt {

struct PTConfig {
    int    initial_block = -1;   // pooled index of |i⟩'s block
    int    initial_state =  0;   // n within that block (0 = halo for production)
    double tau_au        =  0.0; // Gaussian envelope std-dev (atomic units)
    double t_center_au   =  0.0; // Gaussian center (atomic units)
    bool   compute_2nd_order = true;
};

struct PTAmplitudes {
    Eigen::VectorXcd b1;        // 1st-order amplitude per pooled flat index
    Eigen::VectorXcd b2;        // 2nd-order amplitude per pooled flat index
    Eigen::VectorXd  prob_pt;   // recommended population per pooled flat index
};

// Compute closed-form 1st- and 2nd-order PT amplitudes for a Gaussian
// pulse with width tau_au and center t_center_au, given the same
// interaction-picture conventions as PooledTDSE (cfg.omega_au is the
// rotating-frame detuning; cfg.Omega_R_au is the Rabi coupling).
//
// The PulseShape inside cfg is NOT inspected — caller asserts the
// pulse is the Gaussian described by (pt_cfg.tau_au, pt_cfg.t_center_au).
PTAmplitudes compute_pt(const PooledBasis&        pb,
                         const PooledTDSEConfig&   cfg,
                         const PTConfig&           pt_cfg);

// Dawson function F(x) = e^{-x²} ∫_0^x e^{u²} du  (real argument).
// Implemented via power series for |x| < 2.5, composite-Simpson
// quadrature for 2.5 ≤ |x| < 6.5, and asymptotic series for |x| ≥ 6.5.
// Accurate to ~1e-13 across the entire real line.
double dawson(double x);

}  // namespace mc_tdse::pt
