// PooledTDSE.hpp -- TDSE in the interaction picture using a pooled
// eigenstate basis from PooledBasis.
//
// In the interaction picture |ψ_I⟩ = e^{i H_0 t} |ψ_S⟩, so amplitudes
// in the eigenstate basis evolve as
//
//   i ∂_t c_α = Σ_β H_I,αβ(t) c_β
//
// with
//
//   H_I,αβ(t) = (Ω_R/2) χ(t) e^{i(E_α - E_β)t} ·
//                 [ e^{-iωt} d^(+)_{αβ}    (η^(+) absorption: M_F → M_F+1)
//                 + e^{+iωt} (d^(+))^†_{αβ}    (η^(-) emission)  ]
//
// where d^(+)_{αβ} is nonzero only when α is in M_F+1 and β is in M_F
// for some adjacent pair (lookup via PooledBasis::pair_index_of_low_MF).
// d^(-) = (d^(+))^† gives the back-coupling.
//
// We propagate via existing TaylorStepper (matrix-vector exponential
// truncation), with midpoint freezing H(t + Δt/2) for 2nd-order
// accuracy.
#pragma once

#include "PooledBasis.hpp"
#include "Pulse.hpp"
#include "TaylorStepper.hpp"

#include <Eigen/Dense>
#include <functional>

namespace mc_tdse {

struct PooledTDSEConfig {
    // Pulse: χ(t).  Build via mc_tdse::make_gaussian / sin² / etc.
    PulseShape  chi;

    double      omega_au   = 0.0;       // RF carrier (atomic units)
    double      Omega_R_au = 0.0;       // Peak Rabi (atomic units)

    double      t_start    = 0.0;
    double      t_end      = 0.0;
    double      dt         = 0.0;

    TaylorOptions taylor   = {};        // K_max, eps_rel
};

struct PooledTDSEStats {
    int    n_steps  = 0;
    int    K_avg    = 0;
    double max_err  = 0.0;
};

// Build the pooled-basis interaction-picture matrix M(t) = H_I(t).
// Returned matrix is N_total × N_total complex.  This materializes
// H_I as a dense matrix and is intended for tests / debugging.
//
// PRODUCTION PATH (propagate_pooled, below) uses HIApplier instead,
// which avoids materializing the dense matrix and amortizes the
// transcendental phase calls across all K matvecs of a Taylor step.
Eigen::MatrixXcd build_HI(const PooledBasis& pb,
                          double t,
                          const PooledTDSEConfig& cfg);

// Time-frozen applier: precomputes per-state phases φ(α, t) = e^{i E_α t}
// once at construction, then applies H_I(t) · b in O(Σ_pairs Na·Nb)
// per call (no allocation, no transcendentals).  Reuse for all K
// Taylor matvecs of one step.
class HIApplier {
public:
    HIApplier(const PooledBasis& pb,
              const PooledTDSEConfig& cfg,
              double t);

    // out = H_I(t) * b, computed without forming a dense matrix.
    Eigen::VectorXcd apply(const Eigen::VectorXcd& b) const;

private:
    const PooledBasis&       pb_;
    const PooledTDSEConfig&  cfg_;
    double                   t_       = 0.0;
    double                   half_OmR_chi_ = 0.0;
    dcompx                   psi_abs_ = 0.0;     // (Ω_R/2)·χ(t)·e^{-iωt}
    dcompx                   psi_emi_ = 0.0;     // (Ω_R/2)·χ(t)·e^{+iωt}
    Eigen::VectorXcd         phi_;               // e^{i E_α t} for α=0..N_total-1
    Eigen::VectorXcd         phi_conj_;          // conj(phi_)
};

// Propagate c(t_start) -> c(t_end) using midpoint Taylor matrix-vector.
Eigen::VectorXcd propagate_pooled(const PooledBasis& pb,
                                  Eigen::VectorXcd c_start,
                                  const PooledTDSEConfig& cfg,
                                  PooledTDSEStats* stats = nullptr);

// Convenience: |c_α|² aggregated by block (returns a vector indexed
// by the position k in pb.block_MFs, NOT by M_F itself).
std::vector<double> block_populations(const PooledBasis& pb,
                                      const Eigen::VectorXcd& c);

}  // namespace mc_tdse
