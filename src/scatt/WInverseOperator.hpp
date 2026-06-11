// WInverseOperator.hpp -- block-fused application of W_n^{-1}.
//
// Per grid point n the full Numerov matrix is
//
//     W_n = I + (h²/12) Q_n =  [  A(n)   B(n)   ]          N_total × N_total
//                               [  B(n)^T D(n)  ]
//
// with A (N_ψ × N_ψ), B (N_ψ × N_f), D (diagonal, N_f). In production
// N_total can reach ~10^5; materializing W_n^{-1} as a dense matrix is
// impossible. We expose W_n^{-1} as an *operator* on tall-skinny (N_total ×
// ncols) matrices using the block-inverse identity:
//
//     rhs_ψ = Z_ψ − B · (D^{-1} · Z_f)
//     Y_ψ   = Sinv · rhs_ψ
//     Y_f   = D^{-1} · (Z_f − B^T · Y_ψ)
//
// This is FIVE BLAS-3 calls per (n, ncols) pair -- one triangular solve
// replaced by the precomputed Sinv, everything else dense gemm. Nothing
// bigger than (N_total × ncols) is ever allocated.
//
// Where Sinv, B, D come from:
//   - Sinv      : SchurInverter (already built; load-on-demand from its store)
//   - B(n)      : on-fly via ExchangeCoupling::compute_into at grid point n
//   - D(n)_diag : 1 − (h²/12) ℓ_σ(ℓ_σ+1)/r², clamped to W_min (same Johnson
//                 floor as SchurInverter). For ir ≥ n_transition, B = 0.
//
// NOTE ON SIGN:
//   Because B enters twice (once as B, once as B^T) in the block-inverse
//   identity for the off-diagonal blocks, and because D is diagonal, the
//   W^{-1} top-left block (Y_ψ from Z_ψ=I, Z_f=0) is Sinv -- sign-invariant.
//   The off-diagonal blocks flip sign with B, exactly what the PDF predicts
//   and what our +2α convention delivers.

#pragma once

#include "scatt/ExchangeCoupling.hpp"    // ExchangeCoupling, ExchangeCouplingWorkspace
#include "scatt/SchurInverter.hpp"       // SchurInverter
#include "scatt/SolverParams.hpp"
#include "scatt/WavefunctionSetup.hpp"   // ChiRadial

#include <Eigen/Dense>

#include <cstddef>
#include <vector>

namespace scatt {

// Per-thread scratch for WInverseOperator::apply / apply_U. Holds:
//   - ExchangeCoupling scratch.
//   - B, Dinv at the current grid point.
//   - Two transient matrices of shape (N_ψ × ncols) and (N_f × ncols).
struct WInverseOperatorWorkspace {
    ExchangeCouplingWorkspace ec_ws;
    Eigen::MatrixXd B;          // (N_ψ × N_f)
    Eigen::VectorXd Dinv;       // (N_f)
    Eigen::MatrixXd rhs_psi;    // (N_ψ × ncols)
    Eigen::MatrixXd tmp_f;      // (N_f × ncols)
};

class WInverseOperator {
public:
    WInverseOperator(const SolverParams&     sp,
                     SchurInverter&          si,
                     const ExchangeCoupling* ec,       // nullable
                     const ChiRadial*        chi,      // required iff ec != nullptr
                     double                  W_min = 0.001);

    // Y = W_n^{-1} · Z.  Z and Y must have shape (N_total × ncols);
    // Y is resized if necessary.
    void apply(int n, const Eigen::MatrixXd& Z,
               Eigen::MatrixXd& Y,
               WInverseOperatorWorkspace& ws) const;

    // Y = (12 W_n^{-1} − 10 I) · X.  Same shape contract as apply.
    void apply_U(int n, const Eigen::MatrixXd& X,
                 Eigen::MatrixXd& Y,
                 WInverseOperatorWorkspace& ws) const;

    // Materialize the full (N_total × N_total) W_n^{-1}. VALIDATION ONLY --
    // throws if N_total > max_N_total (default 10000).
    Eigen::MatrixXd materialize(int n, std::size_t max_N_total = 10000) const;

    // materialize() without the size guard.  Used by the GPU path, which
    // uploads W_n to device and genuinely needs the dense matrix; the
    // host-side cost is still O(N^2) memory and O(N_f^2 + N_psi·N_f)
    // compute, which is cheap relative to the GPU step.  Caller is
    // responsible for verifying enough host RAM exists.
    void materialize_into(int n, Eigen::MatrixXd& Winv_out,
                          WInverseOperatorWorkspace& ws) const;

    // Factory for a properly-sized workspace.
    WInverseOperatorWorkspace make_workspace() const;

    int N_psi()   const { return sp_.n_mu; }
    int N_f()     const { return sp_.n_occ * sp_.n_sigma; }
    int N_total() const { return N_psi() + N_f(); }

    // Expose the per-n exchange block B_n = (h²/12)·Q_ψf(n) and the f-diagonal
    // D_n (clamped), so external code (e.g. BackPropagator) can construct
    // outer boundary conditions consistent with the Schur reduction in K-
    // extraction. Returns matrices of size (N_ψ × N_f) and (N_f,) respectively.
    // B_out is set to zero if there is no exchange or n ≥ n_transition.
    void load_B_Dinv(int n, Eigen::MatrixXd& B_out, Eigen::VectorXd& Dinv_out,
                     WInverseOperatorWorkspace& ws) const;

    double W_min() const { return W_min_; }

    // Forwarders to the underlying SchurInverter's PotentialStorage so
    // callers (e.g. BackPropagator's backward sweep) can drive async
    // chunk prefetch on Sinv without holding a direct SchurInverter
    // reference.  Bit-identical to legacy synchronous code -- only
    // scheduling changes.  Returns 0 when Sinv is in MEMORY mode.
    void start_prefetch(int chunk_idx) { si_.start_prefetch(chunk_idx); }
    int  si_chunk_size_disk() const { return si_.chunk_size_disk(); }
    int  si_num_chunks_disk() const { return si_.num_chunks_disk(); }

private:
    const SolverParams&     sp_;
    SchurInverter&          si_;
    const ExchangeCoupling* ec_;
    const ChiRadial*        chi_;
    double                  W_min_;
    std::vector<int>        l_sigma_;     // l value per sigma index

    // Fill ws.B (with (h²/12)·Q_ψf(n)) and ws.Dinv (with clamped reciprocal).
    // If n >= n_transition OR ec_ == nullptr, ws.B is zeroed (no allocation).
    void load_B_and_Dinv_(int n, WInverseOperatorWorkspace& ws) const;
};

}  // namespace scatt
