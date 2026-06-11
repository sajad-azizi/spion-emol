// ExchangeCoupling.hpp -- exchange coupling Q_ψf(r) from PDF eq. 18-19.
//
// In the Numerov form Y'' = -Q Y with Y = [ψ; f], the off-diagonal block of
// the super-potential is (rederived 2026-04-23 against the PDF):
//
//     Q_ψf[μ, (i,σ)](r) = +2α · Σ_λ G_{μλσ} · χ^i_λ(r) / r,   α = √(2π).
//
// NOTE ON SIGN vs version_0:
//   version_0/src/RenormalizedNumerov.cpp (CouplingMatrices_Fastest,
//   CouplingMatrices, precomputeCouplingMatrices) uses factor = -2α/r.
//   That sign is a similarity transform by diag(I, -I) and is therefore
//   INVISIBLE to the K- and S-matrices (the diag(I,-I) just flips the sign
//   of f in the output). But any observable that consumes f directly would
//   come out with the wrong sign. We follow the PDF here.
//
// CONSTRUCTION:
//   G_coeff is sparse (~4-5% non-zero for l_orb = 2·l_cont). We stack it
//   into a single CSR matrix indexed by (μ·n_σ + σ, λ):
//     G_combined[μ·n_σ + σ, λ] = G_{μλσ}.
//   A single sparse-dense gemm gives
//     Q_flat[μ·n_σ + σ, i] = Σ_λ G_{μλσ} · χ^i_λ(r_ir),
//   reshaped to (μ, i·n_σ + σ) = Q_ψf[μ, f_idx] with the +2α/r prefactor.
//
// ON-FLY vs CACHED:
//   For the l_cont ~ 100 production runs we compute Q_ψf on demand per
//   grid point ir (version_0's params.onfly=true path). No vector-of-
//   matrices is cached here; a single small dense matrix is returned.

#pragma once

#include "scatt/WavefunctionSetup.hpp"   // AngTriplet, ChiRadial

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cstddef>
#include <vector>

namespace scatt {

// Preallocated workspace for hot-loop callers (one per OpenMP thread).
// Holding chi_T and Q_flat persistent avoids ~2 allocations per grid point.
struct ExchangeCouplingWorkspace {
    Eigen::MatrixXd chi_T;    // (n_lambda_max, n_occ)
    Eigen::MatrixXd Q_flat;   // (N_psi * n_sigma, n_occ)
};

class ExchangeCoupling {
public:
    // Builds G_combined from the sparse triplet list. G_coeff is NOT retained.
    ExchangeCoupling(const std::vector<AngTriplet>& G_coeff,
                     int    N_psi,
                     int    n_sigma,
                     int    n_occ,
                     double r_min,
                     double dr);

    // Returns the (N_psi × N_f) dense Q_ψf(r_ir) matrix. Convenient but
    // allocates a new Q every call. `chi_ir` is the per-radius chi matrix
    // as laid out by load_chi_from_hdf5:
    //   chi_ir(i, λ) = r_ir · F^i_λ(r_ir)   (rows = occupied orbital i,
    //                                         cols = lambda, real-SH packed).
    Eigen::MatrixXd compute(int ir, const Eigen::MatrixXd& chi_ir) const;

    // Allocation-free hot path. `ws` and `Q_out` must be preallocated via
    // make_workspace()/make_output(), OR the caller may pass an already-
    // sized Q_out of shape (N_psi, N_f); compute_into will resize if not.
    // Returns Q_out by reference for chaining.
    void compute_into(int                     ir,
                      const Eigen::MatrixXd&  chi_ir,
                      ExchangeCouplingWorkspace& ws,
                      Eigen::MatrixXd&        Q_out) const;

    // Factory for a workspace sized to this instance's dimensions.
    ExchangeCouplingWorkspace make_workspace() const;

    // Factory for a properly-sized output matrix (zero-initialized).
    Eigen::MatrixXd make_output() const {
        return Eigen::MatrixXd::Zero(N_psi_, N_f_);
    }

    int    N_psi()        const { return N_psi_; }
    int    n_sigma()      const { return n_sigma_; }
    int    n_occ()        const { return n_occ_; }
    int    N_f()          const { return N_f_; }
    int    n_lambda_max() const { return n_lambda_max_; }
    double radius(int ir) const { return r_min_ + ir * dr_; }

    // Memory footprint of the cached sparse matrix (CSR: values + col-idx +
    // row-ptr). Useful for the global memory budget statistic.
    std::size_t sparse_bytes() const;

private:
    int    N_psi_;
    int    n_sigma_;
    int    n_occ_;
    int    N_f_;
    int    n_lambda_max_;
    double r_min_;
    double dr_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> G_combined_;
};

// Slow reference implementation for tests. O(|G_coeff| · n_occ) per ir.
// The literal triple loop over G_coeff — the oracle that the fast sparse
// gemm path is validated against.
Eigen::MatrixXd compute_Q_psi_f_reference(
    const std::vector<AngTriplet>& G_coeff,
    const Eigen::MatrixXd&         chi_ir,
    int                            N_psi,
    int                            n_sigma,
    int                            n_occ,
    double                         r);

}  // namespace scatt
