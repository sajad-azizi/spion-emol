#include "scatt/ExchangeCoupling.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
#  include <omp.h>
#endif

namespace scatt {

ExchangeCoupling::ExchangeCoupling(
    const std::vector<AngTriplet>& G_coeff,
    int    N_psi,
    int    n_sigma,
    int    n_occ,
    double r_min,
    double dr)
    : N_psi_(N_psi),
      n_sigma_(n_sigma),
      n_occ_(n_occ),
      N_f_(n_occ * n_sigma),
      n_lambda_max_(0),
      r_min_(r_min),
      dr_(dr)
{
    // Find max lambda actually appearing in G (for l_orb < Nlm_sce).
    for (const auto& g : G_coeff) {
        if (g.a < N_psi_ && g.c < n_sigma_ && g.b + 1 > n_lambda_max_) {
            n_lambda_max_ = g.b + 1;
        }
    }
    if (n_lambda_max_ == 0) {
        // No triplets touch the (μ, σ) cut window -- keep an empty matrix.
        G_combined_.resize(N_psi_ * n_sigma_, 0);
        return;
    }

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    triplets.reserve(G_coeff.size());
    for (const auto& g : G_coeff) {
        // AngTriplet G: a=μ, b=λ, c=σ.
        if (g.a < N_psi_ && g.c < n_sigma_ && g.b < n_lambda_max_) {
            const int row = g.a * n_sigma_ + g.c;
            triplets.emplace_back(row, g.b, g.value);
        }
    }

    G_combined_.resize(N_psi_ * n_sigma_, n_lambda_max_);
    G_combined_.setFromTriplets(triplets.begin(), triplets.end());
    G_combined_.makeCompressed();
}

Eigen::MatrixXd
ExchangeCoupling::compute(int ir, const Eigen::MatrixXd& chi_ir) const
{
    Eigen::MatrixXd Q = make_output();
    ExchangeCouplingWorkspace ws = make_workspace();
    compute_into(ir, chi_ir, ws, Q);
    return Q;
}

void
ExchangeCoupling::compute_into(int                        ir,
                               const Eigen::MatrixXd&     chi_ir,
                               ExchangeCouplingWorkspace& ws,
                               Eigen::MatrixXd&           Q_out) const
{
    if (Q_out.rows() != N_psi_ || Q_out.cols() != N_f_) {
        Q_out.resize(N_psi_, N_f_);
    }
    Q_out.setZero();

    const double r = r_min_ + ir * dr_;
    if (r < 1e-10 || chi_ir.rows() == 0 || chi_ir.cols() == 0) return;

    const double alpha  = std::sqrt(2.0 * M_PI);
    const double factor = 2.0 * alpha / r;            // PDF: +2α/r

    const int n_lambda_use =
        std::min<int>(n_lambda_max_, static_cast<int>(chi_ir.cols()));
    const int n_occ_use =
        std::min<int>(n_occ_,        static_cast<int>(chi_ir.rows()));
    if (n_lambda_use <= 0 || n_occ_use <= 0) return;

    // Size workspace lazily (first call or dimension growth).
    if (ws.chi_T.rows() < n_lambda_use || ws.chi_T.cols() < n_occ_) {
        ws.chi_T.resize(n_lambda_max_, n_occ_);
    }
    if (ws.Q_flat.rows() != N_psi_ * n_sigma_ ||
        ws.Q_flat.cols() != n_occ_)
    {
        ws.Q_flat.resize(N_psi_ * n_sigma_, n_occ_);
    }

    // chi_T sub-block view: fill (n_lambda_use × n_occ_use) prefix.
    ws.chi_T.topLeftCorner(n_lambda_use, n_occ_use).noalias() =
        chi_ir.topLeftCorner(n_occ_use, n_lambda_use).transpose();

    // One sparse-dense gemm into the Q_flat prefix.
    ws.Q_flat.leftCols(n_occ_use).noalias() =
        G_combined_.leftCols(n_lambda_use) *
        ws.chi_T.topLeftCorner(n_lambda_use, n_occ_use);

    // Reshape + scale: Q_flat[mu*n_sigma+sigma, i] -> Q[mu, i*n_sigma+sigma].
    //
    // THREAD-SAFETY (collapse(2) over (mu, sigma)):
    //   * Each outer (mu, sigma) writes Q_out(mu, i*n_sigma+sigma) for
    //     i ∈ [0, n_occ_use).  Two distinct outer pairs hit either a
    //     different row of Q_out (mu differs) or the same row with
    //     disjoint columns (mu equal, sigma differs => columns
    //     i*n_sigma+sigma1 ≠ i*n_sigma+sigma2 for sigma1 ≠ sigma2).
    //   * No accumulation: each output cell is a single multiply, so
    //     the result is BYTE-IDENTICAL to the serial loop.
    //   * `flat_row` is local per iteration; `ws.Q_flat` and `factor`
    //     are read-only.  No data races.
    //
    // PERF NOTE: at L=100, this serial reshape was ~74M iterations per
    // call × ~3000 calls per Sinv build ≈ tens of minutes of wasted
    // wall time; v0's CouplingMatrices_Fastest had the same parallel
    // pragma and was correspondingly faster.
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int mu = 0; mu < N_psi_; ++mu) {
        for (int sigma = 0; sigma < n_sigma_; ++sigma) {
            const int flat_row = mu * n_sigma_ + sigma;
            for (int i = 0; i < n_occ_use; ++i) {
                Q_out(mu, i * n_sigma_ + sigma) =
                    factor * ws.Q_flat(flat_row, i);
            }
        }
    }
}

ExchangeCouplingWorkspace
ExchangeCoupling::make_workspace() const
{
    ExchangeCouplingWorkspace ws;
    ws.chi_T.resize(n_lambda_max_, n_occ_);
    ws.Q_flat.resize(N_psi_ * n_sigma_, n_occ_);
    return ws;
}

std::size_t ExchangeCoupling::sparse_bytes() const {
    const std::size_t nnz     = static_cast<std::size_t>(G_combined_.nonZeros());
    const std::size_t n_outer = static_cast<std::size_t>(G_combined_.outerSize()) + 1;
    return nnz * (sizeof(double) + sizeof(int)) + n_outer * sizeof(int);
}

Eigen::MatrixXd compute_Q_psi_f_reference(
    const std::vector<AngTriplet>& G_coeff,
    const Eigen::MatrixXd&         chi_ir,
    int                            N_psi,
    int                            n_sigma,
    int                            n_occ,
    double                         r)
{
    const int N_f = n_occ * n_sigma;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N_psi, N_f);

    if (r < 1e-10 || chi_ir.rows() == 0 || chi_ir.cols() == 0) return Q;

    const double alpha  = std::sqrt(2.0 * M_PI);
    const double factor = 2.0 * alpha / r;            // PDF: +2α/r

    const int n_lambda_cut = static_cast<int>(chi_ir.cols());
    const int n_occ_use    = std::min<int>(n_occ, static_cast<int>(chi_ir.rows()));

    for (const auto& g : G_coeff) {
        const int mu     = g.a;
        const int lambda = g.b;
        const int sigma  = g.c;
        if (mu >= N_psi || sigma >= n_sigma || lambda >= n_lambda_cut) continue;

        for (int i = 0; i < n_occ_use; ++i) {
            const int f_idx = i * n_sigma + sigma;
            if (f_idx >= N_f) continue;
            Q(mu, f_idx) += factor * g.value * chi_ir(i, lambda);
        }
    }
    return Q;
}

}  // namespace scatt
