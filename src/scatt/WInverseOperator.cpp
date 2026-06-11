#include "scatt/WInverseOperator.hpp"

#include "angular/Gaunt.hpp"

#include <stdexcept>
#include <string>

namespace scatt {

WInverseOperator::WInverseOperator(const SolverParams&     sp,
                                   SchurInverter&          si,
                                   const ExchangeCoupling* ec,
                                   const ChiRadial*        chi,
                                   double                  W_min)
    : sp_(sp), si_(si), ec_(ec), chi_(chi), W_min_(W_min)
{
    if (ec_ && !chi_) {
        throw std::runtime_error(
            "WInverseOperator: chi must be provided when ExchangeCoupling is non-null");
    }
    l_sigma_.resize(sp_.n_sigma);
    for (int s = 0; s < sp_.n_sigma; ++s) {
        int l, m;
        angular::idx_to_lm(s, l, m);
        l_sigma_[s] = l;
    }
}

WInverseOperatorWorkspace WInverseOperator::make_workspace() const {
    WInverseOperatorWorkspace ws;
    ws.B.resize(N_psi(), N_f());
    ws.Dinv.resize(N_f());
    // rhs_psi and tmp_f resized lazily on first apply (ncols unknown).
    if (ec_) ws.ec_ws = ec_->make_workspace();
    return ws;
}

void
WInverseOperator::load_B_and_Dinv_(int n, WInverseOperatorWorkspace& ws) const
{
    const double r  = sp_.r_min + n * sp_.dr;
    const double r2 = r * r;
    const double h2_12 = sp_.dr * sp_.dr / 12.0;

    // D^{-1} at this n, clamped.
    for (int f_idx = 0; f_idx < N_f(); ++f_idx) {
        const int l = l_sigma_[f_idx % sp_.n_sigma];
        const double centrif = (r2 > 1e-30) ? double(l * (l + 1)) / r2 : 0.0;
        double Df = 1.0 - h2_12 * centrif;
        if (Df < W_min_) Df = W_min_;
        ws.Dinv(f_idx) = 1.0 / Df;
    }

    // B = (h²/12) · Q_ψf(n). Zero if no exchange or ir past transition.
    if (!ec_ || n >= sp_.n_transition) {
        ws.B.setZero();
        return;
    }
    // compute_into writes into ws.ec_ws and an output — use ws.B as output.
    ec_->compute_into(n, (*chi_)[static_cast<std::size_t>(n)], ws.ec_ws, ws.B);
    ws.B *= h2_12;
}

void
WInverseOperator::load_B_Dinv(int n, Eigen::MatrixXd& B_out,
                              Eigen::VectorXd& Dinv_out,
                              WInverseOperatorWorkspace& ws) const
{
    load_B_and_Dinv_(n, ws);
    B_out.noalias()    = ws.B;
    Dinv_out.noalias() = ws.Dinv;
}

void
WInverseOperator::apply(int n, const Eigen::MatrixXd& Z,
                        Eigen::MatrixXd& Y,
                        WInverseOperatorWorkspace& ws) const
{
    const int N_psi = this->N_psi();
    const int N_f   = this->N_f();
    const int N_tot = this->N_total();
    if (Z.rows() != N_tot) {
        throw std::runtime_error(
            "WInverseOperator::apply: Z.rows() must equal N_total (" +
            std::to_string(N_tot) + ")");
    }
    const int ncols = static_cast<int>(Z.cols());
    if (Y.rows() != N_tot || Y.cols() != ncols) Y.resize(N_tot, ncols);

    // Fill workspace B, D^{-1}.
    load_B_and_Dinv_(n, ws);

    if (ws.rhs_psi.rows() != N_psi || ws.rhs_psi.cols() != ncols)
        ws.rhs_psi.resize(N_psi, ncols);
    if (ws.tmp_f.rows() != N_f || ws.tmp_f.cols() != ncols)
        ws.tmp_f.resize(N_f, ncols);

    auto Z_psi = Z.topRows(N_psi);        // (N_ψ × ncols)  view
    auto Z_f   = Z.bottomRows(N_f);       // (N_f × ncols)  view

    // Step 1: tmp_f = D^{-1} · Z_f (row-scale)
    // (If no exchange, this is all we need for Y_f = tmp_f.)
    ws.tmp_f.noalias() = ws.Dinv.asDiagonal() * Z_f;

    if (ec_ && n < sp_.n_transition) {
        // Step 2: rhs_ψ = Z_ψ − B · tmp_f
        ws.rhs_psi.noalias() = Z_psi;
        ws.rhs_psi.noalias() -= ws.B * ws.tmp_f;
    } else {
        // No exchange: rhs_ψ = Z_ψ
        ws.rhs_psi.noalias() = Z_psi;
    }

    // Step 3: Y_ψ = Sinv · rhs_ψ  (single big gemm, reuses Sinv from SchurInverter)
    const Eigen::MatrixXd& Sinv = si_.get(static_cast<std::size_t>(n));
    Y.topRows(N_psi).noalias() = Sinv * ws.rhs_psi;

    if (ec_ && n < sp_.n_transition) {
        // Step 4: Y_f = D^{-1} · (Z_f − B^T · Y_ψ)
        //        = tmp_f − D^{-1} · (B^T · Y_ψ)
        // Use tmp_f (already D^{-1}·Z_f) and subtract D^{-1}·B^T·Y_ψ.
        Eigen::MatrixXd Bt_Ypsi = ws.B.transpose() * Y.topRows(N_psi);  // (N_f × ncols)
        Y.bottomRows(N_f).noalias() = ws.tmp_f - ws.Dinv.asDiagonal() * Bt_Ypsi;
    } else {
        // No exchange: Y_f = D^{-1} · Z_f = tmp_f
        Y.bottomRows(N_f).noalias() = ws.tmp_f;
    }
}

void
WInverseOperator::apply_U(int n, const Eigen::MatrixXd& X,
                          Eigen::MatrixXd& Y,
                          WInverseOperatorWorkspace& ws) const
{
    // Y = 12·W^{-1}·X − 10·X
    apply(n, X, Y, ws);
    Y *= 12.0;
    Y.noalias() -= 10.0 * X;
}

Eigen::MatrixXd
WInverseOperator::materialize(int n, std::size_t max_N_total) const
{
    const std::size_t N_tot = static_cast<std::size_t>(N_total());
    if (N_tot > max_N_total) {
        throw std::runtime_error(
            "WInverseOperator::materialize: N_total=" + std::to_string(N_tot) +
            " exceeds max_N_total=" + std::to_string(max_N_total) +
            " (materialization is for tests only; use apply() in production)");
    }
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N_tot, N_tot);
    Eigen::MatrixXd Winv(N_tot, N_tot);
    auto ws = make_workspace();
    apply(n, I, Winv, ws);
    return Winv;
}

void
WInverseOperator::materialize_into(int n, Eigen::MatrixXd& Winv_out,
                                   WInverseOperatorWorkspace& ws) const
{
    const int N_tot = N_total();
    const int N_psi = this->N_psi();
    const int N_f   = this->N_f();
    if (Winv_out.rows() != N_tot || Winv_out.cols() != N_tot) {
        Winv_out.resize(N_tot, N_tot);
    }
    // Asymptotic-region fast path: when no exchange is active (ec_ ==
    // nullptr) OR we are past the chi-cutoff radius (n >= n_transition),
    // B = 0 by construction, so W_n^{-1} is exactly block-diagonal:
    //   [ Sinv         0          ]
    //   [   0     diag(1/D_clamp) ]
    // The general apply(I) path would compute the same answer through a
    // (N_psi × N_psi × N_total) Sinv·rhs_psi GEMM whose right N_f
    // columns are multiplied by zero -- pure waste at large l_cont.
    // Bit-identical equivalence with apply(I) is verified by
    // test_winv_blockdiag_fastpath.cpp (literal err = 0.0, not just
    // < 1e-12).  Production gain at C8F8 / l_cont=80: ~5 h per
    // energy point on the GPU back-prop path.
    if (!ec_ || n >= sp_.n_transition) {
        Winv_out.setZero();
        Winv_out.topLeftCorner(N_psi, N_psi) =
            si_.get(static_cast<std::size_t>(n));
        const double r  = sp_.r_min + n * sp_.dr;
        const double r2 = r * r;
        const double h2_12 = sp_.dr * sp_.dr / 12.0;
        for (int f = 0; f < N_f; ++f) {
            const int l = l_sigma_[f % sp_.n_sigma];
            const double centrif = (r2 > 1e-30)
                ? double(l * (l + 1)) / r2 : 0.0;
            double Df = 1.0 - h2_12 * centrif;
            if (Df < W_min_) Df = W_min_;
            Winv_out(N_psi + f, N_psi + f) = 1.0 / Df;
        }
        return;
    }
    // Inner-region fallback: original general path via apply(I).
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N_tot, N_tot);
    apply(n, I, Winv_out, ws);
}

}  // namespace scatt
