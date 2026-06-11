#include "scatt/KMatrixExtractor.hpp"

#include "angular/Gaunt.hpp"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace scatt {

KMatrixExtractor::KMatrixExtractor(const SolverParams& sp,
                                   ForwardRPropagator& frp,
                                   int                 n_match_in)
    : sp_(sp), frp_(frp)
{
    build_channel_info_();

    if (sp_.energy <= 0.0) {
        throw std::runtime_error(
            "KMatrixExtractor: energy must be > 0 for scattering (E = " +
            std::to_string(sp_.energy) + ")");
    }
    const int Nrm1 = static_cast<int>(sp_.n_grid) - 1;
    if (n_match_in < 0) {
        n_match_ = Nrm1 - 1;                  // default: (N−2, N−1)
    } else {
        n_match_ = n_match_in;
    }
    if (n_match_ < 0 || n_match_ >= Nrm1) {
        throw std::runtime_error(
            "KMatrixExtractor: n_match (" + std::to_string(n_match_) +
            ") out of range [0, " + std::to_string(Nrm1 - 1) + "]");
    }
}

void KMatrixExtractor::build_channel_info_() {
    l_psi_.resize(sp_.n_mu);
    for (int mu = 0; mu < sp_.n_mu; ++mu) {
        int l, m;
        angular::idx_to_lm(mu, l, m);
        l_psi_[mu] = l;
    }
}

double KMatrixExtractor::riccati_j_(int l, double x) {
    if (x < 1e-300) return 0.0;
    gsl_sf_result res;
    const int status = gsl_sf_bessel_jl_e(l, x, &res);
    if (status != GSL_SUCCESS && status != GSL_EUNDRFLW) {
        throw std::runtime_error(std::string("gsl_sf_bessel_jl_e: ") +
                                 gsl_strerror(status));
    }
    return x * res.val;
}

double KMatrixExtractor::riccati_y_(int l, double x) {
    if (x < 1e-300) {
        // Neumann y_l diverges at x = 0; shouldn't happen at the outer match.
        throw std::runtime_error("riccati_y called at x ≈ 0");
    }
    gsl_sf_result res;
    const int status = gsl_sf_bessel_yl_e(l, x, &res);
    if (status != GSL_SUCCESS && status != GSL_EUNDRFLW) {
        throw std::runtime_error(std::string("gsl_sf_bessel_yl_e: ") +
                                 gsl_strerror(status));
    }
    return x * res.val;
}

ScatteringResult KMatrixExtractor::extract(){
    const int N_psi = sp_.n_mu;
    const int N_f   = sp_.n_occ * sp_.n_sigma;
    const double h  = sp_.dr;
    const double h2_12 = h * h / 12.0;
    const double k     = std::sqrt(2.0 * sp_.energy);
    const double r_in  = sp_.r_min + n_match_       * h;   // inner match radius
    const double r_out = sp_.r_min + (n_match_ + 1) * h;   // outer match radius

    ScatteringResult out;
    out.n_match       = n_match_;
    out.r_match_inner = r_in;
    out.r_match_outer = r_out;

    // ----------------------------------------------------------------------
    // Step 1: R_{n_M}  =  Rinv[n_M]^{-1}  (full N_total × N_total)
    // ----------------------------------------------------------------------
    const Eigen::MatrixXd& Rinv_nM = frp_.get(static_cast<std::size_t>(n_match_));
    const Eigen::MatrixXd  R       = Rinv_nM.partialPivLu().inverse();

    // ----------------------------------------------------------------------
    // Step 2: Schur complement of R on the ψ block.
    //   S_schur = R_ψψ  −  R_ψf · R_ff^{-1} · R_fψ
    // If f-channels are numerically absent (norms below threshold) OR
    // N_f == 0, use R_ψψ directly.
    // ----------------------------------------------------------------------
    Eigen::MatrixXd S_schur;
    constexpr double kCouplingZeroThreshold = 1e-14;

    if (N_f == 0) {
        S_schur = R;
        out.schur_coupling_zero = true;
    } else {
        const auto R_pp = R.topLeftCorner(N_psi, N_psi);
        const auto R_pf = R.topRightCorner(N_psi, N_f);
        const auto R_fp = R.bottomLeftCorner(N_f, N_psi);
        const auto R_ff = R.bottomRightCorner(N_f, N_f);
        const double norm_pf = R_pf.norm();
        const double norm_fp = R_fp.norm();
        if (norm_pf < kCouplingZeroThreshold && norm_fp < kCouplingZeroThreshold) {
            S_schur = R_pp;
            out.schur_coupling_zero = true;
        } else {
            Eigen::MatrixXd R_ff_inv = R_ff.partialPivLu().inverse();
            S_schur = R_pp - R_pf * R_ff_inv * R_fp;
        }
    }
    // Symmetrize (S_schur should be symmetric for real Q, but LU drift).
    S_schur = 0.5 * (S_schur + S_schur.transpose().eval());

    // ----------------------------------------------------------------------
    // Step 3: diagonal Riccati-Bessel J, N and free-W matrices at r_in, r_out.
    //   W^free(μ, μ) = 1 + (h²/12)·(k² − ℓ_μ(ℓ_μ+1)/r²)
    //   J_n(μ, μ)    = ĵ_{ℓμ}(k·r_n)
    //   N_n(μ, μ)    = ŷ_{ℓμ}(k·r_n)
    // ----------------------------------------------------------------------
    Eigen::MatrixXd J_in  = Eigen::MatrixXd::Zero(N_psi, N_psi);
    Eigen::MatrixXd J_out = Eigen::MatrixXd::Zero(N_psi, N_psi);
    Eigen::MatrixXd Y_in  = Eigen::MatrixXd::Zero(N_psi, N_psi);
    Eigen::MatrixXd Y_out = Eigen::MatrixXd::Zero(N_psi, N_psi);
    Eigen::MatrixXd W_in  = Eigen::MatrixXd::Zero(N_psi, N_psi);
    Eigen::MatrixXd W_out = Eigen::MatrixXd::Zero(N_psi, N_psi);

    const double k2    = k * k;
    const double r_in2  = r_in  * r_in;
    const double r_out2 = r_out * r_out;

    for (int mu = 0; mu < N_psi; ++mu) {
        const int l = l_psi_[mu];
        J_in (mu, mu) = riccati_j_(l, k * r_in);
        J_out(mu, mu) = riccati_j_(l, k * r_out);
        Y_in (mu, mu) = riccati_y_(l, k * r_in);
        Y_out(mu, mu) = riccati_y_(l, k * r_out);

        const double centrif_in  = static_cast<double>(l * (l + 1)) / r_in2;
        const double centrif_out = static_cast<double>(l * (l + 1)) / r_out2;
        W_in (mu, mu) = 1.0 + h2_12 * (k2 - centrif_in);
        W_out(mu, mu) = 1.0 + h2_12 * (k2 - centrif_out);
    }

    // ----------------------------------------------------------------------
    // Step 4: extract K.
    //   A_mat = W_out · J_out  −  S_schur · W_in · J_in
    //   B_mat = S_schur · W_in · N_in  −  W_out · N_out
    //   K     = B_mat^{-1} · A_mat
    // ----------------------------------------------------------------------
    Eigen::MatrixXd A_mat = W_out * J_out - S_schur * (W_in * J_in);
    Eigen::MatrixXd B_mat = S_schur * (W_in * Y_in) - W_out * Y_out;

    Eigen::MatrixXd K = B_mat.partialPivLu().solve(A_mat);
    K = 0.5 * (K + K.transpose().eval());   // symmetrize (K is exactly symmetric in continuum limit)
    out.K_matrix       = K;
    out.K_symmetry_err = (K - K.transpose()).cwiseAbs().maxCoeff();

    // ----------------------------------------------------------------------
    // Step 5: S = (I + iK)·(I − iK)^{-1}  (PDF eq. 25). Complex.
    // ----------------------------------------------------------------------
    const std::complex<double> im(0.0, 1.0);
    Eigen::MatrixXcd I_c = Eigen::MatrixXcd::Identity(N_psi, N_psi);
    Eigen::MatrixXcd K_c = K.cast<std::complex<double>>();
    Eigen::MatrixXcd lhs = I_c + im * K_c;   // I + iK
    Eigen::MatrixXcd rhs = I_c - im * K_c;   // I − iK
    // S = lhs · rhs^{-1}
    Eigen::MatrixXcd S = lhs * rhs.partialPivLu().inverse();
    out.S_matrix = S;

    Eigen::MatrixXcd SdS = S.adjoint() * S;
    out.unitarity_err = (SdS - I_c).cwiseAbs().maxCoeff();

    // ----------------------------------------------------------------------
    // Step 6: eigenphases = atan(eigenvalues of K).
    // ----------------------------------------------------------------------
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K);
    out.eigenphases.resize(N_psi);
    for (int i = 0; i < N_psi; ++i) {
        out.eigenphases[i] = std::atan(es.eigenvalues()(i));
    }

    //debug prints symetricic K and unitary S errors
    std::cout << "K symmetric error: " << std::scientific << std::setprecision(10) << out.K_symmetry_err << std::endl;
    std::cout << "S unitarity error: " << std::scientific << std::setprecision(10) << out.unitarity_err << std::endl;

    return out;
}

Eigen::MatrixXd
KMatrixExtractor::make_psi_boundary(const SolverParams&    sp,
                                    const Eigen::MatrixXd& K)
{
    const int N_psi = sp.n_mu;
    if (K.rows() != N_psi || K.cols() != N_psi) {
        throw std::runtime_error("make_psi_boundary: K must be N_psi × N_psi");
    }
    if (sp.energy <= 0.0) {
        throw std::runtime_error("make_psi_boundary: energy must be > 0");
    }
    const double k     = std::sqrt(2.0 * sp.energy);
    const double r_N   = sp.r_min + (sp.n_grid - 1) * sp.dr;

    Eigen::MatrixXd J_N = Eigen::MatrixXd::Zero(N_psi, N_psi);
    Eigen::MatrixXd Y_N = Eigen::MatrixXd::Zero(N_psi, N_psi);
    for (int mu = 0; mu < N_psi; ++mu) {
        int l, m;
        angular::idx_to_lm(mu, l, m);
        J_N(mu, mu) = riccati_j_(l, k * r_N);
        Y_N(mu, mu) = riccati_y_(l, k * r_N);
    }
    // psi_boundary = J_N + N_N · K  (regular normalization A = I, B = K).
    return J_N + Y_N * K;
}

}  // namespace scatt
