#include "scatt/AsymptoticAmplitudes.hpp"

#include "angular/Gaunt.hpp"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace scatt {

AsymptoticAmplitudes::AsymptoticAmplitudes(const SolverParams& sp,
                                           BackPropagator&     bp)
    : sp_(sp), bp_(bp)
{
    build_channel_info_();
    if (sp_.energy <= 0.0) {
        throw std::runtime_error(
            "AsymptoticAmplitudes: energy must be > 0 for scattering (E = "
            + std::to_string(sp_.energy) + ")");
    }
}

void AsymptoticAmplitudes::build_channel_info_() {
    l_psi_.resize(sp_.n_mu);
    for (int mu = 0; mu < sp_.n_mu; ++mu) {
        int l, m;
        angular::idx_to_lm(mu, l, m);
        l_psi_[mu] = l;
    }
}

double AsymptoticAmplitudes::riccati_j_(int l, double x) {
    if (x < 1e-300) return 0.0;
    gsl_sf_result r;
    const int status = gsl_sf_bessel_jl_e(l, x, &r);
    if (status != GSL_SUCCESS && status != GSL_EUNDRFLW) {
        throw std::runtime_error(std::string("gsl_sf_bessel_jl_e: ") +
                                 gsl_strerror(status));
    }
    return x * r.val;
}

double AsymptoticAmplitudes::riccati_y_(int l, double x) {
    if (x < 1e-300) {
        throw std::runtime_error(
            "AsymptoticAmplitudes: Neumann y_l diverges at x = 0");
    }
    gsl_sf_result r;
    const int status = gsl_sf_bessel_yl_e(l, x, &r);
    if (status != GSL_SUCCESS && status != GSL_EUNDRFLW) {
        throw std::runtime_error(std::string("gsl_sf_bessel_yl_e: ") +
                                 gsl_strerror(status));
    }
    return x * r.val;
}

AmplitudeResult AsymptoticAmplitudes::extract(const Config& cfg){
    const int    N_psi = sp_.n_mu;
    const int    Nr    = static_cast<int>(sp_.n_grid);
    const double k     = std::sqrt(2.0 * sp_.energy);

    // Resolve fit window.
    int n_start = cfg.n_fit_start;
    int n_end   = cfg.n_fit_end;
    if (n_start < 0) {
        n_start = std::max(Nr - 200, static_cast<int>(0.9 * Nr));
    }
    if (n_end < 0) {
        n_end = Nr - 1 - cfg.n_exclude_outer;
    }

    // Clamp to the outer-window range of the back-propagator. When the
    // asymptotic-buffer trick is active (n_asym > 0), this is [asym_offset,
    // N]; otherwise it's the full main-store range [n_keep_lo, n_keep_hi].
    n_start = std::max(n_start, bp_.outer_window_lo());
    n_end   = std::min(n_end,   bp_.outer_window_hi());

    if (n_end - n_start + 1 < 10) {
        throw std::runtime_error(
            "AsymptoticAmplitudes: fit window is too small (" +
            std::to_string(n_end - n_start + 1) +
            " points). Extend BackPropagator's n_keep range or lower "
            "n_fit_start.");
    }

    const int n_fit = n_end - n_start + 1;
    const double r_start = sp_.r_min + n_start * sp_.dr;
    const double r_end   = sp_.r_min + n_end   * sp_.dr;

    AmplitudeResult out;
    out.n_fit_start = n_start;
    out.n_fit_end   = n_end;
    out.r_fit_start = r_start;
    out.r_fit_end   = r_end;
    out.A = Eigen::MatrixXd::Zero(N_psi, N_psi);
    out.B = Eigen::MatrixXd::Zero(N_psi, N_psi);

    // Precompute jhat(l, kr_n) and yhat(l, kr_n) at every n in the window,
    // for every unique ℓ_μ in l_psi_. Then for each (μ, j) the LSQ fit
    // reduces to a 2x2 normal-equation solve using these cached tables.
    std::vector<int> unique_ls;
    {
        std::vector<int> sorted_ls = l_psi_;
        std::sort(sorted_ls.begin(), sorted_ls.end());
        sorted_ls.erase(std::unique(sorted_ls.begin(), sorted_ls.end()), sorted_ls.end());
        unique_ls = std::move(sorted_ls);
    }

    // Tables indexed by (l, point_in_window).
    std::vector<Eigen::VectorXd> jhat(unique_ls.size());
    std::vector<Eigen::VectorXd> yhat(unique_ls.size());
    for (std::size_t t = 0; t < unique_ls.size(); ++t) {
        const int l = unique_ls[t];
        jhat[t].resize(n_fit);
        yhat[t].resize(n_fit);
        for (int i = 0; i < n_fit; ++i) {
            const double r  = sp_.r_min + (n_start + i) * sp_.dr;
            const double kr = k * r;
            jhat[t](i) = riccati_j_(l, kr);
            yhat[t](i) = riccati_y_(l, kr);
        }
    }
    // Per-μ lookup: l_idx[mu] = index of l_psi_[mu] within unique_ls.
    std::vector<int> l_idx(N_psi);
    for (int mu = 0; mu < N_psi; ++mu) {
        l_idx[mu] = static_cast<int>(
            std::lower_bound(unique_ls.begin(), unique_ls.end(), l_psi_[mu])
            - unique_ls.begin());
    }

    // Per-ℓ normal-matrix coefficients (2 × 2) independent of j and μ
    // specifically — depend only on ℓ through the (ĵ, ŷ) tables. Cache them.
    //
    // SINGULAR-ℓ HANDLING:
    //   The 2×2 normal matrix M_ℓ becomes singular when the fit window
    //   sits under the centrifugal barrier of channel ℓ
    //   (k·r_max ≲ ℓ) -- ĵ_ℓ and ŷ_ℓ are then both evanescent and lose
    //   the linear independence that fitting requires.
    //
    //   DEFAULT (cfg.allow_closed_channels = false):  abort with the
    //   diagnostic message.  This preserves accuracy: the user must
    //   either extend r_max or reduce l_cont.
    //
    //   OPT-IN  (cfg.allow_closed_channels = true):  treat the affected
    //   ℓ as CLOSED.  Below in the per-μ inversion loop the substitution
    //       A_{μν} := δ_{μν},  B_{μν} := 0   for all μ with l_μ = ℓ_closed
    //   is applied; the K-matrix block for those channels is then 0 and
    //   the channels make no asymptotic contribution.  This is the
    //   SAME as running with a smaller l_cont (the channel doesn't
    //   exist in the asymptotic sense); the equivalence holds as long
    //   as the photoionization observable is converged in ℓ within the
    //   set of OPEN channels (verifiable by running a smaller-box,
    //   smaller-l_cont reference and comparing).
    std::vector<Eigen::Matrix2d> XtX(unique_ls.size());
    std::vector<Eigen::Matrix2d> XtX_inv(unique_ls.size());
    std::vector<bool> ell_is_closed(unique_ls.size(), false);
    int n_closed_ls = 0;
    for (std::size_t t = 0; t < unique_ls.size(); ++t) {
        const Eigen::VectorXd& jv = jhat[t];
        const Eigen::VectorXd& yv = yhat[t];
        Eigen::Matrix2d M;
        M(0, 0) = jv.squaredNorm();
        M(0, 1) = jv.dot(yv);
        M(1, 0) = M(0, 1);
        M(1, 1) = yv.squaredNorm();
        const double det = M.determinant();
        const bool singular = !(std::abs(det) > 1e-20
                                * M.cwiseAbs().maxCoeff()
                                * M.cwiseAbs().maxCoeff());
        if (singular) {
            if (!cfg.allow_closed_channels) {
                throw std::runtime_error(
                    "AsymptoticAmplitudes: singular normal matrix for ℓ=" +
                    std::to_string(unique_ls[t]) +
                    "  (fit window may be too narrow or on a Bessel node). "
                    "Extend the window, shift it, OR opt in to the "
                    "closed-channel approximation by setting "
                    "Config::allow_closed_channels = true (see header).");
            }
            ell_is_closed[t] = true;
            XtX[t]     = Eigen::Matrix2d::Identity();   // unused placeholders
            XtX_inv[t] = Eigen::Matrix2d::Identity();
            ++n_closed_ls;
            if (cfg.verbose) {
                std::cerr << "[AsymptoticAmplitudes] closed channel: ℓ="
                          << unique_ls[t]
                          << "  (det(M)/|M|² = "
                          << std::abs(det) /
                              (M.cwiseAbs().maxCoeff() *
                               M.cwiseAbs().maxCoeff() + 1e-300)
                          << ").  Substituting A=I, B=0 for channels "
                             "with l=" << unique_ls[t] << "  "
                             "(opt-in via allow_closed_channels=true).\n";
            }
        } else {
            XtX[t]     = M;
            XtX_inv[t] = M.inverse();
        }
    }
    if (n_closed_ls > 0 && cfg.verbose) {
        std::cerr << "[AsymptoticAmplitudes] " << n_closed_ls
                  << " ℓ value(s) of " << unique_ls.size()
                  << " marked CLOSED.  Result is equivalent to running "
                     "with l_cont reduced to the highest open ℓ.\n";
    }

    // Fill the fit.  STREAMING PROJECTION (no per-i ψ materialisation).
    //
    // The legacy implementation cached every ψ_n in the fit window into a
    // std::vector<MatrixXd> of size n_fit × N_psi² × 8 B.  At production
    // L=100 (N_psi≈10201, n_fit=300) that is ~250 GB of resident memory --
    // unconditionally OOMs the BP node and is independent of whether
    // psi_asym is MEMORY- or DISK-backed.
    //
    // The fit is fully linear in ψ, so we collapse it into TWO matrices
    // YJ, YY of size N_psi × N_psi and a 2x2 solve per (μ, j):
    //
    //   YJ(μ, j) = Σ_i jhat[l(μ)](i) · ψ_i(μ, j)
    //   YY(μ, j) = Σ_i yhat[l(μ)](i) · ψ_i(μ, j)
    //   [A(μ,j); B(μ,j)] = (XᵀX)_l(μ)⁻¹ · [YJ; YY]
    //
    // Pre-tabulate Jcoef(μ, i) = jhat[l_idx[μ]](i) and Ycoef similarly so
    // the inner accumulation is just a diagonal-scaled GEMM increment.
    // Memory: 2 × N_psi × N_psi × 8 B (YJ + YY) + 2 × N_psi × n_fit × 8 B
    // (Jcoef + Ycoef).  At L=100 that's ~1.7 GB + ~50 MB -- fits trivially.
    //
    // The residual pass below makes a second sequential read of the asym
    // buffer.  On DISK that doubles its read I/O for this stage, but it's
    // sequential-chunk-friendly (matches PotentialStorage's chunk cache);
    // at L=100 the extra wall is tens of seconds vs ~250 GB OOM otherwise.
    double max_resid_abs = 0.0;
    double max_psi_abs   = 0.0;

    Eigen::MatrixXd Jcoef(N_psi, n_fit);
    Eigen::MatrixXd Ycoef(N_psi, n_fit);
    for (int mu = 0; mu < N_psi; ++mu) {
        const int ti = l_idx[mu];
        for (int i = 0; i < n_fit; ++i) {
            Jcoef(mu, i) = jhat[ti](i);
            Ycoef(mu, i) = yhat[ti](i);
        }
    }

    Eigen::MatrixXd YJ = Eigen::MatrixXd::Zero(N_psi, N_psi);
    Eigen::MatrixXd YY = Eigen::MatrixXd::Zero(N_psi, N_psi);
    for (int i = 0; i < n_fit; ++i) {
        const Eigen::MatrixXd& psi_i =
            bp_.get_psi(static_cast<std::size_t>(n_start + i));
        max_psi_abs = std::max(max_psi_abs, psi_i.cwiseAbs().maxCoeff());
        // Per-row scale-and-accumulate.  asDiagonal() avoids materialising
        // a full diag(Jcoef.col(i)) and keeps this O(N_psi²) per i.
        YJ += Jcoef.col(i).asDiagonal() * psi_i;
        YY += Ycoef.col(i).asDiagonal() * psi_i;
    }

    for (int mu = 0; mu < N_psi; ++mu) {
        const int ti = l_idx[mu];
        if (ell_is_closed[ti]) {
            // Opt-in closed-channel substitution: A_{μν}=δ_{μν}, B_{μν}=0.
            // Keeps (A − iB) well-conditioned (identity block on closed
            // channels); K-matrix for those channels is then 0.
            for (int j = 0; j < N_psi; ++j) {
                out.A(mu, j) = (j == mu) ? 1.0 : 0.0;
                out.B(mu, j) = 0.0;
            }
            continue;
        }
        const Eigen::Matrix2d& Mi = XtX_inv[ti];
        for (int j = 0; j < N_psi; ++j) {
            Eigen::Vector2d rhs(YJ(mu, j), YY(mu, j));
            Eigen::Vector2d coeffs = Mi * rhs;
            out.A(mu, j) = coeffs(0);
            out.B(mu, j) = coeffs(1);
        }
    }

    // Residual diagnostic.  Second pass: re-read each ψ_n, compare to
    // ψ_fit_n = diag(Jcoef.col(i)) · A + diag(Ycoef.col(i)) · B.
    // Transient: one N_psi × N_psi temp per i (~830 MB at L=100), released
    // at end of iteration.
    //
    // Closed channels (substituted A=I, B=0) would produce an artificially
    // large residual against the actual ψ -- they are explicitly excluded
    // from this max so the open-channel fit quality is reported faithfully.
    std::vector<bool> mu_is_closed(N_psi, false);
    for (int mu = 0; mu < N_psi; ++mu)
        mu_is_closed[mu] = ell_is_closed[l_idx[mu]];
    for (int i = 0; i < n_fit; ++i) {
        const Eigen::MatrixXd& psi_i =
            bp_.get_psi(static_cast<std::size_t>(n_start + i));
        Eigen::MatrixXd psi_fit = Jcoef.col(i).asDiagonal() * out.A;
        psi_fit             += Ycoef.col(i).asDiagonal() * out.B;
        Eigen::MatrixXd diff = (psi_fit - psi_i).cwiseAbs();
        // Zero out closed-channel rows so they don't contaminate the max.
        for (int mu = 0; mu < N_psi; ++mu)
            if (mu_is_closed[mu]) diff.row(mu).setZero();
        const double r_abs = diff.maxCoeff();
        if (r_abs > max_resid_abs) max_resid_abs = r_abs;
    }

    out.fit_residual_max = max_resid_abs;
    out.fit_residual_rel = max_resid_abs / std::max(max_psi_abs, 1e-30);
    out.A_minus_I_max    = (out.A - Eigen::MatrixXd::Identity(N_psi, N_psi))
                              .cwiseAbs().maxCoeff();

    // K = B · A^{-1}. Do NOT symmetrise: the asymmetry IS a physical
    // indicator of fit quality (V-tail contamination, finite r_max). The
    // raw K lets S = (A + iB)(A − iB)^{-1} = (I + iK)(I − iK)^{-1} hold
    // as an algebraic identity, which is useful for downstream
    // consistency checks. Callers who want a symmetrised K can do:
    //     K_sym = 0.5 * (K + K.transpose())
    // and report K_symmetry_err as their systematic uncertainty.
    Eigen::FullPivLU<Eigen::MatrixXd> lu_A(out.A);
    if (!lu_A.isInvertible()) {
        throw std::runtime_error(
            "AsymptoticAmplitudes: fitted A matrix is singular.");
    }
    out.K = out.B * lu_A.inverse();
    out.K_symmetry_err = (out.K - out.K.transpose()).cwiseAbs().maxCoeff();

    // S = (A + iB) · (A − iB)^{-1}
    const std::complex<double> im(0.0, 1.0);
    Eigen::MatrixXcd Ac = out.A.cast<std::complex<double>>();
    Eigen::MatrixXcd Bc = out.B.cast<std::complex<double>>();
    Eigen::MatrixXcd apb = Ac + im * Bc;
    Eigen::MatrixXcd amb = Ac - im * Bc;
    out.S = apb * amb.partialPivLu().inverse();
    Eigen::MatrixXcd I_c = Eigen::MatrixXcd::Identity(N_psi, N_psi);
    out.S_unitarity_err = (out.S.adjoint() * out.S - I_c)
                              .cwiseAbs().maxCoeff();

    // Eigenphases.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(out.K);
    out.eigenphases.resize(N_psi);
    for (int i = 0; i < N_psi; ++i) {
        out.eigenphases[i] = std::atan(es.eigenvalues()(i));
    }

    if (cfg.verbose) {
        std::cout << "[AsymptoticAmplitudes] fit window: "
                  << "[n=" << n_start << " .. " << n_end << "] "
                  << "(r=" << r_start << " .. " << r_end << " au, "
                  << n_fit << " pts)\n";
        std::cout << "  fit_residual_max = " << std::scientific << std::setprecision(10) << out.fit_residual_max
                  << "  (rel " << out.fit_residual_rel << ")\n";
        std::cout << "  ‖A − I‖_∞       = " << std::scientific << std::setprecision(10) << out.A_minus_I_max << "\n";
        std::cout << "  ‖K − Kᵀ‖_∞      = " << std::scientific << std::setprecision(10) << out.K_symmetry_err << "\n";
        std::cout << "  ‖S†S − I‖_∞    = " << std::scientific << std::setprecision(10) << out.S_unitarity_err << "\n";
    }

    return out;
}

}  // namespace scatt
