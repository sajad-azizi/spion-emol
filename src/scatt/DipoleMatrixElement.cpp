#include "scatt/DipoleMatrixElement.hpp"

#include "angular/Gaunt.hpp"
#include "scatt/GpuDipoleEngine.hpp"
#include "scatt/GpuPropagate.hpp"        // GpuContext
#include "scatt/MklThreadGuard.hpp"

#include <memory>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iomanip>

namespace {
    struct StreamInit {
        StreamInit() {
            std::cout << std::scientific << std::setprecision(10);
            std::cerr << std::scientific << std::setprecision(10);
        }
    } _stream_init;
}

namespace scatt {


namespace {

// Compute ℓ, m from packed index idx = ℓ² + ℓ + m.
inline void idx_to_lm_local(int idx, int& l, int& m) {
    int ll = static_cast<int>(std::sqrt(static_cast<double>(idx)));
    while ((ll + 1) * (ll + 1) <= idx) ++ll;
    while (ll > 0 && ll * ll > idx) --ll;
    l = ll;
    m = idx - ll * ll - ll;
}

// 5-point centered FD with 4th-order boundary formulas, in u-convention.
// df[i] = d/dr f(r_i), for r_i = i·dr on a uniform grid.
void five_point_derivative(const double* f, int n, double dr,
                           double* df)
{
    if (n < 5) {
        // Fallback to 3-point for tiny arrays.
        if (n >= 2) {
            df[0] = (f[1] - f[0]) / dr;
            for (int i = 1; i < n - 1; ++i)
                df[i] = (f[i + 1] - f[i - 1]) / (2.0 * dr);
            df[n - 1] = (f[n - 1] - f[n - 2]) / dr;
        } else if (n == 1) {
            df[0] = 0.0;
        }
        return;
    }
    const double inv_12h = 1.0 / (12.0 * dr);
    // Interior: centered 5-point, O(h⁴).
    for (int i = 2; i < n - 2; ++i) {
        df[i] = (-f[i + 2] + 8.0 * f[i + 1] - 8.0 * f[i - 1] + f[i - 2]) * inv_12h;
    }
    // Boundaries: 4th-order one-sided.
    df[0]     = (-25.0 * f[0] + 48.0 * f[1] - 36.0 * f[2] + 16.0 * f[3] -  3.0 * f[4]) * inv_12h;
    df[1]     = ( -3.0 * f[0] - 10.0 * f[1] + 18.0 * f[2] -  6.0 * f[3] +  1.0 * f[4]) * inv_12h;
    df[n - 2] = (-1.0 * f[n - 5] +  6.0 * f[n - 4] - 18.0 * f[n - 3] + 10.0 * f[n - 2] + 3.0 * f[n - 1]) * inv_12h;
    df[n - 1] = ( 3.0 * f[n - 5] - 16.0 * f[n - 4] + 36.0 * f[n - 3] - 48.0 * f[n - 2] + 25.0 * f[n - 1]) * inv_12h;
}

}  // anon

// ---- static helpers -----------------------------------------------------

double DipoleMatrixElement::angular_dipole(int l_mu, int m_mu,
                                            int q,
                                            int l_nu, int m_nu)
{
    // A_{μν}(q) = √(4π/3) · ⟨Y^R_{ℓ_μ m_μ} | Y^R_{1,q} | Y^R_{ℓ_ν m_ν}⟩
    //           = √(4π/3) · gaunt_real(ℓ_μ, m_μ, 1, q, ℓ_ν, m_ν)
    return std::sqrt(4.0 * M_PI / 3.0) *
           angular::gaunt_real(l_mu, m_mu, 1, q, l_nu, m_nu);
}

double DipoleMatrixElement::velocity_coef(int l_mu, int l_nu)
{
    if (l_mu == l_nu + 1) return -static_cast<double>(l_nu + 1);
    if (l_mu == l_nu - 1) return  static_cast<double>(l_nu);
    return 0.0;  // selection rule violation; will never multiply a nonzero A_{μν}
}

// ---- DipoleMatrixElement --------------------------------------------------

DipoleMatrixElement::DipoleMatrixElement(const SolverParams&   sp,
                                          BackPropagator&       bp,
                                          const Eigen::MatrixXd& chi_init,
                                          const std::vector<OccupiedOrbital>& occ)
    : sp_(sp), bp_(bp), chi_init_(chi_init), occ_(occ)
{
    N_psi_    = sp_.n_mu;
    Nlm_init_ = static_cast<int>(chi_init.cols());

    const int Nr = static_cast<int>(sp.n_grid);
    if (chi_init_.rows() != Nr) {
        throw std::runtime_error(
            "DipoleMatrixElement: chi_init.rows() must equal n_grid (got "
            + std::to_string(chi_init_.rows()) + " vs " + std::to_string(Nr) + ")");
    }
    build_channel_info_();
    compute_initial_state_derivative_();
}

void DipoleMatrixElement::build_channel_info_() {
    l_mu_.resize(N_psi_);
    m_mu_.resize(N_psi_);
    for (int mu = 0; mu < N_psi_; ++mu) {
        idx_to_lm_local(mu, l_mu_[mu], m_mu_[mu]);
    }
    l_nu_.resize(Nlm_init_);
    m_nu_.resize(Nlm_init_);
    for (int nu = 0; nu < Nlm_init_; ++nu) {
        idx_to_lm_local(nu, l_nu_[nu], m_nu_[nu]);
    }
}

void DipoleMatrixElement::compute_initial_state_derivative_() {
    const int Nr = static_cast<int>(sp_.n_grid);
    dchi_init_dr_.resize(Nr, Nlm_init_);
    std::vector<double> fnu(Nr), dfnu(Nr);
    for (int nu = 0; nu < Nlm_init_; ++nu) {
        for (int ir = 0; ir < Nr; ++ir) fnu[ir] = chi_init_(ir, nu);
        five_point_derivative(fnu.data(), Nr, sp_.dr, dfnu.data());
        for (int ir = 0; ir < Nr; ++ir) dchi_init_dr_(ir, nu) = dfnu[ir];
    }
}

double DipoleMatrixElement::simpson_(const std::vector<double>& f, int n_pts) const
{
    // Composite Simpson's 1/3 rule on n_pts samples (n_pts must be ≥ 2).
    // For even intervals (n_pts odd): exact Simpson.
    // For odd intervals (n_pts even): last interval handled by 3/8 Simpson.
    if (n_pts < 2) return 0.0;
    const double h = sp_.dr;

    auto simpson_one_third = [&](int n) {
        // n ≥ 3 odd.
        double s = f[0] + f[n - 1];
        for (int i = 1; i < n - 1; i += 2) s += 4.0 * f[i];
        for (int i = 2; i < n - 2; i += 2) s += 2.0 * f[i];
        return s * h / 3.0;
    };

    if (n_pts == 2) {
        // Trapezoid on single interval.
        return 0.5 * h * (f[0] + f[1]);
    }
    if (n_pts % 2 == 1) {
        return simpson_one_third(n_pts);
    }
    // Even n_pts ⇒ n_pts − 1 intervals ⇒ (n_pts − 4) odd-count Simpson on
    // first part + Simpson 3/8 on last 4 points (3 intervals).
    // Requires n_pts ≥ 4.
    if (n_pts < 4) {
        // Fall back to trapezoid.
        double s = 0.5 * (f[0] + f[n_pts - 1]);
        for (int i = 1; i < n_pts - 1; ++i) s += f[i];
        return s * h;
    }
    double part_simpson = simpson_one_third(n_pts - 3);  // uses f[0..n-4], odd count
    double part_38 = (3.0 * h / 8.0) * (f[n_pts - 4] + 3.0 * f[n_pts - 3] +
                                         3.0 * f[n_pts - 2] + f[n_pts - 1]);
    // simpson_one_third(n-3) ends at f[n-4]; part_38 starts at f[n-4].
    // But the endpoint is double-counted: Simpson-1/3 includes f[n-4] as
    // its final point, Simpson-3/8 includes it as its starting point.
    // The trick: compute Simpson-1/3 ONLY on [0..n-4] and 3/8 on [n-4..n-1],
    // which together cover [0..n-1] with NO double-count if we treat the
    // combined rule as "stitched at f[n-4] on the TRUE boundary". The
    // stitched formula is the sum above, NOT double-counted — because
    // both rules are evaluated independently with distinct coefficients
    // over distinct (non-overlapping open intervals). Algebraically
    // correct: Simpson-1/3 integrates dr ∈ [0, (n-4)·h] and Simpson-3/8
    // integrates dr ∈ [(n-4)·h, (n-1)·h], with the boundary point f[n-4]
    // counted once in each rule's endpoint contribution — but Simpson's
    // rules are DEFINED with those endpoints as part of the quadrature.
    // We want total ∫₀^((n-1)h), which is the two partial integrals added.
    return part_simpson + part_38;
}

// ---- shared scalar-reduction helpers -------------------------------------
//
// These three helpers carry every inline scalar reduction shared between
// compute() and compute_six().  Each is a SINGLE compiled function body so
// both call sites see exactly the same FP order under -ffast-math
// (verified bit-for-bit by test_dipole_compute_six on the H2O fixture
// after icpx + -fsycl + -ffast-math built it).  See the header for the
// noinline rationale.

Eigen::MatrixXd
DipoleMatrixElement::build_xi_(const Eigen::MatrixXd& Ang,
                                DipoleGauge gauge,
                                int n_lo, int n_hi, int n_pts) const
{
    Eigen::MatrixXd Xi(n_pts, N_psi_);
    Xi.setZero();
    #pragma omp parallel for schedule(static)
    for (int ir = n_lo; ir < n_hi; ++ir) {
        const int row = ir - n_lo;
        const double r = sp_.r_min + ir * sp_.dr;
        const double inv_r = (r > 1e-30) ? 1.0 / r : 0.0;
        for (int mu = 0; mu < N_psi_; ++mu) {
            double s = 0.0;
            for (int nu = 0; nu < Nlm_init_; ++nu) {
                const double a = Ang(mu, nu);
                if (a == 0.0) continue;
                const double chi = chi_init_(ir, nu);
                if (gauge == DipoleGauge::Length) {
                    s += a * chi;
                } else {
                    const double dchi = dchi_init_dr_(ir, nu);
                    const double c = velocity_coef(l_mu_[mu], l_nu_[nu]);
                    s += a * (dchi + c * inv_r * chi);
                }
            }
            Xi(row, mu) = s;
        }
    }
    return Xi;
}

Eigen::VectorXd
DipoleMatrixElement::compute_d_correction_(const Eigen::MatrixXd& Xi,
                                            DipoleGauge gauge,
                                            int n_lo, int n_hi, int n_pts) const
{
    const int n_occ = static_cast<int>(occ_.size());
    Eigen::VectorXd d_correction = Eigen::VectorXd::Zero(n_occ);
    for (int alpha = 0; alpha < n_occ; ++alpha) {
        const auto& phi_a = occ_[alpha].phi;
        const int N_lambda_orb = static_cast<int>(phi_a.cols());
        const int mu_hi = std::min(N_psi_, N_lambda_orb);
        std::vector<double> integrand_corr(n_pts, 0.0);
        #pragma omp parallel for schedule(static)
        for (int ir = n_lo; ir < n_hi; ++ir) {
            const int row = ir - n_lo;
            const double r = sp_.r_min + ir * sp_.dr;
            const double rfac = (gauge == DipoleGauge::Length) ? r : 1.0;
            double s = 0.0;
            for (int mu = 0; mu < mu_hi; ++mu) {
                s += phi_a(ir, mu) * Xi(row, mu);
            }
            integrand_corr[row] = rfac * s;
        }
        d_correction(alpha) = simpson_(integrand_corr, n_pts);
    }
    return d_correction;
}

Eigen::VectorXcd
DipoleMatrixElement::apply_ortho_subtract_(
    const Eigen::VectorXcd& D_raw,
    const Eigen::MatrixXcd& o_coeff,
    const Eigen::VectorXd& d_correction) const
{
    using cxd = std::complex<double>;
    const int n_occ = static_cast<int>(occ_.size());
    Eigen::VectorXcd D(N_psi_);
    for (int mu = 0; mu < N_psi_; ++mu) {
        cxd sub = 0.0;
        for (int alpha = 0; alpha < n_occ; ++alpha) {
            sub += occ_[alpha].spin_factor *
                   o_coeff(mu, alpha) *
                   d_correction(alpha);
        }
        D(mu) = D_raw(mu) - sub;
    }
    return D;
}

DipoleResult DipoleMatrixElement::compute(const Eigen::MatrixXd& A,
                                          const Eigen::MatrixXd& B,
                                          DipoleGauge  gauge,
                                          Polarization pol,
                                          const Config& cfg)
{
    using cxd = std::complex<double>;
    const cxd I(0.0, 1.0);
    const int Nr_all = static_cast<int>(sp_.n_grid);
    const int q = q_of(pol);

    // Decide radial cutoff for the overlap integrals. Default: use every
    // kept ψ point below cfg.n_overlap_hi (or Nr_all if unset).
    int n_hi = (cfg.n_overlap_hi > 0) ? cfg.n_overlap_hi : Nr_all;
    n_hi = std::min(n_hi, bp_.n_keep_hi() + 1);   // exclusive upper bound
    const int n_lo = std::max(0, bp_.n_keep_lo());
    if (n_hi - n_lo < 5) {
        throw std::runtime_error(
            "DipoleMatrixElement: too few radial points for integration");
    }

    // Reset stats for this call.
    stats_ = Stats{};
    using clk = std::chrono::steady_clock;
    auto t_begin = clk::now();

    // --- build angular table A_{μν}(q)  (sparse -- most entries zero) ---
    //
    // BUG FIX (2026-04-27): the previous gate `if (mmu != mnu + q) continue;`
    // is the COMPLEX-Y selection rule (m_mu = m_nu + q).  We use REAL Y_lm
    // throughout (q-map x→+1, y→-1, z→0), and the real-Y Gaunt has a more
    // permissive m-rule -- e.g. for q = -1, real-Y allows
    //     m_mu ∈ {±|m_nu - 1|, ±|m_nu + 1|}
    // depending on signs of m_nu and m_mu.  The complex-Y rule keeps only
    // ONE of those four, so the old gate dropped ~80% of the legitimate
    // y-polarisation Gaunt couplings (numerically verified on every l ≤ 3
    // triple: 21 of 26 nonzero G^R for q=-1 were dropped, including the
    // largest, G^R(l=0,m=0; 1,-1; l=1,m=-1) = +0.282).  This was the cause
    // of the σ_y → 1e-17 anomaly on C8F8 (kept entries on their own
    // happened to integrate to machine zero against the propagated ψ).
    //
    // The triangle rule |Δl| = 1 is the only structural shortcut; let
    // `angular_dipole` (which delegates to `gaunt_real` with the proper
    // U_real_to_complex transforms) decide every m-combination.
    Eigen::MatrixXd Ang = Eigen::MatrixXd::Zero(N_psi_, Nlm_init_);
    for (int mu = 0; mu < N_psi_; ++mu) {
        const int lmu = l_mu_[mu], mmu = m_mu_[mu];
        for (int nu = 0; nu < Nlm_init_; ++nu) {
            const int lnu = l_nu_[nu], mnu = m_nu_[nu];
            if (std::abs(lmu - lnu) != 1) continue;
            const double a = angular_dipole(lmu, mmu, q, lnu, mnu);
            if (a != 0.0) Ang(mu, nu) = a;
        }
    }
    stats_.t_angular_table_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        clk::now() - t_begin).count();

    // --- precompute per-ir vectors Ξ_μ(r) for length and velocity gauges ---
    //   Length:   Ξ_μ(r)    = Σ_ν Ang_{μν} · χ_{i,ν}(r)
    //   Velocity: Ξ_μ^V(r)  = Σ_ν Ang_{μν} · w_{μν}(r)
    //             w_{μν}(r) = χ'_{i,ν}(r) + c(ℓ_μ,ℓ_ν)/r · χ_{i,ν}(r)
    // Store Ξ over [n_lo, n_hi) as (n_pts, N_psi).  Delegated to a shared
    // helper so compute_six() gets byte-identical Xi under -ffast-math.
    const int n_pts = n_hi - n_lo;
    auto t_xi0 = clk::now();
    const Eigen::MatrixXd Xi = build_xi_(Ang, gauge, n_lo, n_hi, n_pts);
    stats_.t_xi_build_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        clk::now() - t_xi0).count();

    // --- raw dipole d_β = ∫ dr [r or 1] · Σ_μ ψ_{β,μ}(r) · Ξ_μ(r) ---
    //
    // Length: extra factor of r.
    // Velocity: no factor of r.
    DipoleResult out;
    out.gauge = gauge;
    out.pol   = pol;
    out.d_raw = Eigen::VectorXd::Zero(N_psi_);
    std::vector<std::vector<double>> integrand_beta(N_psi_,
        std::vector<double>(n_pts, 0.0));

    // Iterate ir BACKWARD so the DISK-storage local_idx = n_keep_hi − ir
    // goes FORWARD. This matches the OS's sequential-read prefetcher and
    // cuts DISK-mode cost roughly in half. For MEMORY mode the direction
    // makes no difference (each integrand[beta][row] write is independent).
    //
    // Parallelism (accuracy-preserving):
    //   * MEMORY mode: BackPropagator::get_psi is thread-safe (returns
    //     a const reference into psi_memory_[]).  Parallelize OUTER ir.
    //     Each iteration writes to integrand_beta[*][row] for its own
    //     row (disjoint memory); inner mu sum is per-thread and serial
    //     -> bit-identical to single-thread serial.
    //   * DISK mode: get_psi hits an internal chunk cache and is NOT
    //     thread-safe.  Run the OUTER ir loop SERIALLY and parallelize
    //     the INNER beta loop (each beta writes to a distinct
    //     integrand_beta[beta][row]).  Same bit-identity property.
    // Inner mat-vec  s_β = Σ_μ ψ(μ,β)·Ξ(row,μ)  is an Eigen GEMV
    // (1×N_ψ) · (N_ψ×N_ψ) -> (1×N_ψ).  With EIGEN_USE_MKL_ALL this
    // dispatches to MKL's dgemv (hand-tuned SIMD); without MKL, Eigen's
    // native dgemv (also SIMD-vectorized).  In either case
    // mkl_set_dynamic(1) (set in main.cpp) ensures MKL runs SINGLE-
    // THREADED inside our omp parallel-over-ir region — no oversub.
    // Eigen's vector*matrix is not threaded by Eigen itself.
    //
    // Determinism: MKL/Eigen GEMV is deterministic for fixed inputs and
    // matches the rest of this project's MKL summation order.  It does
    // NOT match the previous scalar `s += a*b` order bit-for-bit; the
    // difference is O(ε·N_ψ) in the LSB.  All gauge / dipole
    // regression tests stay well within their tolerances.
    if (bp_.psi_in_memory()) {
        #pragma omp parallel for schedule(static)
        for (int ir = n_hi - 1; ir >= n_lo; --ir) {
            const int row = ir - n_lo;
            const double r = sp_.r_min + ir * sp_.dr;
            const double rfac = (gauge == DipoleGauge::Length) ? r : 1.0;
            const Eigen::MatrixXd& psi_ir =
                bp_.get_psi(static_cast<std::size_t>(ir));
            const Eigen::RowVectorXd row_result = Xi.row(row) * psi_ir;
            for (int beta = 0; beta < N_psi_; ++beta) {
                integrand_beta[beta][row] = rfac * row_result(beta);
            }
        }
        // Fetch timing not meaningful in MEMORY-parallel path.
    } else {
        for (int ir = n_hi - 1; ir >= n_lo; --ir) {
            const int row = ir - n_lo;
            const double r = sp_.r_min + ir * sp_.dr;
            const double rfac = (gauge == DipoleGauge::Length) ? r : 1.0;
            auto t_f0 = clk::now();
            const Eigen::MatrixXd& psi_ir =
                bp_.get_psi(static_cast<std::size_t>(ir));
            auto t_f1 = clk::now();
            stats_.t_psi_fetch_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_f1 - t_f0).count();
            // SERIAL outer ir -> MKL can use full thread count here.
            const Eigen::RowVectorXd row_result = Xi.row(row) * psi_ir;
            for (int beta = 0; beta < N_psi_; ++beta) {
                integrand_beta[beta][row] = rfac * row_result(beta);
            }
        }
    }
    // Integrand assembly time = total of this backward loop minus psi fetch.
    stats_.t_integrand_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t_xi0).count()
        - stats_.t_xi_build_ns - stats_.t_psi_fetch_ns;

    auto t_simp0 = clk::now();
    // Each beta integrates an independent integrand_beta[beta] vector
    // and writes to a distinct out.d_raw(beta) slot. Bit-identical.
    #pragma omp parallel for schedule(static)
    for (int beta = 0; beta < N_psi_; ++beta) {
        out.d_raw(beta) = simpson_(integrand_beta[beta], n_pts);
    }
    stats_.t_simpson_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        clk::now() - t_simp0).count();

    auto t_m0 = clk::now();
    Eigen::MatrixXcd AmiB = A.cast<cxd>() - I * B.cast<cxd>();
    Eigen::MatrixXcd AmiB_inv = AmiB.partialPivLu().inverse();
    Eigen::MatrixXcd M = AmiB_inv.adjoint();
    stats_.t_M_lu_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        clk::now() - t_m0).count();

    auto t_ma0 = clk::now();
    out.D_reduced_raw = M * out.d_raw.cast<cxd>();
    stats_.t_M_apply_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        clk::now() - t_ma0).count();

    // --- orthogonalization ---
    auto t_o0 = clk::now();
    const int n_occ = static_cast<int>(occ_.size());
    if (cfg.orthogonalize && n_occ > 0) {
        // Overlap b_{βα} = Σ_μ ∫ dr · φ_{α,μ}(r) · ψ_{β,μ}(r)
        // on the same [n_lo, n_hi) window.
        //
        // FAST PATH: if the caller supplied a cached b_overlap matrix
        // (from a previous gauge×pol call at the SAME energy), skip the
        // recomputation -- b_overlap depends only on (φ_α, ψ_β), not on
        // gauge or polarization.  This is the dominant cost at
        // production size; saving 5 of 6 calls cuts the dipole step
        // by ~83%.
        Eigen::MatrixXd b_overlap;
        const bool have_cache =
            cfg.cached_b_overlap != nullptr
            && cfg.cached_b_overlap->rows() == N_psi_
            && cfg.cached_b_overlap->cols() == n_occ;
        if (have_cache) {
            b_overlap = *cfg.cached_b_overlap;
        } else {
        b_overlap = Eigen::MatrixXd::Zero(N_psi_, n_occ);
        std::vector<double> integrand_overlap(n_pts, 0.0);
        for (int alpha = 0; alpha < n_occ; ++alpha) {
            const auto& phi_a = occ_[alpha].phi;
            const int N_lambda_orb = static_cast<int>(phi_a.cols());
            // truncate μ-summation at phi's angular extent
            const int mu_hi = std::min(N_psi_, N_lambda_orb);
            // Same reverse-ir trick for DISK-friendly reads. The inner
            // beta loop now reads psi_ir(mu, beta) for many beta per ir,
            // so we restructure to iterate ir outermost and beta innermost
            // (cheaper since each get_psi call serves all beta).
            std::vector<std::vector<double>> all_integrands(
                N_psi_, std::vector<double>(n_pts, 0.0));
            // Same MEMORY/DISK split as the integrand loop above; same
            // GEMV substitution.  Inner mat-vec is
            //   s_β = Σ_{μ<mu_hi} φ_a(ir, μ) · ψ(μ, β)
            //       = (φ_a.row(ir).head(mu_hi) · ψ.topRows(mu_hi))(β)
            // Eigen GEMV; MKL when EIGEN_USE_MKL_ALL is on.
            if (bp_.psi_in_memory()) {
                #pragma omp parallel for schedule(static)
                for (int ir = n_hi - 1; ir >= n_lo; --ir) {
                    const int row = ir - n_lo;
                    const Eigen::MatrixXd& psi_ir =
                        bp_.get_psi(static_cast<std::size_t>(ir));
                    const Eigen::RowVectorXd row_result =
                        phi_a.row(ir).head(mu_hi) * psi_ir.topRows(mu_hi);
                    for (int beta = 0; beta < N_psi_; ++beta) {
                        all_integrands[beta][row] = row_result(beta);
                    }
                }
            } else {
                for (int ir = n_hi - 1; ir >= n_lo; --ir) {
                    const int row = ir - n_lo;
                    const Eigen::MatrixXd& psi_ir =
                        bp_.get_psi(static_cast<std::size_t>(ir));
                    const Eigen::RowVectorXd row_result =
                        phi_a.row(ir).head(mu_hi) * psi_ir.topRows(mu_hi);
                    for (int beta = 0; beta < N_psi_; ++beta) {
                        all_integrands[beta][row] = row_result(beta);
                    }
                }
            }
            #pragma omp parallel for schedule(static)
            for (int beta = 0; beta < N_psi_; ++beta) {
                b_overlap(beta, alpha) = simpson_(all_integrands[beta], n_pts);
            }
        }
        }   // end !have_cache branch
        out.b_overlap = b_overlap;

        // Dipole correction d_α = ⟨φ_α | O | Φ_i⟩
        //   Length:   d_α = Σ_{μν} Ang_{μν} ∫ dr · r · φ_{α,μ} · χ_{i,ν}
        //   Velocity: d_α = Σ_{μν} Ang_{μν} ∫ dr · φ_{α,μ} · w_{μν}
        // Shared helper for byte-identity under -ffast-math.
        out.d_correction = compute_d_correction_(Xi, gauge, n_lo, n_hi, n_pts);

        // o_{μα} = Σ_β M_{μβ} · b_{βα}
        Eigen::MatrixXcd o_coeff = M * b_overlap.cast<cxd>();

        // D_μ = D^(raw)_μ − Σ_α N_α · o_{μα} · d_α  (shared helper).
        out.D_reduced = apply_ortho_subtract_(
            out.D_reduced_raw, o_coeff, out.d_correction);
    } else {
        out.D_reduced = out.D_reduced_raw;
        out.b_overlap = Eigen::MatrixXd(0, 0);
        out.d_correction = Eigen::VectorXd(0);
    }

    // Partial cross-section-like quantity (Σ |D_μ|²).
    out.partial_sigma = out.D_reduced.squaredNorm();

    stats_.t_ortho_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        clk::now() - t_o0).count();

    if (cfg.verbose) {
        std::cout << "[DipoleMatrixElement] gauge="
                  << (gauge == DipoleGauge::Length ? "length" : "velocity")
                  << "  pol=" << name_of(pol)
                  << "  N_psi=" << N_psi_
                  << "  n_occ=" << n_occ
                  << "  n_pts=" << n_pts
                  << "  Σ|D|²="<< std::scientific << std::setprecision(10) << out.partial_sigma << "\n";
    }

    return out;
}

// ===========================================================================
// compute_six: batched 6-(gauge × pol) computation in one ir-pass.
//
// Strict bit-identity contract with calling compute(A, B, g, p, cfg) six
// times.  Achieved by preserving every individual GEMV's input bytes and
// every Simpson input vector's order:
//
//   * Per-ir integrand GEMV `Xi_gp.row(row) * psi_ir` is the same operand
//     pair the legacy path would have used for that (g, p).
//   * Per-ir-per-α b_overlap GEMV `phi_a.row(ir).head(mu_hi) *
//     psi_ir.topRows(mu_hi)` is the same operand pair the legacy α-outer
//     loop would have used.
//   * Simpson is deterministic per input vector; the input vectors are
//     accumulated in the same row order as the legacy path.
//   * M = (A − iB)⁻¹†, Xi build, d_correction, and the final ortho
//     subtraction all use the same operand bytes in the same order.
//
// The only loop change is: (g, p) outer and α outer become ir-outer, with
// 6 + n_occ GEMVs per ir against a single shared psi_ir.  The b_overlap
// path materialises per-α integrand buffers of size n_occ × N_psi × n_pts
// (13.5 GB at L=100); legacy held one such buffer per α and freed it
// before moving to α+1.  Caller is responsible for confirming this fits.
// ===========================================================================
std::array<DipoleResult, 6>
DipoleMatrixElement::compute_six(const Eigen::MatrixXd& A,
                                  const Eigen::MatrixXd& B,
                                  const Config& cfg)
{
    // Force the main thread's MKL pool to the full OMP thread count for
    // the serial-outer per-ir loop (8 chunked GEMMs / ir at L=100).  See
    // MklThreadGuard.hpp for the rationale: globally mkl_set_dynamic(1)
    // protects OMP regions but can throttle serial MKL calls.
#if defined(_OPENMP)
    MklThreadGuard _mkl_local(omp_get_max_threads());
#else
    MklThreadGuard _mkl_local(1);
#endif

    using cxd = std::complex<double>;
    const cxd I(0.0, 1.0);
    const int Nr_all = static_cast<int>(sp_.n_grid);

    // Slot order MUST match DipoleWriter::slice_index().
    constexpr std::array<std::pair<DipoleGauge, Polarization>, 6> combos = {{
        {DipoleGauge::Length,   Polarization::X},
        {DipoleGauge::Length,   Polarization::Y},
        {DipoleGauge::Length,   Polarization::Z},
        {DipoleGauge::Velocity, Polarization::X},
        {DipoleGauge::Velocity, Polarization::Y},
        {DipoleGauge::Velocity, Polarization::Z}
    }};

    // ---- ir window (same logic as compute()) ----
    int n_hi = (cfg.n_overlap_hi > 0) ? cfg.n_overlap_hi : Nr_all;
    n_hi = std::min(n_hi, bp_.n_keep_hi() + 1);
    const int n_lo  = std::max(0, bp_.n_keep_lo());
    if (n_hi - n_lo < 5) {
        throw std::runtime_error(
            "DipoleMatrixElement: too few radial points for integration");
    }
    const int n_pts = n_hi - n_lo;
    const int n_occ = static_cast<int>(occ_.size());

    stats_ = Stats{};
    using clk = std::chrono::steady_clock;
    const auto t_begin = clk::now();

    // ---- Angular tables: one per q (3 unique values across 6 slots) ----
    auto build_ang = [&](int q) {
        Eigen::MatrixXd Ang = Eigen::MatrixXd::Zero(N_psi_, Nlm_init_);
        for (int mu = 0; mu < N_psi_; ++mu) {
            const int lmu = l_mu_[mu], mmu = m_mu_[mu];
            for (int nu = 0; nu < Nlm_init_; ++nu) {
                const int lnu = l_nu_[nu], mnu = m_nu_[nu];
                if (std::abs(lmu - lnu) != 1) continue;
                const double a = angular_dipole(lmu, mmu, q, lnu, mnu);
                if (a != 0.0) Ang(mu, nu) = a;
            }
        }
        return Ang;
    };
    const std::array<Eigen::MatrixXd, 3> Ang_q = {
        build_ang(+1),  // X (q = +1)
        build_ang(-1),  // Y (q = -1)
        build_ang( 0)   // Z (q =  0)
    };
    stats_.t_angular_table_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            clk::now() - t_begin).count();

    auto ang_for_pol = [&](Polarization p) -> const Eigen::MatrixXd& {
        switch (p) {
            case Polarization::X: return Ang_q[0];
            case Polarization::Y: return Ang_q[1];
            case Polarization::Z: return Ang_q[2];
        }
        return Ang_q[0];
    };

    // ---- 6 Ξ tables (one per slot).  Each (n_pts × N_psi). ----
    // Shared helper -- same compiled body as compute() so the 6 Xi
    // matrices here are byte-identical to what 6x compute() would build,
    // even under -ffast-math.
    const auto t_xi0 = clk::now();
    std::array<Eigen::MatrixXd, 6> Xi_arr;
    for (int slot = 0; slot < 6; ++slot) {
        Xi_arr[slot] = build_xi_(ang_for_pol(combos[slot].second),
                                 combos[slot].first,
                                 n_lo, n_hi, n_pts);
    }
    stats_.t_xi_build_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        clk::now() - t_xi0).count();

    // ---- M = (A − iB)⁻¹†  (computed ONCE, reused for all 6) ----
    const auto t_m0 = clk::now();
    Eigen::MatrixXcd AmiB = A.cast<cxd>() - I * B.cast<cxd>();
    Eigen::MatrixXcd M    = AmiB.partialPivLu().inverse().adjoint();
    stats_.t_M_lu_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        clk::now() - t_m0).count();

    // ---- 6 integrand_beta arrays for the (g, p) integrals ----
    std::array<std::vector<std::vector<double>>, 6> integrand_beta_arr;
    for (int slot = 0; slot < 6; ++slot) {
        integrand_beta_arr[slot].assign(
            N_psi_, std::vector<double>(n_pts, 0.0));
    }

    // ---- b_overlap path: use cache, or materialise per-α integrand
    //      buffers and rebuild b_overlap from a single ir-pass. ----
    const bool ortho_active = cfg.orthogonalize && n_occ > 0;
    const bool have_b_cache =
        cfg.cached_b_overlap != nullptr
        && cfg.cached_b_overlap->rows() == N_psi_
        && cfg.cached_b_overlap->cols() == n_occ;
    const bool build_b_overlap = ortho_active && !have_b_cache;

    // all_integrands_per_alpha[α][β][row]:  n_occ × N_psi × n_pts × 8 B
    // ~13.5 GB at L=100.  Allocated ONLY when ortho is on and the cache
    // is absent; freed before the per-(g, p) work begins.
    std::vector<std::vector<std::vector<double>>> all_integrands_per_alpha;
    if (build_b_overlap) {
        all_integrands_per_alpha.assign(
            n_occ, std::vector<std::vector<double>>(
                N_psi_, std::vector<double>(n_pts, 0.0)));
    }

    // ---- Single ir-pass: read ψ_n once, do 6 + n_occ GEMVs ----
    //
    // Per ir we do:
    //   1) ONE bp_.get_psi(ir) read.
    //   2) 6 integrand GEMVs  (Xi_arr[slot].row(row) * psi_ir).
    //   3) n_occ b_overlap GEMVs (phi_a.row(ir).head(mu_hi) *
    //                             psi_ir.topRows(mu_hi))  [only if building].
    //
    // CPU path:
    //   MEMORY-ψ mode: outer ir loop is parallel-for; each thread writes
    //                  only to its own row.  Disjoint -> bit-identical.
    //   DISK-ψ mode:   outer ir loop is SERIAL (chunk cache is not
    //                  thread-safe) -- inner GEMVs use MKL's 112 threads.
    //                  Both paths produce the same bytes.
    //
    // GPU path (cfg.use_gpu && GpuContext::gpu_available()):
    //   Outer ir loop is STRICTLY SERIAL (one GPU stream, in-order queue).
    //   Per ir we build a single V (n_slots × N_psi) matrix on the host
    //   from the (6 dipole + n_occ overlap) rows that the CPU path would
    //   GEMV individually -- with the length-gauge r-factor baked into the
    //   6 dipole rows so no scaling is needed on the output -- then call
    //   GpuDipoleEngine::step which uploads V + psi_ir, runs ONE oneMKL /
    //   cuBLAS DGEMM on the device, and downloads the (n_slots × N_psi)
    //   result.  Bit-equivalent to the CPU GEMV path up to the GEMM-vs-GEMV
    //   summation-order rounding (ε_mach × N, ~1e-13 relative).  See
    //   test_gpu_dme for the tolerance gate.
    //
    //   DISK-ψ contract: bp_.get_psi(ir) reads exactly one chunk at a time
    //   into a single-threaded chunk cache; the GPU step then runs to
    //   completion before the next get_psi() call, so no race window
    //   exists.  Mirrors the CPU DISK-ψ serial pattern.
    const auto t_integ0 = clk::now();

    // ---- Decide: GPU offload vs. CPU path ---------------------------------
    const int n_slots_gpu = 6 + (build_b_overlap ? n_occ : 0);
    const bool want_gpu_requested = cfg.use_gpu;
    const bool gpu_available_now  = GpuContext::gpu_available();
    const bool want_gpu = want_gpu_requested && gpu_available_now;
    if (want_gpu_requested && !gpu_available_now) {
        std::cerr << "[DipoleMatrixElement] (compute_six) use_gpu=true but no "
                     "GPU is visible at runtime -- falling back to CPU.  "
                     "Build with -DSCATT_WITH_SYCL=ON and run on a GPU node.\n";
    }

    std::unique_ptr<GpuContext>      gpu_ctx;
    std::unique_ptr<GpuDipoleEngine> gpu_eng;
    Eigen::MatrixXd                  V_gpu;       // (n_slots_gpu × N_psi), host scratch
    Eigen::MatrixXd                  R_gpu;       // (n_slots_gpu × N_psi), host scratch

    if (want_gpu) {
        gpu_ctx = std::make_unique<GpuContext>(/*prefer_gpu=*/true);
        gpu_eng = std::make_unique<GpuDipoleEngine>(
            *gpu_ctx, N_psi_, n_slots_gpu);
        V_gpu = Eigen::MatrixXd::Zero(n_slots_gpu, N_psi_);
        R_gpu = Eigen::MatrixXd::Zero(n_slots_gpu, N_psi_);
        if (cfg.verbose) {
            std::cout << "[DipoleMatrixElement] (compute_six) GPU offload "
                         "enabled: " << gpu_ctx->info().device_name
                      << "  N_psi=" << N_psi_
                      << "  n_slots=" << n_slots_gpu
                      << "  (DGEMM per ir; CPU GEMV path bypassed)\n";
        }
    }

    auto do_ir_step_cpu = [&](int ir) {
        const int row = ir - n_lo;
        const double r = sp_.r_min + ir * sp_.dr;
        const Eigen::MatrixXd& psi_ir =
            bp_.get_psi(static_cast<std::size_t>(ir));
        for (int slot = 0; slot < 6; ++slot) {
            const double rfac =
                (combos[slot].first == DipoleGauge::Length) ? r : 1.0;
            const Eigen::RowVectorXd row_result =
                Xi_arr[slot].row(row) * psi_ir;
            for (int beta = 0; beta < N_psi_; ++beta) {
                integrand_beta_arr[slot][beta][row] = rfac * row_result(beta);
            }
        }
        if (build_b_overlap) {
            for (int alpha = 0; alpha < n_occ; ++alpha) {
                const auto& phi_a = occ_[alpha].phi;
                const int N_lambda_orb = static_cast<int>(phi_a.cols());
                const int mu_hi = std::min(N_psi_, N_lambda_orb);
                const Eigen::RowVectorXd row_result =
                    phi_a.row(ir).head(mu_hi) * psi_ir.topRows(mu_hi);
                for (int beta = 0; beta < N_psi_; ++beta) {
                    all_integrands_per_alpha[alpha][beta][row] =
                        row_result(beta);
                }
            }
        }
    };

    auto do_ir_step_gpu = [&](int ir) {
        const int row = ir - n_lo;
        const double r = sp_.r_min + ir * sp_.dr;
        const Eigen::MatrixXd& psi_ir =
            bp_.get_psi(static_cast<std::size_t>(ir));
        // ---- Build V_gpu(n_slots × N_psi) on host ------------------------
        // Rows 0..5: r-factor-pre-multiplied Xi rows (length gauge gets r,
        // velocity gauge gets 1.0).  This bakes the rfac scaling into the
        // GEMM input so no host-side rescale is needed on the output.
        for (int slot = 0; slot < 6; ++slot) {
            const double rfac =
                (combos[slot].first == DipoleGauge::Length) ? r : 1.0;
            V_gpu.row(slot) = rfac * Xi_arr[slot].row(row);
        }
        // Rows 6..6+n_occ-1: zero-padded φ_α(ir, 0..mu_hi-1).  Zero padding
        // beyond mu_hi makes  V_row · ψ_ir(:, β) = Σ_{μ<mu_hi} φ_α(ir, μ)
        // · ψ_ir(μ, β)  =  φ_α.row(ir).head(mu_hi) · ψ_ir.topRows(mu_hi)(:, β),
        // i.e. exactly the CPU expression.
        if (build_b_overlap) {
            for (int alpha = 0; alpha < n_occ; ++alpha) {
                const auto& phi_a = occ_[alpha].phi;
                const int N_lambda_orb = static_cast<int>(phi_a.cols());
                const int mu_hi = std::min(N_psi_, N_lambda_orb);
                V_gpu.row(6 + alpha).setZero();
                V_gpu.row(6 + alpha).head(mu_hi) = phi_a.row(ir).head(mu_hi);
            }
        }
        // ---- ONE DGEMM: R = V · ψ_ir   (device round-trip) --------------
        gpu_eng->step(V_gpu, psi_ir, R_gpu);

        // ---- Scatter result into the per-(slot, β) integrand buffers ----
        // R_gpu(slot, β) is already rfac-multiplied (rfac was baked into
        // V_gpu above), so we store it verbatim.
        for (int slot = 0; slot < 6; ++slot) {
            for (int beta = 0; beta < N_psi_; ++beta) {
                integrand_beta_arr[slot][beta][row] = R_gpu(slot, beta);
            }
        }
        if (build_b_overlap) {
            for (int alpha = 0; alpha < n_occ; ++alpha) {
                for (int beta = 0; beta < N_psi_; ++beta) {
                    all_integrands_per_alpha[alpha][beta][row] =
                        R_gpu(6 + alpha, beta);
                }
            }
        }
    };

    if (want_gpu) {
        // Strictly serial across ir (one GPU stream, one chunk cache).
        // Time the psi_fetch the same way as the CPU DISK-ψ path for
        // apples-to-apples bench output.
        for (int ir = n_hi - 1; ir >= n_lo; --ir) {
            if (!bp_.psi_in_memory()) {
                const auto t_f0 = clk::now();
                (void)bp_.get_psi(static_cast<std::size_t>(ir));  // warm cache for timing
                const auto t_f1 = clk::now();
                stats_.t_psi_fetch_ns +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t_f1 - t_f0).count();
            }
            do_ir_step_gpu(ir);
        }
    } else if (bp_.psi_in_memory()) {
        #pragma omp parallel for schedule(static)
        for (int ir = n_hi - 1; ir >= n_lo; --ir) {
            do_ir_step_cpu(ir);
        }
    } else {
        for (int ir = n_hi - 1; ir >= n_lo; --ir) {
            const auto t_f0 = clk::now();
            (void)bp_.get_psi(static_cast<std::size_t>(ir));  // warm cache for timing
            const auto t_f1 = clk::now();
            stats_.t_psi_fetch_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_f1 - t_f0).count();
            do_ir_step_cpu(ir);
        }
    }
    stats_.t_integrand_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            clk::now() - t_integ0).count();

    // ---- Build b_overlap from the per-α integrand buffers ----
    Eigen::MatrixXd b_overlap;
    if (have_b_cache) {
        b_overlap = *cfg.cached_b_overlap;
    } else if (ortho_active) {
        b_overlap = Eigen::MatrixXd::Zero(N_psi_, n_occ);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int alpha = 0; alpha < n_occ; ++alpha) {
            for (int beta = 0; beta < N_psi_; ++beta) {
                b_overlap(beta, alpha) =
                    simpson_(all_integrands_per_alpha[alpha][beta], n_pts);
            }
        }
        all_integrands_per_alpha.clear();
        all_integrands_per_alpha.shrink_to_fit();
    }

    // ---- Per-slot finalisation: Simpson, M·d, d_correction, ortho ----
    const auto t_post0 = clk::now();
    std::array<DipoleResult, 6> results;
    for (int slot = 0; slot < 6; ++slot) {
        DipoleResult& out = results[slot];
        out.gauge = combos[slot].first;
        out.pol   = combos[slot].second;

        // d_raw via Simpson over β.
        out.d_raw = Eigen::VectorXd::Zero(N_psi_);
        const auto t_s0 = clk::now();
        #pragma omp parallel for schedule(static)
        for (int beta = 0; beta < N_psi_; ++beta) {
            out.d_raw(beta) = simpson_(integrand_beta_arr[slot][beta], n_pts);
        }
        stats_.t_simpson_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            clk::now() - t_s0).count();
        // Free this slot's integrand buffer now that d_raw is materialised.
        integrand_beta_arr[slot].clear();
        integrand_beta_arr[slot].shrink_to_fit();

        // D_reduced_raw = M · d_raw.
        const auto t_ma0 = clk::now();
        out.D_reduced_raw = M * out.d_raw.cast<cxd>();
        stats_.t_M_apply_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            clk::now() - t_ma0).count();

        if (!ortho_active) {
            out.D_reduced    = out.D_reduced_raw;
            out.b_overlap    = Eigen::MatrixXd(0, 0);
            out.d_correction = Eigen::VectorXd(0);
            out.partial_sigma = out.D_reduced.squaredNorm();
            continue;
        }

        // d_correction[α] for this (g, p), using Xi_arr[slot] and phi_α.
        // Shared helper for byte-identity with compute() under -ffast-math.
        const auto t_o0 = clk::now();
        out.d_correction = compute_d_correction_(
            Xi_arr[slot], combos[slot].first, n_lo, n_hi, n_pts);

        // o_coeff = M · b_overlap.
        Eigen::MatrixXcd o_coeff = M * b_overlap.cast<cxd>();

        // D = D_raw − Σ_α N_α · o_{μα} · d_α  (shared helper).
        out.D_reduced     = apply_ortho_subtract_(
            out.D_reduced_raw, o_coeff, out.d_correction);
        out.b_overlap     = b_overlap;
        out.partial_sigma = out.D_reduced.squaredNorm();
        stats_.t_ortho_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            clk::now() - t_o0).count();
    }
    (void)t_post0;

    if (cfg.verbose) {
        for (int slot = 0; slot < 6; ++slot) {
            const auto& out = results[slot];
            std::cout << "[DipoleMatrixElement] (compute_six) gauge="
                      << (out.gauge == DipoleGauge::Length ? "length" : "velocity")
                      << "  pol=" << name_of(out.pol)
                      << "  Σ|D|²="
                      << std::scientific << std::setprecision(10)
                      << out.partial_sigma << "\n";
        }
    }
    return results;
}

}  // namespace scatt
