#include "ComplexResolventEa.hpp"

#include "Common.hpp"

#include <array>
#include <limits>

#if __has_include(<gsl/gsl_sf_dawson.h>)
#include <gsl/gsl_sf_dawson.h>
#define ZEPE_COMPLEX_RESOLVENT_HAVE_GSL_DAWSON 1
#endif

namespace zepe {
namespace {

using cdouble = std::complex<double>;

std::vector<double> uniform_simpson38_weights(int n, double dx)
{
    std::vector<double> w(n, 0.0);
    const int intervals = n - 1;
    if (n < 2) return w;

    if (intervals == 1) {
        w[0] = 0.5 * dx;
        w[1] = 0.5 * dx;
        return w;
    }

    if (intervals % 2 == 0) {
        for (int i = 0; i < n; ++i) {
            if (i == 0 || i == n - 1) w[i] = dx / 3.0;
            else if (i % 2 == 1)      w[i] = 4.0 * dx / 3.0;
            else                      w[i] = 2.0 * dx / 3.0;
        }
        return w;
    }

    if (intervals == 3) {
        w[0] = 3.0 * dx / 8.0;
        w[1] = 9.0 * dx / 8.0;
        w[2] = 9.0 * dx / 8.0;
        w[3] = 3.0 * dx / 8.0;
        return w;
    }

    const int last13 = n - 4;
    for (int i = 0; i <= last13; ++i) {
        if (i == 0 || i == last13) w[i] += dx / 3.0;
        else if (i % 2 == 1)       w[i] += 4.0 * dx / 3.0;
        else                       w[i] += 2.0 * dx / 3.0;
    }
    w[n - 4] += 3.0 * dx / 8.0;
    w[n - 3] += 9.0 * dx / 8.0;
    w[n - 2] += 9.0 * dx / 8.0;
    w[n - 1] += 3.0 * dx / 8.0;
    return w;
}

void append_shifted_log_grid(double offset, double E_shift_lo,
                             double E_shift_hi, int N,
                             std::vector<double>& E,
                             std::vector<double>& w)
{
    if (E_shift_hi <= E_shift_lo || N < 4) {
        throw std::runtime_error(
            "append_shifted_log_grid: bad range or too few points");
    }

    const double xlo = std::log(E_shift_lo);
    const double xhi = std::log(E_shift_hi);
    const double dx = (xhi - xlo) / (N - 1);
    const std::vector<double> wx = uniform_simpson38_weights(N, dx);

    for (int i = 0; i < N; ++i) {
        const double E_shift = std::exp(xlo + i * dx);
        E.push_back(offset + E_shift);
        w.push_back(wx[i] * E_shift);
    }
}

double radial_weight(int i, int n, double dr)
{
    const int intervals = n - 1;
    if (n < 2) return 0.0;
    if (intervals == 1) return 0.5 * dr;

    if (intervals % 2 == 0) {
        if (i == 0 || i == n - 1) return dr / 3.0;
        return (i % 2 == 1) ? 4.0 * dr / 3.0 : 2.0 * dr / 3.0;
    }

    if (intervals == 3) {
        if (i == 0 || i == 3) return 3.0 * dr / 8.0;
        return 9.0 * dr / 8.0;
    }

    const int last13 = n - 4;
    double wi = 0.0;
    if (i <= last13) {
        if (i == 0 || i == last13) wi += dr / 3.0;
        else if (i % 2 == 1)       wi += 4.0 * dr / 3.0;
        else                       wi += 2.0 * dr / 3.0;
    }
    if (i == n - 4 || i == n - 1) wi += 3.0 * dr / 8.0;
    if (i == n - 3 || i == n - 2) wi += 9.0 * dr / 8.0;
    return wi;
}

double dawson_real(double x)
{
#ifdef ZEPE_COMPLEX_RESOLVENT_HAVE_GSL_DAWSON
    return gsl_sf_dawson(x);
#else
    const double ax = std::abs(x);
    if (ax < 0.2) {
        const double xx = x * x;
        return x * (1.0 - (2.0 / 3.0) * xx
                  * (1.0 - 0.4 * xx * (1.0 - (2.0 / 7.0) * xx)));
    }

    constexpr int NMAX = 12;
    constexpr double H = 0.4;
    constexpr double INV_SQRT_PI = 0.56418958354775628695;
    static const std::array<double, NMAX> c = [] {
        std::array<double, NMAX> values{};
        for (int i = 0; i < NMAX; ++i) {
            const double u = (2.0 * i + 1.0) * H;
            values[i] = std::exp(-u * u);
        }
        return values;
    }();

    const int n0 = 2 * (int)(0.5 * ax / H + 0.5);
    const double xp = ax - n0 * H;
    double e1 = std::exp(2.0 * xp * H);
    const double e2 = e1 * e1;
    double d1 = n0 + 1.0;
    double d2 = d1 - 2.0;
    double sum = 0.0;
    for (int i = 0; i < NMAX; ++i) {
        sum += c[i] * (e1 / d1 + 1.0 / (d2 * e1));
        d1 += 2.0;
        d2 -= 2.0;
        e1 *= e2;
    }
    const double ans = INV_SQRT_PI * std::exp(-xp * xp) * sum;
    return std::copysign(ans, x);
#endif
}

cdouble faddeeva_real(double z)
{
    const double real_part = (z * z > 745.0) ? 0.0 : std::exp(-z * z);
    const double imag_part = 2.0 / std::sqrt(M_PI) * dawson_real(z);
    return {real_part, imag_part};
}

cdouble second_order_J_real_energy(double Ef, double En, double Eh,
                                   double omega, double tau0)
{
    // Emit then absorb through M_F=-5: sigma1=+1, sigma2=-1.
    const double Omega1 = En - Eh + omega;
    const double Omega2 = Ef - En - omega;
    const double sum = Omega1 + Omega2;
    const double z = (Omega2 - Omega1) * tau0 / (2.0 * std::sqrt(2.0));
    const double envelope = std::exp(-0.125 * sum * sum * tau0 * tau0);
    return 0.5 * M_PI * tau0 * tau0 * envelope * faddeeva_real(z);
}

cdouble emit_absorb_green_pulse_factor(double Ef, double Eh, double tau0)
{
    const double delta = Ef - Eh;
    const double envelope = std::exp(-0.125 * delta * delta * tau0 * tau0);
    return {0.0, -0.25 * std::sqrt(0.5 * M_PI) * tau0 * envelope};
}

Eigen::Matrix2d square_well_matrix2_at_r(const Potentials& pot, double r)
{
    Eigen::Matrix2d threshold = Eigen::Matrix2d::Zero();
    threshold(0, 0) = pot.thresholds()(0);
    threshold(1, 1) = pot.thresholds()(1);
    const Eigen::Matrix2d s1s2 = pot.s1s2();

    if (r == 0.0) return Eigen::Matrix2d::Zero();
    if (r <= pot.r0()) {
        return -pot.V_bar() * Eigen::Matrix2d::Identity()
             + pot.dV() * s1s2
             + threshold;
    }
    return threshold;
}

cdouble bilinear2(const Eigen::Vector2d& a, const Eigen::Vector2cd& b)
{
    return a(0) * b(0) + a(1) * b(1);
}

std::vector<Eigen::Vector2cd> solve_complex_resolvent_2ch(
    const Potentials& pot,
    const Parameters& p,
    cdouble z,
    double dr,
    int N,
    const std::vector<Eigen::Vector2d>& source)
{
    if (pot.num_channels() != 2) {
        throw std::runtime_error(
            "solve_complex_resolvent_2ch requires a two-channel block");
    }
    if ((int)source.size() != N) {
        throw std::runtime_error(
            "solve_complex_resolvent_2ch: source has wrong grid size");
    }

    const int M = N - 1;
    const double h2 = dr * dr;
    const double a_coef = -1.0 / (2.0 * p.mu * h2);
    const Eigen::Matrix2cd I2 = Eigen::Matrix2cd::Identity();
    const Eigen::Matrix2cd A = a_coef * I2;
    const Eigen::Matrix2cd C = A;

    Eigen::Matrix2cd tail = Eigen::Matrix2cd::Zero();
    for (int a = 0; a < 2; ++a) {
        cdouble kappa = std::sqrt(2.0 * p.mu * (pot.thresholds()(a) - z));
        if (kappa.real() < 0.0) kappa = -kappa;
        tail(a, a) = std::exp(-kappa * dr);
    }

    std::vector<Eigen::Matrix2cd> Bmod(M);
    std::vector<Eigen::Vector2cd> dmod(M);

    for (int j = 0; j < M; ++j) {
        const int i = j + 1;
        const double r = i * dr;
        Eigen::Matrix2cd Bj =
            (1.0 / (p.mu * h2)) * I2
            + square_well_matrix2_at_r(pot, r).cast<cdouble>()
            - z * I2;
        if (j == M - 1) {
            Bj += C * tail;
        }

        Eigen::Vector2cd dj = -source[i].cast<cdouble>();
        if (j == 0) {
            Bmod[j] = Bj;
            dmod[j] = dj;
        } else {
            Eigen::PartialPivLU<Eigen::Matrix2cd> lu_prev(Bmod[j - 1]);
            const Eigen::Matrix2cd M_mat =
                lu_prev.solve(A.transpose()).transpose();
            Bmod[j] = Bj - M_mat * C;
            dmod[j] = dj - M_mat * dmod[j - 1];
        }
    }

    std::vector<Eigen::Vector2cd> X(N, Eigen::Vector2cd::Zero());
    {
        Eigen::PartialPivLU<Eigen::Matrix2cd> lu_last(Bmod[M - 1]);
        X[N - 1] = lu_last.solve(dmod[M - 1]);
    }
    for (int j = M - 2; j >= 0; --j) {
        const int i = j + 1;
        Eigen::PartialPivLU<Eigen::Matrix2cd> lu(Bmod[j]);
        X[i] = lu.solve(dmod[j] - C * X[i + 1]);
    }
    return X;
}

} // namespace

ComplexResolventEaResult compute_complex_resolvent_ea(
    Potentials& pot_m5,
    Parameters& p_m5,
    Potentials& pot_m4,
    Parameters& p_m4,
    const AnalyticState& halo_state,
    const std::vector<double>& Ef_grid,
    const Eigen::MatrixXd& eta_m4_m5,
    double E_halo,
    double omega,
    double tau0,
    double rf_energy2,
    const ComplexResolventEaOptions& opt)
{
    ComplexResolventEaResult result;
    result.enabled = opt.enabled;
    result.opt = opt;
    if (!opt.enabled) return result;
    if (pot_m5.num_channels() != 2 || pot_m4.num_channels() != 2) {
        throw std::runtime_error(
            "compute_complex_resolvent_ea requires two-channel M_F=-5 and M_F=-4 blocks");
    }
    if (eta_m4_m5.rows() != 2 || eta_m4_m5.cols() != 2) {
        throw std::runtime_error(
            "compute_complex_resolvent_ea requires a 2x2 eta_m4_m5 matrix");
    }

    const int Nf = (int)Ef_grid.size();
    const int N = (int)std::llround(opt.R_a0 / opt.dr_a0) + 1;
    if (N < 4) {
        throw std::runtime_error(
            "compute_complex_resolvent_ea: radial grid has too few points");
    }

    result.radial_N = N;
    result.c_cont_scaled.assign(Nf, {0.0, 0.0});
    result.c_LO_cont_scaled.assign(Nf, {0.0, 0.0});
    result.exact_cont_kernel.assign(Nf, {0.0, 0.0});
    result.LO_cont_kernel.assign(Nf, 0.0);

    const double eps = AU::kHz_to_au(opt.eps_kHz);
    const double eta = AU::kHz_to_au(opt.eta_kHz);
    const double thr0 = pot_m5.thresholds()(0);
    const double thr1 = pot_m5.thresholds()(1);
    const double E_max = AU::GHz_to_au(opt.E_max_GHz);
    const double E1_shift_hi = (thr1 - thr0) - eps;
    const double E2_shift_hi = E_max - thr1;
    if (E1_shift_hi <= eps || E2_shift_hi <= eps) {
        throw std::runtime_error(
            "compute_complex_resolvent_ea: continuum thresholds and Emax are inconsistent");
    }

    std::vector<double> E_cont;
    std::vector<double> w_cont;
    append_shifted_log_grid(thr0, eps, E1_shift_hi, opt.N_seg1, E_cont, w_cont);
    append_shifted_log_grid(thr1, eps, E2_shift_hi, opt.N_seg2, E_cont, w_cont);
    result.continuum_N = (int)E_cont.size();

    const Eigen::Matrix2d eta_m5_m4 = eta_m4_m5.transpose();
    std::vector<Eigen::Vector2d> source(N, Eigen::Vector2d::Zero());
    std::vector<double> wr(N, 0.0);
    for (int i = 0; i < N; ++i) {
        const double r = i * opt.dr_a0;
        wr[i] = radial_weight(i, N, opt.dr_a0);
        source[i] = eta_m5_m4 * halo_state.evaluate2(r);
    }

    AnalyticSquareWell solver_m4(&pot_m4, &p_m4);
    std::vector<Eigen::Vector2d> final_eta((size_t)Nf * (size_t)N,
                                           Eigen::Vector2d::Zero());
    for (int kf = 0; kf < Nf; ++kf) {
        const AnalyticState uf = solver_m4.build_scattering_state(Ef_grid[kf], 0);
        const size_t base = (size_t)kf * (size_t)N;
        for (int i = 0; i < N; ++i) {
            const double r = i * opt.dr_a0;
            final_eta[base + (size_t)i] =
                eta_m4_m5.transpose() * uf.evaluate2(r);
        }
    }

    std::vector<cdouble> I_exact(Nf, {0.0, 0.0});
    std::vector<double> I_LO(Nf, 0.0);
    double source_norm = 0.0;

    #pragma omp parallel
    {
        std::vector<cdouble> I_exact_local(Nf, {0.0, 0.0});
        std::vector<double> I_LO_local(Nf, 0.0);
        double source_norm_local = 0.0;

        #pragma omp for schedule(dynamic)
        for (int ie = 0; ie < (int)E_cont.size(); ++ie) {
            const cdouble z(E_cont[ie], eta);
            const auto X = solve_complex_resolvent_2ch(
                pot_m5, p_m5, z, opt.dr_a0, N, source);

            cdouble source_G_source(0.0, 0.0);
            for (int i = 0; i < N; ++i) {
                source_G_source += wr[i] * bilinear2(source[i], X[i]);
            }
            const double rho_hh = -source_G_source.imag() / M_PI;
            source_norm_local += w_cont[ie] * rho_hh;

            for (int kf = 0; kf < Nf; ++kf) {
                const size_t base = (size_t)kf * (size_t)N;
                cdouble M_z(0.0, 0.0);
                for (int i = 0; i < N; ++i) {
                    M_z += wr[i] * bilinear2(final_eta[base + (size_t)i], X[i]);
                }

                const double rho_fh = -M_z.imag() / M_PI;
                I_exact_local[kf] += w_cont[ie] * rho_fh
                    * second_order_J_real_energy(Ef_grid[kf], E_cont[ie],
                                                 E_halo, omega, tau0);
                I_LO_local[kf] += w_cont[ie] * rho_fh
                    / (E_halo - omega - E_cont[ie]);
            }
        }

        #pragma omp critical
        {
            source_norm += source_norm_local;
            for (int kf = 0; kf < Nf; ++kf) {
                I_exact[kf] += I_exact_local[kf];
                I_LO[kf] += I_LO_local[kf];
            }
        }
    }

    result.source_continuum_norm = source_norm;
    for (int kf = 0; kf < Nf; ++kf) {
        result.exact_cont_kernel[kf] = -0.25 * I_exact[kf];
        result.LO_cont_kernel[kf] = I_LO[kf];
        result.c_cont_scaled[kf] = rf_energy2 * result.exact_cont_kernel[kf];
        result.c_LO_cont_scaled[kf] =
            rf_energy2
            * emit_absorb_green_pulse_factor(Ef_grid[kf], E_halo, tau0)
            * I_LO[kf];
    }

    return result;
}

} // namespace zepe
