#include "Common.hpp"
#include "Parameters.hpp"
#include "SpinAlgebra.hpp"
#include "Potentials.hpp"
#include "AnalyticSquareWell.hpp"
#include "ClosedChannelInhom.hpp"
#include "ComplexResolventEa.hpp"
#include <fstream>
#include <complex>
#include <limits>
#if __has_include(<gsl/gsl_sf_dawson.h>)
#include <gsl/gsl_sf_dawson.h>
#define ZEPE_HAVE_GSL_DAWSON 1
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// Uniform-grid quadrature. Uses composite Simpson 1/3 when possible and
// a Simpson 3/8 closure when the number of intervals is odd.
// ============================================================
template <class F>
static double integrate_uniform_grid(int N_grid, double dr, F&& f)
{
    const int intervals = N_grid - 1;
    if (N_grid < 2) return 0.0;
    if (intervals == 1) {
        return 0.5 * dr * (f(0) + f(1));
    }

    if (intervals % 2 == 0) {
        double s = f(0) + f(N_grid - 1);
        for (int ir = 1; ir < N_grid - 1; ++ir) {
            s += (ir % 2 == 1 ? 4.0 : 2.0) * f(ir);
        }
        return s * dr / 3.0;
    }

    if (intervals == 3) {
        return 3.0 * dr / 8.0
             * (f(0) + 3.0 * f(1) + 3.0 * f(2) + f(3));
    }

    // Use Simpson 1/3 up to N_grid-4, then Simpson 3/8 for the last
    // three intervals. This keeps fourth-order accuracy on uniform grids.
    const int last13 = N_grid - 4;
    double s = f(0) + f(last13);
    for (int ir = 1; ir < last13; ++ir) {
        s += (ir % 2 == 1 ? 4.0 : 2.0) * f(ir);
    }
    double total13 = s * dr / 3.0;
    double total38 = 3.0 * dr / 8.0
        * (f(N_grid - 4) + 3.0 * f(N_grid - 3)
         + 3.0 * f(N_grid - 2) + f(N_grid - 1));
    return total13 + total38;
}

static std::vector<double> uniform_simpson38_weights(int n, double dx)
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

static std::vector<double> log_energy_quadrature_weights(
    const std::vector<double>& E_grid)
{
    const int n = (int)E_grid.size();
    std::vector<double> wE(n, 0.0);
    if (n < 2) return wE;
    if (E_grid.front() <= 0.0 || E_grid.back() <= E_grid.front()) {
        throw std::runtime_error("log_energy_quadrature_weights: invalid log grid");
    }

    const double dx = std::log(E_grid.back() / E_grid.front()) / (n - 1);
    const std::vector<double> wx = uniform_simpson38_weights(n, dx);
    for (int i = 0; i < n; ++i) {
        wE[i] = wx[i] * E_grid[i]; // dE = E d(log E)
    }
    return wE;
}

static int env_int_or_default(const char* name, int fallback)
{
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') return fallback;
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || value <= 1) {
        throw std::runtime_error(std::string("Invalid integer environment variable ") + name);
    }
    if (value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("Environment variable too large: ") + name);
    }
    return (int)value;
}

static double env_double_or_default(const char* name, double fallback)
{
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') return fallback;
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(raw, &end);
    if (errno != 0 || end == raw || *end != '\0' || !(value > 0.0)) {
        throw std::runtime_error(std::string("Invalid positive environment variable ") + name);
    }
    return value;
}

static bool env_flag_enabled(const char* name)
{
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') return false;
    const std::string v(raw);
    return v == "1" || v == "true" || v == "TRUE"
        || v == "yes" || v == "YES" || v == "on" || v == "ON";
}

// Matrix element <u_A | eta | u_B> = sum_{a,b} eta(a,b) int u_A^(a) u_B^(b) dr
// on the current uniform grid. The wavefunctions and RF matrices are real here.
static double overlap_eta(const std::vector<Eigen::VectorXd>& uA,
                           const std::vector<Eigen::VectorXd>& uB,
                           const Eigen::MatrixXd& eta,
                           double dr, int N_grid)
{
    int Na = eta.rows(), Nb = eta.cols();
    if ((int)uA.size() != N_grid || (int)uB.size() != N_grid) {
        throw std::runtime_error("overlap_eta: wavefunction grid size mismatch");
    }
    if (eta.rows() == 0 || eta.cols() == 0
        || uA[0].size() < eta.rows() || uB[0].size() < eta.cols()) {
        throw std::runtime_error("overlap_eta: channel dimension mismatch");
    }
    if (N_grid < 2) return 0.0;
    double total = 0.0;

    for (int a = 0; a < Na; ++a) {
        for (int b = 0; b < Nb; ++b) {
            double eab = eta(a, b);
            if (std::abs(eab) < 1e-16) continue;
            total += eab * integrate_uniform_grid(N_grid, dr, [&](int ir) {
                return uA[ir](a) * uB[ir](b);
            });
        }
    }
    return total;
}

static double overlap_eta_state_grid2(const AnalyticState& uA_state,
                                      const std::vector<Eigen::Vector2d>& uB,
                                      const Eigen::MatrixXd& eta,
                                      double dr, int N_grid)
{
    if ((int)uB.size() != N_grid || eta.rows() != 2 || eta.cols() != 2) {
        throw std::runtime_error("overlap_eta_state_grid2: dimension mismatch");
    }
    const Eigen::Matrix2d eta2 = eta.topLeftCorner<2,2>();
    return integrate_uniform_grid(N_grid, dr, [&](int ir) {
        const double r = ir * dr;
        const Eigen::Vector2d uA = uA_state.evaluate2(r);
        return uA.dot(eta2 * uB[ir]);
    });
}

static double overlap_eta_states2(const AnalyticState& uA_state,
                                  const AnalyticState& uB_state,
                                  const Eigen::MatrixXd& eta,
                                  double dr, int N_grid)
{
    if (eta.rows() != 2 || eta.cols() != 2) {
        throw std::runtime_error("overlap_eta_states2: eta must be 2x2");
    }
    const Eigen::Matrix2d eta2 = eta.topLeftCorner<2,2>();
    return integrate_uniform_grid(N_grid, dr, [&](int ir) {
        const double r = ir * dr;
        const Eigen::Vector2d uA = uA_state.evaluate2(r);
        const Eigen::Vector2d uB = uB_state.evaluate2(r);
        return uA.dot(eta2 * uB);
    });
}

#if 0
static void write_tdpt_finite_cache_seed(
    const std::string& node_path,
    const std::string& coupling_path,
    const std::string& manifest_path,
    const std::vector<double>& final_m4_E,
    const std::vector<double>& final_m4_wE,
    double E_halo,
    const AnalyticState& halo_state,
    AnalyticSquareWell& solver_m3,
    const Eigen::MatrixXd& eta_m4_m3,
    const tdpt::GaussianRfPulse& pulse,
    double odd_m3_E_min,
    double odd_m3_E_max,
    int odd_m3_N)
{
    if (final_m4_E.size() != final_m4_wE.size()) {
        throw std::runtime_error("write_tdpt_finite_cache_seed: final grid size mismatch");
    }
    if (odd_m3_N < 2 || odd_m3_E_max <= odd_m3_E_min) {
        throw std::runtime_error("write_tdpt_finite_cache_seed: invalid odd M_F=-3 grid");
    }
    if (eta_m4_m3.rows() != 2 || eta_m4_m3.cols() != 2) {
        throw std::runtime_error("write_tdpt_finite_cache_seed: eta_m4_m3 must be 2x2");
    }

    std::ofstream nodes(node_path);
    if (!nodes) {
        throw std::runtime_error("write_tdpt_finite_cache_seed: cannot open " + node_path);
    }
    std::ofstream couplings(coupling_path);
    if (!couplings) {
        throw std::runtime_error("write_tdpt_finite_cache_seed: cannot open " + coupling_path);
    }
    std::ofstream manifest(manifest_path);
    if (!manifest) {
        throw std::runtime_error("write_tdpt_finite_cache_seed: cannot open " + manifest_path);
    }

    nodes << std::scientific << std::setprecision(12);
    couplings << std::scientific << std::setprecision(12);
    manifest << std::scientific << std::setprecision(12);

    nodes << "# TDPT finite spectral node cache seed\n";
    nodes << "# columns: index  M_F  kind  E_Hartree  E_kHz  weight_Hartree  "
             "include_probability  label\n";

    int index = 0;
    const int halo_index = index++;
    nodes << halo_index << "  -4  bound  "
          << E_halo << "  " << E_halo * AU::Hartree_in_kHz
          << "  1.000000000000e+00  0  halo_m4\n";

    const int m4_final_start = index;
    for (int i = 0; i < (int)final_m4_E.size(); ++i) {
        nodes << index++ << "  -4  continuum  "
              << final_m4_E[i] << "  " << final_m4_E[i] * AU::Hartree_in_kHz
              << "  " << final_m4_wE[i] << "  1  m4_final_cont_" << i << "\n";
    }

    const int m3_odd_start = index;
    const double dE_odd = (odd_m3_E_max - odd_m3_E_min) / (odd_m3_N - 1);
    const std::vector<double> odd_wE = uniform_simpson38_weights(odd_m3_N, dE_odd);
    for (int i = 0; i < odd_m3_N; ++i) {
        const double E = odd_m3_E_min + i * dE_odd;
        nodes << index++ << "  -3  continuum  "
              << E << "  " << E * AU::Hartree_in_kHz
              << "  " << odd_wE[i] << "  1  m3_odd_cont_" << i << "\n";
    }

    couplings << "# TDPT finite spectral coupling cache seed\n";
    couplings << "# Only exact square-well halo <-> M_F=-3 bound-continuum "
                 "matrix elements are written here.\n";
    couplings << "# columns: to_index  from_index  eta_to_from  abs_eta  representation_note\n";

    const Eigen::MatrixXd eta_m3_m4 = eta_m4_m3.transpose();
    double max_recip_abs = 0.0;
    double max_recip_rel = 0.0;
    double P_odd_order1_from_cache = 0.0;
    for (int i = 0; i < odd_m3_N; ++i) {
        const double E = odd_m3_E_min + i * dE_odd;
        AnalyticState cont_m3 = solver_m3.build_scattering_state(E, 0);

        const double D_m3_h =
            tdpt::exact_overlap_eta_2ch(cont_m3, halo_state, eta_m3_m4);
        const double D_h_m3 =
            tdpt::exact_overlap_eta_2ch(halo_state, cont_m3, eta_m4_m3);
        const int node_m3 = m3_odd_start + i;

        couplings << node_m3 << "  " << halo_index << "  "
                  << D_m3_h << "  " << std::abs(D_m3_h)
                  << "  exact_square_well_bound_continuum_m3_from_halo\n";
        couplings << halo_index << "  " << node_m3 << "  "
                  << D_h_m3 << "  " << std::abs(D_h_m3)
                  << "  exact_square_well_bound_continuum_halo_from_m3\n";

        const double abs_diff = std::abs(D_m3_h - D_h_m3);
        const double scale = std::max({std::abs(D_m3_h), std::abs(D_h_m3), 1e-300});
        max_recip_abs = std::max(max_recip_abs, abs_diff);
        max_recip_rel = std::max(max_recip_rel, abs_diff / scale);

        const std::complex<double> c1 =
            tdpt::first_order_linear_x(D_m3_h, E, E_halo, pulse);
        P_odd_order1_from_cache += odd_wE[i] * std::norm(c1);
    }

    manifest << "# TDPT finite spectral cache seed manifest\n";
    manifest << "nodes_file " << node_path << "\n";
    manifest << "couplings_file " << coupling_path << "\n";
    manifest << "node_count " << index << "\n";
    manifest << "coupling_count " << 2 * odd_m3_N << "\n";
    manifest << "halo_index " << halo_index << "\n";
    manifest << "m4_final_start " << m4_final_start << "\n";
    manifest << "m4_final_count " << final_m4_E.size() << "\n";
    manifest << "m3_odd_start " << m3_odd_start << "\n";
    manifest << "m3_odd_count " << odd_m3_N << "\n";
    manifest << "m3_odd_E_min_Hartree " << odd_m3_E_min << "\n";
    manifest << "m3_odd_E_max_Hartree " << odd_m3_E_max << "\n";
    manifest << "max_reciprocity_abs_error " << max_recip_abs << "\n";
    manifest << "max_reciprocity_rel_error " << max_recip_rel << "\n";
    manifest << "P_odd_order1_reconstructed_from_cache "
             << P_odd_order1_from_cache << "\n";
    manifest << "status seed_cache_only\n";
    manifest << "note continuum-continuum couplings are not included here; "
                "they require a declared distribution/quadrature or box representation.\n";
}

static void write_m4_m3_finite_window_coupling_cache(
    const std::string& path,
    const std::string& manifest_path,
    const std::vector<double>& final_m4_E,
    double odd_m3_E_min,
    double odd_m3_E_max,
    int odd_m3_N,
    AnalyticSquareWell& solver_m4,
    AnalyticSquareWell& solver_m3,
    const Eigen::MatrixXd& eta_m4_m3,
    double R_window,
    bool full_cache)
{
    if (final_m4_E.empty() || odd_m3_N < 2 || odd_m3_E_max <= odd_m3_E_min) {
        throw std::runtime_error("write_m4_m3_finite_window_coupling_cache: invalid grids");
    }
    if (eta_m4_m3.rows() != 2 || eta_m4_m3.cols() != 2) {
        throw std::runtime_error("write_m4_m3_finite_window_coupling_cache: eta must be 2x2");
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("write_m4_m3_finite_window_coupling_cache: cannot open " + path);
    }
    std::ofstream manifest(manifest_path);
    if (!manifest) {
        throw std::runtime_error("write_m4_m3_finite_window_coupling_cache: cannot open " + manifest_path);
    }

    out << std::scientific << std::setprecision(12);
    manifest << std::scientific << std::setprecision(12);

    const int m4_final_start = 1;
    const int m3_odd_start = 1 + (int)final_m4_E.size();
    const double dE_odd = (odd_m3_E_max - odd_m3_E_min) / (odd_m3_N - 1);
    const int kf_stride = full_cache ? 1 : std::max(1, (int)final_m4_E.size() / 20);
    const int kn_stride = full_cache ? 1 : std::max(1, odd_m3_N / 20);

    out << "# Finite-window M_F=-4 continuum <-> M_F=-3 continuum coupling cache\n";
    out << "# This is a declared finite radial window representation on [0,R_window].\n";
    out << "# It is NOT the infinite-volume continuum-continuum distribution matrix element.\n";
    out << "# columns: to_index  from_index  eta_to_from  abs_eta  representation_note\n";

    std::vector<AnalyticState> m3_states;
    m3_states.reserve(odd_m3_N);
    for (int in = 0; in < odd_m3_N; ++in) {
        const double En = odd_m3_E_min + in * dE_odd;
        m3_states.push_back(solver_m3.build_scattering_state(En, 0));
    }

    const Eigen::MatrixXd eta_m3_m4 = eta_m4_m3.transpose();
    long long pair_count = 0;
    long long coupling_count = 0;
    double max_recip_abs = 0.0;
    double max_recip_rel = 0.0;
    double max_abs_eta = 0.0;

    for (int kf = 0; kf < (int)final_m4_E.size(); ++kf) {
        const bool keep_kf = full_cache
            || kf == 0 || kf == (int)final_m4_E.size() - 1
            || (kf % kf_stride == 0);
        if (!keep_kf) continue;

        const AnalyticState m4_state =
            solver_m4.build_scattering_state(final_m4_E[kf], 0);
        const int node_m4 = m4_final_start + kf;

        for (int in = 0; in < odd_m3_N; ++in) {
            const bool keep_in = full_cache
                || in == 0 || in == odd_m3_N - 1
                || (in % kn_stride == 0);
            if (!keep_in) continue;

            const int node_m3 = m3_odd_start + in;
            const AnalyticState& m3_state = m3_states[in];
            const double D_m4_m3 =
                tdpt::finite_window_overlap_eta_2ch(
                    m4_state, m3_state, eta_m4_m3, R_window);
            const double D_m3_m4 =
                tdpt::finite_window_overlap_eta_2ch(
                    m3_state, m4_state, eta_m3_m4, R_window);

            out << node_m4 << "  " << node_m3 << "  "
                << D_m4_m3 << "  " << std::abs(D_m4_m3)
                << "  finite_radial_window_m4_from_m3_R_" << R_window << "\n";
            out << node_m3 << "  " << node_m4 << "  "
                << D_m3_m4 << "  " << std::abs(D_m3_m4)
                << "  finite_radial_window_m3_from_m4_R_" << R_window << "\n";

            const double abs_diff = std::abs(D_m4_m3 - D_m3_m4);
            const double scale = std::max({std::abs(D_m4_m3), std::abs(D_m3_m4), 1e-300});
            max_recip_abs = std::max(max_recip_abs, abs_diff);
            max_recip_rel = std::max(max_recip_rel, abs_diff / scale);
            max_abs_eta = std::max({max_abs_eta, std::abs(D_m4_m3), std::abs(D_m3_m4)});
            ++pair_count;
            coupling_count += 2;
        }
    }

    manifest << "# Finite-window continuum-continuum coupling cache manifest\n";
    manifest << "couplings_file " << path << "\n";
    manifest << "full_cache " << (full_cache ? 1 : 0) << "\n";
    manifest << "R_window_a0 " << R_window << "\n";
    manifest << "m4_final_count_total " << final_m4_E.size() << "\n";
    manifest << "m3_odd_count_total " << odd_m3_N << "\n";
    manifest << "kf_stride " << kf_stride << "\n";
    manifest << "kn_stride " << kn_stride << "\n";
    manifest << "pair_count_written " << pair_count << "\n";
    manifest << "coupling_count_written " << coupling_count << "\n";
    manifest << "max_abs_eta " << max_abs_eta << "\n";
    manifest << "max_reciprocity_abs_error " << max_recip_abs << "\n";
    manifest << "max_reciprocity_rel_error " << max_recip_rel << "\n";
    manifest << "status finite_window_regularized_not_infinite_volume_exact\n";
}

static void write_m4_m3_window_convergence_probe(
    const std::string& path,
    const std::vector<double>& final_m4_E,
    double odd_m3_E_min,
    double odd_m3_E_max,
    int odd_m3_N,
    AnalyticSquareWell& solver_m4,
    AnalyticSquareWell& solver_m3,
    const Eigen::MatrixXd& eta_m4_m3,
    const std::vector<double>& windows)
{
    if (final_m4_E.empty() || odd_m3_N < 2 || windows.empty()) {
        throw std::runtime_error("write_m4_m3_window_convergence_probe: invalid input");
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("write_m4_m3_window_convergence_probe: cannot open " + path);
    }
    out << std::scientific << std::setprecision(12);
    out << "# Finite-window continuum-continuum convergence probe\n";
    out << "# Values are sparse diagnostics for selected M_F=-4 final and M_F=-3 odd-continuum indices.\n";
    out << "# Stable physics should not be inferred unless these values converge or a box/distribution prescription is supplied.\n";
    out << "# columns: kf_index  kn_index  E_f_kHz  E_n_kHz  ";
    for (double R : windows) out << "D_R_" << R << "  ";
    out << "rel_spread\n";

    const double dE_odd = (odd_m3_E_max - odd_m3_E_min) / (odd_m3_N - 1);
    const std::vector<int> kf_indices{
        0,
        (int)final_m4_E.size() / 4,
        (int)final_m4_E.size() / 2,
        3 * (int)final_m4_E.size() / 4,
        (int)final_m4_E.size() - 1
    };
    const std::vector<int> kn_indices{
        0,
        odd_m3_N / 4,
        odd_m3_N / 2,
        3 * odd_m3_N / 4,
        odd_m3_N - 1
    };

    for (int kf : kf_indices) {
        const AnalyticState m4_state =
            solver_m4.build_scattering_state(final_m4_E[kf], 0);
        for (int kn : kn_indices) {
            const double En = odd_m3_E_min + kn * dE_odd;
            const AnalyticState m3_state =
                solver_m3.build_scattering_state(En, 0);

            std::vector<double> vals;
            vals.reserve(windows.size());
            double min_abs = std::numeric_limits<double>::infinity();
            double max_abs = 0.0;
            for (double R : windows) {
                const double D =
                    tdpt::finite_window_overlap_eta_2ch(
                        m4_state, m3_state, eta_m4_m3, R);
                vals.push_back(D);
                min_abs = std::min(min_abs, std::abs(D));
                max_abs = std::max(max_abs, std::abs(D));
            }
            const double rel_spread =
                (max_abs - min_abs) / std::max(max_abs, 1e-300);

            out << kf << "  " << kn << "  "
                << final_m4_E[kf] * AU::Hartree_in_kHz << "  "
                << En * AU::Hartree_in_kHz << "  ";
            for (double v : vals) out << v << "  ";
            out << rel_spread << "\n";
        }
    }
}

#endif

static double dawson_real(double x)
{
#ifdef ZEPE_HAVE_GSL_DAWSON
    return gsl_sf_dawson(x);
#else
    // Numerical Recipes Dawson approximation. The production cluster build
    // uses GSL when available; this fallback keeps local source checks working.
    const double ax = std::abs(x);
    if (ax < 0.2) {
        const double xx = x * x;
        return x * (1.0 - (2.0/3.0) * xx
                  * (1.0 - 0.4 * xx * (1.0 - (2.0/7.0) * xx)));
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

static std::complex<double> faddeeva_real(double z)
{
    const double real_part = (z * z > 745.0) ? 0.0 : std::exp(-z * z);
    const double imag_part = 2.0 / std::sqrt(M_PI) * dawson_real(z);
    return {real_part, imag_part};
}

// PDF Eq. (8), atomic units with hbar = 1.
static std::complex<double> second_order_J(double Ef, double En, double Eh,
                                           double omega, int sigma1, int sigma2,
                                           double tau0)
{
    const double Omega1 = En - Eh + sigma1 * omega;
    const double Omega2 = Ef - En + sigma2 * omega;
    const double sum = Omega1 + Omega2;
    const double z = (Omega2 - Omega1) * tau0 / (2.0 * std::sqrt(2.0));
    const double envelope = std::exp(-0.125 * sum * sum * tau0 * tau0);
    return 0.5 * M_PI * tau0 * tau0 * envelope * faddeeva_real(z);
}

template <class F>
static std::complex<double> integrate_uniform_grid_complex(int N_grid, double dx, F&& f)
{
    const int intervals = N_grid - 1;
    if (N_grid < 2) return {0.0, 0.0};
    if (intervals == 1) {
        return 0.5 * dx * (f(0) + f(1));
    }

    if (intervals % 2 == 0) {
        std::complex<double> s = f(0) + f(N_grid - 1);
        for (int i = 1; i < N_grid - 1; ++i) {
            s += (i % 2 == 1 ? 4.0 : 2.0) * f(i);
        }
        return s * dx / 3.0;
    }

    if (intervals == 3) {
        return 3.0 * dx / 8.0
             * (f(0) + 3.0 * f(1) + 3.0 * f(2) + f(3));
    }

    const int last13 = N_grid - 4;
    std::complex<double> s = f(0) + f(last13);
    for (int i = 1; i < last13; ++i) {
        s += (i % 2 == 1 ? 4.0 : 2.0) * f(i);
    }
    const std::complex<double> total13 = s * dx / 3.0;
    const std::complex<double> total38 = 3.0 * dx / 8.0
        * (f(N_grid - 4) + 3.0 * f(N_grid - 3)
         + 3.0 * f(N_grid - 2) + f(N_grid - 1));
    return total13 + total38;
}

static std::complex<double> two_photon_m3_pdf(
    const AnalyticState& final_state,
    const AnalyticState& halo_state,
    AnalyticSquareWell& solver_m3,
    const Eigen::MatrixXd& eta_m4_m3,
    double Ef, double Eh, double omega, double tau0,
    int sigma1, int sigma2,
    int N_En, double En_halfwidth, double overlap_dr, int overlap_N)
{
    if (sigma1 + sigma2 != 0) {
        throw std::runtime_error("two_photon_m3_pdf: expected a ZEPE +- pathway");
    }

    // From PDF Eq. (8):
    // z = [Ef + Eh - 2 En + (sigma2 - sigma1) omega] tau0/(2 sqrt(2)).
    // The Faddeeva kernel is centred at z=0.
    const double En_center = 0.5 * (Ef + Eh + (sigma2 - sigma1) * omega);
    double En_lo = En_center - En_halfwidth;
    double En_hi = En_center + En_halfwidth;
    const double thr0 = solver_m3.thresholds()(0);
    if (En_lo <= thr0) {
        En_lo = std::nextafter(thr0, std::numeric_limits<double>::infinity());
    }
    if (En_hi <= En_lo) {
        throw std::runtime_error("two_photon_m3_pdf: intermediate energy window is closed");
    }
    const double dEn = (En_hi - En_lo) / (N_En - 1);

    const Eigen::MatrixXd eta_m3_m4 = eta_m4_m3.transpose();
    const std::complex<double> integral =
        integrate_uniform_grid_complex(N_En, dEn, [&](int iEn) {
            const double En = En_lo + iEn * dEn;
            AnalyticState n_state = solver_m3.build_scattering_state(En, 0);
            const double d_fn = overlap_eta_states2(final_state, n_state,
                                                    eta_m4_m3, overlap_dr, overlap_N);
            const double d_nh = overlap_eta_states2(n_state, halo_state,
                                                    eta_m3_m4, overlap_dr, overlap_N);
            return d_fn * d_nh
                 * second_order_J(Ef, En, Eh, omega,
                                  sigma1, sigma2, tau0);
        });

    return -0.25 * integral;
}

static std::complex<double> absorb_emit_m3_pdf(
    const AnalyticState& final_state,
    const AnalyticState& halo_state,
    AnalyticSquareWell& solver_m3,
    const Eigen::MatrixXd& eta_m4_m3,
    double Ef, double Eh, double omega, double tau0,
    int N_En, double En_halfwidth, double overlap_dr, int overlap_N)
{
    // PDF convention: absorption first is sigma1=-1; emission second is sigma2=+1.
    return two_photon_m3_pdf(final_state, halo_state, solver_m3, eta_m4_m3,
                             Ef, Eh, omega, tau0, -1, +1,
                             N_En, En_halfwidth, overlap_dr, overlap_N);
}

static void write_abs_emit_m3_window_probe(
    const std::string& path,
    const std::vector<double>& Ef_grid,
    const AnalyticState& halo_state,
    AnalyticSquareWell& solver_m4,
    AnalyticSquareWell& solver_m3,
    const Eigen::MatrixXd& eta_m4_m3,
    double Eh,
    double omega,
    double tau0,
    int N_En,
    double En_halfwidth,
    double overlap_dr,
    const std::vector<double>& windows)
{
    if (Ef_grid.empty() || windows.empty()) {
        throw std::runtime_error("write_abs_emit_m3_window_probe: empty input");
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("write_abs_emit_m3_window_probe: cannot open " + path);
    }
    out << std::scientific << std::setprecision(12);
    out << "# Absorb-emission M_F=-3 energy-integrated window probe\n";
    out << "# This probes the full second-order continuum integral, not pointwise continuum-continuum overlaps.\n";
    out << "# If this converges with R, the finite-window representation may be usable as a distribution action.\n";
    out << "# columns: kf_index  E_f_kHz  ";
    for (double R : windows) {
        out << "Re_c_R_" << R << "  Im_c_R_" << R << "  Abs_c_R_" << R << "  ";
    }
    out << "rel_abs_spread  rel_complex_spread_vs_last\n";

    const std::vector<int> kf_indices{
        0,
        (int)Ef_grid.size() / 4,
        (int)Ef_grid.size() / 2,
        3 * (int)Ef_grid.size() / 4,
        (int)Ef_grid.size() - 1
    };

    for (int kf : kf_indices) {
        const double Ef = Ef_grid[kf];
        const AnalyticState final_state =
            solver_m4.build_scattering_state(Ef, 0);
        std::vector<std::complex<double>> vals;
        vals.reserve(windows.size());

        double min_abs = std::numeric_limits<double>::infinity();
        double max_abs = 0.0;
        for (double R : windows) {
            const int overlap_N = (int)std::llround(R / overlap_dr) + 1;
            const std::complex<double> c =
                absorb_emit_m3_pdf(final_state, halo_state, solver_m3,
                                   eta_m4_m3, Ef, Eh, omega, tau0,
                                   N_En, En_halfwidth,
                                   overlap_dr, overlap_N);
            vals.push_back(c);
            min_abs = std::min(min_abs, std::abs(c));
            max_abs = std::max(max_abs, std::abs(c));
        }

        const std::complex<double> ref = vals.back();
        double max_complex_diff = 0.0;
        for (const auto& c : vals) {
            max_complex_diff = std::max(max_complex_diff, std::abs(c - ref));
        }
        const double rel_abs_spread =
            (max_abs - min_abs) / std::max(max_abs, 1e-300);
        const double rel_complex_spread =
            max_complex_diff / std::max(std::abs(ref), 1e-300);

        out << kf << "  " << Ef * AU::Hartree_in_kHz << "  ";
        for (const auto& c : vals) {
            out << c.real() << "  " << c.imag() << "  " << std::abs(c) << "  ";
        }
        out << rel_abs_spread << "  " << rel_complex_spread << "\n";
    }
}

static std::complex<double> emit_absorb_m5_green_pulse_factor(
    double Ef, double Eh, double tau0)
{
    // For the closed M_F=-5 block the intermediate detuning is GHz while the
    // ZEPE bandwidth is kHz. Using the large-|z| limit of PDF Eq. (8),
    //
    //   J^(+,-) ~= -i sqrt(pi/2) tau0 exp[-(Ef-Eh)^2 tau0^2/8]
    //              /(E_n - E_h + omega).
    //
    // Since X solves G(E_h-omega)S = (E_h-omega-H)^(-1)S, the spectral
    // denominator is already included in M_emit_abs_m5_green. The factor
    // below converts that Green-function matrix element into the finite-pulse
    // second-order amplitude, up to the separately applied (mu_B B_RF)^2.
    const double delta = Ef - Eh;
    const double envelope = std::exp(-0.125 * delta * delta * tau0 * tau0);
    return {0.0, -0.25 * std::sqrt(0.5 * M_PI) * tau0 * envelope};
}

int main(int argc, char** argv)
{
    using namespace AU;
    cout << std::fixed << std::setprecision(8);

    // ------------------------------------------------------------
    // Optional command-line arguments: omega (kHz), tau_0 (us), and B_RF (mG).
    // Usage:
    //   ./zepe_mc                                  # defaults: omega=E_b/2, tau_0=30 us, B_RF=1 mG
    //   ./zepe_mc <omega_kHz>                      # set omega
    //   ./zepe_mc <omega_kHz> <tau0_us>            # set omega and tau_0
    //   ./zepe_mc <omega_kHz> <tau0_us> <B_RF_mG>  # set all three
    // ------------------------------------------------------------
    double omega_override_kHz = -1.0;  // -1 => use default E_b/2
    double tau0_override_us   = -1.0;  // -1 => use default 30 us
    double B_rf_mG            = 1.0;   // reference RF amplitude
    if (argc >= 2) {
        omega_override_kHz = std::atof(argv[1]);
        cout << "CMDLINE omega = " << omega_override_kHz << " kHz\n";
    }
    if (argc >= 3) {
        tau0_override_us = std::atof(argv[2]);
        cout << "CMDLINE tau_0 = " << tau0_override_us << " us\n";
    }
    if (argc >= 4) {
        B_rf_mG = std::atof(argv[3]);
        cout << "CMDLINE B_RF = " << B_rf_mG << " mG\n";
    }
    cout << "\n";


    Parameters p;
    p.N_grid       = 100000000;        // tail BC point = N_grid * dr
    p.dr           = 0.01;
    p.mu           = mu_Rb85;
    p.r0           = 82.1;
    p.smooth_width = 0.0;
    p.B_gauss      = 155.04;
    p.MF_target    = -4;
    p.N_ch_keep    = 2;
    p.store_potential_grid = false;
    p.Nroots = 100; p.NTHREADS = 1; p.divide = 1; p.p = 3;
    p.Emin = 0; p.Emax = 0; p.external_parameter = 0;
    p.N_grid = env_int_or_default("ZEPE_N_GRID", p.N_grid);
    p.dr = env_double_or_default("ZEPE_DR", p.dr);
    // V_T sets the square-well depth. The actual halo binding energy is found
    // below and printed; retune this number if a different E_b is required
    // after changing the channel basis or singlet/triplet convention.
    p.V_T = GHz_to_au(9.6930959056);
    p.V_S = 1.02 * p.V_T;

    const double r_grid_max = (p.N_grid - 1) * p.dr;
    const double r_tail_point = p.N_grid * p.dr;
    cout << "R_grid_max = " << r_grid_max << " a_0\n";
    cout << "Tail BC point = " << r_tail_point << " a_0\n";
    cout << "Grid dr = " << p.dr << " a_0\n";
    cout << "N_grid  = " << p.N_grid << "\n";
    cout << "Store V(r) grid = " << (p.store_potential_grid ? "yes" : "no") << "\n";
    cout << "Well V_T = " << p.V_T * Hartree_in_GHz << " GHz\n";
    cout << "Well r_0 = " << p.r0 << " a_0\n\n";


    SpinAlgebra spin(p.B_gauss);
    auto chans_m4 = spin.channels(-4);
    double E_global_ref_MHz = chans_m4[0].E_th_MHz;
    cout << "Global energy reference: " << E_global_ref_MHz << " MHz (M_F=-4 open)\n\n";

    // ------------------------------------------------------------
    // Analytic halo (same as Method A)
    // ------------------------------------------------------------
    Parameters p_halo = p; p_halo.MF_target = -4; p_halo.N_ch_keep = 2;
    Potentials pot_m4(&p_halo, &spin, E_global_ref_MHz);
    AnalyticSquareWell solver_m4(&pot_m4, &p_halo);

    double E_halo;
    try {
        E_halo = solver_m4.find_highest_bound_state(-MHz_to_au(1.0),
                                                     -kHz_to_au(0.001), 20000);
    } catch (std::exception& e) {
        cout << "ERROR finding halo: " << e.what() << "\n"; return 1;
    }
    cout << "---- Analytic halo ----\n";
    cout << "  E_halo = " << std::scientific << std::setprecision(8) << E_halo * Hartree_in_kHz
         << " kHz   (E_b = " << -E_halo * Hartree_in_kHz << " kHz)\n" << std::fixed;



    AnalyticState halo_state = solver_m4.build_bound_state(E_halo);

    double P0 = integrate_uniform_grid(p.N_grid, p.dr, [&](int ir) {
        const Eigen::Vector2d u = halo_state.evaluate2(ir * p.dr);
        return u(0) * u(0);
    });
    double P1 = integrate_uniform_grid(p.N_grid, p.dr, [&](int ir) {
        const Eigen::Vector2d u = halo_state.evaluate2(ir * p.dr);
        return u(1) * u(1);
    });
    cout << "  P_open (grid-normalized) = " << P0 << ", P_closed = " << P1 << "\n";
    cout << "  halo total after grid normalization = " << P0 + P1 << "\n\n";

    // ------------------------------------------------------------
    // Potentials for M_F=-3 and M_F=-5 blocks (for intermediates)
    // ------------------------------------------------------------
    Parameters p_m3 = p; p_m3.MF_target = -3; p_m3.N_ch_keep = 2;
    Parameters p_m5 = p; p_m5.MF_target = -5; p_m5.N_ch_keep = 2;
    Potentials pot_m3(&p_m3, &spin, E_global_ref_MHz);
    Potentials pot_m5(&p_m5, &spin, E_global_ref_MHz);
    AnalyticSquareWell solver_m3_main(&pot_m3, &p_m3);
    AnalyticSquareWell solver_m5(&pot_m5, &p_m5);
    if (pot_m4.num_channels() != 2 || pot_m5.num_channels() != 2) {
        throw std::runtime_error("This memory-compact path currently requires two kept channels");
    }
    // M_F=-5 has no open final channel in the sub-MHz window, but it still
    // contributes as a virtual closed intermediate through its resolvent.

    cout << "-- Block thresholds (relative to global M_F=-4 open) --\n";
    cout << "  M_F=-4: " << (pot_m4.thresholds() * Hartree_in_MHz).transpose() << " MHz\n";
    cout << "  M_F=-3: " << (pot_m3.thresholds() * Hartree_in_MHz).transpose() << " MHz\n";
    cout << "  M_F=-5: " << (pot_m5.thresholds() * Hartree_in_MHz).transpose() << " MHz\n\n";

    // ------------------------------------------------------------
    // RF coupling eta
    // ------------------------------------------------------------
    Eigen::MatrixXd rf_m3_m4 = spin.rf_matrix(pot_m4.channels(), pot_m3.channels());
    cout << "eta_x(M_F=-4 <- M_F=-3), linear x polarization:\n"
         << rf_m3_m4 << "\n\n";

    // Linear-x coupling to the closed M_F=-5 block:
    //   eta_m4_m5(alpha, beta) = <m4,alpha | (mu_x1 + mu_x2)/mu_B | m5,beta>
    // Pure circular polarization requires mu_+ or mu_- operators, not mu_x.
    Eigen::MatrixXd eta_m4_m5 = spin.rf_matrix(pot_m4.channels(), pot_m5.channels());
    cout << "eta_x(M_F=-4 <-> M_F=-5), linear x polarization:\n"
         << eta_m4_m5 << "\n\n";


    // ------------------------------------------------------------
    // Test T0: check scattering state at a test energy
    //   - does it asymptote to sqrt(2mu/(pi k)) sin(kr+delta)?
    //   - is the closed channel exponentially decaying?
    // ------------------------------------------------------------
    double E_test = kHz_to_au(1.0);
    AnalyticState test_state = solver_m4.build_scattering_state(E_test, 0);
    double k_test = std::sqrt(2.0 * p.mu * E_test);
    double expected_amp = std::sqrt(2.0 * p.mu / (M_PI * k_test));
    cout << "[T0] Scattering state at E = 1 kHz:\n";
    cout << "  k = " << std::scientific << k_test << " a.u.\n";
    cout << "  expected open-channel asymptotic amp = sqrt(2mu/pi k) = " << expected_amp << "\n";
    double peak = 0;
    int ip = 0;
    for (int ir = p.N_grid/2; ir < p.N_grid; ++ir) {
        const Eigen::Vector2d u = test_state.evaluate2(ir * p.dr);
        if (std::abs(u(0)) > peak) {
            peak = std::abs(u(0));
            ip = ir;
        }
    }
    cout << "  numerical peak in second half of grid = " << peak
         << " at r = " << ip * p.dr << " a_0\n";
    cout << "  ratio (numerical/expected) = " << peak / expected_amp << "\n";
    cout << "  closed-channel max |u_closed| just outside/after r0 = ";
    double cc_max = 0;
    for (int ir = (int)(p.r0 / p.dr) + 10; ir < p.N_grid; ++ir) {
        const Eigen::Vector2d u = test_state.evaluate2(ir * p.dr);
        if (std::abs(u(1)) > cc_max) cc_max = std::abs(u(1));
    }
    cout << cc_max << " (finite near r0; should decay exponentially)\n" << std::fixed;
    cout << "\n";

   // ------------------------------------------------------------
    // Energy grids
    // ------------------------------------------------------------
    // Final states: log-spaced from 1 Hz to 100 kHz.
    int N_Ef = 4000;
    double Ef_lo_kHz = 1e-3;   // 1 Hz
    double Ef_hi_kHz = 200.0;  // 100 kHz
    std::vector<double> Ef_grid(N_Ef);
    for (int i = 0; i < N_Ef; ++i) {
        double t = (double)i / (N_Ef - 1);
        Ef_grid[i] = kHz_to_au(Ef_lo_kHz * std::pow(Ef_hi_kHz / Ef_lo_kHz, t));
    }
    const std::vector<double> Ef_weight_Hartree =
        log_energy_quadrature_weights(Ef_grid);
    cout << "Final-state grid: " << N_Ef << " points, log-spaced ["
         << Ef_lo_kHz << ", " << Ef_hi_kHz << "] kHz\n";



    // ------------------------------------------------------------
    // Pulse parameters
    // ------------------------------------------------------------
    double E_b = -E_halo;
    double omega;
    double tau0_us;
    if (omega_override_kHz > 0) omega = kHz_to_au(omega_override_kHz);
    else                        omega = 0.5 * E_b;
    if (tau0_override_us > 0)   tau0_us = tau0_override_us;
    else                        tau0_us = 30.0;
    double tau0 = tau0_us * 1e-6 / 2.4188843e-17;
    double beta = E_b * tau0 / 2.0;
    const double B_rf_gauss = B_rf_mG * 1e-3;
    const double rf_energy = MHz_to_au(mu_B_MHz_per_G * B_rf_gauss);
    const double rf_energy2 = rf_energy * rf_energy;
    cout << "omega = " << omega * Hartree_in_kHz << " kHz, tau0 = " << tau0_us << " us, beta = " << beta << "\n";
    cout << "B_RF = " << B_rf_mG << " mG, mu_B B_RF = "
         << rf_energy * Hartree_in_kHz << " kHz = "
         << rf_energy << " Hartree\n";
    double x_pk_an = 0.5 * (std::sqrt(1.0 + 1.0/(beta*beta)) - 1.0);
    cout << "Analytic peak: x = " << x_pk_an << ", E_f = " << x_pk_an * E_b * Hartree_in_kHz << " kHz\n\n";
#if 0
    try {
        tdpt::write_continuum_tdpt_equations(
            "tdpt_higher_order_formalism.txt", p.MF_target, p.MF_target, 6);
        cout << "Wrote exact higher-order TDPT formalism to "
                "tdpt_higher_order_formalism.txt\n\n";
    } catch (const std::exception& e) {
        cout << "WARNING: could not write higher-order TDPT formalism: "
             << e.what() << "\n\n";
    }
    try {
        tdpt::write_finite_tdpt_cache_contract(
            "tdpt_finite_spectrum_contract.txt", 6);
        cout << "Wrote finite-spectrum TDPT cache contract to "
                "tdpt_finite_spectrum_contract.txt\n\n";
    } catch (const std::exception& e) {
        cout << "WARNING: could not write finite-spectrum TDPT cache contract: "
             << e.what() << "\n\n";
    }
    try {
        const double tdpt_selftest_rel =
            tdpt::write_finite_tdpt_first_order_selftest(
                "tdpt_finite_tdpt_selftest.txt");
        cout << "Wrote finite-spectrum TDPT first-order self-test to "
                "tdpt_finite_tdpt_selftest.txt\n";
        cout << "  self-test relative error = " << std::scientific
             << tdpt_selftest_rel << std::fixed << "\n\n";
    } catch (const std::exception& e) {
        cout << "WARNING: finite-spectrum TDPT self-test failed: "
             << e.what() << "\n\n";
    }
    const tdpt::GaussianRfPulse pulse{omega, tau0, rf_energy};
    const int N_odd_m3 = 401;
    const double odd_center_emit = E_halo - omega;
    const double odd_center_abs = E_halo + omega;
    const double odd_pad = 8.0 / tau0;
    double odd_E_min = std::min(odd_center_emit, odd_center_abs) - odd_pad;
    double odd_E_max = std::max(odd_center_emit, odd_center_abs) + odd_pad;
    const double thr_m3 = pot_m3.thresholds()(0);
    if (odd_E_min <= thr_m3) {
        odd_E_min = std::nextafter(thr_m3, std::numeric_limits<double>::infinity());
    }

    double P_first_order_odd_m3 = std::numeric_limits<double>::quiet_NaN();
    try {
        P_first_order_odd_m3 = tdpt::write_first_order_continuum_spectrum_2ch(
            "tdpt_first_order_odd_m3.txt",
            solver_m3_main,
            halo_state,
            rf_m3_m4.transpose(),
            pulse,
            odd_E_min,
            odd_E_max,
            N_odd_m3,
            "First-order odd-sector spectrum: M_F=-4 halo -> M_F=-3 continuum");
        cout << "Wrote first-order odd-sector M_F=-3 continuum spectrum to "
                "tdpt_first_order_odd_m3.txt\n";
        cout << "  E range = [" << odd_E_min * Hartree_in_kHz << ", "
             << odd_E_max * Hartree_in_kHz << "] kHz relative to global M_F=-4 open\n\n";
        cout << "  Integrated P_odd_order1_m3 = "
             << std::scientific << P_first_order_odd_m3
             << std::fixed << "\n\n";
    } catch (const std::exception& e) {
        cout << "WARNING: could not write first-order odd-sector M_F=-3 spectrum: "
             << e.what() << "\n\n";
    }
    try {
        write_tdpt_finite_cache_seed(
            "tdpt_cache_nodes.txt",
            "tdpt_cache_couplings_halo_m3.txt",
            "tdpt_cache_manifest.txt",
            Ef_grid,
            Ef_weight_Hartree,
            E_halo,
            halo_state,
            solver_m3_main,
            rf_m3_m4,
            pulse,
            odd_E_min,
            odd_E_max,
            N_odd_m3);
        cout << "Wrote finite TDPT cache seed files:\n";
        cout << "  tdpt_cache_nodes.txt\n";
        cout << "  tdpt_cache_couplings_halo_m3.txt\n";
        cout << "  tdpt_cache_manifest.txt\n\n";
        const tdpt::FiniteTdptSystem cache_system =
            tdpt::read_finite_tdpt_cache(
                "tdpt_cache_nodes.txt",
                "tdpt_cache_couplings_halo_m3.txt",
                0);
        const tdpt::FiniteTdptOptions cache_check_options{
            1,
            -8.0 * tau0,
            +8.0 * tau0,
            tau0 / 500.0
        };
        tdpt::write_finite_tdpt_order_report(
            "tdpt_cache_order1_report.txt",
            cache_system,
            pulse,
            cache_check_options,
            {-4, -3});
        cout << "Wrote finite TDPT cache order-1 report to "
                "tdpt_cache_order1_report.txt\n\n";

        const bool full_cc_cache =
            env_flag_enabled("ZEPE_WRITE_FULL_CC_CACHE");
        const double cc_window_R =
            env_double_or_default(
                "ZEPE_CC_WINDOW_R",
                env_double_or_default("ZEPE_OVERLAP_R", 5000.0));
        const std::string cc_path = full_cc_cache
            ? "tdpt_cache_couplings_m4_m3_finite_window.txt"
            : "tdpt_cache_couplings_m4_m3_finite_window_sample.txt";
        const std::string cc_manifest = full_cc_cache
            ? "tdpt_cache_couplings_m4_m3_finite_window_manifest.txt"
            : "tdpt_cache_couplings_m4_m3_finite_window_sample_manifest.txt";
        write_m4_m3_finite_window_coupling_cache(
            cc_path,
            cc_manifest,
            Ef_grid,
            odd_E_min,
            odd_E_max,
            N_odd_m3,
            solver_m4,
            solver_m3_main,
            rf_m3_m4,
            cc_window_R,
            full_cc_cache);
        cout << "Wrote finite-window M_F=-4/M_F=-3 continuum-continuum "
                "coupling "
             << (full_cc_cache ? "cache" : "sample") << " to "
             << cc_path << "\n";
        cout << "  This file is a finite-window regularization, not an "
                "infinite-volume exact continuum matrix element.\n\n";
        write_m4_m3_window_convergence_probe(
            "tdpt_cache_couplings_m4_m3_window_probe.txt",
            Ef_grid,
            odd_E_min,
            odd_E_max,
            N_odd_m3,
            solver_m4,
            solver_m3_main,
            rf_m3_m4,
            {500.0, 1000.0, 2500.0, cc_window_R});
        cout << "Wrote finite-window M_F=-4/M_F=-3 convergence probe to "
                "tdpt_cache_couplings_m4_m3_window_probe.txt\n\n";
    } catch (const std::exception& e) {
        cout << "WARNING: could not write finite TDPT cache seed: "
             << e.what() << "\n\n";
    }
#endif
    cout << "Note: M_emit_abs_m5_green is the closed-resolvent matrix element. "
            "The code also prints c_emit_abs_m5_green, obtained by applying "
            "the large-detuning finite-pulse factor to that Green object.\n";

    // ------------------------------------------------------------
    // ZEPE pathways kept distinct:
    //   absorb -> emit:  M_F=-4 -> M_F=-3 -> M_F=-4
    //   emit   -> absorb: M_F=-4 -> M_F=-5 -> M_F=-4
    //
    // The M_F=-3 continuum pathway below uses the finite-pulse kernel from
    // the PDF formula. The M_F=-5 amplitude uses the closed-block Green
    // function with the large-detuning finite-pulse factor. The coherent
    // sum printed below is therefore:
    //
    //   c_sum = c_ae + c_ea
    //
    // with c_ae = absorb-emit through M_F=-3 and c_ea = emit-absorb
    // through M_F=-5 in the current closed-Green approximation.
    // ------------------------------------------------------------

    std::vector<double> Ef_kHz(N_Ef), M_emit_abs_m5_green_kernel(N_Ef, 0.0);
    std::vector<double> abs_M_emit_abs_m5_green_kernel(N_Ef, 0.0);
    std::vector<double> M_emit_abs_m5_green_scaled(N_Ef, 0.0);
    std::vector<double> abs_M_emit_abs_m5_green_scaled(N_Ef, 0.0);
    std::vector<std::complex<double>> c_emit_abs_m5_green_scaled(N_Ef, {0.0, 0.0});
    std::vector<double> abs_c_emit_abs_m5_green_scaled(N_Ef, 0.0);
    std::vector<double> dP_dE_Hartree_emit_abs_m5_green(N_Ef, 0.0);
    std::vector<double> dP_dE_kHz_emit_abs_m5_green(N_Ef, 0.0);
    std::vector<std::complex<double>> c_abs_emit_m3_kernel(N_Ef, {0.0, 0.0});
    std::vector<std::complex<double>> c_abs_emit_m3_scaled(N_Ef, {0.0, 0.0});
    std::vector<double> abs_c_abs_emit_m3_kernel(N_Ef, 0.0);
    std::vector<double> abs_c_abs_emit_m3_scaled(N_Ef, 0.0);
    std::vector<double> dP_dE_Hartree_abs_emit_m3(N_Ef, 0.0);
    std::vector<double> dP_dE_kHz_abs_emit_m3(N_Ef, 0.0);
    std::vector<std::complex<double>> c_sum_ae_ea_scaled(N_Ef, {0.0, 0.0});
    std::vector<double> abs_c_sum_ae_ea_scaled(N_Ef, 0.0);
    std::vector<double> dP_dE_Hartree_sum_ae_ea(N_Ef, 0.0);
    std::vector<double> dP_dE_kHz_sum_ae_ea(N_Ef, 0.0);

    // ---- M_F=-5 emit-absorb diagnostic / "best" reconstructions ----
    //
    // _exact_bound : -1/4 (mu_B B_RF)^2 (pi tau0^2/2) exp[-(E_f-E_h)^2 tau0^2/8]
    //                  * sum_n d_{f n} d_{n h} w(z_n),  bound states only.
    //                z_n = (E_f + E_h - 2 E_n - 2 omega) tau0/(2 sqrt(2)).
    //                NOTE: this is incomplete because M_F=-5 has a continuum
    //                whose contribution is NOT bound (and turns out to dominate
    //                for an extended halo source). Kept as a diagnostic column.
    //
    // _best        : c_ea_green + (c_ea_exact_bound - c_ea_LO_bound).
    //                Combines exact Faddeeva on bound states with the leading-
    //                order closed-Green treatment of the entire spectrum.
    //                This is the recommended "exact-as-far-as-we-can-go" output
    //                when |z| >> 1 across the M_F=-5 continuum (true for the
    //                B=155 G calibration; check the kf=0 diagnostic).
    std::vector<std::complex<double>> c_emit_abs_m5_exact_kernel(N_Ef, {0.0, 0.0});
    std::vector<std::complex<double>> c_emit_abs_m5_exact_scaled(N_Ef, {0.0, 0.0});
    std::vector<double> abs_c_emit_abs_m5_exact_scaled(N_Ef, 0.0);
    std::vector<double> dP_dE_Hartree_emit_abs_m5_exact(N_Ef, 0.0);
    std::vector<double> dP_dE_kHz_emit_abs_m5_exact(N_Ef, 0.0);
    std::vector<std::complex<double>> c_sum_ae_ea_exact_scaled(N_Ef, {0.0, 0.0});
    std::vector<double> abs_c_sum_ae_ea_exact_scaled(N_Ef, 0.0);
    std::vector<double> dP_dE_Hartree_sum_ae_ea_exact(N_Ef, 0.0);
    std::vector<double> dP_dE_kHz_sum_ae_ea_exact(N_Ef, 0.0);

    std::vector<std::complex<double>> c_emit_abs_m5_LO_bound_scaled(N_Ef, {0.0, 0.0});
    std::vector<std::complex<double>> c_emit_abs_m5_best_scaled(N_Ef, {0.0, 0.0});
    std::vector<double> abs_c_emit_abs_m5_best_scaled(N_Ef, 0.0);
    std::vector<double> dP_dE_Hartree_emit_abs_m5_best(N_Ef, 0.0);
    std::vector<double> dP_dE_kHz_emit_abs_m5_best(N_Ef, 0.0);
    std::vector<std::complex<double>> c_sum_ae_ea_best_scaled(N_Ef, {0.0, 0.0});
    std::vector<double> abs_c_sum_ae_ea_best_scaled(N_Ef, 0.0);
    std::vector<double> dP_dE_Hartree_sum_ae_ea_best(N_Ef, 0.0);
    std::vector<double> dP_dE_kHz_sum_ae_ea_best(N_Ef, 0.0);

    // ---- Complex-resolvent continuum diagnostic for M_F=-5 ea ----
    // Implemented in ComplexResolventEa.cpp. It keeps the bound exact result
    // from the spectral sum below and replaces the finite-window continuum-
    // continuum object by the off-real-axis spectral density
    //   -Im <u_f|eta (E+i eta-H_-5)^(-1) eta|u_h> / pi.
    std::vector<std::complex<double>> c_emit_abs_m5_cr_cont_scaled(N_Ef, {0.0, 0.0});
    std::vector<std::complex<double>> c_emit_abs_m5_cr_LO_cont_scaled(N_Ef, {0.0, 0.0});
    std::vector<std::complex<double>> c_emit_abs_m5_cr_scaled(N_Ef, {0.0, 0.0});
    std::vector<double> abs_c_emit_abs_m5_cr_scaled(N_Ef, 0.0);
    std::vector<double> dP_dE_Hartree_emit_abs_m5_cr(N_Ef, 0.0);
    std::vector<double> dP_dE_kHz_emit_abs_m5_cr(N_Ef, 0.0);
    std::vector<std::complex<double>> c_sum_ae_ea_cr_scaled(N_Ef, {0.0, 0.0});
    std::vector<double> abs_c_sum_ae_ea_cr_scaled(N_Ef, 0.0);
    std::vector<double> dP_dE_Hartree_sum_ae_ea_cr(N_Ef, 0.0);
    std::vector<double> dP_dE_kHz_sum_ae_ea_cr(N_Ef, 0.0);

    // ---- Truly exact M_F=-5 emit-absorb (exact bound + exact continuum) ----
    // c_ea_truly = c_ea_exact_bound + c_ea_cont_exact, where the continuum
    // piece uses the full Faddeeva on energy-normalized M_F=-5 scattering
    // states integrated over E ∈ [thr0_m5+eps, E_max_cont].
    std::vector<std::complex<double>> c_emit_abs_m5_cont_exact_scaled(N_Ef, {0.0, 0.0});
    std::vector<std::complex<double>> c_emit_abs_m5_truly_scaled(N_Ef, {0.0, 0.0});
    std::vector<double> abs_c_emit_abs_m5_truly_scaled(N_Ef, 0.0);
    std::vector<double> dP_dE_Hartree_emit_abs_m5_truly(N_Ef, 0.0);
    std::vector<double> dP_dE_kHz_emit_abs_m5_truly(N_Ef, 0.0);
    std::vector<std::complex<double>> c_sum_ae_ea_truly_scaled(N_Ef, {0.0, 0.0});
    std::vector<double> abs_c_sum_ae_ea_truly_scaled(N_Ef, 0.0);
    std::vector<double> dP_dE_Hartree_sum_ae_ea_truly(N_Ef, 0.0);
    std::vector<double> dP_dE_kHz_sum_ae_ea_truly(N_Ef, 0.0);

    const int N_En_abs_emit = 201;
    const double En_halfwidth_abs_emit_kHz = 150.0;
    const double En_halfwidth_abs_emit = kHz_to_au(En_halfwidth_abs_emit_kHz);
    const double overlap_dr = env_double_or_default("ZEPE_OVERLAP_DR", 0.5);
    const double overlap_R = env_double_or_default("ZEPE_OVERLAP_R", 5000.0);
    const int overlap_N = (int)std::llround(overlap_R / overlap_dr) + 1;
    cout << "M_F=-3 absorb-emit pathway: PDF Eq. (6)-(9)\n";
    cout << "  sigma1=-1, sigma2=+1, En*=(Ef+Eh)/2+omega\n";
    cout << "  En quadrature: " << N_En_abs_emit << " Simpson points over +/- "
         << En_halfwidth_abs_emit_kHz << " kHz around each En*\n";
    cout << "  radial overlap grid: R = " << (overlap_N - 1) * overlap_dr
         << " a_0, dr = " << overlap_dr << " a_0\n";
    cout << "  c_abs_emit_m3_kernel is unscaled; c_abs_emit_m3_scaled = "
            "(mu_B B_RF)^2 c_abs_emit_m3_kernel.\n";
    try {
        write_abs_emit_m3_window_probe(
            "second_order_abs_emit_m3_window_probe.txt",
            Ef_grid,
            halo_state,
            solver_m4,
            solver_m3_main,
            rf_m3_m4,
            E_halo,
            omega,
            tau0,
            N_En_abs_emit,
            En_halfwidth_abs_emit,
            overlap_dr,
            {500.0, 1000.0, 2500.0, overlap_R});
        cout << "Wrote absorb-emission M_F=-3 full-integral window probe to "
                "second_order_abs_emit_m3_window_probe.txt\n";
    } catch (const std::exception& e) {
        cout << "WARNING: could not write absorb-emission M_F=-3 window probe: "
             << e.what() << "\n";
    }
    cout << "M_F=-5 emit-absorb pathway:\n";
    cout << "  route: M_F=-4 --emit--> M_F=-5 --absorb--> M_F=-4\n";
    cout << "  Green kernel: G_-5(E_halo-omega) = (E_halo-omega-H_-5)^(-1)\n";
    cout << "  c_emit_abs_m5_green uses the large-detuning PDF-kernel limit; "
            "exact J(H_-5) is still not silently approximated as exact.\n";
    cout << "Coherent spectrum aliases in the output file:\n";
    cout << "  ae  = absorb-emit  = M_F=-4 -> -3 -> -4\n";
    cout << "  ea  = emit-absorb  = M_F=-4 -> -5 -> -4\n";
    cout << "  sum = ae + ea, using the current closed-Green ea approximation\n";

    #ifdef _OPENMP
    cout << "Using " << omp_get_max_threads() << " OpenMP threads\n";
    #endif
    cout << "Computing M_F=-5 emit-absorb closed-resolvent matrix element for "
         << N_Ef << " final states...\n";



    // -------------------------------------------------------------------
    // Precompute the M_F=-5 closed-channel response field:
    //
    //   S_beta(r) = sum_alpha eta_m4_m5(alpha, beta) * u_halo_alpha(r)
    //
    // For the linear-x closed-block pathway tested here, the intermediate
    // energy is
    //
    //   E_int = E_halo - ℏω
    //
    // This S is only the dimensionless spin/source shape. The physical RF
    // energy scale (mu_B * B_RF, Rabi frequency, pulse envelope, etc.) is
    // applied later. Therefore X has units of 1/energy.
    //
    // We solve the inhomogeneous closed-channel equation
    //
    //   (E_int - H_{-5}) X(r) = S(r)
    //
    // The solution X(r) is the finite-difference action of the closed-block
    // resolvent G_{-5}(E_int) on S. Since all M_F=-5 thresholds are far above
    // E_int, this is a strictly closed problem and no outgoing-wave boundary
    // condition is needed.
    //
    // X depends on E_halo, ω, and the M_F=-5 Hamiltonian — none of these
    // depend on E_f, so we precompute X once here.
    // -------------------------------------------------------------------
    const int N_m5 = pot_m5.num_channels();
    std::vector<Eigen::Vector2d> X_m5_src;
    double E_int_m5;
    // Hoisted closed-Green diagnostics (filled inside the block below; reused in
    // the # SUMMARY line at end of main).
    double diag_nS = 0.0, diag_nX = 0.0, diag_gap_min = 0.0;
    double diag_response = 0.0, diag_local_rel = 0.0, diag_residual = 0.0;
    {
        int Na = eta_m4_m5.rows(), Nb = eta_m4_m5.cols();
        if (Na != 2 || Nb != 2 || N_m5 != 2) {
            throw std::runtime_error("M_F=-5 compact inhom source requires 2x2 eta");
        }
        const Eigen::Matrix2d eta_m4_m5_2 = eta_m4_m5.topLeftCorner<2,2>();
        auto source_m5 = [&](int ir) -> Eigen::Vector2d {
            const Eigen::Vector2d u = halo_state.evaluate2(ir * p.dr);
            return eta_m4_m5_2.transpose() * u;
        };

        // Intermediate energy for the closed M_F=-5 resolvent used here.
        E_int_m5 = E_halo - omega;
        cout << "\nClosed-channel M_F=-5 inhom setup:\n";
        cout << "  E_int = E_halo - ℏω = "
             << E_int_m5 * Hartree_in_kHz << " kHz\n";
        cout << "  Thresholds - E_int = "
             << ((pot_m5.thresholds().array() - E_int_m5) * Hartree_in_GHz).transpose()
             << " GHz\n";

        // Solve (E_int - H_{-5}) X = S
        ClosedChannelInhom inhom_m5(&pot_m5, &p_m5, E_int_m5);
        X_m5_src = inhom_m5.solve_2ch(source_m5);

        // Diagnostics: norm of X vs S. The ratio has units Hartree^{-1}.
        // For a far-detuned closed block, |X|/|S| should be on the order of
        // 1/min(threshold - E_int), where the gap is measured in Hartree.
        const Eigen::VectorXd gap_vec = (pot_m5.thresholds().array() - E_int_m5).matrix();
        const int ir_asym = std::min(p.N_grid - 1,
            std::max(1, (int)((p.r0 + 100.0) / p.dr)));

        double nS2 = integrate_uniform_grid(p.N_grid, p.dr, [&](int ir) {
            return source_m5(ir).squaredNorm();
        });
        double nX2 = integrate_uniform_grid(p.N_grid, p.dr, [&](int ir) {
            return X_m5_src[ir].squaredNorm();
        });
        double local_err2 = integrate_uniform_grid(p.N_grid, p.dr, [&](int ir) {
            if (ir < ir_asym) return 0.0;
            const Eigen::Vector2d S = source_m5(ir);
            Eigen::Vector2d local = Eigen::Vector2d::Zero();
            for (int beta = 0; beta < Nb; ++beta) {
                local(beta) = -S(beta) / gap_vec(beta);
            }
            return (X_m5_src[ir] - local).squaredNorm();
        });
        double local_ref2 = integrate_uniform_grid(p.N_grid, p.dr, [&](int ir) {
            if (ir < ir_asym) return 0.0;
            const Eigen::Vector2d S = source_m5(ir);
            Eigen::Vector2d local = Eigen::Vector2d::Zero();
            for (int beta = 0; beta < Nb; ++beta) {
                local(beta) = -S(beta) / gap_vec(beta);
            }
            return local.squaredNorm();
        });
        diag_nS       = std::sqrt(nS2);
        diag_nX       = std::sqrt(nX2);
        diag_local_rel = std::sqrt(local_err2)
                       / std::max(std::sqrt(local_ref2), 1e-300);
        diag_gap_min  = gap_vec.minCoeff();
        diag_response = diag_nX / std::max(diag_nS, 1e-300);
        diag_residual = inhom_m5.relative_residual_2ch(X_m5_src, source_m5);
        cout << "  |S| = " << diag_nS << ",  |X| = " << diag_nX << "\n";
        cout << "  |X|/|S| = " << diag_response << " Hartree^-1\n";
        cout << "  min gap = " << diag_gap_min * Hartree_in_GHz
             << " GHz = " << diag_gap_min << " Hartree\n";
        cout << "  dimensionless check (|X|/|S|)*min_gap = "
             << diag_response * diag_gap_min << "  (order 1 expected)\n";
        cout << "  outside-well local check ||X + S/gap||/||S/gap|| = "
             << diag_local_rel << "  (small expected far from r0)\n";
        cout << "  finite-difference relative residual ||(E-H)X-S||/||S|| = "
             << diag_residual << "\n\n";
    }


    // -------------------------------------------------------------------
    // Exact M_F=-5 emit-absorb spectral data (precomputed once, reused per E_f)
    //
    // Replace the LO factor 1/(E_h - omega - E_n) by the full Faddeeva
    // J^(+,-)(E_f, E_n, E_h) inside a discrete bound-state sum:
    //
    //   c_ea_exact(E_f) = -1/4 (mu_B B_RF)^2 sum_n d_{f n} d_{n h}
    //                       J^(+,-)(E_f, E_n, E_h),
    //   J^(+,-) = (pi tau0^2/2) exp[-(E_f-E_h)^2 tau0^2/8] w(z_n),
    //   z_n = (E_f + E_h - 2 E_n - 2 omega) tau0 / (2 sqrt(2)).
    //
    // The closed M_F=-5 block has a finite (discrete) bound-state spectrum
    // strictly below pot_m5.thresholds()(0). Above that threshold there is
    // a continuum which we do NOT include here; the existing closed-Green
    // columns capture the full spectrum at leading order in 1/(E_n+omega),
    // so the two sets together cover both effects.
    // -------------------------------------------------------------------
    std::vector<double> m5_bound_E;
    std::vector<AnalyticState> m5_bound_states;
    std::vector<double> d_nh_m5;
    double m5_bound_dnh2_sum = 0.0;
    double m5_eta2_sumrule_target = 0.0;
    {
        cout << "---- Exact M_F=-5 emit-absorb spectral data ----\n";

        // Step 3a: scan for all bound states of the closed M_F=-5 block.
        // Lower edge: deeper than the well floor (well depth ~ V_T relative
        // to the threshold), with a small safety margin. Upper edge: just
        // below the lowest M_F=-5 threshold.
        const double m5_thr0 = pot_m5.thresholds()(0);
        const double m5_E_lo = m5_thr0 - 1.10 * p.V_T;
        const double m5_E_hi = m5_thr0 - kHz_to_au(1.0);
        const int    m5_n_scan = 20000;

        m5_bound_E = solver_m5.find_all_bound_states(
            m5_E_lo, m5_E_hi, m5_n_scan);

        cout << "  scan range = ["
             << m5_E_lo * Hartree_in_GHz << ", "
             << m5_E_hi * Hartree_in_GHz << "] GHz\n";
        cout << "  M_F=-5 bound states found: " << m5_bound_E.size() << "\n";
        if (m5_bound_E.empty()) {
            cout << "  WARNING: no bound states found; exact ea will be zero.\n";
        }
        for (size_t i = 0; i < m5_bound_E.size(); ++i) {
            const double E_n = m5_bound_E[i];
            cout << "    n=" << std::setw(2) << i << "  E_n = "
                 << std::scientific << std::setprecision(8)
                 << E_n * Hartree_in_GHz << " GHz   "
                 << "(below thresh by "
                 << (m5_thr0 - E_n) * Hartree_in_GHz << " GHz)\n";
        }
        cout << std::fixed;

        // Step 3b: build AnalyticState for each bound state and pre-compute
        // d_{n h} = <u_n | eta_{m5<-m4} | u_halo>. This is E_f-independent.
        m5_bound_states.reserve(m5_bound_E.size());
        d_nh_m5.reserve(m5_bound_E.size());
        const Eigen::MatrixXd eta_m5_m4 = eta_m4_m5.transpose();

        double sum_dnh2 = 0.0;
        for (size_t n = 0; n < m5_bound_E.size(); ++n) {
            AnalyticState n_state =
                solver_m5.build_bound_state(m5_bound_E[n]);
            const double d_nh = overlap_eta_states2(
                n_state, halo_state, eta_m5_m4, p.dr, p.N_grid);
            m5_bound_states.push_back(std::move(n_state));
            d_nh_m5.push_back(d_nh);
            sum_dnh2 += d_nh * d_nh;
        }

        // Step 3c: bound-state sum-rule diagnostic.
        // Completeness on the closed-block Hilbert space gives
        //   sum_{n bound} |d_{nh}|^2 + integral_continuum |d_E|^2 dE
        //     = <u_h | eta_{m4<->m5} eta_{m5<->m4} | u_h>.
        // The bound part should account for most of the right-hand side
        // when most of eta*u_halo lives at short range (where bound states
        // dominate); the continuum gap is the rest.
        const Eigen::Matrix2d eta_eta_T =
            eta_m4_m5.topLeftCorner<2,2>() * eta_m5_m4.topLeftCorner<2,2>();
        const double full_eta2 = integrate_uniform_grid(p.N_grid, p.dr,
            [&](int ir) {
                const Eigen::Vector2d u = halo_state.evaluate2(ir * p.dr);
                return u.dot(eta_eta_T * u);
            });
        cout << "  Sum_n d_{nh}^2                  = "
             << std::scientific << sum_dnh2 << "\n";
        cout << "  <u_h|eta_m4_m5 eta_m5_m4|u_h>   = "
             << full_eta2 << "\n";
        cout << "  bound completeness ratio        = "
             << sum_dnh2 / std::max(full_eta2, 1e-300)
             << "  (rest = M_F=-5 continuum contribution)\n";
        m5_bound_dnh2_sum = sum_dnh2;
        m5_eta2_sumrule_target = full_eta2;
        cout << std::fixed;

        // Step 3d: Green's-function consistency check at one E_f.
        // Comparing
        //   M_spectral_bound = sum_n d_{f n} d_{n h} / (E_h - omega - E_n)
        // (bound part of the spectral expansion of <u_f|eta G_-5(E_h-omega)
        // eta|u_h>) against the closed-Green value computed above. The
        // gap is the M_F=-5 continuum contribution at leading order.
        const double Ef_test = Ef_grid[0];
        AnalyticState uf_test = solver_m4.build_scattering_state(Ef_test, 0);
        double M_spec_bound = 0.0;
        for (size_t n = 0; n < m5_bound_states.size(); ++n) {
            const double d_fn = overlap_eta_states2(
                uf_test, m5_bound_states[n], eta_m4_m5, p.dr, p.N_grid);
            M_spec_bound += d_fn * d_nh_m5[n]
                          / (E_halo - omega - m5_bound_E[n]);
        }
        const double M_inhom = overlap_eta_state_grid2(
            uf_test, X_m5_src, eta_m4_m5, p.dr, p.N_grid);
        cout << "  Green-function consistency (E_f="
             << Ef_test * Hartree_in_kHz << " kHz):\n";
        cout << "    bound-state spectral sum      = "
             << std::scientific << M_spec_bound << "\n";
        cout << "    closed-Green (bound+continuum) = "
             << M_inhom << "\n";
        cout << "    ratio (bound/full)            = "
             << M_spec_bound / std::max(std::abs(M_inhom), 1e-300) << "\n\n";
        cout << std::fixed;
    }

    // -------------------------------------------------------------------
    // Exact M_F=-5 continuum spectral data (precomputed once, reused per E_f)
    //
    //   c_ea_cont_exact(E_f) = -1/4 (mu_B B_RF)^2 * sum_{i, beta} w_i
    //                            d_{f, E_i beta}(E_f) d_{E_i beta, h}
    //                            * J^(+,-)(E_f, E_i, E_h),
    //   J^(+,-) = (pi tau0^2/2) exp[-(E_f-E_h)^2 tau0^2/8] * w(z),
    //   z = (E_f + E_h - 2 E_i - 2 omega) tau0/(2 sqrt(2)).
    //
    // Two segments in E:
    //   seg 1: [thr0_m5 + eps, thr1_m5 - eps]   1 open channel
    //   seg 2: [thr1_m5 + eps, E_max_cont]       2 open channels
    // Each uses Simpson-1/3 + 3/8 weights. Use the K-matrix energy-
    // normalized states from solve_scattering_states_kmatrix.
    //
    // Diagnostic check at one E_f: bound spectral sum + LO-continuum
    // integral should reproduce closed-Green M_inhom (sum rule).
    // -------------------------------------------------------------------
    const double thr0_m5 = pot_m5.thresholds()(0);
    const double thr1_m5 = pot_m5.thresholds()(1);
    const double E_max_cont_GHz = env_double_or_default(
        "ZEPE_M5_CONT_E_MAX_GHZ", 50.0);
    const double E_max_cont = MHz_to_au(E_max_cont_GHz * 1.0e3);
    const int N_cont_seg1 = env_int_or_default("ZEPE_M5_CONT_N1", 401);
    const int N_cont_seg2 = env_int_or_default("ZEPE_M5_CONT_N2", 401);
    // Lower bounds for the log-spaced shifted grid (E - threshold).
    // At threshold k -> 0 and the matrix element has Wigner suppression d ~ sqrt(k);
    // a peak sets in around E - thresh ~ kappa_halo^2/(2 mu) ~ kHz, of width ~ kHz.
    // Default 1 Hz lower edge in (E-thresh) covers the halo threshold structure.
    const double cont_eps_kHz = env_double_or_default(
        "ZEPE_M5_CONT_EPS_KHZ", 0.001);   // 1 Hz default

    std::vector<double> E_cont;
    std::vector<double> w_cont;
    std::vector<int>    Nopen_cont;
    std::vector<std::vector<AnalyticState>> m5_cont_states;
    std::vector<std::vector<double>> d_Eh_cont;     // d_{E beta, h}
    const double m5_cont_norm_pass_tol = env_double_or_default(
        "ZEPE_M5_CONT_NORM_TOL", 0.05);
    const double m5_cc_window_pass_tol = env_double_or_default(
        "ZEPE_M5_CC_WINDOW_TOL", 0.10);
    double m5_cont_norm_ratio = std::numeric_limits<double>::quiet_NaN();
    double m5_cc_window_total_rel_error = std::numeric_limits<double>::quiet_NaN();
    double m5_cc_window_required_ratio = std::numeric_limits<double>::quiet_NaN();
    bool m5_cont_norm_pass = false;
    bool m5_cc_window_pass = false;
    {
        cout << "---- Exact M_F=-5 continuum spectral data ----\n";
        const double eps = kHz_to_au(cont_eps_kHz);

        // Build log-spaced grid in (E - threshold) for each segment, with
        // composite Simpson weights in log E. Weight in E: w_x * (E - threshold).
        auto build_shifted_log_grid = [&](double offset,
                                           double E_shift_lo,
                                           double E_shift_hi,
                                           int N,
                                           int Nopen)
        {
            if (E_shift_hi <= E_shift_lo || N < 4) {
                throw std::runtime_error(
                    "build_shifted_log_grid: bad range or too few points");
            }
            const double xlo = std::log(E_shift_lo);
            const double xhi = std::log(E_shift_hi);
            const double dx  = (xhi - xlo) / (N - 1);
            const std::vector<double> wx = uniform_simpson38_weights(N, dx);
            for (int i = 0; i < N; ++i) {
                const double x = xlo + i * dx;
                const double E_shift = std::exp(x);
                E_cont.push_back(offset + E_shift);
                w_cont.push_back(wx[i] * E_shift);   // dE = (E - offset) dx
                Nopen_cont.push_back(Nopen);
            }
        };

        const double E1_shift_hi = (thr1_m5 - thr0_m5) - eps;
        const double E2_shift_hi = (E_max_cont - thr1_m5);
        if (E1_shift_hi <= eps || E2_shift_hi <= eps) {
            throw std::runtime_error(
                "continuum E grid: thresholds and E_max_cont inconsistent");
        }

        cout << "  segment 1 (1 open ch): log-spaced E - thresh in ["
             << eps * Hartree_in_kHz << " kHz, "
             << E1_shift_hi * Hartree_in_GHz << " GHz], "
             << N_cont_seg1 << " points\n";
        cout << "  segment 2 (2 open ch): log-spaced E - thr1 in ["
             << eps * Hartree_in_kHz << " kHz, "
             << E2_shift_hi * Hartree_in_GHz << " GHz], "
             << N_cont_seg2 << " points\n";

        build_shifted_log_grid(thr0_m5, eps, E1_shift_hi, N_cont_seg1, 1);
        build_shifted_log_grid(thr1_m5, eps, E2_shift_hi, N_cont_seg2, 2);

        const int N_E = (int)E_cont.size();
        m5_cont_states.resize(N_E);
        d_Eh_cont.resize(N_E);

        const Eigen::MatrixXd eta_m5_m4_local = eta_m4_m5.transpose();
        long total_states = 0;
        for (int i = 0; i < N_E; ++i) {
            auto states = solver_m5.build_scattering_states_kmatrix(E_cont[i]);
            if ((int)states.size() != Nopen_cont[i]) {
                throw std::runtime_error(
                    "continuum: open-channel count mismatch at E_cont["
                    + std::to_string(i) + "] (expected "
                    + std::to_string(Nopen_cont[i])
                    + ", got " + std::to_string(states.size()) + ")");
            }
            m5_cont_states[i] = std::move(states);
            d_Eh_cont[i].resize(m5_cont_states[i].size());
            for (size_t b = 0; b < m5_cont_states[i].size(); ++b) {
                d_Eh_cont[i][b] = overlap_eta_states2(
                    m5_cont_states[i][b], halo_state,
                    eta_m5_m4_local, overlap_dr, overlap_N);
                ++total_states;
            }
        }
        cout << "  total continuum states pre-built: " << total_states
             << "  (overlap radial grid: R = "
             << (overlap_N - 1) * overlap_dr << " a_0, dr = "
             << overlap_dr << " a_0)\n";

        // Diagnostic: sum-rule check at Ef_test.
        // Σ d_{Eβ,h}^2 weighted by dE captured by the continuum:
        double sum_dEh2 = 0.0;
        for (int i = 0; i < N_E; ++i) {
            for (size_t b = 0; b < m5_cont_states[i].size(); ++b) {
                sum_dEh2 += w_cont[i] * d_Eh_cont[i][b] * d_Eh_cont[i][b];
            }
        }
        cout << "  Integral_continuum d_{E,h}^2 dE = "
             << std::scientific << sum_dEh2 << "\n";
        const double cont_norm_target =
            m5_eta2_sumrule_target - m5_bound_dnh2_sum;
        const double total_norm_sum =
            m5_bound_dnh2_sum + sum_dEh2;
        m5_cont_norm_ratio =
            total_norm_sum / std::max(m5_eta2_sumrule_target, 1e-300);
        m5_cont_norm_pass =
            std::abs(m5_cont_norm_ratio - 1.0) <= m5_cont_norm_pass_tol;
        cout << "  Continuum norm target            = "
             << cont_norm_target << "\n";
        cout << "  bound + continuum norm sum       = "
             << total_norm_sum << "\n";
        cout << "  norm completeness ratio          = "
             << m5_cont_norm_ratio
             << "  (close to 1 validates d_Eh and E-continuum quadrature)\n";
        cout << "  M5 continuum norm check: "
             << (m5_cont_norm_pass ? "PASS" : "FAIL")
             << "  (tol |ratio-1| <= " << m5_cont_norm_pass_tol << ")\n";
        cout << "  (should be close to "
             << "<u_h|eta_m4_m5 eta_m5_m4|u_h> minus bound sum, modulo finite E_max"
             << ")\n";

        // Energy-normalization sanity check: for any orthonormalized state,
        //   sum_{alpha open} [P_alpha^2 + Q_alpha^2] / (2 mu/(pi k_alpha)) = 1
        // (follows from u^beta = sum_gamma u^gamma_raw O_{gamma,beta} with
        //  O = (1+KK^T)^{-1/2}). Print at one E in each segment.
        auto norm_check = [&](const AnalyticState& s) {
            double total = 0.0;
            for (int a = 0; a < (int)s.is_open.size(); ++a) {
                if (!s.is_open[a]) continue;
                const double k_a = s.k_out(a);
                const double per_ch = 2.0 * p.mu / (M_PI * k_a);
                total += (s.P_out(a)*s.P_out(a) + s.Q_out(a)*s.Q_out(a)) / per_ch;
            }
            return total;
        };
        if (!m5_cont_states.empty() && !m5_cont_states[0].empty()) {
            cout << "  Norm check (1-open, E="
                 << E_cont[0] * Hartree_in_GHz << " GHz, beta=0): "
                 "sum (P^2+Q^2)/(2mu/pi k) = "
                 << norm_check(m5_cont_states[0][0]) << "  (expected 1.0)\n";
        }
        if (N_cont_seg1 < (int)m5_cont_states.size()
            && m5_cont_states[N_cont_seg1].size() == 2)
        {
            cout << "  Norm check (2-open, E="
                 << E_cont[N_cont_seg1] * Hartree_in_GHz
                 << " GHz, beta=0): "
                 << norm_check(m5_cont_states[N_cont_seg1][0])
                 << "  (expected 1.0)\n";
            cout << "  Norm check (2-open, E="
                 << E_cont[N_cont_seg1] * Hartree_in_GHz
                 << " GHz, beta=1): "
                 << norm_check(m5_cont_states[N_cont_seg1][1])
                 << "  (expected 1.0)\n";
        }

        // LO continuum vs (closed-Green - bound) at Ef_test
        const double Ef_test_c = Ef_grid[0];
        AnalyticState uf_test_c = solver_m4.build_scattering_state(Ef_test_c, 0);
        double M_cont_LO = 0.0;
        for (int i = 0; i < N_E; ++i) {
            for (size_t b = 0; b < m5_cont_states[i].size(); ++b) {
                const double d_fE = overlap_eta_states2(
                    uf_test_c, m5_cont_states[i][b],
                    eta_m4_m5, overlap_dr, overlap_N);
                M_cont_LO += w_cont[i] * d_fE * d_Eh_cont[i][b]
                            / (E_halo - omega - E_cont[i]);
            }
        }
        // Recompute bound + closed-Green at the same E_f for a clean comparison
        double M_spec_bound_c = 0.0;
        for (size_t n = 0; n < m5_bound_states.size(); ++n) {
            const double d_fn = overlap_eta_states2(
                uf_test_c, m5_bound_states[n],
                eta_m4_m5, overlap_dr, overlap_N);
            M_spec_bound_c += d_fn * d_nh_m5[n]
                            / (E_halo - omega - m5_bound_E[n]);
        }
        const double M_inhom_c = overlap_eta_state_grid2(
            uf_test_c, X_m5_src, eta_m4_m5, p.dr, p.N_grid);
        const double M_spec_total_LO = M_spec_bound_c + M_cont_LO;
        const double M_cont_required = M_inhom_c - M_spec_bound_c;
        const double M_cont_required_den =
            (std::abs(M_cont_required) > 1e-300)
                ? M_cont_required
                : (M_cont_required < 0.0 ? -1e-300 : 1e-300);
        const double M_cont_rel_error =
            (M_cont_LO - M_cont_required)
            / std::max(std::abs(M_cont_required), 1e-300);
        const double M_total_rel_error =
            (M_spec_total_LO - M_inhom_c)
            / std::max(std::abs(M_inhom_c), 1e-300);
        m5_cc_window_total_rel_error = M_total_rel_error;
        m5_cc_window_required_ratio = M_cont_LO / M_cont_required_den;
        m5_cc_window_pass =
            std::abs(m5_cc_window_total_rel_error) <= m5_cc_window_pass_tol;
        cout << "  Sum-rule check at E_f = "
             << Ef_test_c * Hartree_in_kHz << " kHz:\n";
        cout << "    LO bound (overlap_N grid)   = " << M_spec_bound_c << "\n";
        cout << "    LO continuum (integral)     = " << M_cont_LO << "\n";
        cout << "    LO continuum required       = " << M_cont_required
             << "  (closed-Green - LO bound)\n";
        cout << "    continuum finite-window / required = "
             << m5_cc_window_required_ratio
             << "\n";
        cout << "    continuum relative error    = " << M_cont_rel_error
             << "  (large => continuum-continuum window artifact)\n";
        cout << "    bound + continuum (LO)      = " << M_spec_total_LO << "\n";
        cout << "    closed-Green (full p_grid)  = " << M_inhom_c << "\n";
        cout << "    ratio (LO_total/closed)     = "
             << M_spec_total_LO / std::max(std::abs(M_inhom_c), 1e-300)
             << "  (1 if finite-window continuum-continuum object were valid)\n";
        cout << "    total relative error        = " << M_total_rel_error
             << "  (same test for the finite-window continuum-continuum object)\n";
        cout << "    M5 continuum-continuum finite-window check: "
             << (m5_cc_window_pass ? "PASS" : "FAIL")
             << "  (tol |total relative error| <= "
             << m5_cc_window_pass_tol << ")\n";
        cout << "    recommended ea column: best\n";
        cout << "    truly_exact status: "
             << (m5_cc_window_pass
                 ? "LO check passed; still verify R/N/Emax convergence"
                 : "diagnostic only") << "\n\n";
        cout << std::fixed;

        // Early one-point complex-resolvent probe. This runs before the
        // expensive all-final-state loop, so a production log immediately
        // shows whether the off-real-axis continuum method reproduces the
        // closed-Green continuum at leading order.
        zepe::ComplexResolventEaOptions cr_probe_opt;
        cr_probe_opt.R_a0 = env_double_or_default("ZEPE_M5_CR_R", overlap_R);
        cr_probe_opt.dr_a0 = env_double_or_default("ZEPE_M5_CR_DR", overlap_dr);
        cr_probe_opt.eta_kHz = env_double_or_default("ZEPE_M5_CR_ETA_KHZ", 10.0);
        cr_probe_opt.N_seg1 = env_int_or_default("ZEPE_M5_CR_N1", N_cont_seg1);
        cr_probe_opt.N_seg2 = env_int_or_default("ZEPE_M5_CR_N2", N_cont_seg2);
        cr_probe_opt.eps_kHz = env_double_or_default("ZEPE_M5_CR_EPS_KHZ", cont_eps_kHz);
        cr_probe_opt.E_max_GHz = env_double_or_default("ZEPE_M5_CR_E_MAX_GHZ", E_max_cont_GHz);
        if (const char* raw = std::getenv("ZEPE_M5_COMPLEX_RESOLVENT")) {
            const std::string value(raw);
            cr_probe_opt.enabled = !(value == "0" || value == "false"
                                  || value == "FALSE" || value == "off"
                                  || value == "OFF" || value == "no"
                                  || value == "NO");
        }

        cout << "---- M_F=-5 complex-resolvent early kf=0 probe ----\n";
        cout << "  status = " << (cr_probe_opt.enabled ? "enabled" : "disabled")
             << "  (set ZEPE_M5_COMPLEX_RESOLVENT=0 to skip)\n";
        if (cr_probe_opt.enabled) {
            cout << "  radial grid: R = " << cr_probe_opt.R_a0
                 << " a_0, dr = " << cr_probe_opt.dr_a0 << " a_0\n";
            cout << "  spectral broadening eta = "
                 << cr_probe_opt.eta_kHz << " kHz\n";
            cout << "  continuum grid: N1 = " << cr_probe_opt.N_seg1
                 << ", N2 = " << cr_probe_opt.N_seg2
                 << ", eps = " << cr_probe_opt.eps_kHz
                 << " kHz, Emax = " << cr_probe_opt.E_max_GHz << " GHz\n";

            const std::vector<double> Ef_probe_grid{Ef_test_c};
            const zepe::ComplexResolventEaResult cr_probe =
                zepe::compute_complex_resolvent_ea(
                    pot_m5, p_m5, pot_m4, p_halo,
                    halo_state, Ef_probe_grid, eta_m4_m5,
                    E_halo, omega, tau0, rf_energy2, cr_probe_opt);

            const double target_cont_norm =
                m5_eta2_sumrule_target - m5_bound_dnh2_sum;
            const double cr_norm_ratio =
                cr_probe.source_continuum_norm
                / std::max(std::abs(target_cont_norm), 1e-300);
            const std::complex<double> LO_factor_probe =
                emit_absorb_m5_green_pulse_factor(Ef_test_c, E_halo, tau0);
            const std::complex<double> c_green_probe =
                rf_energy2 * LO_factor_probe * M_inhom_c;
            const std::complex<double> c_LO_bound_probe =
                rf_energy2 * LO_factor_probe * M_spec_bound_c;
            const std::complex<double> c_LO_required_probe =
                c_green_probe - c_LO_bound_probe;
            const std::complex<double> c_CR_LO_total_probe =
                c_LO_bound_probe + cr_probe.c_LO_cont_scaled[0];
            const double rel_CR_cont_required =
                std::abs(cr_probe.c_LO_cont_scaled[0] - c_LO_required_probe)
                / std::max(std::abs(c_LO_required_probe), 1e-300);
            const double rel_CR_total_green =
                std::abs(c_CR_LO_total_probe - c_green_probe)
                / std::max(std::abs(c_green_probe), 1e-300);

            cout << std::scientific << std::setprecision(6);
            cout << "  complex-resolvent states: radial_N = "
                 << cr_probe.radial_N
                 << ", continuum_N = " << cr_probe.continuum_N << "\n";
            cout << "  source continuum norm (broadened) = "
                 << cr_probe.source_continuum_norm << "\n";
            cout << "  target continuum norm             = "
                 << target_cont_norm << "\n";
            cout << "  source norm ratio                 = "
                 << cr_norm_ratio
                 << "  (check eta/R/Emax convergence)\n";
            cout << "  |c_CR_cont_exact_kernel|          = "
                 << std::abs(cr_probe.c_cont_scaled[0]) << "\n";
            cout << "  |c_CR_LO_cont|                    = "
                 << std::abs(cr_probe.c_LO_cont_scaled[0]) << "\n";
            cout << "  |LO continuum required|           = "
                 << std::abs(c_LO_required_probe)
                 << "  (= green - LO_bound)\n";
            cout << "  relative CR_LO_cont vs required   = "
                 << rel_CR_cont_required << "\n";
            cout << "  relative (LO_bound+CR_LO_cont) vs green = "
                 << rel_CR_total_green << "\n\n";
            cout << std::fixed;
        }
    }

    #pragma omp parallel
    {
        // Thread-local solver wrapper, sharing the already-built M_F=-4
        // potential. Rebuilding Potentials here would duplicate enormous
        // grid storage.
        AnalyticSquareWell solver_local_m4(&pot_m4, &p_halo);
        AnalyticSquareWell solver_local_m3(&pot_m3, &p_m3);

        

        #pragma omp for schedule(dynamic)
        for (int kf = 0; kf < N_Ef; ++kf) {
            double Ef = Ef_grid[kf];
            Ef_kHz[kf] = Ef * Hartree_in_kHz;

            // Final state u_f
            AnalyticState u_f_state = solver_local_m4.build_scattering_state(Ef, 0);

            std::complex<double> c_abs_emit_kernel_val = absorb_emit_m3_pdf(
                u_f_state, halo_state, solver_local_m3, rf_m3_m4,
                Ef, E_halo, omega, tau0,
                N_En_abs_emit, En_halfwidth_abs_emit, overlap_dr, overlap_N);
            const std::complex<double> c_abs_emit_scaled_val =
                rf_energy2 * c_abs_emit_kernel_val;
            c_abs_emit_m3_kernel[kf] = c_abs_emit_kernel_val;
            c_abs_emit_m3_scaled[kf] = c_abs_emit_scaled_val;
            abs_c_abs_emit_m3_kernel[kf] = std::abs(c_abs_emit_kernel_val);
            abs_c_abs_emit_m3_scaled[kf] = std::abs(c_abs_emit_scaled_val);
            dP_dE_Hartree_abs_emit_m3[kf] = std::norm(c_abs_emit_scaled_val);
            dP_dE_kHz_abs_emit_m3[kf] =
                dP_dE_Hartree_abs_emit_m3[kf] * kHz_to_au(1.0);

            // --------------------------------------------------------
            // Linear-x matrix element through the CLOSED M_F=-5 block.
            //
            // X_m5_src(r) was precomputed as G_{-5}(E_halo-ℏω) applied to
            // the source eta_x u_halo. The final-state matrix element is
            //   M_emit_abs_m5_green(E_f) = ∫ dr u_f^*(r) eta_x X_m5_src(r).
            // This is the exact spin/radial closed-resolvent response at the
            // chosen intermediate energy. Below, c_emit_abs_m5_green_scaled
            // applies the large-detuning finite-pulse factor from
            // emit_absorb_m5_green_pulse_factor().
            // --------------------------------------------------------
            {
                double M_emit_abs_m5_green_val = overlap_eta_state_grid2(
                    u_f_state, X_m5_src, eta_m4_m5, p.dr, p.N_grid);
                M_emit_abs_m5_green_kernel[kf] = M_emit_abs_m5_green_val;
                abs_M_emit_abs_m5_green_kernel[kf] = std::abs(M_emit_abs_m5_green_val);
                M_emit_abs_m5_green_scaled[kf] = rf_energy2 * M_emit_abs_m5_green_val;
                abs_M_emit_abs_m5_green_scaled[kf] =
                    std::abs(M_emit_abs_m5_green_scaled[kf]);
                c_emit_abs_m5_green_scaled[kf] =
                    rf_energy2
                    * emit_absorb_m5_green_pulse_factor(Ef, E_halo, tau0)
                    * M_emit_abs_m5_green_val;
                abs_c_emit_abs_m5_green_scaled[kf] =
                    std::abs(c_emit_abs_m5_green_scaled[kf]);
                dP_dE_Hartree_emit_abs_m5_green[kf] =
                    std::norm(c_emit_abs_m5_green_scaled[kf]);
                dP_dE_kHz_emit_abs_m5_green[kf] =
                    dP_dE_Hartree_emit_abs_m5_green[kf] * kHz_to_au(1.0);

                c_sum_ae_ea_scaled[kf] =
                    c_abs_emit_m3_scaled[kf] + c_emit_abs_m5_green_scaled[kf];
                abs_c_sum_ae_ea_scaled[kf] = std::abs(c_sum_ae_ea_scaled[kf]);
                dP_dE_Hartree_sum_ae_ea[kf] =
                    std::norm(c_sum_ae_ea_scaled[kf]);
                dP_dE_kHz_sum_ae_ea[kf] =
                    dP_dE_Hartree_sum_ae_ea[kf] * kHz_to_au(1.0);

                // -------- ea via bound-state spectral sum --------
                // Two flavours, plus a "best-of-both" combination:
                //
                //   _exact_bound:  -1/4 sum_n d_{f n} d_{n h} J^(+,-)(E_f,E_n,E_h)
                //                  with the FULL Faddeeva (no large-|z|
                //                  factorization). Bound states only;
                //                  excludes M_F=-5 continuum.
                //
                //   _LO_bound:     bound-state piece of c_ea_green, i.e.
                //                  LO_factor(E_f) * sum_n d_{f n} d_{n h}
                //                                  / (E_h - omega - E_n).
                //                  Bound contribution to closed-Green at LO.
                //
                //   _best:         c_ea_green + (c_ea_exact_bound
                //                                - c_ea_LO_bound).
                //                  -> bound part exact, continuum part LO.
                //
                // For |z_n| >> 1 across the entire M_F=-5 spectrum (true at
                // B=155 G, tau0~30us), c_ea_exact_bound ~ c_ea_LO_bound and
                // c_ea_best ~ c_ea_green. Differences quantify higher-order
                // corrections to the bound-state Faddeeva.
                std::complex<double> sum_J_dd(0.0, 0.0);
                double sum_d_d_over_dE = 0.0;
                for (size_t n = 0; n < m5_bound_states.size(); ++n) {
                    const double d_fn = overlap_eta_states2(
                        u_f_state, m5_bound_states[n],
                        eta_m4_m5, p.dr, p.N_grid);
                    const std::complex<double> J_n = second_order_J(
                        Ef, m5_bound_E[n], E_halo, omega,
                        +1, -1, tau0);
                    sum_J_dd += d_fn * d_nh_m5[n] * J_n;
                    sum_d_d_over_dE += d_fn * d_nh_m5[n]
                                     / (E_halo - omega - m5_bound_E[n]);
                }
                const std::complex<double> c_ea_exact_kernel = -0.25 * sum_J_dd;
                const std::complex<double> c_ea_exact_scaled =
                    rf_energy2 * c_ea_exact_kernel;
                c_emit_abs_m5_exact_kernel[kf] = c_ea_exact_kernel;
                c_emit_abs_m5_exact_scaled[kf] = c_ea_exact_scaled;
                abs_c_emit_abs_m5_exact_scaled[kf] = std::abs(c_ea_exact_scaled);
                dP_dE_Hartree_emit_abs_m5_exact[kf] = std::norm(c_ea_exact_scaled);
                dP_dE_kHz_emit_abs_m5_exact[kf] =
                    dP_dE_Hartree_emit_abs_m5_exact[kf] * kHz_to_au(1.0);

                c_sum_ae_ea_exact_scaled[kf] =
                    c_abs_emit_m3_scaled[kf] + c_ea_exact_scaled;
                abs_c_sum_ae_ea_exact_scaled[kf] =
                    std::abs(c_sum_ae_ea_exact_scaled[kf]);
                dP_dE_Hartree_sum_ae_ea_exact[kf] =
                    std::norm(c_sum_ae_ea_exact_scaled[kf]);
                dP_dE_kHz_sum_ae_ea_exact[kf] =
                    dP_dE_Hartree_sum_ae_ea_exact[kf] * kHz_to_au(1.0);

                // LO bound piece using the same kernel as closed-Green:
                //   c_ea_LO_bound = LO_factor(E_f) * sum_n d d / (E_h-w-E_n) * rf^2
                const std::complex<double> LO_factor =
                    emit_absorb_m5_green_pulse_factor(Ef, E_halo, tau0);
                const std::complex<double> c_ea_LO_bound_scaled_val =
                    rf_energy2 * LO_factor * sum_d_d_over_dE;
                c_emit_abs_m5_LO_bound_scaled[kf] = c_ea_LO_bound_scaled_val;

                // best-of-both: continuum at LO via closed-Green, bound at full Faddeeva
                const std::complex<double> c_ea_best_scaled =
                    c_emit_abs_m5_green_scaled[kf]
                    + (c_ea_exact_scaled - c_ea_LO_bound_scaled_val);
                c_emit_abs_m5_best_scaled[kf] = c_ea_best_scaled;
                abs_c_emit_abs_m5_best_scaled[kf] = std::abs(c_ea_best_scaled);
                dP_dE_Hartree_emit_abs_m5_best[kf] = std::norm(c_ea_best_scaled);
                dP_dE_kHz_emit_abs_m5_best[kf] =
                    dP_dE_Hartree_emit_abs_m5_best[kf] * kHz_to_au(1.0);

                c_sum_ae_ea_best_scaled[kf] =
                    c_abs_emit_m3_scaled[kf] + c_ea_best_scaled;
                abs_c_sum_ae_ea_best_scaled[kf] =
                    std::abs(c_sum_ae_ea_best_scaled[kf]);
                dP_dE_Hartree_sum_ae_ea_best[kf] =
                    std::norm(c_sum_ae_ea_best_scaled[kf]);
                dP_dE_kHz_sum_ae_ea_best[kf] =
                    dP_dE_Hartree_sum_ae_ea_best[kf] * kHz_to_au(1.0);

                // ---- Continuum (exact Faddeeva over energy integration) ----
                // I_cont = sum_i sum_beta w_i d_{f, E_i beta} d_{E_i beta, h}
                //          * J^(+,-)(E_f, E_i, E_h)
                // c_ea_cont_exact = -1/4 I_cont; c_ea_truly = bound_exact + cont_exact.
                std::complex<double> I_cont_exact(0.0, 0.0);
                std::complex<double> I_cont_LO(0.0, 0.0);
                for (size_t i = 0; i < E_cont.size(); ++i) {
                    const double E_i = E_cont[i];
                    const std::complex<double> J_E = second_order_J(
                        Ef, E_i, E_halo, omega, +1, -1, tau0);
                    const double inv_dE = 1.0 / (E_halo - omega - E_i);
                    for (size_t b = 0; b < m5_cont_states[i].size(); ++b) {
                        const double d_fE = overlap_eta_states2(
                            u_f_state, m5_cont_states[i][b],
                            eta_m4_m5, overlap_dr, overlap_N);
                        const double dd = d_fE * d_Eh_cont[i][b];
                        I_cont_exact += w_cont[i] * dd * J_E;
                        I_cont_LO    += w_cont[i] * dd * inv_dE;
                    }
                }
                const std::complex<double> c_ea_cont_exact_kernel = -0.25 * I_cont_exact;
                const std::complex<double> c_ea_cont_exact_scaled =
                    rf_energy2 * c_ea_cont_exact_kernel;
                c_emit_abs_m5_cont_exact_scaled[kf] = c_ea_cont_exact_scaled;

                const std::complex<double> c_ea_truly_scaled =
                    c_ea_exact_scaled + c_ea_cont_exact_scaled;
                c_emit_abs_m5_truly_scaled[kf] = c_ea_truly_scaled;
                abs_c_emit_abs_m5_truly_scaled[kf] = std::abs(c_ea_truly_scaled);
                dP_dE_Hartree_emit_abs_m5_truly[kf] = std::norm(c_ea_truly_scaled);
                dP_dE_kHz_emit_abs_m5_truly[kf] =
                    dP_dE_Hartree_emit_abs_m5_truly[kf] * kHz_to_au(1.0);

                c_sum_ae_ea_truly_scaled[kf] =
                    c_abs_emit_m3_scaled[kf] + c_ea_truly_scaled;
                abs_c_sum_ae_ea_truly_scaled[kf] =
                    std::abs(c_sum_ae_ea_truly_scaled[kf]);
                dP_dE_Hartree_sum_ae_ea_truly[kf] =
                    std::norm(c_sum_ae_ea_truly_scaled[kf]);
                dP_dE_kHz_sum_ae_ea_truly[kf] =
                    dP_dE_Hartree_sum_ae_ea_truly[kf] * kHz_to_au(1.0);

                if (kf == 0) {
                    #pragma omp critical
                    {
                        const double abs_green = abs_c_emit_abs_m5_green_scaled[kf];
                        const double abs_LO_bound = std::abs(c_ea_LO_bound_scaled_val);
                        const double abs_correction =
                            std::abs(c_ea_exact_scaled - c_ea_LO_bound_scaled_val);
                        // Sum-rule check at this kf:
                        //   LO total spectral = LO bound + LO continuum
                        //                     should equal closed-Green ea (LO_factor * M_inhom)
                        const std::complex<double> c_ea_cont_LO_scaled_val =
                            rf_energy2 * LO_factor * I_cont_LO;
                        const std::complex<double> c_ea_total_LO_check =
                            c_ea_LO_bound_scaled_val + c_ea_cont_LO_scaled_val;
                        const double abs_best = std::abs(c_ea_best_scaled);
                        const double rel_LO_total_complex =
                            std::abs(c_ea_total_LO_check
                                     - c_emit_abs_m5_green_scaled[kf])
                            / std::max(abs_green, 1e-300);
                        const double rel_best_vs_green =
                            std::abs(c_ea_best_scaled
                                     - c_emit_abs_m5_green_scaled[kf])
                            / std::max(abs_green, 1e-300);
                        const double rel_truly_vs_best =
                            std::abs(c_ea_truly_scaled - c_ea_best_scaled)
                            / std::max(abs_best, 1e-300);
                        cout << "  [ea kf=0] E_f="
                             << std::scientific << std::setprecision(3)
                             << Ef * Hartree_in_kHz << " kHz\n"
                             << "    |c_ea_green|         = " << abs_green << "\n"
                             << "    |c_ea_exact_bound|   = "
                             << abs_c_emit_abs_m5_exact_scaled[kf] << "\n"
                             << "    |c_ea_LO_bound|      = " << abs_LO_bound << "\n"
                             << "    |c_ea_best|          = "
                             << abs_c_emit_abs_m5_best_scaled[kf]
                             << "  (recommended controlled result)\n"
                             << "    |c_ea_cont_exact|    = "
                             << std::abs(c_ea_cont_exact_scaled) << "\n"
                             << "    |c_ea_truly_exact|   = "
                             << abs_c_emit_abs_m5_truly_scaled[kf] << "\n"
                             << "    relative |best - green|/|green| = "
                             << rel_best_vs_green << "\n"
                             << "    relative |truly - best|/|best| = "
                             << rel_truly_vs_best
                             << "  (large => finite-window continuum-continuum effect)\n"
                             << "    sum-rule (LO_bound + LO_cont) / green:\n"
                             << "      |LO_total_check| = "
                             << std::abs(c_ea_total_LO_check)
                             << "  ratio = "
                             << (abs_green > 0
                                 ? std::abs(c_ea_total_LO_check) / abs_green
                                 : 0.0)
                             << "  (1 if finite-window LO continuum reproduces closed Green)\n"
                             << "      complex relative error vs green = "
                             << rel_LO_total_complex
                             << "  (phase/sign-sensitive)\n"
                             << "    |Faddeeva correction (exact_bound - LO_bound)| = "
                             << abs_correction
                             << " (relative to green: "
                             << (abs_green > 0 ? abs_correction / abs_green : 0.0)
                             << ")\n"
                             << std::fixed;
                    }
                }

                // Diagnostic print at kf=0 only
                if (kf == 0) {
                    #pragma omp critical
                    {
                        /* cout << "  [inhom-GF] E_int=" << std::scientific
                             << std::setprecision(3)
                             << E_int_m5*Hartree_in_kHz << " kHz"
                             << "  M_emit_abs_m5_green_kernel=" << M_emit_abs_m5_green_val
                             << "  |M_emit_abs_m5_green_scaled|="
                             << abs_M_emit_abs_m5_green_scaled[kf]
                             << "  |c_emit_abs_m5_green_scaled|="
                             << abs_c_emit_abs_m5_green_scaled[kf]
                             << "  dP/dE_kHz_emit_abs_m5_green="
                             << dP_dE_kHz_emit_abs_m5_green[kf]
                             << "  c_abs_emit_m3_kernel=" << c_abs_emit_kernel_val
                             << "  |c_abs_emit_m3_scaled|="
                             << abs_c_abs_emit_m3_scaled[kf]
                             << "  dP/dE_kHz_abs_emit_m3="
                             << dP_dE_kHz_abs_emit_m3[kf]
                             << "  |c_sum_ae_ea_scaled|="
                             << abs_c_sum_ae_ea_scaled[kf]
                             << "  dP/dE_kHz_sum_ae_ea="
                             << dP_dE_kHz_sum_ae_ea[kf] << "\n"; */
                    }
                }
            }

            #pragma omp critical
            {
                if (kf % 20 == 0){
                    /* cout << "  kf=" << kf << "/" << N_Ef
                         << "  E_f = " << std::scientific << std::setprecision(3)
                         << Ef * Hartree_in_kHz
                         << " kHz  |M_emit_abs_m5_green_scaled| = "
                         << abs_M_emit_abs_m5_green_scaled[kf]
                         << "  |c_emit_abs_m5_green_scaled| = "
                         << abs_c_emit_abs_m5_green_scaled[kf]
                         << "  dP/dE_kHz_emit_abs_m5_green = "
                         << dP_dE_kHz_emit_abs_m5_green[kf]
                         << "  |c_abs_emit_m3_scaled| = "
                         << abs_c_abs_emit_m3_scaled[kf]
                         << "  dP/dE_kHz_abs_emit_m3 = "
                         << dP_dE_kHz_abs_emit_m3[kf]
                         << "  |c_sum_ae_ea_scaled| = "
                         << abs_c_sum_ae_ea_scaled[kf]
                         << "  dP/dE_kHz_sum_ae_ea = "
                         << dP_dE_kHz_sum_ae_ea[kf] << std::fixed << "\n"; */
                }
            }


        }
    }

    zepe::ComplexResolventEaResult m5_cr_result;
    {
        zepe::ComplexResolventEaOptions cr_opt;
        cr_opt.R_a0 = env_double_or_default("ZEPE_M5_CR_R", overlap_R);
        cr_opt.dr_a0 = env_double_or_default("ZEPE_M5_CR_DR", overlap_dr);
        cr_opt.eta_kHz = env_double_or_default("ZEPE_M5_CR_ETA_KHZ", 10.0);
        cr_opt.N_seg1 = env_int_or_default("ZEPE_M5_CR_N1", N_cont_seg1);
        cr_opt.N_seg2 = env_int_or_default("ZEPE_M5_CR_N2", N_cont_seg2);
        cr_opt.eps_kHz = env_double_or_default("ZEPE_M5_CR_EPS_KHZ", cont_eps_kHz);
        cr_opt.E_max_GHz = env_double_or_default("ZEPE_M5_CR_E_MAX_GHZ", E_max_cont_GHz);
        if (const char* raw = std::getenv("ZEPE_M5_COMPLEX_RESOLVENT")) {
            const std::string value(raw);
            cr_opt.enabled = !(value == "0" || value == "false"
                            || value == "FALSE" || value == "off"
                            || value == "OFF" || value == "no"
                            || value == "NO");
        }

        cout << "---- M_F=-5 complex-resolvent continuum diagnostic ----\n";
        cout << "  status = " << (cr_opt.enabled ? "enabled" : "disabled")
             << "  (set ZEPE_M5_COMPLEX_RESOLVENT=0 to skip)\n";
        if (cr_opt.enabled) {
            cout << "  radial grid: R = " << cr_opt.R_a0
                 << " a_0, dr = " << cr_opt.dr_a0 << " a_0\n";
            cout << "  spectral broadening eta = " << cr_opt.eta_kHz << " kHz\n";
            cout << "  continuum grid: N1 = " << cr_opt.N_seg1
                 << ", N2 = " << cr_opt.N_seg2
                 << ", eps = " << cr_opt.eps_kHz
                 << " kHz, Emax = " << cr_opt.E_max_GHz << " GHz\n";

            m5_cr_result = zepe::compute_complex_resolvent_ea(
                pot_m5, p_m5, pot_m4, p_halo,
                halo_state, Ef_grid, eta_m4_m5,
                E_halo, omega, tau0, rf_energy2, cr_opt);

            for (int kf = 0; kf < N_Ef; ++kf) {
                c_emit_abs_m5_cr_cont_scaled[kf] =
                    m5_cr_result.c_cont_scaled[kf];
                c_emit_abs_m5_cr_LO_cont_scaled[kf] =
                    m5_cr_result.c_LO_cont_scaled[kf];
                c_emit_abs_m5_cr_scaled[kf] =
                    c_emit_abs_m5_exact_scaled[kf]
                    + c_emit_abs_m5_cr_cont_scaled[kf];
                abs_c_emit_abs_m5_cr_scaled[kf] =
                    std::abs(c_emit_abs_m5_cr_scaled[kf]);
                dP_dE_Hartree_emit_abs_m5_cr[kf] =
                    std::norm(c_emit_abs_m5_cr_scaled[kf]);
                dP_dE_kHz_emit_abs_m5_cr[kf] =
                    dP_dE_Hartree_emit_abs_m5_cr[kf] * kHz_to_au(1.0);

                c_sum_ae_ea_cr_scaled[kf] =
                    c_abs_emit_m3_scaled[kf] + c_emit_abs_m5_cr_scaled[kf];
                abs_c_sum_ae_ea_cr_scaled[kf] =
                    std::abs(c_sum_ae_ea_cr_scaled[kf]);
                dP_dE_Hartree_sum_ae_ea_cr[kf] =
                    std::norm(c_sum_ae_ea_cr_scaled[kf]);
                dP_dE_kHz_sum_ae_ea_cr[kf] =
                    dP_dE_Hartree_sum_ae_ea_cr[kf] * kHz_to_au(1.0);
            }

            const double target_cont_norm =
                m5_eta2_sumrule_target - m5_bound_dnh2_sum;
            const double cr_norm_ratio =
                m5_cr_result.source_continuum_norm
                / std::max(std::abs(target_cont_norm), 1e-300);
            const std::complex<double> cr_LO_total_kf0 =
                c_emit_abs_m5_LO_bound_scaled[0]
                + c_emit_abs_m5_cr_LO_cont_scaled[0];
            const std::complex<double> cr_LO_required_kf0 =
                c_emit_abs_m5_green_scaled[0]
                - c_emit_abs_m5_LO_bound_scaled[0];
            const double rel_cr_LO_total =
                std::abs(cr_LO_total_kf0 - c_emit_abs_m5_green_scaled[0])
                / std::max(std::abs(c_emit_abs_m5_green_scaled[0]), 1e-300);
            const double rel_cr_cont_required =
                std::abs(c_emit_abs_m5_cr_LO_cont_scaled[0]
                         - cr_LO_required_kf0)
                / std::max(std::abs(cr_LO_required_kf0), 1e-300);
            const double rel_cr_vs_best =
                std::abs(c_emit_abs_m5_cr_scaled[0]
                         - c_emit_abs_m5_best_scaled[0])
                / std::max(std::abs(c_emit_abs_m5_best_scaled[0]), 1e-300);

            cout << std::scientific << std::setprecision(6);
            cout << "  complex-resolvent states: radial_N = "
                 << m5_cr_result.radial_N
                 << ", continuum_N = " << m5_cr_result.continuum_N << "\n";
            cout << "  source continuum norm (broadened) = "
                 << m5_cr_result.source_continuum_norm << "\n";
            cout << "  target continuum norm             = "
                 << target_cont_norm << "\n";
            cout << "  source norm ratio                 = "
                 << cr_norm_ratio
                 << "  (check R, eta, Emax convergence)\n";
            cout << "  [ea complex-resolvent kf=0]\n";
            cout << "    |c_CR_cont|            = "
                 << std::abs(c_emit_abs_m5_cr_cont_scaled[0]) << "\n";
            cout << "    |c_CR_total|           = "
                 << abs_c_emit_abs_m5_cr_scaled[0]
                 << "  (= exact_bound + CR continuum)\n";
            cout << "    |c_CR_LO_cont|         = "
                 << std::abs(c_emit_abs_m5_cr_LO_cont_scaled[0]) << "\n";
            cout << "    |LO continuum required|= "
                 << std::abs(cr_LO_required_kf0)
                 << "  (= green - LO_bound)\n";
            cout << "    relative CR_LO_cont vs required = "
                 << rel_cr_cont_required << "\n";
            cout << "    relative (LO_bound + CR_LO_cont) vs green = "
                 << rel_cr_LO_total << "\n";
            cout << "    relative CR_total vs best = "
                 << rel_cr_vs_best
                 << "  (small only after CR convergence checks pass)\n\n";
            cout << std::fixed;
        }
    }

    double P_grid_abs_emit_m3 = 0.0;
    double P_grid_emit_abs_m5_green = 0.0;
    double P_grid_sum_ae_ea = 0.0;
    double P_grid_emit_abs_m5_exact = 0.0;
    double P_grid_sum_ae_ea_exact = 0.0;
    double P_grid_emit_abs_m5_best = 0.0;
    double P_grid_sum_ae_ea_best = 0.0;
    double P_grid_emit_abs_m5_cr = 0.0;
    double P_grid_sum_ae_ea_cr = 0.0;
    double P_grid_emit_abs_m5_truly = 0.0;
    double P_grid_sum_ae_ea_truly = 0.0;
    for (int kf = 0; kf < N_Ef; ++kf) {
        P_grid_abs_emit_m3 +=
            dP_dE_Hartree_abs_emit_m3[kf] * Ef_weight_Hartree[kf];
        P_grid_emit_abs_m5_green +=
            dP_dE_Hartree_emit_abs_m5_green[kf] * Ef_weight_Hartree[kf];
        P_grid_sum_ae_ea +=
            dP_dE_Hartree_sum_ae_ea[kf] * Ef_weight_Hartree[kf];
        P_grid_emit_abs_m5_exact +=
            dP_dE_Hartree_emit_abs_m5_exact[kf] * Ef_weight_Hartree[kf];
        P_grid_sum_ae_ea_exact +=
            dP_dE_Hartree_sum_ae_ea_exact[kf] * Ef_weight_Hartree[kf];
        P_grid_emit_abs_m5_best +=
            dP_dE_Hartree_emit_abs_m5_best[kf] * Ef_weight_Hartree[kf];
        P_grid_sum_ae_ea_best +=
            dP_dE_Hartree_sum_ae_ea_best[kf] * Ef_weight_Hartree[kf];
        P_grid_emit_abs_m5_cr +=
            dP_dE_Hartree_emit_abs_m5_cr[kf] * Ef_weight_Hartree[kf];
        P_grid_sum_ae_ea_cr +=
            dP_dE_Hartree_sum_ae_ea_cr[kf] * Ef_weight_Hartree[kf];
        P_grid_emit_abs_m5_truly +=
            dP_dE_Hartree_emit_abs_m5_truly[kf] * Ef_weight_Hartree[kf];
        P_grid_sum_ae_ea_truly +=
            dP_dE_Hartree_sum_ae_ea_truly[kf] * Ef_weight_Hartree[kf];
    }
    cout << "\nIntegrated probabilities over final-state grid "
            "(log-grid Simpson/3-8 weights):\n";
    cout << "  P_abs_emit_m3                                  = " << std::scientific
         << P_grid_abs_emit_m3 << "\n";
    cout << "  P_emit_abs_m5_green (LO total)                 = "
         << P_grid_emit_abs_m5_green << "\n";
    cout << "  P_emit_abs_m5_exact_bound (incomplete)         = "
         << P_grid_emit_abs_m5_exact << "\n";
    cout << "  P_emit_abs_m5_best  (recommended: LO continuum + exact bound) = "
         << P_grid_emit_abs_m5_best << "\n";
    cout << "  P_emit_abs_m5_CR    (exact bound + complex-resolvent continuum) = "
         << P_grid_emit_abs_m5_cr << "\n";
    cout << "  P_emit_abs_m5_truly (exact bound + exact continuum) = "
         << P_grid_emit_abs_m5_truly << "\n";
    cout << "  P_sum_ae_ea  (with LO ea)                      = "
         << P_grid_sum_ae_ea << "\n";
    cout << "  P_sum_ae_ea_exact (with bound ea)              = "
         << P_grid_sum_ae_ea_exact << "\n";
    cout << "  P_sum_ae_ea_best  (recommended ea)             = "
         << P_grid_sum_ae_ea_best << "\n";
    cout << "  P_sum_ae_ea_CR    (with complex-resolvent ea)  = "
         << P_grid_sum_ae_ea_cr << "\n";
    cout << "  P_sum_ae_ea_truly (with truly-exact ea)        = "
         << P_grid_sum_ae_ea_truly
         << std::fixed << "\n\n";

    std::ofstream fout("output_m5_m3.txt");
    fout << "# B_RF_mG = " << B_rf_mG << "\n";
    fout << "# c_scaled = (mu_B B_RF)^2 c_kernel, with mu_B B_RF in Hartree\n";
    fout << "# c_emit_abs_m5_green uses the large-detuning closed-Green finite-pulse factor\n";
    fout << "# aliases: ae=absorb-emit M_F=-4->-3->-4, ea=emit-absorb M_F=-4->-5->-4, sum=ae+ea\n";
    fout << "# final-energy quadrature: log-grid Simpson/3-8; "
            "weighted_P columns equal dP_dE_Hartree * quadrature_weight_Hartree\n";
    fout << "# integrated_P_abs_emit_m3 = " << std::scientific
         << P_grid_abs_emit_m3 << "\n";
    fout << "# integrated_P_emit_abs_m5_green = "
         << P_grid_emit_abs_m5_green << "\n";
    fout << "# integrated_P_sum_ae_ea = " << P_grid_sum_ae_ea << "\n";
    fout << "# integrated_P_emit_abs_m5_exact (bound spectral sum, full Faddeeva) = "
         << P_grid_emit_abs_m5_exact << "\n";
    fout << "# integrated_P_sum_ae_ea_exact = "
         << P_grid_sum_ae_ea_exact << "\n";
    fout << "# integrated_P_emit_abs_m5_best (recommended: closed-Green + exact-bound Faddeeva correction) = "
         << P_grid_emit_abs_m5_best << "\n";
    fout << "# integrated_P_sum_ae_ea_best = "
         << P_grid_sum_ae_ea_best << "\n";
    fout << "# integrated_P_emit_abs_m5_CR (exact bound + complex-resolvent continuum) = "
         << P_grid_emit_abs_m5_cr << "\n";
    fout << "# integrated_P_sum_ae_ea_CR = "
         << P_grid_sum_ae_ea_cr << "\n";
    fout << "# M5_complex_resolvent_enabled = "
         << (m5_cr_result.enabled ? "yes" : "no") << "\n";
    if (m5_cr_result.enabled) {
        const double target_cont_norm =
            m5_eta2_sumrule_target - m5_bound_dnh2_sum;
        fout << "# M5_complex_resolvent_radial_N = "
             << m5_cr_result.radial_N << "\n";
        fout << "# M5_complex_resolvent_continuum_N = "
             << m5_cr_result.continuum_N << "\n";
        fout << "# M5_complex_resolvent_R_a0 = "
             << m5_cr_result.opt.R_a0 << "\n";
        fout << "# M5_complex_resolvent_dr_a0 = "
             << m5_cr_result.opt.dr_a0 << "\n";
        fout << "# M5_complex_resolvent_eta_kHz = "
             << m5_cr_result.opt.eta_kHz << "\n";
        fout << "# M5_complex_resolvent_source_continuum_norm = "
             << m5_cr_result.source_continuum_norm << "\n";
        fout << "# M5_complex_resolvent_target_continuum_norm = "
             << target_cont_norm << "\n";
        fout << "# M5_complex_resolvent_source_norm_ratio = "
             << m5_cr_result.source_continuum_norm
                / std::max(std::abs(target_cont_norm), 1e-300)
             << "\n";
    }
    fout << "# integrated_P_emit_abs_m5_truly (exact bound + exact continuum) = "
         << P_grid_emit_abs_m5_truly << "\n";
    fout << "# integrated_P_sum_ae_ea_truly = "
         << P_grid_sum_ae_ea_truly << "\n";
    fout << "# M5_continuum_norm_check = "
         << (m5_cont_norm_pass ? "PASS" : "FAIL")
         << "  ratio = " << m5_cont_norm_ratio
         << "  tolerance_abs_ratio_minus_1 = "
         << m5_cont_norm_pass_tol << "\n";
    fout << "# M5_continuum_continuum_finite_window_check = "
         << (m5_cc_window_pass ? "PASS" : "FAIL")
         << "  total_relative_error = "
         << m5_cc_window_total_rel_error
         << "  finite_window_over_required = "
         << m5_cc_window_required_ratio
         << "  tolerance_abs_total_relative_error = "
         << m5_cc_window_pass_tol << "\n";
    fout << "# recommended_ea_column = best\n";
    fout << "# truly_exact_status = "
         << (m5_cc_window_pass
             ? "LO_check_passed_still_verify_R_N_Emax_convergence"
             : "diagnostic_only") << "\n";
    fout << "# Notes:\n"
            "#   - green ea: leading-order closed-Green kernel; covers BOTH bound\n"
            "#     and continuum of M_F=-5 at LO in 1/(E_n + omega).\n"
            "#   - exact_bound ea: discrete bound-state sum with full Faddeeva\n"
            "#     w(z_n), z_n = (E_f + E_h - 2 E_n - 2 omega) tau0/(2 sqrt(2));\n"
            "#     INCOMPLETE because it omits the M_F=-5 continuum.\n"
            "#   - best ea: RECOMMENDED controlled result: green + (exact_bound - LO_bound),\n"
            "#     continuum piece LO.\n"
            "#   - CR ea: exact_bound plus the M_F=-5 continuum obtained from the\n"
            "#     broadened off-real-axis spectral density -Im <f|eta G(E+i eta) eta|h>/pi.\n"
            "#     Treat as a convergence diagnostic unless its LO continuum check agrees\n"
            "#     with green - LO_bound as eta/R/Emax are varied.\n"
            "#   - truly ea: c_ea_exact_bound + c_ea_cont_exact, both with the full\n"
            "#     Faddeeva. Kept for comparison as a finite-window continuum-continuum\n"
            "#     diagnostic; trust it only if the printed LO sum-rule diagnostics pass.\n";
    fout << "# E_f_kHz  M_emit_abs_m5_green_kernel  |M_emit_abs_m5_green_kernel|  "
            "M_emit_abs_m5_green_scaled  |M_emit_abs_m5_green_scaled|  "
            "Re(c_emit_abs_m5_green_scaled)  Im(c_emit_abs_m5_green_scaled)  "
            "|c_emit_abs_m5_green_scaled|  dP_dE_Hartree_emit_abs_m5_green  "
            "dP_dE_kHz_emit_abs_m5_green  "
            "Re(c_abs_emit_m3_kernel)  Im(c_abs_emit_m3_kernel)  "
            "|c_abs_emit_m3_kernel|  Re(c_abs_emit_m3_scaled)  "
            "Im(c_abs_emit_m3_scaled)  |c_abs_emit_m3_scaled|  "
            "dP_dE_Hartree_abs_emit_m3  dP_dE_kHz_abs_emit_m3  "
            "Re(c_ae_scaled)  Im(c_ae_scaled)  |c_ae_scaled|  "
            "dP_dE_Hartree_ae  dP_dE_kHz_ae  "
            "Re(c_ea_scaled)  Im(c_ea_scaled)  |c_ea_scaled|  "
            "dP_dE_Hartree_ea  dP_dE_kHz_ea  "
            "Re(c_sum_ae_ea_scaled)  Im(c_sum_ae_ea_scaled)  "
            "|c_sum_ae_ea_scaled|  dP_dE_Hartree_sum_ae_ea  "
            "dP_dE_kHz_sum_ae_ea  quadrature_weight_Hartree  "
            "quadrature_weight_kHz  weighted_P_ae  weighted_P_ea  "
            "weighted_P_sum_ae_ea  "
            "Re(c_emit_abs_m5_exact_scaled)  Im(c_emit_abs_m5_exact_scaled)  "
            "|c_emit_abs_m5_exact_scaled|  dP_dE_Hartree_emit_abs_m5_exact  "
            "dP_dE_kHz_emit_abs_m5_exact  "
            "Re(c_sum_ae_ea_exact_scaled)  Im(c_sum_ae_ea_exact_scaled)  "
            "|c_sum_ae_ea_exact_scaled|  dP_dE_Hartree_sum_ae_ea_exact  "
            "dP_dE_kHz_sum_ae_ea_exact  weighted_P_ea_exact  "
            "weighted_P_sum_ae_ea_exact  "
            "Re(c_emit_abs_m5_LO_bound_scaled)  Im(c_emit_abs_m5_LO_bound_scaled)  "
            "Re(c_emit_abs_m5_best_scaled)  Im(c_emit_abs_m5_best_scaled)  "
            "|c_emit_abs_m5_best_scaled|  dP_dE_Hartree_emit_abs_m5_best  "
            "dP_dE_kHz_emit_abs_m5_best  "
            "Re(c_sum_ae_ea_best_scaled)  Im(c_sum_ae_ea_best_scaled)  "
            "|c_sum_ae_ea_best_scaled|  dP_dE_Hartree_sum_ae_ea_best  "
            "dP_dE_kHz_sum_ae_ea_best  weighted_P_ea_best  "
            "weighted_P_sum_ae_ea_best  "
            "Re(c_emit_abs_m5_CR_cont_scaled)  Im(c_emit_abs_m5_CR_cont_scaled)  "
            "Re(c_emit_abs_m5_CR_LO_cont_scaled)  Im(c_emit_abs_m5_CR_LO_cont_scaled)  "
            "Re(c_emit_abs_m5_CR_scaled)  Im(c_emit_abs_m5_CR_scaled)  "
            "|c_emit_abs_m5_CR_scaled|  dP_dE_Hartree_emit_abs_m5_CR  "
            "dP_dE_kHz_emit_abs_m5_CR  "
            "Re(c_sum_ae_ea_CR_scaled)  Im(c_sum_ae_ea_CR_scaled)  "
            "|c_sum_ae_ea_CR_scaled|  dP_dE_Hartree_sum_ae_ea_CR  "
            "dP_dE_kHz_sum_ae_ea_CR  weighted_P_ea_CR  "
            "weighted_P_sum_ae_ea_CR  "
            "Re(c_emit_abs_m5_cont_exact_scaled)  Im(c_emit_abs_m5_cont_exact_scaled)  "
            "Re(c_emit_abs_m5_truly_scaled)  Im(c_emit_abs_m5_truly_scaled)  "
            "|c_emit_abs_m5_truly_scaled|  dP_dE_Hartree_emit_abs_m5_truly  "
            "dP_dE_kHz_emit_abs_m5_truly  "
            "Re(c_sum_ae_ea_truly_scaled)  Im(c_sum_ae_ea_truly_scaled)  "
            "|c_sum_ae_ea_truly_scaled|  dP_dE_Hartree_sum_ae_ea_truly  "
            "dP_dE_kHz_sum_ae_ea_truly  weighted_P_ea_truly  "
            "weighted_P_sum_ae_ea_truly\n";
    for (int kf = 0; kf < N_Ef; ++kf) {
        fout << std::scientific << std::setprecision(6)
             << Ef_kHz[kf] << "  "
             << M_emit_abs_m5_green_kernel[kf] << "  "
             << abs_M_emit_abs_m5_green_kernel[kf] << "  "
             << M_emit_abs_m5_green_scaled[kf] << "  "
             << abs_M_emit_abs_m5_green_scaled[kf] << "  "
             << c_emit_abs_m5_green_scaled[kf].real() << "  "
             << c_emit_abs_m5_green_scaled[kf].imag() << "  "
             << abs_c_emit_abs_m5_green_scaled[kf] << "  "
             << dP_dE_Hartree_emit_abs_m5_green[kf] << "  "
             << dP_dE_kHz_emit_abs_m5_green[kf] << "  "
             << c_abs_emit_m3_kernel[kf].real() << "  "
             << c_abs_emit_m3_kernel[kf].imag() << "  "
             << abs_c_abs_emit_m3_kernel[kf] << "  "
             << c_abs_emit_m3_scaled[kf].real() << "  "
             << c_abs_emit_m3_scaled[kf].imag() << "  "
             << abs_c_abs_emit_m3_scaled[kf] << "  "
             << dP_dE_Hartree_abs_emit_m3[kf] << "  "
             << dP_dE_kHz_abs_emit_m3[kf] << "  "
             << c_abs_emit_m3_scaled[kf].real() << "  "
             << c_abs_emit_m3_scaled[kf].imag() << "  "
             << abs_c_abs_emit_m3_scaled[kf] << "  "
             << dP_dE_Hartree_abs_emit_m3[kf] << "  "
             << dP_dE_kHz_abs_emit_m3[kf] << "  "
             << c_emit_abs_m5_green_scaled[kf].real() << "  "
             << c_emit_abs_m5_green_scaled[kf].imag() << "  "
             << abs_c_emit_abs_m5_green_scaled[kf] << "  "
             << dP_dE_Hartree_emit_abs_m5_green[kf] << "  "
             << dP_dE_kHz_emit_abs_m5_green[kf] << "  "
             << c_sum_ae_ea_scaled[kf].real() << "  "
             << c_sum_ae_ea_scaled[kf].imag() << "  "
             << abs_c_sum_ae_ea_scaled[kf] << "  "
             << dP_dE_Hartree_sum_ae_ea[kf] << "  "
             << dP_dE_kHz_sum_ae_ea[kf] << "  "
             << Ef_weight_Hartree[kf] << "  "
             << Ef_weight_Hartree[kf] * Hartree_in_kHz << "  "
             << dP_dE_Hartree_abs_emit_m3[kf] * Ef_weight_Hartree[kf] << "  "
             << dP_dE_Hartree_emit_abs_m5_green[kf] * Ef_weight_Hartree[kf] << "  "
             << dP_dE_Hartree_sum_ae_ea[kf] * Ef_weight_Hartree[kf] << "  "
             << c_emit_abs_m5_exact_scaled[kf].real() << "  "
             << c_emit_abs_m5_exact_scaled[kf].imag() << "  "
             << abs_c_emit_abs_m5_exact_scaled[kf] << "  "
             << dP_dE_Hartree_emit_abs_m5_exact[kf] << "  "
             << dP_dE_kHz_emit_abs_m5_exact[kf] << "  "
             << c_sum_ae_ea_exact_scaled[kf].real() << "  "
             << c_sum_ae_ea_exact_scaled[kf].imag() << "  "
             << abs_c_sum_ae_ea_exact_scaled[kf] << "  "
             << dP_dE_Hartree_sum_ae_ea_exact[kf] << "  "
             << dP_dE_kHz_sum_ae_ea_exact[kf] << "  "
             << dP_dE_Hartree_emit_abs_m5_exact[kf] * Ef_weight_Hartree[kf] << "  "
             << dP_dE_Hartree_sum_ae_ea_exact[kf] * Ef_weight_Hartree[kf] << "  "
             << c_emit_abs_m5_LO_bound_scaled[kf].real() << "  "
             << c_emit_abs_m5_LO_bound_scaled[kf].imag() << "  "
             << c_emit_abs_m5_best_scaled[kf].real() << "  "
             << c_emit_abs_m5_best_scaled[kf].imag() << "  "
             << abs_c_emit_abs_m5_best_scaled[kf] << "  "
             << dP_dE_Hartree_emit_abs_m5_best[kf] << "  "
             << dP_dE_kHz_emit_abs_m5_best[kf] << "  "
             << c_sum_ae_ea_best_scaled[kf].real() << "  "
             << c_sum_ae_ea_best_scaled[kf].imag() << "  "
             << abs_c_sum_ae_ea_best_scaled[kf] << "  "
             << dP_dE_Hartree_sum_ae_ea_best[kf] << "  "
             << dP_dE_kHz_sum_ae_ea_best[kf] << "  "
             << dP_dE_Hartree_emit_abs_m5_best[kf] * Ef_weight_Hartree[kf] << "  "
             << dP_dE_Hartree_sum_ae_ea_best[kf] * Ef_weight_Hartree[kf] << "  "
             << c_emit_abs_m5_cr_cont_scaled[kf].real() << "  "
             << c_emit_abs_m5_cr_cont_scaled[kf].imag() << "  "
             << c_emit_abs_m5_cr_LO_cont_scaled[kf].real() << "  "
             << c_emit_abs_m5_cr_LO_cont_scaled[kf].imag() << "  "
             << c_emit_abs_m5_cr_scaled[kf].real() << "  "
             << c_emit_abs_m5_cr_scaled[kf].imag() << "  "
             << abs_c_emit_abs_m5_cr_scaled[kf] << "  "
             << dP_dE_Hartree_emit_abs_m5_cr[kf] << "  "
             << dP_dE_kHz_emit_abs_m5_cr[kf] << "  "
             << c_sum_ae_ea_cr_scaled[kf].real() << "  "
             << c_sum_ae_ea_cr_scaled[kf].imag() << "  "
             << abs_c_sum_ae_ea_cr_scaled[kf] << "  "
             << dP_dE_Hartree_sum_ae_ea_cr[kf] << "  "
             << dP_dE_kHz_sum_ae_ea_cr[kf] << "  "
             << dP_dE_Hartree_emit_abs_m5_cr[kf] * Ef_weight_Hartree[kf] << "  "
             << dP_dE_Hartree_sum_ae_ea_cr[kf] * Ef_weight_Hartree[kf] << "  "
             << c_emit_abs_m5_cont_exact_scaled[kf].real() << "  "
             << c_emit_abs_m5_cont_exact_scaled[kf].imag() << "  "
             << c_emit_abs_m5_truly_scaled[kf].real() << "  "
             << c_emit_abs_m5_truly_scaled[kf].imag() << "  "
             << abs_c_emit_abs_m5_truly_scaled[kf] << "  "
             << dP_dE_Hartree_emit_abs_m5_truly[kf] << "  "
             << dP_dE_kHz_emit_abs_m5_truly[kf] << "  "
             << c_sum_ae_ea_truly_scaled[kf].real() << "  "
             << c_sum_ae_ea_truly_scaled[kf].imag() << "  "
             << abs_c_sum_ae_ea_truly_scaled[kf] << "  "
             << dP_dE_Hartree_sum_ae_ea_truly[kf] << "  "
             << dP_dE_kHz_sum_ae_ea_truly[kf] << "  "
             << dP_dE_Hartree_emit_abs_m5_truly[kf] * Ef_weight_Hartree[kf] << "  "
             << dP_dE_Hartree_sum_ae_ea_truly[kf] * Ef_weight_Hartree[kf] << "\n";
    }
    fout.close();

#if 0
    {
        std::ofstream manifest("tdpt_higher_order_manifest.txt");
        manifest << std::scientific << std::setprecision(12);
        manifest << "# TDPT run manifest\n";
        manifest << "# This file records which coherent sectors were generated in this run.\n";
        manifest << "# Atomic units are used internally; energies printed below are also shown in kHz where useful.\n\n";

        manifest << "initial_M_F " << p.MF_target << "\n";
        manifest << "initial_halo_energy_Hartree " << E_halo << "\n";
        manifest << "initial_halo_energy_kHz " << E_halo * Hartree_in_kHz << "\n";
        manifest << "omega_Hartree " << omega << "\n";
        manifest << "omega_kHz " << omega * Hartree_in_kHz << "\n";
        manifest << "tau0_au " << tau0 << "\n";
        manifest << "tau0_us " << tau0_us << "\n";
        manifest << "B_RF_mG " << B_rf_mG << "\n";
        manifest << "muB_BRF_Hartree " << rf_energy << "\n\n";
        manifest << "radial_grid_N " << p.N_grid << "\n";
        manifest << "radial_grid_dr_a0 " << p.dr << "\n";
        manifest << "overlap_grid_R_a0 " << overlap_R << "\n";
        manifest << "overlap_grid_dr_a0 " << overlap_dr << "\n\n";

        manifest << "files\n";
        manifest << "  formal_equations tdpt_higher_order_formalism.txt\n";
        manifest << "  finite_spectrum_contract tdpt_finite_spectrum_contract.txt\n";
        manifest << "  finite_spectrum_selftest tdpt_finite_tdpt_selftest.txt\n";
        manifest << "  finite_cache_nodes tdpt_cache_nodes.txt\n";
        manifest << "  finite_cache_halo_m3_couplings tdpt_cache_couplings_halo_m3.txt\n";
        manifest << "  finite_cache_manifest tdpt_cache_manifest.txt\n";
        manifest << "  finite_cache_order1_report tdpt_cache_order1_report.txt\n";
        manifest << "  finite_window_m4_m3_cc_sample tdpt_cache_couplings_m4_m3_finite_window_sample.txt\n";
        manifest << "  finite_window_m4_m3_cc_sample_manifest tdpt_cache_couplings_m4_m3_finite_window_sample_manifest.txt\n";
        manifest << "  finite_window_m4_m3_cc_probe tdpt_cache_couplings_m4_m3_window_probe.txt\n";
        manifest << "  abs_emit_m3_full_integral_window_probe tdpt_abs_emit_m3_window_probe.txt\n";
        manifest << "  odd_order1_m3_spectrum tdpt_first_order_odd_m3.txt\n";
        manifest << "  even_order2_m3_m5_spectrum output_m5_m3.txt\n\n";

        manifest << "coherent_sectors\n";
        if (std::isfinite(P_first_order_odd_m3)) {
            manifest << "  odd order 1 path -4->-3 integrated_probability "
                     << P_first_order_odd_m3
                     << " status exact_Gaussian_time_exact_square_well_bound_continuum_overlap\n";
        } else {
            manifest << "  odd order 1 path -4->-3 integrated_probability nan status not_written\n";
        }
        manifest << "  even order 2 path ae:-4->-3->-4 integrated_probability "
                 << P_grid_abs_emit_m3
                 << " status finite_En_quadrature_and_finite_radial_overlap_grid\n";
        manifest << "  even order 2 path ea:-4->-5->-4 integrated_probability "
                 << P_grid_emit_abs_m5_green
                 << " status closed_resolvent_with_large_detuning_time_kernel\n";
        manifest << "  even order 2 coherent_sum ae+ea integrated_probability "
                 << P_grid_sum_ae_ea
                 << " status coherent_sum_of_current_ae_and_ea_representations\n\n";

        manifest << "higher_order_rule\n";
        manifest << "  even_orders end in M_F=-4 and must be added coherently only with even_orders.\n";
        manifest << "  odd_orders end in neighboring M_F blocks and must be added coherently only within the same final block and spectral label.\n";
        manifest << "  exact continuum higher orders require continuum-continuum distribution kernels or a declared quadrature/box representation with weights and convergence checks.\n";
    }
#endif

    // ===================================================================
    // Machine-readable run summary (one line, key=value tokens)
    //
    // Designed to be grep'd by the sweep scripts:
    //     grep "^# SUMMARY" run.log
    // and parsed token-by-token (split on whitespace, then on '=').
    //
    // Numerical peak E_f comes from argmax of dP_dE_kHz_*, restricted to
    // E_f >= 100 Hz to avoid the Wigner low-energy tail (which is monotonic
    // and would otherwise dominate the argmax for ae's small-E behaviour).
    // ===================================================================
    auto find_peak_Ef_kHz = [&](const std::vector<double>& dP_dE_kHz) {
        // Wigner threshold law gives dP/dE ∝ E_f near 0, so the argmax
        // can't sit at the lowest grid point. A tiny 10 mHz floor guards
        // against numerical noise at the very lowest E_f only.
        const double Ef_floor_kHz = 1.0e-5;   // 10 mHz
        double best = -1.0; double Ef_peak = 0.0;
        for (int kf = 0; kf < N_Ef; ++kf) {
            if (Ef_kHz[kf] < Ef_floor_kHz) continue;
            if (dP_dE_kHz[kf] > best) {
                best = dP_dE_kHz[kf];
                Ef_peak = Ef_kHz[kf];
            }
        }
        return Ef_peak;
    };
    const double peak_ae_Ef_kHz   = find_peak_Ef_kHz(dP_dE_kHz_abs_emit_m3);
    const double peak_ea_Ef_kHz   = find_peak_Ef_kHz(dP_dE_kHz_emit_abs_m5_green);
    const double peak_sum_Ef_kHz  = find_peak_Ef_kHz(dP_dE_kHz_sum_ae_ea);
    const double peak_best_Ef_kHz = find_peak_Ef_kHz(dP_dE_kHz_sum_ae_ea_best);

    cout << std::scientific << std::setprecision(8);
    cout << "# SUMMARY"
         << "  omega_kHz="            << omega * Hartree_in_kHz
         << "  tau0_us="              << tau0_us
         << "  B_RF_mG="              << B_rf_mG
         << "  beta="                 << beta
         << "  E_b_kHz="              << E_b * Hartree_in_kHz
         << "  E_halo_kHz="           << E_halo * Hartree_in_kHz
         << "  V_T_GHz="              << p.V_T * Hartree_in_GHz
         << "  P_open="               << P0
         << "  P_closed="             << P1
         << "  P_ae="                 << P_grid_abs_emit_m3
         << "  P_ea="                 << P_grid_emit_abs_m5_green
         << "  P_sum="                << P_grid_sum_ae_ea
         << "  P_ea_best="            << P_grid_emit_abs_m5_best
         << "  P_sum_best="           << P_grid_sum_ae_ea_best
         << "  P_ea_truly="           << P_grid_emit_abs_m5_truly
         << "  P_sum_truly="          << P_grid_sum_ae_ea_truly
         << "  ratio_ea_over_ae="     << P_grid_emit_abs_m5_green
                                       / std::max(P_grid_abs_emit_m3, 1e-300)
         << "  peak_ae_Ef_kHz="       << peak_ae_Ef_kHz
         << "  peak_ea_Ef_kHz="       << peak_ea_Ef_kHz
         << "  peak_sum_Ef_kHz="      << peak_sum_Ef_kHz
         << "  peak_best_Ef_kHz="     << peak_best_Ef_kHz
         << "  E_int_m5_kHz="         << E_int_m5 * Hartree_in_kHz
         << "  S_norm="               << diag_nS
         << "  X_norm="               << diag_nX
         << "  X_over_S_au="          << diag_response
         << "  min_gap_GHz="          << diag_gap_min * Hartree_in_GHz
         << "  X_over_S_x_min_gap="   << diag_response * diag_gap_min
         << "  local_rel="            << diag_local_rel
         << "  residual="             << diag_residual
         << "\n";
    cout << std::fixed;

    return 0;
}
