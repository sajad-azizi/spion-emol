#include "BlockEigenstates.hpp"

#include "Rb85Spin.hpp"

#include "AnalyticSquareWell.hpp"
#include "Equations.hpp"
#include "Eigenvalues.hpp"
#include "Wavefunctions.hpp"
#include "Potentials.hpp"
#include "SpinAlgebra.hpp"

#include <cmath>
#include <stdexcept>

#include <chrono>
#include <cstdio>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace mc_tdse {

namespace {

::Parameters make_ref_params(const BlockBuildOptions& opt, int MF, int N_ch) {
    ::Parameters p;
    p.N_grid       = opt.N_grid;
    p.dr           = opt.dr_a0;
    p.mu           = AU::mu_Rb85;
    p.r0           = opt.r0_a0;
    p.smooth_width = 0.0;
    p.B_gauss      = opt.B_gauss;
    p.MF_target    = MF;
    p.N_ch_keep    = (opt.N_ch_keep > 0 ? opt.N_ch_keep : N_ch);
    p.store_potential_grid = true;        // Numerov needs V on grid
    p.Nroots       = 1;
    p.NTHREADS     = 1;
    p.divide       = 1;
    p.p            = opt.p_init;
    p.external_parameter = 0;
    p.V_T          = AU::GHz_to_au(opt.V_T_GHz);
    p.V_S          = opt.V_S_over_T * p.V_T;
    return p;
}

double mf4_entrance_MHz(double B_gauss) {
    ::SpinAlgebra spin(B_gauss);
    auto chs = spin.channels(-4);
    if (chs.empty()) throw std::runtime_error("M_F=-4 has no channels");
    return chs.front().E_th_MHz;
}

// Normalize an eigenstate u (N_grid × N_ch) to Σ_c ∫|u|² dr = 1 by
// composite Simpson on the uniform grid (or trapezoid fallback).
void normalize_unit_integral(Eigen::MatrixXd& u, double dr) {
    const int N = u.rows();
    double sum = 0.0;
    for (int ir = 0; ir < N; ++ir) {
        double w = 1.0;
        if (ir == 0 || ir == N - 1)            w = 0.5;          // trapezoid
        for (int c = 0; c < u.cols(); ++c)
            sum += w * u(ir, c) * u(ir, c);
    }
    sum *= dr;
    if (sum <= 0.0)
        throw std::runtime_error("normalize_unit_integral: zero norm");
    const double s = std::sqrt(sum);
    u /= s;
}

}  // namespace

int block_node_count(int M_F, double E_au, const BlockBuildOptions& opt) {
    ::SpinAlgebra spin(opt.B_gauss);
    auto chs = spin.channels(M_F);
    if (chs.empty()) throw std::runtime_error("block_node_count: empty M_F");
    const int N_ch = (opt.N_ch_keep > 0 ? opt.N_ch_keep
                                        : static_cast<int>(chs.size()));
    auto p = make_ref_params(opt, M_F, N_ch);
    const double E_ref_MHz = mf4_entrance_MHz(opt.B_gauss);
    ::Potentials pot(&p, &spin, E_ref_MHz);
    ::Equations  eqs(&pot, &p);
    auto [n, pos] = eqs.OutwardNodeCounting(E_au);
    return n;
}

BlockEigenstates build_block_eigenstates(int M_F,
                                         const BlockBuildOptions& opt) {
    ::SpinAlgebra spin(opt.B_gauss);
    auto chs = spin.channels(M_F);
    if (chs.empty())
        throw std::runtime_error("build_block_eigenstates: no channels for M_F");
    const int N_ch = (opt.N_ch_keep > 0 ? opt.N_ch_keep
                                        : static_cast<int>(chs.size()));

    BlockEigenstates out;
    out.M_F    = M_F;
    out.N_ch   = N_ch;
    out.N_grid = opt.N_grid;
    out.dr     = opt.dr_a0;

    auto p = make_ref_params(opt, M_F, N_ch);
    const double E_ref_MHz = mf4_entrance_MHz(opt.B_gauss);

    // ---- Halo route (M_F = -4 + analytic flag) ---------------------
    // For M_F=-4 with use_analytic_halo, we PREPEND the analytic halo
    // and let the normal Numerov box-quantization route fill in the
    // continuum above the entrance threshold.  The two methods cover
    // complementary regions:
    //   * Analytic halo : single near-threshold bound state at -E_b,
    //                     where Numerov can't resolve 1-part-in-10^6.
    //   * Numerov scan  : continuum levels above the entrance threshold.
    if (M_F == -4 && opt.use_analytic_halo) {
        ::Potentials pot(&p, &spin, E_ref_MHz);
        ::AnalyticSquareWell solver(&pot, &p);
        const double E_lo = AU::kHz_to_au(-1000.0);
        const double E_hi = AU::kHz_to_au(-0.001);
        const double E_h  = solver.find_highest_bound_state(E_lo, E_hi, 20000);
        ::AnalyticState st = solver.build_bound_state(E_h);

        Eigen::MatrixXd u(opt.N_grid, N_ch);
        for (int ir = 0; ir < opt.N_grid; ++ir) {
            const double r = ir * opt.dr_a0;
            const Eigen::Vector2d v = st.evaluate2(r);
            for (int c = 0; c < N_ch; ++c) u(ir, c) = v(c);
        }
        normalize_unit_integral(u, opt.dr_a0);

        out.E_au.push_back(E_h);
        out.u.push_back(std::move(u));
        // FALL THROUGH to the continuum scan below.  The scan starts
        // just above the M_F=-4 entrance threshold (recipe origin = 0),
        // so it never tries to bracket the halo (Numerov can't resolve
        // it anyway).
    }

    // ---- Numerov box-quantization route ---------------------------
    // For every other block (and M_F=-4 if use_analytic_halo=false),
    // build levels by node-count bisection above the block's lowest
    // threshold and reconstruct via calculate_eigenfunction_continuum.
    //
    // OutwardNodeCounting(E) increments by one at each box-quantized
    // open-channel level.  For an energy E_lo just below the block
    // threshold and E_hi at threshold + E_max_kHz_above_threshold, the
    // sequence nc(E_lo), nc(E_lo + dE), ... , nc(E_hi) increases by an
    // integer at each level; we bisect each transition to recover the
    // level energy, then use Wavefunctions::calculate_eigenfunction_continuum.
    ::Potentials pot(&p, &spin, E_ref_MHz);
    ::Equations  eqs(&pot, &p);
    ::Wavefunctions wfn(&eqs, &p);

    // Block thresholds in atomic units (recipe origin).
    // CAUTION: ::SpinAlgebra::channels(MF) returns ABSOLUTE thresholds;
    // we need them shifted to the recipe origin (M_F=-4 entrance).  Use
    // Rb85Spin which already does this normalization.
    Rb85Spin spin_bridge(opt.B_gauss);
    auto chs_recipe = spin_bridge.channels(M_F);
    const double E_th_lo_au = AU::MHz_to_au(chs_recipe.front().E_th_MHz);

    double E_lo_au, E_max_au;
    const bool absolute_window =
        (opt.E_window_kHz_lo != 0.0 || opt.E_window_kHz_hi != 0.0)
        && (opt.E_window_kHz_hi > opt.E_window_kHz_lo);
    if (absolute_window) {
        E_lo_au  = AU::kHz_to_au(opt.E_window_kHz_lo);
        E_max_au = AU::kHz_to_au(opt.E_window_kHz_hi);
        // Clamp to be at or above the block's threshold (no states below).
        if (E_lo_au < E_th_lo_au - AU::kHz_to_au(0.1))
            E_lo_au = E_th_lo_au - AU::kHz_to_au(0.1);
    } else {
        E_max_au = E_th_lo_au + AU::kHz_to_au(opt.E_max_kHz_above_threshold);
        E_lo_au  = E_th_lo_au - AU::kHz_to_au(0.1);
    }

    // Coarse scan: precompute nc(E) at all sample points in PARALLEL.
    const double window_kHz = AU::au_to_kHz(E_max_au - E_lo_au);
    int n_scan = std::max(4000, 4 * static_cast<int>(window_kHz + 1));
    if (n_scan > 100000) n_scan = 100000;

    std::fprintf(stderr,
        "[BlockEigenstates] M_F=%+d  scan window [E_th + %+.1f, E_th + %+.1f] kHz "
        "(width %.1f kHz)  n_scan=%d  N_grid=%d  N_ch=%d\n",
        M_F,
        AU::au_to_kHz(E_lo_au - E_th_lo_au),
        AU::au_to_kHz(E_max_au - E_th_lo_au),
        window_kHz, n_scan, opt.N_grid, N_ch);
    std::fprintf(stderr, "[BlockEigenstates] M_F=%+d  starting scan ...\n", M_F);
    auto t_scan0 = std::chrono::steady_clock::now();
    const double dE_scan = (E_max_au - E_lo_au) / n_scan;
    std::vector<int> nc_arr(n_scan + 1, 0);

    #pragma omp parallel
    {
        ::Equations eqs_local(&pot, &p);
        #pragma omp for schedule(static)
        for (int s = 0; s <= n_scan; ++s) {
            const double E_s = E_lo_au + s * dE_scan;
            auto [n, pos] = eqs_local.OutwardNodeCounting(E_s);
            nc_arr[s] = n;
        }
    }

    // Sequential pass over the precomputed nc[] to find each (E_prev, E_curr)
    // bracket where nc transitions.  Collect all bisection tasks (one per
    // crossing) into a flat list so they can be done in parallel.
    struct BisectTask {
        double e_lo;
        double e_hi;
        int    target_n;
    };
    std::vector<BisectTask> tasks;
    int nc_prev = nc_arr[0];
    for (int s = 1; s <= n_scan; ++s) {
        const int nc_curr = nc_arr[s];
        if (nc_curr > nc_prev) {
            for (int t = nc_prev + 1; t <= nc_curr; ++t) {
                tasks.push_back({E_lo_au + (s - 1) * dE_scan,
                                 E_lo_au + s * dE_scan, t});
            }
        }
        nc_prev = nc_curr;
    }

    auto dt_scan = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_scan0).count();
    const int n_lvl = static_cast<int>(tasks.size());
    std::fprintf(stderr,
        "[BlockEigenstates] M_F=%+d  scan done in %.1f s   found %d levels\n",
        M_F, dt_scan, n_lvl);
    // Memory footprint estimate before allocating.
    const double per_state_MB = static_cast<double>(opt.N_grid) * N_ch
                              * sizeof(double) / (1024.0 * 1024.0);
    const double total_GB = n_lvl * per_state_MB / 1024.0;
    std::fprintf(stderr,
        "[BlockEigenstates] M_F=%+d  storing %d × %.2f MB = %.2f GB\n",
        M_F, n_lvl, per_state_MB, total_GB);
    if (total_GB > 200.0) {
        std::fprintf(stderr,
            "[BlockEigenstates] WARNING: M_F=%+d state storage %.0f GB; check node RAM.\n",
            M_F, total_GB);
    }

    std::vector<double> E_levels(n_lvl, 0.0);

    // Bisect each task in parallel; each thread has its own Equations.
    auto t_bis0 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[BlockEigenstates] M_F=%+d  bisecting %d levels ...\n", M_F, n_lvl);
    #pragma omp parallel
    {
        ::Equations eqs_local(&pot, &p);
        #pragma omp for schedule(dynamic, 1)
        for (int j = 0; j < n_lvl; ++j) {
            double e_lo = tasks[j].e_lo;
            double e_hi = tasks[j].e_hi;
            const int target_n = tasks[j].target_n;
            for (int it = 0; it < 70; ++it) {
                const double e_mid = 0.5 * (e_lo + e_hi);
                auto [n_mid, pos] = eqs_local.OutwardNodeCounting(e_mid);
                if (n_mid < target_n) e_lo = e_mid;
                else                  e_hi = e_mid;
                if (e_hi - e_lo < 1e-22) break;
            }
            E_levels[j] = 0.5 * (e_lo + e_hi);
        }
    }
    auto dt_bis = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_bis0).count();
    std::fprintf(stderr, "[BlockEigenstates] M_F=%+d  bisection done in %.1f s\n",
                 M_F, dt_bis);

    // Reconstruct each eigenfunction in parallel.  Wavefunctions::
    // calculate_eigenfunction_continuum requires its own Equations
    // (forward propagator with save=true) and its own Wavefunctions
    // workspace; allocate one per thread.
    auto t_rec0 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "[BlockEigenstates] M_F=%+d  reconstructing %d wavefunctions ...\n",
        M_F, n_lvl);
    std::vector<Eigen::MatrixXd> u_levels(n_lvl);
    #pragma omp parallel
    {
        ::Equations eqs_local(&pot, &p);
        ::Wavefunctions wfn_local(&eqs_local, &p);
        #pragma omp for schedule(dynamic, 1)
        for (int j = 0; j < n_lvl; ++j) {
            wfn_local.calculate_eigenfunction_continuum(E_levels[j]);
            wfn_local.Normalization();
            Eigen::MatrixXd u(opt.N_grid, N_ch);
            for (int ir = 0; ir < opt.N_grid; ++ir)
                for (int c = 0; c < N_ch; ++c)
                    u(ir, c) = wfn_local.eigfunc[ir](c).real();
            u_levels[j] = std::move(u);
        }
    }
    auto dt_rec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_rec0).count();
    std::fprintf(stderr,
        "[BlockEigenstates] M_F=%+d  reconstruction done in %.1f s\n",
        M_F, dt_rec);

    // Move into output (preserve ascending energy order from tasks).
    for (int j = 0; j < n_lvl; ++j) {
        out.E_au.push_back(E_levels[j]);
        out.u.push_back(std::move(u_levels[j]));
    }
    (void)wfn;       // unused now that the loop is parallelized
    return out;
}

}  // namespace mc_tdse
