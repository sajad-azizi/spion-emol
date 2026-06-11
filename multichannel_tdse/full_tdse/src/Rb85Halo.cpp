#include "Rb85Halo.hpp"

// Reference code (brings dcompx, AU::*, ::SpinAlgebra, ::Parameters,
// ::Potentials, ::AnalyticSquareWell, ::Equations, ::Eigenvalues at
// global scope).  Confined to this translation unit.
#include "AnalyticSquareWell.hpp"
#include "Eigenvalues.hpp"
#include "Equations.hpp"
#include "Potentials.hpp"
#include "SpinAlgebra.hpp"

#include <cmath>
#include <stdexcept>

namespace mc_tdse {

namespace {

// Build the canonical reference-code Parameters object from our
// Rb85HaloOptions.  The reference Parameters is a plain struct with
// many fields -- we set them explicitly so nothing depends on
// whatever default the reference happened to choose.
::Parameters make_ref_params(const Rb85HaloOptions& opt) {
    ::Parameters p;
    p.N_grid       = opt.N_grid_numerov;
    p.dr           = opt.dr_numerov;
    p.mu           = AU::mu_Rb85;
    p.r0           = opt.r0_a0;
    p.smooth_width = 0.0;
    p.B_gauss      = opt.B_gauss;
    p.MF_target    = -4;
    p.N_ch_keep    = opt.N_ch_keep;
    p.store_potential_grid = false;       // analytic doesn't need V on grid
    p.Nroots       = 1;       // unused
    p.NTHREADS     = 1;
    p.divide       = 1;
    p.p            = opt.p_init_numerov;
    p.Emin         = AU::kHz_to_au(opt.E_bracket_lo_kHz);
    p.Emax         = AU::kHz_to_au(opt.E_bracket_hi_kHz);
    p.external_parameter = 0;
    p.V_T          = AU::GHz_to_au(opt.V_T_GHz);
    p.V_S          = opt.V_S_over_T * p.V_T;
    return p;
}

// Look up the M_F=-4 entrance threshold in MHz (used as global zero).
double mf4_entrance_MHz(double B_gauss) {
    ::SpinAlgebra spin(B_gauss);
    auto chs = spin.channels(-4);
    if (chs.empty()) throw std::runtime_error("M_F=-4 has no channels");
    return chs.front().E_th_MHz;
}

// Composite Simpson's rule on a uniform grid of N_grid samples (one
// at each ir = 0..N_grid-1).  Falls back to trapezoid when N_grid==2.
template<typename F>
double simpson_uniform(int N_grid, double dr, F&& f) {
    if (N_grid < 2) return 0.0;
    if (N_grid == 2) return 0.5 * dr * (f(0) + f(1));
    if (N_grid % 2 == 1) {
        double s = f(0) + f(N_grid - 1);
        for (int ir = 1; ir < N_grid - 1; ++ir)
            s += (ir & 1 ? 4.0 : 2.0) * f(ir);
        return s * dr / 3.0;
    }
    // Mix Simpson 1/3 (first N_grid-3 points) + Simpson 3/8 (last 4 pts).
    double s13 = f(0) + f(N_grid - 4);
    for (int ir = 1; ir < N_grid - 4; ++ir)
        s13 += (ir & 1 ? 4.0 : 2.0) * f(ir);
    s13 *= dr / 3.0;
    const double s38 = (3.0 * dr / 8.0) *
        (f(N_grid - 4) + 3.0 * f(N_grid - 3)
       + 3.0 * f(N_grid - 2) + f(N_grid - 1));
    return s13 + s38;
}

}  // namespace

double halo_binding_analytic(const Rb85HaloOptions& opt) {
    auto p = make_ref_params(opt);
    ::SpinAlgebra spin(opt.B_gauss);
    const double E_ref_MHz = mf4_entrance_MHz(opt.B_gauss);
    ::Potentials pot(&p, &spin, E_ref_MHz);
    ::AnalyticSquareWell solver(&pot, &p);
    return solver.find_highest_bound_state(p.Emin, p.Emax, /*n_scan*/ 20000);
}

// ---- Numerov-stack validator ------------------------------------
// (See header for rationale.)  We do not use Numerov for the halo
// itself; the analytic state is exact for the model.  The stack is
// validated by checking that every analytic eigenvalue is registered
// as a node-count transition by Numerov.

std::vector<double> mf4_bound_states_analytic(const Rb85HaloOptions& opt,
                                              double E_lo_kHz,
                                              double E_hi_kHz) {
    auto p = make_ref_params(opt);
    p.Emin = AU::kHz_to_au(E_lo_kHz);
    p.Emax = AU::kHz_to_au(E_hi_kHz);
    ::SpinAlgebra spin(opt.B_gauss);
    const double E_ref_MHz = mf4_entrance_MHz(opt.B_gauss);
    ::Potentials pot(&p, &spin, E_ref_MHz);
    ::AnalyticSquareWell solver(&pot, &p);
    return solver.find_all_bound_states(p.Emin, p.Emax, /*n_scan*/ 20000);
}

int mf4_node_count_numerov(double E_au, const Rb85HaloOptions& opt) {
    auto p = make_ref_params(opt);
    p.store_potential_grid = true;
    ::SpinAlgebra spin(opt.B_gauss);
    const double E_ref_MHz = mf4_entrance_MHz(opt.B_gauss);
    ::Potentials pot(&p, &spin, E_ref_MHz);
    ::Equations eqs(&pot, &p);
    auto [nc, npos] = eqs.OutwardNodeCounting(E_au);
    return nc;
}

HaloWeights halo_weights_analytic(const Rb85HaloOptions& opt) {
    auto p = make_ref_params(opt);
    ::SpinAlgebra spin(opt.B_gauss);
    const double E_ref_MHz = mf4_entrance_MHz(opt.B_gauss);
    ::Potentials pot(&p, &spin, E_ref_MHz);
    ::AnalyticSquareWell solver(&pot, &p);
    const double E_h = solver.find_highest_bound_state(p.Emin, p.Emax, 20000);

    // Build the (un-normalized) analytic state and integrate |u_α|² on
    // a fine grid to compute channel weights.  The state's evaluate2()
    // returns the 2-component radial wavefunction at any r.
    ::AnalyticState state = solver.build_bound_state(E_h);

    // Integrate over [0, N_grid·dr] from the options.  This window
    // must extend several halo lengths (~1/κ ≈ 2050 a_0) past r_0 so
    // the closed-channel exponential tail is captured.
    const int    N  = opt.N_grid_numerov;
    const double dr = opt.dr_numerov;
    const double P0 = simpson_uniform(N, dr, [&](int ir) {
        const Eigen::Vector2d u = state.evaluate2(ir * dr);
        return u(0) * u(0);
    });
    const double P1 = simpson_uniform(N, dr, [&](int ir) {
        const Eigen::Vector2d u = state.evaluate2(ir * dr);
        return u(1) * u(1);
    });
    const double total = P0 + P1;
    HaloWeights w;
    w.P_open   = P0 / total;
    w.P_closed = P1 / total;
    return w;
}

}  // namespace mc_tdse
