// test_cube_c8f8.cpp -- cubic well sized to mimic the inner-cube of
// C₈F₈ for a one-electron model of the C₈F₈⁻ HOMO photodetachment.
// User-specified parameters: L = 2.94 a.u. (cross-pointed inner cube),
// V₀ = 0.75 a.u. (target HOMO IP ~ 1.99 eV ≈ 0.073 Hartree).
//
// Reports: ground-state energy, channel composition (A_1g of O_h),
// and norm.  No reference value comparison (V₀ is a tunable parameter
// chosen to land near the C₈F₈⁻ EA).
#include "Angular.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"
#include "Equations.hpp"
#include "Eigenvalues.hpp"
#include "Wavefunctions.hpp"

#include <cstdio>

int main() {
    Parameters p;
    p.N_grid = 4001;        // r_max = 40 a.u. (well past box edge L/2 = 1.47)
    p.dr     = 0.01;
    p.l_max  = 6;           // (l_max+1)² = 49 channels
    p.n_channels = ang3d::n_channels(p.l_max);
    p.Emin = -0.75;         // bottom of the well
    p.Emax = -0.001;
    p.N_theta = 32; p.N_phi = 64;
    p.p = 9; p.external_parameter = 0;
    p.n_threads = 1; p.out_decimation = 1;

    const double V0 = 0.75;
    const double L  = 2.94;

    Potentials pot(p);
    pot.set_V0(V0);
    pot.set_L(L);
    pot.set_potential("cubic");
    pot.build();

    Equations eqs(pot, p);
    Eigenvalues eig(eqs, p);
    eig.groundstate_finder(/*desire_node*/ 1, /*tol*/ 1e-10);

    Wavefunctions wfs(eqs, p);
    wfs.calculate_eigenfunction(eig.gsEnergy, eig.i_match);

    double norm_pre = 0.0;
    for (int idx = 0; idx < p.n_channels; ++idx) {
        for (int ir = 0; ir < p.N_grid; ++ir) {
            const double a = std::abs(wfs.eigfunc[ir](idx));
            norm_pre += a * a * p.dr;
        }
    }
    wfs.Normalization(wfs.eigfunc);
    double norm_post = 0.0;
    std::vector<double> w_lm(p.n_channels, 0.0);
    for (int idx = 0; idx < p.n_channels; ++idx) {
        for (int ir = 0; ir < p.N_grid; ++ir) {
            const double a = std::abs(wfs.eigfunc[ir](idx));
            w_lm[idx] += a * a * p.dr;
        }
        norm_post += w_lm[idx];
    }

    double w_even_m0 = 0.0, w_even_mNonzero = 0.0, w_odd = 0.0;
    for (int idx = 0; idx < p.n_channels; ++idx) {
        int l, m; ang3d::idx_to_lm(idx, l, m);
        if (l & 1)        w_odd          += w_lm[idx];
        else if (m == 0)  w_even_m0      += w_lm[idx];
        else              w_even_mNonzero+= w_lm[idx];
    }

    constexpr double Hartree_to_eV = 27.211386;

    std::printf("[cube_c8f8]  V₀ = %.3f a.u.   L = %.3f a.u.   (half-side %.3f)\n",
                V0, L, 0.5 * L);
    std::printf("  E_numerov = %+.10f a.u.   = %.4f eV\n",
                eig.gsEnergy, eig.gsEnergy * Hartree_to_eV);
    std::printf("  IP (= -E) = %+.10f a.u.   = %.4f eV\n",
                -eig.gsEnergy, -eig.gsEnergy * Hartree_to_eV);
    std::printf("  i_match   = %d   (r = %.4f a.u.)\n",
                eig.i_match, eig.i_match * p.dr);
    std::printf("  norm pre  = %.10f\n", norm_pre);
    std::printf("  norm post = %.10f\n", norm_post);
    std::printf("\n  channel composition (top-15 by weight):\n");

    std::vector<int> order(p.n_channels);
    for (int i = 0; i < p.n_channels; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return w_lm[a] > w_lm[b]; });
    for (int k = 0; k < std::min(15, p.n_channels); ++k) {
        const int idx = order[k];
        int l, m; ang3d::idx_to_lm(idx, l, m);
        if (w_lm[idx] < 1e-10) break;
        std::printf("    (l=%d, m=%+d)  weight = %.6f\n", l, m, w_lm[idx]);
    }
    std::printf("\n  Σ weight  even-l, m=0       : %.6f\n", w_even_m0);
    std::printf("  Σ weight  even-l, m≠0       : %.6f\n", w_even_mNonzero);
    std::printf("  Σ weight  odd-l (forbidden) : %.3e\n", w_odd);

    const bool ok_E    = eig.gsEnergy < 0.0;
    const bool ok_norm = std::fabs(norm_post - 1.0) < 1e-12;
    const bool ok_par  = w_odd < 1e-6;
    const bool ok = ok_E && ok_norm && ok_par;
    std::printf("\n  result : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
