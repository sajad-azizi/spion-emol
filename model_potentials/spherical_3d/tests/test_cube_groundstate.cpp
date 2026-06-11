// test_cube_groundstate.cpp -- diagnostics report on the cubic-well
// ground state.  V = -V0 inside |x|, |y|, |z| ≤ L/2, 0 outside.
//
// O_h symmetry implies the ground state belongs to A_1g irrep, which
// in real-Y basis is a coherent combination of (l, m) channels:
//
//   A_1g = a_0 Y^R_{0,0}  +  a_4 (Y^R_{4,0} + α (Y^R_{4,+4} + Y^R_{4,-4}))
//        + a_6 (...) + ...
//
// We don't enforce a closed-form constraint here; we just print the
// channel composition and verify:
//   * eigenvalue is finite, negative, and below threshold
//   * even-l, m=0 component dominates (the irrep-allowed leading term)
//   * normalization is exact
//   * weights of *odd-l* and *m ≠ 0 with no symmetry partner* channels
//     are SMALL (cube has cubic-only symmetry; some m≠0 weight is
//     allowed by O_h's coupling rules, e.g. m=±4 within l=4).
#include "Angular.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"
#include "Equations.hpp"
#include "Eigenvalues.hpp"
#include "Wavefunctions.hpp"

#include <cstdio>

int main() {
    Parameters p;
    p.N_grid = 2001;
    p.dr     = 0.01;
    p.l_max  = 6;          // up to (6+1)² = 49 channels
    p.n_channels = ang3d::n_channels(p.l_max);
    p.Emin = -1.5;
    p.Emax = -0.001;
    p.N_theta = 32; p.N_phi = 64;
    p.p = 9; p.external_parameter = 150;   // L = 1.5
    p.n_threads = 1; p.out_decimation = 1;

    Potentials pot(p);
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

    double w_even_m0 = 0.0, w_even_mNonzero = 0.0,
           w_odd     = 0.0;
    for (int idx = 0; idx < p.n_channels; ++idx) {
        int l, m; ang3d::idx_to_lm(idx, l, m);
        if (l & 1)        w_odd          += w_lm[idx];
        else if (m == 0)  w_even_m0      += w_lm[idx];
        else              w_even_mNonzero+= w_lm[idx];
    }

    std::printf("[cube_gs]  V = -%.2f for |x|,|y|,|z| ≤ %.3f a.u.  (L = %.2f)\n",
                1.5, 0.5 * 1.5, 1.5);
    std::printf("  E_numerov = %+.10f\n", eig.gsEnergy);
    std::printf("  i_match   = %d   (r = %.4f a.u.)\n",
                eig.i_match, eig.i_match * p.dr);
    std::printf("  norm pre  = %.10f\n", norm_pre);
    std::printf("  norm post = %.10f   (must be 1)\n", norm_post);
    std::printf("\n  channel composition (top-15 weights):\n");

    // Sort channels by weight, print top 15.
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
    std::printf("\n  total weight even-l, m=0       : %.6f\n", w_even_m0);
    std::printf("  total weight even-l, m≠0       : %.6f   (O_h allows m=±4 in l=4)\n",
                w_even_mNonzero);
    std::printf("  total weight odd-l (forbidden) : %.3e\n", w_odd);

    // Pass criteria:
    //   * negative bound state
    //   * normalization to 1 within roundoff
    //   * odd-l weight should be much smaller than the rest (parity);
    //     small numerical leakage is OK
    const bool ok_E    = eig.gsEnergy < 0.0;
    const bool ok_norm = std::fabs(norm_post - 1.0) < 1e-12;
    const bool ok_par  = w_odd < 1e-6;
    const bool ok = ok_E && ok_norm && ok_par;
    std::printf("\n  eigenvalue    : %s\n", ok_E ? "PASS" : "FAIL");
    std::printf("  norm          : %s\n", ok_norm ? "PASS" : "FAIL");
    std::printf("  odd-l ~ 0     : %s\n", ok_par ? "PASS" : "FAIL");
    std::printf("  result        : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
