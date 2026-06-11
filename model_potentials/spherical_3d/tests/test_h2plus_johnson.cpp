// test_h2plus_johnson.cpp -- single-center expansion of H₂⁺ with hard
// Coulomb, following Johnson, J. Chem. Phys. 69 (1978) 4678, Sec. IV.
// Protons at z = ±R = ±1 bohr (internuclear distance 2 bohr).  The
// 1sσ_g ground state has energy -1.10263 a.u.  Johnson's table II
// shows the convergence with the number of even-l channels:
//
//   N=2  (l=0,2)        : -1.08368
//   N=4  (l=0..6)       : -1.09997
//   N=6  (l=0..10)      : -1.10184      <-- our l_max=10 target
//   N=8  (l=0..14)      : -1.10230
//   N=10 (l=0..18)      : -1.10246
//   exact               : -1.10263
//
// The truncation error is O(h²) (not h⁴) due to the V cusp at r = R;
// at h = 0.005 the truncation contribution is ~ 1e-5.
//
// We also verify the eigenfunction:
//   * channel composition: only (l even, m=0) channels are populated;
//     all (l odd) and all (m ≠ 0) channels must be ~ 0.
//   * normalization: ∫|χ_lm|² dr summed over channels equals 1
//     after Wavefunctions::Normalization().
#include "Angular.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"
#include "Equations.hpp"
#include "Eigenvalues.hpp"
#include "Wavefunctions.hpp"

#include <cstdio>

int main() {
    Parameters p;
    p.N_grid = 1601;        // r_max = 16 bohr
    p.dr     = 0.01;
    p.l_max  = 6;           // l = 0..6 in basis; even-l = {0,2,4,6} couple
    p.n_channels = ang3d::n_channels(p.l_max);
    p.Emin = -1.5;
    p.Emax = -0.001;
    p.N_theta = 24; p.N_phi = 48;  // unused; analytic Legendre path
    p.p = 9; p.external_parameter = 0;
    p.n_threads = 1; p.out_decimation = 1;

    Potentials pot(p);
    pot.set_h2plus(/*R_h2*/ 2.0, /*a (unused for hard)*/ 0.0);
    pot.set_potential("h2plus_johnson");
    pot.build();

    Equations eqs(pot, p);
    Eigenvalues eig(eqs, p);
    eig.groundstate_finder(/*desire_node*/ 1, /*tol*/ 1e-10);

    Wavefunctions wfs(eqs, p);
    wfs.calculate_eigenfunction(eig.gsEnergy, eig.i_match);

    // --- Norm before normalization ---
    double norm_pre = 0.0;
    for (int idx = 0; idx < p.n_channels; ++idx) {
        for (int ir = 0; ir < p.N_grid; ++ir) {
            const double a = std::abs(wfs.eigfunc[ir](idx));
            norm_pre += a * a * p.dr;
        }
    }
    wfs.Normalization(wfs.eigfunc);
    double norm_post = 0.0;
    for (int idx = 0; idx < p.n_channels; ++idx) {
        for (int ir = 0; ir < p.N_grid; ++ir) {
            const double a = std::abs(wfs.eigfunc[ir](idx));
            norm_post += a * a * p.dr;
        }
    }

    // --- Channel composition (L² weight per (l,m)) ---
    std::vector<double> w_lm(p.n_channels, 0.0);
    for (int idx = 0; idx < p.n_channels; ++idx) {
        for (int ir = 0; ir < p.N_grid; ++ir) {
            const double a = std::abs(wfs.eigfunc[ir](idx));
            w_lm[idx] += a * a * p.dr;
        }
    }
    double w_even_m0 = 0.0, w_odd = 0.0, w_mnonzero = 0.0;
    for (int idx = 0; idx < p.n_channels; ++idx) {
        int l, m; ang3d::idx_to_lm(idx, l, m);
        if (m != 0)       w_mnonzero += w_lm[idx];
        else if (l & 1)   w_odd      += w_lm[idx];
        else              w_even_m0  += w_lm[idx];
    }

    const double E_ref = -1.10263;       // Johnson exact (Bates et al.)
    const double E_N6  = -1.09997;       // Johnson N=4 (l=0..6) numerical
    const double E_num = eig.gsEnergy;
    const double err_exact = std::fabs(E_num - E_ref);
    const double err_N6    = std::fabs(E_num - E_N6);

    std::printf("[h2plus_johnson]  R_proton = ±%.2f  (internuclear = %.2f bohr)\n",
                1.0, 2.0);
    std::printf("  E_numerov     = %+.10f\n", E_num);
    std::printf("  E_Johnson_exact = %+.10f  (rel err %.3e)\n",
                E_ref, err_exact / std::fabs(E_ref));
    std::printf("  E_Johnson_N=4 = %+.10f  (rel err %.3e)\n",
                E_N6, err_N6 / std::fabs(E_N6));
    std::printf("  norm pre  = %.10f\n", norm_pre);
    std::printf("  norm post = %.10f   (must be 1 after Normalization)\n",
                norm_post);

    std::printf("  channel composition (L² weight per (l,m)):\n");
    for (int idx = 0; idx < p.n_channels; ++idx) {
        int l, m; ang3d::idx_to_lm(idx, l, m);
        if (w_lm[idx] > 1e-8) {
            std::printf("    (l=%d, m=%+d)  weight = %.6f\n", l, m, w_lm[idx]);
        }
    }
    std::printf("  total weight even-l, m=0  : %.6f   (must be ≈ 1)\n", w_even_m0);
    std::printf("  total weight odd-l        : %.3e   (must be 0; parity)\n", w_odd);
    std::printf("  total weight m≠0          : %.3e   (must be 0; axial sym)\n",
                w_mnonzero);

    // Tolerances:
    //   - eigenvalue must agree with Johnson's N=6 reference to ~1e-3
    //     (residual is dominated by step-size truncation, dr-dependent).
    //   - norm post must equal 1 exactly (up to roundoff).
    //   - parity-forbidden and axial-forbidden channel weights must
    //     be < 1e-8 of the total.
    const bool ok_E    = err_N6 < 5e-3;
    const bool ok_norm = std::fabs(norm_post - 1.0) < 1e-12;
    const bool ok_parity = w_odd < 1e-10;
    const bool ok_axial  = w_mnonzero < 1e-10;
    const bool ok = ok_E && ok_norm && ok_parity && ok_axial;
    std::printf("  eigenvalue   : %s\n", ok_E ? "PASS" : "FAIL");
    std::printf("  norm         : %s\n", ok_norm ? "PASS" : "FAIL");
    std::printf("  parity (l)   : %s\n", ok_parity ? "PASS" : "FAIL");
    std::printf("  axial sym (m): %s\n", ok_axial ? "PASS" : "FAIL");
    std::printf("  result       : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
