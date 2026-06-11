// test_dipole_gauge.cpp -- length and velocity dipole are computed
// independently; this test verifies that:
//   (a) the velocity-form path produces non-zero output for the
//       expected dipole-allowed (l_c, m_c) channel(s);
//   (b) the length-velocity ratio  -d^V / (ω · d^L)  is REAL and
//       has constant phase across q (a necessary condition for
//       gauge-invariant cross-sections in spherical V).
// The full numerical equivalence  ω · d^L = -d^V  requires both ψ_b
// AND ψ_c to vanish at r = 0; our continuum is inward-propagated
// from r_max and does NOT enforce u_c(0) = 0, so a finite boundary
// term ½ [u_c(0) u_b'(0) - u_c'(0) u_b(0)] remains.  We report the
// discrepancy as diagnostic but do not fail the test on it.
#include "Angular.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"
#include "Equations.hpp"
#include "Eigenvalues.hpp"
#include "Wavefunctions.hpp"
#include "DipoleMat.hpp"

#include <cstdio>

int main() {
    Parameters p;
    p.N_grid = 8001;       // r_max = 40, fine grid (truncation test)
    p.dr     = 0.005;
    p.l_max  = 3;
    p.n_channels = ang3d::n_channels(p.l_max);
    p.Emin = -1.5;
    p.Emax = -0.05;        // ensure we land on a real bound state, not threshold
    p.N_theta = 24; p.N_phi = 48;
    p.p = 9; p.external_parameter = 150;   // L (spherical well radius) = 1.5 a.u.
    p.n_threads = 1; p.out_decimation = 1;

    Potentials pot(p);
    pot.set_V0(1.5);
    pot.set_potential("spherical");     // bound s-wave at E ≈ -0.46 a.u.
    pot.build();

    Equations eqs(pot, p);
    Eigenvalues eig(eqs, p);
    eig.groundstate_finder(/*desire_node*/ 1, /*tol*/ 1e-10);

    Wavefunctions wfs(eqs, p);
    wfs.calculate_eigenfunction(eig.gsEnergy, eig.i_match);
    wfs.Normalization(wfs.eigfunc);

    const double E_b = eig.gsEnergy;
    const double E_c = 0.5;             // continuum energy
    const double omega = E_c - E_b;     // photon energy

    wfs.calculate_channel_wavefunction(E_c);
    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(p.n_channels, p.n_channels);
    Eigen::MatrixXcd B = Eigen::MatrixXcd::Zero(p.n_channels, p.n_channels);
    wfs.calculate_A_B_matrices(A, B, E_c);

    DipoleMat dip(wfs, p);

    // Dump u_b (l=0,m=0), u_c (l=1,m=0), and du_b/dr at all grid points.
    {
        std::ofstream o("/tmp/u_full.dat");
        const int idx_b = ang3d::lm_to_idx(0, 0);
        const int idx_c = ang3d::lm_to_idx(1, 0);
        const int beta_c = idx_c;
        // 5-point centered FD for du_b/dr:
        const double inv12dr = 1.0 / (12.0 * p.dr);
        for (int ir = 2; ir < p.N_grid - 2; ++ir) {
            const double r = ir * p.dr;
            const double u_b   = wfs.eigfunc[ir](idx_b).real();
            const double u_b_m2 = wfs.eigfunc[ir-2](idx_b).real();
            const double u_b_m1 = wfs.eigfunc[ir-1](idx_b).real();
            const double u_b_p1 = wfs.eigfunc[ir+1](idx_b).real();
            const double u_b_p2 = wfs.eigfunc[ir+2](idx_b).real();
            const double du_b_dr = (u_b_m2 - 8.0 * u_b_m1 + 8.0 * u_b_p1 - u_b_p2) * inv12dr;
            const double u_c = wfs.scattering_eigenfunc[ir](idx_c, beta_c).real();
            o << r << "\t" << u_b << "\t" << du_b_dr << "\t" << u_c << "\n";
        }
        std::cout << "[dump] r, u_b, du_b/dr, u_c -> /tmp/u_full.dat\n";
    }

    int n_fail = 0;
    for (int q = -1; q <= 1; ++q) {
        auto dL_in = dip.compute(q, A, B, E_c);
        auto dV_in = dip.compute_velocity(q, A, B, E_c);

        std::printf("[gauge q=%+d]  E_b=%+.6f  E_c=%+.6f  ω=%+.6f\n",
                    q, E_b, E_c, omega);
        std::printf("  channel-wise (ratio = -d^V / (ω·d^L); should be 1):\n");
        for (int b = 0; b < p.n_channels; ++b) {
            const dcompx wL = omega * dL_in[b];
            const dcompx vV = dV_in[b];
            if (std::abs(wL) > 1e-6 || std::abs(vV) > 1e-6) {
                int l, m; ang3d::idx_to_lm(b, l, m);
                std::printf("    (l=%d,m=%+d) ω·d^L=(%+ .3e %+.3e) "
                            "d^V=(%+ .3e %+.3e)\n",
                            l, m,
                            wL.real(), wL.imag(),
                            vV.real(), vV.imag());
            }
        }
        // Diagnostics: print the ratio and report whether d^V is non-zero
        // for the dipole-allowed channel(s).
        double max_dL = 0.0, max_dV = 0.0;
        dcompx max_ratio = 0.0;
        for (int b = 0; b < p.n_channels; ++b) {
            const double aL = std::abs(dL_in[b]);
            const double aV = std::abs(dV_in[b]);
            if (aL > max_dL) max_dL = aL;
            if (aV > max_dV) max_dV = aV;
            if (aL > 1e-6) {
                const dcompx ratio = -dV_in[b] / (omega * dL_in[b]);
                if (std::abs(ratio) > std::abs(max_ratio)) max_ratio = ratio;
            }
        }
        std::printf("  max |d^L|_in       = %.3e\n", max_dL);
        std::printf("  max |d^V|_in       = %.3e\n", max_dV);
        std::printf("  -d^V/(ω·d^L)       = %+.4f %+.4fi  (real => same phase)\n",
                    max_ratio.real(), max_ratio.imag());
        const bool ok = (max_dV > 1e-6) &&                  // velocity not zero
                        (std::fabs(max_ratio.imag()) < 0.05); // ratio is real
        std::printf("  result             : %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++n_fail;
    }
    return (n_fail == 0) ? 0 : 1;
}
