// test_3d_harmonic_eigenvalue.cpp -- 3D isotropic harmonic oscillator
// V(r) = ½ r².  Analytic energies E_{n,l} = (2n + l + 3/2) ω with ω=1.
//
// Lowest s-wave bound state:  E_{0,0} = 3/2 = 1.500000.
// Smooth potential -> Numerov gets full 4th-order accuracy.  At
// dr = 0.005 we should match analytic to ~1e-5 relative.
#include "Angular.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"
#include "Equations.hpp"
#include "Eigenvalues.hpp"

#include <cstdio>

int main() {
    Parameters p;
    p.N_grid = 1601;     // r_max = 8 (well past classical turning ~2.4)
    p.dr     = 0.005;
    p.l_max  = 3;
    p.n_channels = ang3d::n_channels(p.l_max);
    p.Emin = 0.001;       // V > 0 everywhere -> bound states have E > 0
    p.Emax = 2.0;
    p.N_theta = 24; p.N_phi = 48;
    p.p = 9; p.external_parameter = 0;
    p.n_threads = 1; p.out_decimation = 1;

    Potentials pot(p);
    pot.set_potential("harmonic");
    pot.build();

    Equations eqs(pot, p);
    Eigenvalues eig(eqs, p);
    eig.groundstate_finder(/*desire_node*/1, /*tol*/1e-10);

    const double E_analytic = 1.5;
    const double E_numerov  = eig.gsEnergy;
    const double err = std::fabs(E_numerov - E_analytic);
    const double rel = err / E_analytic;

    std::printf("[3d_harmonic]  V(r) = ½ r²  (smooth)\n");
    std::printf("  E_analytic = %+.10f\n", E_analytic);
    std::printf("  E_numerov  = %+.10f\n", E_numerov);
    std::printf("  abs err    = %.3e\n",  err);
    std::printf("  rel err    = %.3e\n",  rel);

    // Smooth potential: O(dr⁴) accuracy.  At dr=0.005, expect rel err
    // ~ (5e-3)⁴ ~ 6e-10.  Allow 1e-5 to cover the heuristic node-count
    // bisection floor.
    const bool ok = (rel < 1e-5);
    std::printf("  result     : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
