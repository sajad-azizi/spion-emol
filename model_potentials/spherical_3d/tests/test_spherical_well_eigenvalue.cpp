// test_spherical_well_eigenvalue.cpp -- 3D s-wave spherical well.
//
// Inside r < R0:   chi(r) = sin(K r),       K = sqrt(2(E + V0))
// Outside r > R0:  chi(r) = exp(-κ r),      κ = sqrt(-2 E)
// Continuity of chi'/chi at R0  =>  K cot(K R0) = -κ.
//
// Here V0 = 1.5, R0 = 1.5 (matches our default cubic-well shell).
// The lowest l=0 root of the transcendental equation is computed
// once below by bisection and compared to the Numerov solver.
#include "Angular.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"
#include "Equations.hpp"
#include "Eigenvalues.hpp"

#include <cstdio>

namespace {

double analytic_swave_E(double V0, double R0) {
    // Solve K cot(K R0) + κ = 0 for E in (-V0, 0).
    auto f = [&](double E) {
        const double K  = std::sqrt(2.0 * (E + V0));
        const double k2 = std::sqrt(-2.0 * E);
        return K / std::tan(K * R0) + k2;
    };
    double Elo = -V0 + 1e-6;
    double Ehi = -1e-6;
    double flo = f(Elo);
    double fhi = f(Ehi);
    // Bracket: the lowest s-wave root is where f transitions sign as
    // K R0 approaches the first π/2 value crossing from below.
    // f(Elo) = +∞ (K small, cot huge positive) ; f(Ehi) ≈ small.
    // We do dumb bisection assuming f is monotone in (Elo, Ehi) for
    // the lowest state.  If V0 R0² > π²/8, the lowest bound state
    // exists; for V0=1.5, R0=1.5 we have V0 R0² = 3.375 > 1.234 -- yes.
    if (flo * fhi >= 0.0) {
        // Walk Ehi down until we bracket.
        for (int it = 0; it < 80; ++it) {
            Ehi -= 0.02;
            fhi = f(Ehi);
            if (flo * fhi < 0.0) break;
        }
    }
    for (int it = 0; it < 200; ++it) {
        const double Em = 0.5 * (Elo + Ehi);
        const double fm = f(Em);
        if (flo * fm <= 0.0) { Ehi = Em; fhi = fm; }
        else                 { Elo = Em; flo = fm; }
        if (std::fabs(Ehi - Elo) < 1e-12) break;
    }
    return 0.5 * (Elo + Ehi);
}

}  // namespace

int main() {
    // Step potential: Numerov loses its 4th-order convergence at the
    // V-discontinuity at r = R0; the eigenvalue error scales as O(dr).
    // dr = 0.001 puts the step error at ~1e-3 relative, which is the
    // realistic accuracy for this potential.  See test_3d_harmonic_*
    // for the smooth-potential case where dr⁴ scaling holds.
    Parameters p;
    p.N_grid = 20001;
    p.dr     = 0.001;
    p.l_max  = 3;          // single irreducible s-wave but allow l>0 for coupling test
    p.n_channels = ang3d::n_channels(p.l_max);
    p.Emin = -1.5;
    p.Emax = -0.001;
    p.N_theta = 32; p.N_phi = 64;
    p.p = 9; p.external_parameter = 150;  // R0 = 1.5
    p.n_threads = 1; p.out_decimation = 1;

    const double V0 = 1.5;
    const double R0 = p.external_parameter * 0.01;

    Potentials pot(p);
    pot.set_V0(V0);   // override default; default V0_ moved to 0.75 for the C8F8 cube test
    pot.set_potential("spherical");
    pot.build();

    Equations eqs(pot, p);
    Eigenvalues eig(eqs, p);
    eig.groundstate_finder(/*desire_node*/1, /*tol*/1e-9);

    const double E_analytic = analytic_swave_E(V0, R0);
    const double E_numerov  = eig.gsEnergy;
    const double err        = std::fabs(E_numerov - E_analytic);
    const double rel        = err / std::fabs(E_analytic);

    std::printf("[spherical_well]  V0=%.2f  R0=%.2f\n", V0, R0);
    std::printf("  E_analytic = %+.10f\n", E_analytic);
    std::printf("  E_numerov  = %+.10f\n", E_numerov);
    std::printf("  abs err    = %.3e\n",  err);
    std::printf("  rel err    = %.3e\n",  rel);

    // Numerov on a STEP potential: the radial wavefunction's higher
    // derivatives are discontinuous at r = R0.  Empirically the
    // eigenvalue error scales as O(dr), giving ~1e-3 relative at
    // dr=0.001.  The smooth-potential HO test next door verifies that
    // Numerov reaches its full 4th-order accuracy on smooth V.
    const bool ok = (rel < 2e-3);
    std::printf("  result     : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
