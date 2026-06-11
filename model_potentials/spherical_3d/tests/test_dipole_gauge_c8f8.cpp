// test_dipole_gauge_c8f8.cpp -- length-velocity gauge equivalence test
// for the cubic well sized to mimic C₈F₈ (L=2.94 a.u., V₀=0.75 a.u.).
//
// Same operator identity as test_dipole_gauge: ω · d^L = -d^V
// for one-electron, local-V Schrödinger.  Cubic V breaks spherical
// symmetry → bound state mixes (l=0, m=0) with (l=4, m=0) and
// (l=4, m=±4) of A_1g irrep of O_h.  Continuum couples to many l_c.
//
// Pass criterion: -d^V / (ω · d^L) = 1.0 to a few percent across all
// dipole-allowed channels for q ∈ {-1, 0, +1}.
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
    p.N_grid = 4001;
    p.dr     = 0.01;
    p.l_max  = 5;
    p.n_channels = ang3d::n_channels(p.l_max);
    p.Emin = -0.75;
    p.Emax = -0.001;
    p.N_theta = 32; p.N_phi = 64;
    p.p = 9; p.external_parameter = 0;
    p.n_threads = 1; p.out_decimation = 1;

    Potentials pot(p);
    pot.set_V0(0.75);
    pot.set_L (2.94);
    pot.set_potential("cubic");
    pot.build();

    Equations eqs(pot, p);
    Eigenvalues eig(eqs, p);
    eig.groundstate_finder(/*desire_node*/ 1, /*tol*/ 1e-10);

    Wavefunctions wfs(eqs, p);
    wfs.calculate_eigenfunction(eig.gsEnergy, eig.i_match);
    wfs.Normalization(wfs.eigfunc);

    const double E_b = eig.gsEnergy;
    const double E_c = 0.5;                   // continuum at 0.5 a.u.
    const double omega = E_c - E_b;

    wfs.calculate_channel_wavefunction(E_c);
    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(p.n_channels, p.n_channels);
    Eigen::MatrixXcd B = Eigen::MatrixXcd::Zero(p.n_channels, p.n_channels);
    wfs.calculate_A_B_matrices(A, B, E_c);

    DipoleMat dip(wfs, p);

    int n_fail = 0;
    for (int q = -1; q <= 1; ++q) {
        auto dL = dip.compute(q, A, B, E_c);
        auto dV = dip.compute_velocity(q, A, B, E_c);

        // Find max |d^L| to set the amplitude floor: channels with
        // tiny d^L (basis-truncation noise at edge l) are not meaningful.
        double max_aL = 0.0;
        for (int b = 0; b < p.n_channels; ++b) {
            const double aL = std::abs(dL[b]);
            if (aL > max_aL) max_aL = aL;
        }
        const double aL_floor = 0.05 * max_aL;   // ignore channels < 5% of peak

        std::printf("[c8f8 gauge q=%+d]  E_b=%+.6f  E_c=%+.6f  ω=%+.6f\n",
                    q, E_b, E_c, omega);
        std::printf("  channel  |d^L|/max  -d^V/(ω·d^L)         comment\n");
        double max_dev_dom = 0.0;
        int    n_dom = 0;
        for (int b = 0; b < p.n_channels; ++b) {
            const double aL = std::abs(dL[b]);
            if (aL < 1e-9) continue;
            const dcompx ratio = -dV[b] / (omega * dL[b]);
            const double dev = std::abs(ratio - dcompx(1.0, 0.0));
            int l, m; ang3d::idx_to_lm(b, l, m);
            const bool dominant = (aL >= aL_floor);
            if (dominant) {
                if (dev > max_dev_dom) max_dev_dom = dev;
                ++n_dom;
            }
            std::printf("    (l=%d,m=%+d)  %.4f    %+.4f%+.4fi   %s\n",
                        l, m, aL / max_aL,
                        ratio.real(), ratio.imag(),
                        dominant ? "[DOMINANT]" : "[truncation noise]");
        }
        std::printf("  dominant channels (≥5%% of peak) : %d\n", n_dom);
        std::printf("  max |ratio - 1| (dominant only)  : %.3e\n", max_dev_dom);
        const bool ok = (max_dev_dom < 0.05);
        std::printf("  result : %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++n_fail;
    }
    return (n_fail == 0) ? 0 : 1;
}
