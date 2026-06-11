// test_wavefunction_matching_continuity.cpp -- after computing the
// bound-state chi_lm(r) by gluing forward+backward sweeps at i_match,
// the channel vector must be continuous across i_match.
#include "Angular.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"
#include "Equations.hpp"
#include "Eigenvalues.hpp"
#include "Wavefunctions.hpp"

#include <cstdio>

int main() {
    Parameters p;
    p.N_grid = 4001;
    p.dr     = 0.005;
    p.l_max  = 3;
    p.n_channels = ang3d::n_channels(p.l_max);
    p.Emin = -1.5;
    p.Emax = -0.001;
    p.N_theta = 32; p.N_phi = 64;
    p.p = 9; p.external_parameter = 150;
    p.n_threads = 1; p.out_decimation = 1;

    Potentials pot(p);
    pot.set_potential("spherical");
    pot.build();
    Equations eqs(pot, p);
    Eigenvalues eig(eqs, p);
    eig.groundstate_finder();

    Wavefunctions wfs(eqs, p);
    wfs.calculate_eigenfunction(eig.gsEnergy, eig.i_match);
    wfs.Normalization(wfs.eigfunc);

    const int im = eig.i_match;
    if (im < 2 || im > p.N_grid - 3) {
        std::printf("[matching] i_match=%d out of range -- abort\n", im);
        return 1;
    }
    // Compare the magnitude of (chi[im+1]-chi[im-1])/2 against (forward
    // FD with chi[im+2]-chi[im]) -- the curve should be locally smooth
    // (no kink at the matching point); equivalent to comparing the
    // backward-side and forward-side derivatives at im.
    Eigen::VectorXcd der_back  = (wfs.eigfunc[im]   - wfs.eigfunc[im-1]) / p.dr;
    Eigen::VectorXcd der_fwd   = (wfs.eigfunc[im+1] - wfs.eigfunc[im])   / p.dr;

    double max_kink = (der_fwd - der_back).cwiseAbs().maxCoeff();
    double max_amp  = wfs.eigfunc[im].cwiseAbs().maxCoeff();
    double rel_kink = max_kink / std::max(max_amp / p.dr, 1e-30);

    std::printf("[matching]  i_match=%d  r=%.4f\n", im, im * p.dr);
    std::printf("  max|chi(i_match)| = %.3e\n", max_amp);
    std::printf("  max|der_fwd - der_back| = %.3e\n", max_kink);
    std::printf("  ratio (kink / typical chi'/dr scale) = %.3e\n", rel_kink);

    // Tolerance: the glue is exact in arithmetic up to numerical noise;
    // O(1e-8) relative is comfortable.
    const bool ok = (rel_kink < 1e-3);
    std::printf("  result : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
