// test_free_particle_K_zero.cpp -- with V == 0 the radial solver must
// reproduce the free-particle solution chi_l(r) = A·S_l(kr); B == 0,
// hence K = B A^{-1} = 0 and S = identity.
//
// We propagate forward and back-extract A,B; the test asserts that
// |B|_max is small and |A^*A - I|_max  (in the energy-normalized sense
// of K-matrix scattering) is small.
#include "Angular.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"
#include "Equations.hpp"
#include "Wavefunctions.hpp"

#include <cstdio>

int main() {
    Parameters p;
    p.N_grid = 3001;
    p.dr     = 0.01;
    p.l_max  = 3;
    p.n_channels = ang3d::n_channels(p.l_max);
    p.Emin = -3.0; p.Emax = 0.5;
    p.N_theta = 24; p.N_phi = 48;
    p.p = 9; p.external_parameter = 100;
    p.n_threads = 1; p.out_decimation = 1;

    Potentials pot(p);
    pot.set_potential("free");
    pot.build();
    Equations eqs(pot, p);

    const double E = 0.5;   // k = 1.0
    Wavefunctions wfs(eqs, p);
    wfs.calculate_channel_wavefunction(E);

    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(p.n_channels, p.n_channels);
    Eigen::MatrixXcd B = Eigen::MatrixXcd::Zero(p.n_channels, p.n_channels);
    wfs.calculate_A_B_matrices(A, B, E);

    Eigen::MatrixXcd K = B * A.inverse();
    Eigen::MatrixXcd In = Eigen::MatrixXcd::Identity(p.n_channels,
                                                     p.n_channels);
    Eigen::MatrixXcd S  = (In + I_unit * K) * (In - I_unit * K).inverse();
    const double Kmax  = K.cwiseAbs().maxCoeff();
    const double unit  = (S.adjoint() * S - In).cwiseAbs().maxCoeff();
    const double Imag  = (S - In).cwiseAbs().maxCoeff();

    std::printf("[free_particle]  E=%.3f  k=%.3f  l_max=%d\n",
                E, std::sqrt(2.0 * E), p.l_max);
    std::printf("  |K|_max          = %.3e\n", Kmax);
    std::printf("  |S^*S - I|_max   = %.3e\n", unit);
    std::printf("  |S - I|_max      = %.3e\n", Imag);

    // K should be ~ small.  Tolerance: with the heuristic getCoefficients
    // window the projection has finite-size noise ~ O(1/N).  3e-3 is
    // safe at N_grid=3001.
    const bool ok = (Kmax < 5e-3 && unit < 5e-3);
    std::printf("  result : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
