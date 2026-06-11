// test_potential_hermiticity.cpp -- the V-matrix V^R_{(lm),(l'm')}(r)
// must be real-symmetric at every r (V is real, real-Y is real).
#include "Angular.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"

#include <cstdio>

namespace {
int run_one(const std::string& kind, double abs_tol) {
    Parameters p;
    p.N_grid = 11;
    p.dr     = 0.5;
    p.l_max  = 3;
    p.n_channels = ang3d::n_channels(p.l_max);
    p.Emin = -3.0; p.Emax = 0.5;
    p.N_theta = 32; p.N_phi = 64;
    p.p = 9; p.external_parameter = 150;
    p.n_threads = 1; p.out_decimation = 1;

    Potentials pot(p);
    pot.set_potential(kind);
    pot.build();

    double max_skew = 0.0;
    double max_imag = 0.0;
    for (int ir = 0; ir < p.N_grid; ++ir) {
        const Eigen::MatrixXcd& V = pot.VR(ir);
        Eigen::MatrixXd V_re = V.real();
        Eigen::MatrixXd V_im = V.imag();
        const double skew = (V_re - V_re.transpose()).cwiseAbs().maxCoeff();
        const double im   = V_im.cwiseAbs().maxCoeff();
        if (skew > max_skew) max_skew = skew;
        if (im   > max_imag) max_imag = im;
    }
    const bool ok = (max_skew < abs_tol && max_imag < abs_tol);
    std::printf("[hermiticity] %-12s  max|V-V^T|=%.2e  max|Im V|=%.2e  %s\n",
                kind.c_str(), max_skew, max_imag, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
}  // namespace

int main() {
    int n_fail = 0;
    n_fail += run_one("cubic",      1e-10);
    n_fail += run_one("spherical",  1e-10);
    n_fail += run_one("gaussian",   1e-10);
    n_fail += run_one("anis_gauss", 1e-10);
    n_fail += run_one("soft_coul",  1e-10);
    return (n_fail == 0) ? 0 : 1;
}
