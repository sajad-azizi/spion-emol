// test_potential_axisymmetric.cpp -- for any V(r,θ) with no φ-dep
// (e.g. spherical, isotropic Gaussian, harmonic, soft_coul, free), the
// V^R-matrix must be diagonal in m (couple only same-m channels).
//
// Why: V(r,θ) is invariant under φ-rotation, so it commutes with L_z.
// In the real-Y basis, that means V^R_{(l,m),(l',m')} ∝ δ_{m,m'} (and
// can mix l, l' freely subject to parity).
#include "Angular.hpp"
#include "Parameters.hpp"
#include "Potentials.hpp"

#include <cstdio>

namespace {
int run_one(const std::string& kind, double abs_tol) {
    Parameters p;
    p.N_grid = 7;
    p.dr     = 0.5;
    p.l_max  = 4;
    p.n_channels = ang3d::n_channels(p.l_max);
    p.Emin = -3.0; p.Emax = 0.5;
    p.N_theta = 32; p.N_phi = 64;
    p.p = 9; p.external_parameter = 150;
    p.n_threads = 1; p.out_decimation = 1;

    Potentials pot(p);
    pot.set_potential(kind);
    pot.build();

    double max_off_m = 0.0;
    for (int ir = 0; ir < p.N_grid; ++ir) {
        const Eigen::MatrixXcd& V = pot.VR(ir);
        for (int a = 0; a < p.n_channels; ++a) {
            int la, ma; ang3d::idx_to_lm(a, la, ma);
            for (int b = 0; b < p.n_channels; ++b) {
                int lb, mb; ang3d::idx_to_lm(b, lb, mb);
                if (ma == mb) continue;
                const double mag = std::abs(V(a, b));
                if (mag > max_off_m) max_off_m = mag;
            }
        }
    }
    const bool ok = (max_off_m < abs_tol);
    std::printf("[axisymmetric] %-12s  max|V_{m≠m'}| = %.3e  %s\n",
                kind.c_str(), max_off_m, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
}  // namespace

int main() {
    int n_fail = 0;
    n_fail += run_one("spherical",  1e-10);
    n_fail += run_one("gaussian",   1e-10);
    n_fail += run_one("harmonic",   1e-10);
    n_fail += run_one("soft_coul",  1e-10);
    n_fail += run_one("free",       1e-12);
    return (n_fail == 0) ? 0 : 1;
}
