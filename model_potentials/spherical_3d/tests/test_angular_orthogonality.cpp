// test_angular_orthogonality.cpp -- numerical confirmation that the
// real-Y^R basis Y^R_{l,m}(θ,φ) is orthonormal on the sphere up to
// l_max = 6, on the same Gauss-Legendre × trapezoid grid the V-matrix
// builder uses.
#include "Angular.hpp"
#include <cstdio>

int main() {
    constexpr int Lmax  = 6;
    constexpr int Nth   = 32;
    constexpr int Nph   = 64;
    constexpr double abs_tol = 1e-12;

    std::vector<double> ct, wct;
    ang3d::gauss_legendre(Nth, ct, wct);
    std::vector<double> theta(Nth);
    for (int i = 0; i < Nth; ++i) theta[i] = std::acos(ct[i]);
    const double dphi = 2.0 * M_PI / Nph;

    const int Nlm = ang3d::n_channels(Lmax);
    int n_fail = 0;
    double max_err = 0.0;

    for (int a = 0; a < Nlm; ++a) {
        int la, ma; ang3d::idx_to_lm(a, la, ma);
        for (int b = a; b < Nlm; ++b) {
            int lb, mb; ang3d::idx_to_lm(b, lb, mb);
            double sum = 0.0;
            for (int i = 0; i < Nth; ++i) {
                double row = 0.0;
                for (int j = 0; j < Nph; ++j) {
                    const double Ya = ang3d::real_Ylm(la, ma, theta[i], j * dphi);
                    const double Yb = ang3d::real_Ylm(lb, mb, theta[i], j * dphi);
                    row += Ya * Yb;
                }
                sum += wct[i] * row;
            }
            sum *= dphi;
            const double expect = (a == b) ? 1.0 : 0.0;
            const double err = std::fabs(sum - expect);
            if (err > max_err) max_err = err;
            if (err > abs_tol) {
                std::printf("FAIL  (%d,%+d).(%d,%+d) = %.3e  expect %g\n",
                            la, ma, lb, mb, sum, expect);
                ++n_fail;
            }
        }
    }
    std::printf("[orthogonality] Lmax=%d  pairs=%d  max|err|=%.3e  fails=%d\n",
                Lmax, Nlm * (Nlm + 1) / 2, max_err, n_fail);
    return (n_fail == 0) ? 0 : 1;
}
