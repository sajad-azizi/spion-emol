// test_gaunt.cpp -- correctness checks for real-Gaunt coefficients.
//
// Known values:
//   G^R(0 0; 0 0; 0 0) = 1 / sqrt(4 pi)         (trivial)
//   G^R(l m; l m; 0 0) = 1 / sqrt(4 pi)         (completeness)
// Symmetry:
//   G^R is symmetric under permutation of any two (l, m) legs.
// Selection rules:
//   zero unless triangle inequality, parity, m-sum compatible.
// Cross-check against version_0's implementation (line-by-line port) by
// computing a handful of specific (l1 m1; l2 m2; l3 m3) triples and
// comparing against hand-computed values.

#include "angular/Gaunt.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using scatt::angular::gaunt_real;
using scatt::angular::real_Ylm;

static int fails = 0;

static void expect(bool cond, const char* what, double got, double want, double tol) {
    const double diff = std::abs(got - want);
    const bool ok = cond && diff <= tol;
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what
              << "   got=" << got << "  want=" << want << "  |diff|=" << diff << "\n";
    if (!ok) ++fails;
}

int main() {
    const double inv_sqrt_4pi = 1.0 / std::sqrt(4.0 * M_PI);

    std::cout << "--- G^R(0 0; 0 0; 0 0) ---\n";
    {
        const double g = gaunt_real(0, 0, 0, 0, 0, 0);
        expect(true, "G(s,s,s)", g, inv_sqrt_4pi, 1e-14);
    }

    std::cout << "--- G^R(l m; l m; 0 0) completeness ---\n";
    for (int l = 0; l <= 3; ++l) {
        for (int m = -l; m <= l; ++m) {
            const double g = gaunt_real(l, m, l, m, 0, 0);
            char buf[64]; std::snprintf(buf, sizeof(buf), "G(%d%d,%d%d,0 0)", l, m, l, m);
            expect(true, buf, g, inv_sqrt_4pi, 1e-14);
        }
    }

    std::cout << "--- permutation symmetry ---\n";
    struct Triple { int l1, m1, l2, m2, l3, m3; };
    const Triple cases[] = {
        {1, +1, 1, -1, 2, -2},
        {2,  0, 2,  0, 2,  0},
        {1,  0, 1,  0, 2,  0},
        {1, +1, 2, +1, 1,  0},
        {3, -2, 2, +1, 1, -1},
    };
    for (const auto& c : cases) {
        const double g123 = gaunt_real(c.l1, c.m1, c.l2, c.m2, c.l3, c.m3);
        const double g213 = gaunt_real(c.l2, c.m2, c.l1, c.m1, c.l3, c.m3);
        const double g132 = gaunt_real(c.l1, c.m1, c.l3, c.m3, c.l2, c.m2);
        const double g321 = gaunt_real(c.l3, c.m3, c.l2, c.m2, c.l1, c.m1);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "(%d%+d,%d%+d,%d%+d) 12<->21", c.l1, c.m1, c.l2, c.m2, c.l3, c.m3);
        expect(true, buf, g213, g123, 1e-14);
        std::snprintf(buf, sizeof(buf), "(%d%+d,%d%+d,%d%+d) 23<->32", c.l1, c.m1, c.l2, c.m2, c.l3, c.m3);
        expect(true, buf, g132, g123, 1e-14);
        std::snprintf(buf, sizeof(buf), "(%d%+d,%d%+d,%d%+d) reverse",  c.l1, c.m1, c.l2, c.m2, c.l3, c.m3);
        expect(true, buf, g321, g123, 1e-14);
    }

    std::cout << "--- selection rules ---\n";
    // l1+l2+l3 must be even.
    expect(gaunt_real(1, 0, 1, 0, 1, 0) == 0.0,
           "odd-parity (1,1,1) zero", gaunt_real(1, 0, 1, 0, 1, 0), 0.0, 1e-14);
    // triangle inequality: (1, 0; 1, 0; 4, 0) => no
    expect(gaunt_real(1, 0, 1, 0, 4, 0) == 0.0,
           "triangle violation zero", gaunt_real(1, 0, 1, 0, 4, 0), 0.0, 1e-14);

    std::cout << "--- numeric cross-check via angular quadrature ---\n";
    // Numerically integrate Y^R_{l1,m1} * Y^R_{l2,m2} * Y^R_{l3,m3} on a
    // dense (theta, phi) grid using Gauss-Legendre x uniform trapezoidal
    // and compare to gaunt_real(...). Tensor-product exact for bandlimited
    // products up to Lmax = ntheta - 1.
    auto num_gaunt = [](int l1, int m1, int l2, int m2, int l3, int m3) {
        const int nT = 24, nP = 48;
        // simple GL nodes and weights: Newton recurrence on P_n (same as
        // preprocessing/src/angular/GaussLegendre.hpp; inlined minimally).
        std::vector<double> x(nT), w(nT);
        for (int i = 0; i < (nT + 1) / 2; ++i) {
            double z = std::cos(M_PI * (i + 0.75) / (nT + 0.5));
            for (int it = 0; it < 200; ++it) {
                double p1 = 1, p2 = 0, p3;
                for (int j = 0; j < nT; ++j) {
                    p3 = p2; p2 = p1;
                    p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
                }
                double pp = nT * (z * p1 - p2) / (z * z - 1.0);
                double zn = z - p1 / pp;
                if (std::abs(zn - z) < 1e-15) { z = zn; break; }
                z = zn;
            }
            double p1 = 1, p2 = 0, p3;
            for (int j = 0; j < nT; ++j) {
                p3 = p2; p2 = p1;
                p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
            }
            double pp = nT * (z * p1 - p2) / (z * z - 1.0);
            x[i]          = -z;    w[i]          = 2.0 / ((1.0 - z * z) * pp * pp);
            x[nT - 1 - i] = +z;    w[nT - 1 - i] = w[i];
        }
        double sum = 0.0;
        for (int it = 0; it < nT; ++it) {
            const double th = std::acos(x[it]);
            for (int jp = 0; jp < nP; ++jp) {
                const double ph = 2.0 * M_PI * jp / nP;
                sum += w[it] * (2.0 * M_PI / nP)
                     * real_Ylm(l1, m1, th, ph)
                     * real_Ylm(l2, m2, th, ph)
                     * real_Ylm(l3, m3, th, ph);
            }
        }
        return sum;
    };
    for (const auto& c : cases) {
        const double g_alg = gaunt_real (c.l1, c.m1, c.l2, c.m2, c.l3, c.m3);
        const double g_num = num_gaunt  (c.l1, c.m1, c.l2, c.m2, c.l3, c.m3);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "(%d%+d,%d%+d,%d%+d) analytic vs quadrature",
                      c.l1, c.m1, c.l2, c.m2, c.l3, c.m3);
        expect(true, buf, g_num, g_alg, 1e-12);
    }

    std::cout << (fails == 0 ? "\n==> PASS\n" : "\n==> FAIL (" + std::to_string(fails) + " checks)\n");
    return fails ? 1 : 0;
}
