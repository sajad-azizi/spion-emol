// test_real_gaunt_full_l50.cpp -- exhaustive numerical validation of
// `scatt::angular::gaunt_real` for the dipole channel (l2 = 1, q ∈ {-1,0,+1}),
// for every (l_mu, m_mu, l_nu, m_nu) with l_mu, l_nu ∈ [0, L_MAX] and
// |l_mu - l_nu| = 1.
//
// We call the EXACT in-code `gaunt_real` (3j-symbol path) and compare to
// a brute numerical reference computed from the EXACT in-code `real_Ylm`
// evaluator on a Gauss-Legendre × trapezoid grid.  Both are in
// `src/angular/Gaunt.hpp`, so this is end-to-end self-consistency: if
// the analytic and numerical paths disagree, EITHER the 3j formula OR
// the U_real_to_complex transform OR the real_Ylm phase convention is
// wrong -- and any of those would have leaked into production code.
//
// Default L_MAX = 50; pass a different value as argv[1].
//
// Output format (one line per triple):
//   l_mu  m_mu  q  l_nu  m_nu   G_analytic   G_numerical   abs_err   rel_err
//
// At end:
//   # SUMMARY  compared=...  failed=...  (and a list of failures)
// Exit code 0 if all pass, 1 if any fail.

#include "angular/Gaunt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using scatt::angular::gaunt_real;
using scatt::angular::real_Ylm;

namespace {

// Numerical-recipes-style Gauss-Legendre nodes/weights on [-1, 1].
void gauss_legendre(int N, std::vector<double>& x, std::vector<double>& w) {
    x.assign(N, 0.0);
    w.assign(N, 0.0);
    constexpr double eps = 3e-16;
    const int m = (N + 1) / 2;
    for (int i = 1; i <= m; ++i) {
        double z  = std::cos(M_PI * (i - 0.25) / (N + 0.5));
        double z1 = 0.0;
        double pp = 0.0;
        for (int it = 0; it < 100; ++it) {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 1; j <= N; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            pp = N * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z  = z1 - p1 / pp;
            if (std::fabs(z - z1) < eps) break;
        }
        x[i - 1] = -z;
        x[N - i] =  z;
        w[i - 1] = 2.0 / ((1.0 - z * z) * pp * pp);
        w[N - i] = w[i - 1];
    }
}

}  // namespace

int main(int argc, char** argv) {
    int L_MAX = (argc > 1) ? std::atoi(argv[1]) : 50;
    if (L_MAX < 1)  L_MAX = 1;
    if (L_MAX > 80) L_MAX = 80;  // memory guard for the precomputed Y table

    // Grid sized to integrate triple Y_l_max products exactly:
    //   Gauss-Legendre N_theta in cos(theta) is exact for polynomials up to
    //   degree 2*N_theta - 1; triple Y_lm has degree up to 3*L_MAX in cos.
    //   N_theta = ceil((3 L_MAX + 1) / 2) is the analytical minimum; we
    //   round up generously.  Trapezoid in phi is exact for periodic
    //   integrand of order < N_phi; max frequency is 3*L_MAX.
    const int N_theta = std::max(96, 2 * L_MAX + 8);
    const int N_phi   = std::max(192, 4 * L_MAX);

    std::vector<double> ct, wct;
    gauss_legendre(N_theta, ct, wct);

    std::vector<double> theta_v(N_theta), sin_th(N_theta);
    for (int i = 0; i < N_theta; ++i) {
        theta_v[i] = std::acos(ct[i]);
        sin_th[i]  = std::sqrt(std::max(0.0, 1.0 - ct[i] * ct[i]));
    }
    std::vector<double> phi_v(N_phi);
    const double dphi = 2.0 * M_PI / N_phi;
    for (int j = 0; j < N_phi; ++j) phi_v[j] = j * dphi;

    const int Nlm = (L_MAX + 1) * (L_MAX + 1);
    const std::size_t G = static_cast<std::size_t>(N_theta) * N_phi;

    std::printf("# Gaunt full l_max comparison\n");
    std::printf("# L_MAX  = %d\n", L_MAX);
    std::printf("# Nlm   = %d\n", Nlm);
    std::printf("# grid  = %d (theta, Gauss-Legendre) x %d (phi, trapezoid)\n",
                N_theta, N_phi);
    std::printf("# precompute table size = %.2f GB\n",
                double(Nlm) * G * 8.0 / (1ull << 30));
    std::fflush(stdout);

    std::vector<double> Y(static_cast<std::size_t>(Nlm) * G);
    for (int l = 0; l <= L_MAX; ++l) {
        for (int m = -l; m <= l; ++m) {
            const int idx = l * l + l + m;
            double* dst = &Y[static_cast<std::size_t>(idx) * G];
            for (int i = 0; i < N_theta; ++i) {
                for (int j = 0; j < N_phi; ++j) {
                    dst[i * N_phi + j] = real_Ylm(l, m, theta_v[i], phi_v[j]);
                }
            }
        }
        if (l % 5 == 0) {
            std::printf("# precomputed real_Ylm up to l=%d\n", l);
            std::fflush(stdout);
        }
    }

    auto numerical_gaunt = [&](int l1, int m1, int l2, int m2, int l3, int m3) -> double {
        const int i1 = l1 * l1 + l1 + m1;
        const int i2 = l2 * l2 + l2 + m2;
        const int i3 = l3 * l3 + l3 + m3;
        const double* Y1 = &Y[static_cast<std::size_t>(i1) * G];
        const double* Y2 = &Y[static_cast<std::size_t>(i2) * G];
        const double* Y3 = &Y[static_cast<std::size_t>(i3) * G];
        double sum = 0.0;
        for (int i = 0; i < N_theta; ++i) {
            double row = 0.0;
            const std::size_t base = static_cast<std::size_t>(i) * N_phi;
            for (int j = 0; j < N_phi; ++j) {
                row += Y1[base + j] * Y2[base + j] * Y3[base + j];
            }
            sum += row * wct[i];
        }
        return sum * dphi;
    };

    long long n_compared = 0, n_fail = 0;
    double max_abs_err = 0.0, max_rel_err_for_big = 0.0;
    constexpr double abs_tol = 5e-7;
    constexpr double rel_tol = 1e-4;
    constexpr double big_abs = 1e-6;   // only count rel_err for "big" gaunts

    struct Failure {
        int    l_mu, m_mu, q, l_nu, m_nu;
        double Ga, Gn;
        double abs_err, rel_err;
    };
    std::vector<Failure> failures;

    std::printf("# columns: l_mu m_mu q l_nu m_nu  G_analytic  G_numerical  abs_err  rel_err\n");
    std::printf("# tol: |err| > %.0e AND |rel| > %.0e (when |G| > %.0e) flags as FAIL\n",
                abs_tol, rel_tol, big_abs);
    std::fflush(stdout);

    for (int l_mu = 0; l_mu <= L_MAX; ++l_mu) {
        for (int m_mu = -l_mu; m_mu <= l_mu; ++m_mu) {
            for (int q = -1; q <= 1; ++q) {
                for (int dl = -1; dl <= 1; dl += 2) {
                    const int l_nu = l_mu + dl;
                    if (l_nu < 0 || l_nu > L_MAX) continue;
                    for (int m_nu = -l_nu; m_nu <= l_nu; ++m_nu) {
                        const double Ga = gaunt_real(l_mu, m_mu, 1, q, l_nu, m_nu);
                        const double Gn = numerical_gaunt(l_mu, m_mu, 1, q, l_nu, m_nu);
                        const double abs_err = std::fabs(Ga - Gn);
                        const double scale   = std::max({std::fabs(Ga), std::fabs(Gn), 1e-30});
                        const double rel_err = abs_err / scale;

                        std::printf("%3d %+3d  q=%+d  %3d %+3d   "
                                    "%+.12e  %+.12e   %+.3e   %+.3e\n",
                                    l_mu, m_mu, q, l_nu, m_nu, Ga, Gn, abs_err, rel_err);

                        ++n_compared;
                        if (abs_err > max_abs_err) max_abs_err = abs_err;
                        if (std::fabs(Ga) > big_abs && rel_err > max_rel_err_for_big)
                            max_rel_err_for_big = rel_err;

                        const bool big  = std::fabs(Ga) > big_abs;
                        const bool fail = (abs_err > abs_tol) && (!big || rel_err > rel_tol);
                        if (fail) {
                            ++n_fail;
                            if (failures.size() < 200u) {
                                failures.push_back({l_mu, m_mu, q, l_nu, m_nu,
                                                    Ga, Gn, abs_err, rel_err});
                            }
                        }
                    }
                }
            }
        }
        if (l_mu % 5 == 0) std::fflush(stdout);
    }

    std::printf("\n# === SUMMARY ===\n");
    std::printf("# compared:           %lld\n", n_compared);
    std::printf("# failed:             %lld\n", n_fail);
    std::printf("# max |abs_err|:      %.3e\n", max_abs_err);
    std::printf("# max rel_err (|G|>%.0e): %.3e\n", big_abs, max_rel_err_for_big);
    if (!failures.empty()) {
        std::printf("# first %zu failures:\n", failures.size());
        for (const auto& f : failures) {
            std::printf("#   FAIL  l_mu=%2d m_mu=%+3d q=%+d  l_nu=%2d m_nu=%+3d  "
                        "Ga=%+.12e  Gn=%+.12e  abs=%+.3e  rel=%+.3e\n",
                        f.l_mu, f.m_mu, f.q, f.l_nu, f.m_nu, f.Ga, f.Gn,
                        f.abs_err, f.rel_err);
        }
        std::printf("# RESULT: FAIL\n");
        return 1;
    }
    std::printf("# RESULT: ALL PASS\n");
    return 0;
}
