// test_angular.cpp — unit tests for the angular module (Ylm + GL grid).
//
// Three independent checks:
//   (A) Ylm hand-values at theta = pi/4, phi = pi/3 for l <= 2,
//       compared to closed-form expressions.
//   (B) Orthonormality: integrate Y^R_{l,m} * Y^R_{l',m'} on the grid;
//       must equal Kronecker delta to ~1e-14 for Lmax = 6.
//   (C) Single-mode projection round-trip: sample F(theta, phi) = Y^R_{L,M}
//       on the grid, project onto every (l, m), must recover 1 at (L, M)
//       and 0 elsewhere to ~1e-14.

#include "angular/Grid.hpp"
#include "angular/Ylm.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using preproc::angular::AngGrid;
using preproc::angular::lm_index;
using preproc::angular::real_Ylm;

namespace {
struct Fail { int n = 0; };
void check(bool c, const std::string& s, double detail, Fail& f) {
    if (c) std::cerr << "  [ok] " << s << "  (" << detail << ")\n";
    else   { std::cerr << "  [FAIL] " << s << "  (" << detail << ")\n"; ++f.n; }
}
bool close(double a, double b, double atol) { return std::abs(a - b) <= atol; }
}

int main() {
    Fail F;
    std::cerr << std::setprecision(17);

    // ---------- (A) hand-check Ylm values -------------------------------
    {
        const double th = M_PI / 4.0;
        const double ph = M_PI / 3.0;
        const double s = std::sin(th), c = std::cos(th);
        const double cp = std::cos(ph), sp = std::sin(ph);

        // Standard real SH (no CS) at this point, from closed-form.
        const double exp_00 = 1.0 / std::sqrt(4.0 * M_PI);
        const double exp_1p1 = std::sqrt(3.0 / (4.0 * M_PI)) * s * cp;   // p_x angular
        const double exp_1n1 = std::sqrt(3.0 / (4.0 * M_PI)) * s * sp;   // p_y
        const double exp_10  = std::sqrt(3.0 / (4.0 * M_PI)) * c;        // p_z
        // d_{z^2} = sqrt(5/(16 pi)) * (3 cos^2 t - 1)
        const double exp_2_0  = std::sqrt(5.0 / (16.0 * M_PI)) * (3.0 * c * c - 1.0);
        // d_{x^2-y^2} = sqrt(15/(16 pi)) * sin^2 t * cos(2 phi)
        const double exp_2_p2 = std::sqrt(15.0 / (16.0 * M_PI)) * s * s * std::cos(2.0 * ph);
        // d_{xy} = sqrt(15/(16 pi)) * sin^2 t * sin(2 phi)
        const double exp_2_n2 = std::sqrt(15.0 / (16.0 * M_PI)) * s * s * std::sin(2.0 * ph);
        // d_{xz} = sqrt(15/(4 pi)) * sin t cos t cos phi
        const double exp_2_p1 = std::sqrt(15.0 / (4.0 * M_PI)) * s * c * cp;
        // d_{yz} = sqrt(15/(4 pi)) * sin t cos t sin phi
        const double exp_2_n1 = std::sqrt(15.0 / (4.0 * M_PI)) * s * c * sp;

        struct Row { int l, m; double expected; const char* label; };
        Row rows[] = {
            {0,  0, exp_00,  "Y^R_{0,0}   = 1/sqrt(4pi)"},
            {1,  1, exp_1p1, "Y^R_{1,+1}  = sqrt(3/4pi) sin t cos phi"},
            {1, -1, exp_1n1, "Y^R_{1,-1}  = sqrt(3/4pi) sin t sin phi"},
            {1,  0, exp_10,  "Y^R_{1, 0}  = sqrt(3/4pi) cos t"},
            {2,  0, exp_2_0,  "Y^R_{2, 0}  = d_{z^2}"},
            {2,  1, exp_2_p1, "Y^R_{2,+1}  = d_{xz}"},
            {2, -1, exp_2_n1, "Y^R_{2,-1}  = d_{yz}"},
            {2,  2, exp_2_p2, "Y^R_{2,+2}  = d_{x^2-y^2}"},
            {2, -2, exp_2_n2, "Y^R_{2,-2}  = d_{xy}"},
        };
        std::cerr << "\n[A] Ylm hand-check at theta=pi/4 phi=pi/3\n";
        for (const auto& r : rows) {
            const double got = real_Ylm(r.l, r.m, th, ph);
            const double diff = std::abs(got - r.expected);
            check(diff < 1e-14, r.label, diff, F);
        }
    }

    // ---------- (B) orthonormality ---------------------------------------
    {
        const int Lmax = 6;
        auto grid = AngGrid::build(Lmax);
        const int N = grid.nTheta * grid.nPhi;

        std::cerr << "\n[B] Ylm orthonormality, Lmax=" << Lmax << " (nTheta="
                  << grid.nTheta << " nPhi=" << grid.nPhi << ")\n";

        double max_offdiag = 0.0, max_diag_err = 0.0;
        for (int l1 = 0; l1 <= Lmax; ++l1)
        for (int m1 = -l1; m1 <= l1; ++m1) {
            std::vector<double> Y1(N);
            for (int i = 0; i < grid.nTheta; ++i)
                for (int j = 0; j < grid.nPhi; ++j)
                    Y1[i * grid.nPhi + j] = grid.Y(l1, m1, i, j);
            for (int l2 = 0; l2 <= Lmax; ++l2)
            for (int m2 = -l2; m2 <= l2; ++m2) {
                std::vector<double> P(N);
                for (int k = 0; k < N; ++k) {
                    int i = k / grid.nPhi, j = k % grid.nPhi;
                    P[k] = Y1[k] * grid.Y(l2, m2, i, j);
                }
                const double I = grid.integrate(P);
                if (l1 == l2 && m1 == m2) {
                    max_diag_err = std::max(max_diag_err, std::abs(I - 1.0));
                } else {
                    max_offdiag = std::max(max_offdiag, std::abs(I));
                }
            }
        }
        check(max_diag_err < 1e-13, "diag error < 1e-13", max_diag_err, F);
        check(max_offdiag  < 1e-13, "offdiag max < 1e-13", max_offdiag,  F);
    }

    // ---------- (C) round-trip on single mode ---------------------------
    {
        const int Lmax = 6;
        auto grid = AngGrid::build(Lmax);
        const int N = grid.nTheta * grid.nPhi;
        std::cerr << "\n[C] Round-trip: project Y^R_{L,M} onto all channels\n";

        int fails = 0;
        for (int L = 0; L <= Lmax; ++L)
        for (int M = -L; M <= L; ++M) {
            // Sample F = Y^R_{L, M} on grid
            std::vector<double> Fval(N);
            for (int i = 0; i < grid.nTheta; ++i)
                for (int j = 0; j < grid.nPhi; ++j)
                    Fval[i * grid.nPhi + j] = grid.Y(L, M, i, j);

            // Project onto (l, m) for all l <= Lmax.
            double max_offdiag = 0.0, diag = 0.0;
            for (int l = 0; l <= Lmax; ++l)
            for (int m = -l; m <= l; ++m) {
                std::vector<double> P(N);
                for (int k = 0; k < N; ++k) {
                    int i = k / grid.nPhi, j = k % grid.nPhi;
                    P[k] = Fval[k] * grid.Y(l, m, i, j);
                }
                const double I = grid.integrate(P);
                if (l == L && m == M) diag = I;
                else                  max_offdiag = std::max(max_offdiag, std::abs(I));
            }
            if (std::abs(diag - 1.0) > 1e-13 || max_offdiag > 1e-13) {
                std::cerr << "    [FAIL] (L,M)=(" << L << "," << M << ")  diag-1=" << (diag-1.0)
                          << "  max_offdiag=" << max_offdiag << "\n";
                ++fails;
            }
        }
        check(fails == 0, "all (L,M) pairs round-trip", static_cast<double>(fails), F);
    }

    std::cerr << "\n==> " << (F.n == 0 ? "PASS" : "FAIL: " + std::to_string(F.n)) << "\n";
    return F.n == 0 ? 0 : 1;
}
