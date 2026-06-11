// test_lapack_inverse.cpp
//
// Verifies that inverse_general (LAPACK or Eigen fallback) matches Eigen's
// partialPivLu().inverse() to near-machine precision across a range of
// matrix sizes and conditioning.
//
// Both paths should also be self-consistent with the defining equation:
//     A · inverse_general(A)  ≈  I
// to ~1e-12 relative for well-conditioned matrices.

#include "scatt/LapackInverse.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <algorithm>

using namespace scatt;

static int g_fail = 0;
static void check(bool cond, const std::string& what) {
    if (!cond) { std::cerr << "FAIL  " << what << "\n"; ++g_fail; }
    else       { std::cout << "ok    " << what << "\n"; }
}

static Eigen::MatrixXd random_spd(int n, std::mt19937& rng) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    Eigen::MatrixXd X(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            X(i, j) = U(rng);
    // A = X^T X + n*I is SPD and well-conditioned.
    return X.transpose() * X + static_cast<double>(n) * Eigen::MatrixXd::Identity(n, n);
}

static Eigen::MatrixXd random_general(int n, std::mt19937& rng) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    Eigen::MatrixXd X(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            X(i, j) = U(rng);
    X += static_cast<double>(n) * Eigen::MatrixXd::Identity(n, n);
    return X;
}

int main() {
    std::mt19937 rng(0xBEEF);

    for (int n : {1, 4, 17, 49, 121, 256}) {
        auto A_spd = random_spd(n, rng);
        auto A_gen = random_general(n, rng);

        // 1. inverse · A ≈ I
        for (const auto& A : { A_spd, A_gen }) {
            auto Ainv = inverse_general(A);
            const Eigen::MatrixXd residual =
                A * Ainv - Eigen::MatrixXd::Identity(n, n);
            const double relerr = residual.cwiseAbs().maxCoeff() /
                                  std::max(A.cwiseAbs().maxCoeff(), 1.0);
            std::ostringstream os;
            os << "n=" << n << "  ‖A·A⁻¹ - I‖∞ = " << relerr
               << " (tol 1e-11)";
            check(relerr < 1.0e-11, os.str());
        }

        // 2. Agreement with Eigen reference (identical when no MKL, close when MKL).
        {
            auto ref    = A_gen.partialPivLu().inverse();
            auto via_us = inverse_general(A_gen);
            const double scale = std::max(ref.cwiseAbs().maxCoeff(), 1e-30);
            const double rel   = (ref - via_us).cwiseAbs().maxCoeff() / scale;
            std::ostringstream os;
            os << "n=" << n << "  ‖Eigen - inverse_general‖∞/‖Eigen‖∞ = " << rel
               << " (tol 1e-11)";
            check(rel < 1.0e-11, os.str());
        }
    }

    // NOTE on singular matrices: the MKL path throws std::runtime_error
    // (dgetrf returns info > 0). The Eigen-fallback path does NOT throw;
    // partialPivLu returns a matrix filled with NaN/Inf. We don't exercise
    // this here because the test has to pass on both build flavors.

    std::cout << "\n" << (g_fail == 0 ? "PASS" : "FAIL")
              << " lapack_inverse  (" << g_fail << " failures)\n";
    return g_fail == 0 ? 0 : 1;
}
