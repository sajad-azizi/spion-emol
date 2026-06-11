// test_gpu_steppers.cpp -- direct math unit tests for the GPU steppers.
//
// The CPU-vs-GPU propagator test catches regressions in "the whole stack,"
// but if BOTH paths share a bug (e.g. a wrong kernel) it stays silent.
// This file fills that gap: each GPU step is checked directly against a
// separately-computed Eigen reference on small random matrices.
//
// Checks, in order of strength:
//   (A) GpuForwardStepper::step (one shot):
//         expected = sym( (12 W_inv - 10 I - R_prev)^-1 )
//         gpu      = stepper.step(W_inv, out)
//         require ||gpu - expected|| < 1e-11 * ||expected||
//
//   (B) GpuForwardStepper::step repeated (5 iterations, stacked):
//         same formula applied recursively on host vs device
//         require drift < 1e-10 after 5 steps
//
//   (C) GpuBackStepper::step (one shot):
//         expected Y = W_inv * (Rinv * Z)
//         require top(expected) ≈ psi_gpu  AND  bottom(expected) ≈ f_gpu
//
//   (D) GpuBackStepper::step with compute_f = false:
//         require the f buffer is NOT touched (no UB, no crash, and psi
//         still equals Y.top(N_psi))
//
// These run at N=128 (small enough to finish in <1s even on host CPU
// emulation of SYCL, large enough to exercise LU pivot branches).
//
// Gracefully skip when no SYCL GPU is visible (same convention as
// test_gpu_propagate).

#include "scatt/GpuPropagate.hpp"
#include "scatt/LapackInverse.hpp"

#include <Eigen/Dense>

#include <iomanip>
#include <iostream>
#include <random>
#include <string>

using scatt::GpuBackStepper;
using scatt::GpuContext;
using scatt::GpuForwardStepper;
using scatt::inverse_general;

static int fails = 0;
static void check(bool ok, const std::string& what) {
    std::cout << "  [" << (ok ? "ok  " : "FAIL") << "] " << what << "\n";
    if (!ok) ++fails;
}

// Generate a random symmetric positive-definite N×N matrix with
// condition number ≈ given value.  Matches the structure of W_inv(n)
// in production: SPD and well-conditioned.
static Eigen::MatrixXd make_random_spd(int N, std::uint64_t seed, double cond = 50.0) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    Eigen::MatrixXd A(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) A(i, j) = gauss(rng);

    // A := A * A^T + eps * I  so it's SPD.  Scale eigenspectrum so that
    // cond(A) ≈ cond (smallest eigenvalue = largest/cond).
    Eigen::MatrixXd S = (A * A.transpose()) / double(N);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    Eigen::VectorXd lam = es.eigenvalues();
    const double lmax = lam.maxCoeff();
    const double lmin_target = lmax / cond;
    for (int k = 0; k < N; ++k) {
        if (lam(k) < lmin_target) lam(k) = lmin_target;
    }
    return es.eigenvectors() * lam.asDiagonal() * es.eigenvectors().transpose();
}

// Reference for ONE forward step -- exactly what the GPU stepper should
// produce.
static Eigen::MatrixXd forward_reference(const Eigen::MatrixXd& W_inv,
                                         const Eigen::MatrixXd& R_prev_inv)
{
    const int N = static_cast<int>(W_inv.rows());
    Eigen::MatrixXd U = 12.0 * W_inv - 10.0 * Eigen::MatrixXd::Identity(N, N);
    Eigen::MatrixXd R_current = U - R_prev_inv;
    Eigen::MatrixXd R_new = inverse_general(R_current);
    // Symmetrize (same as GPU kernel_symmetrize).
    R_new = 0.5 * (R_new + R_new.transpose().eval());
    return R_new;
}

// Reference for ONE back step.
static void back_reference(const Eigen::MatrixXd& Rinv,
                           const Eigen::MatrixXd& W_inv,
                           const Eigen::MatrixXd& Z,
                           int N_psi, int N_f,
                           Eigen::MatrixXd& Z_next_expected,
                           Eigen::MatrixXd& psi_expected,
                           Eigen::MatrixXd& f_expected)
{
    Z_next_expected = Rinv * Z;
    Eigen::MatrixXd Y = W_inv * Z_next_expected;
    psi_expected = Y.topRows(N_psi);
    f_expected   = Y.bottomRows(N_f);
}

int main(int /*argc*/, char** /*argv*/) {
    std::cout << "=== GPU steppers direct-math unit tests ===\n";

    if (!GpuContext::gpu_available()) {
        std::cout << "  [skip] no SYCL GPU visible.\n";
        return 0;
    }
    GpuContext ctx(/*prefer_gpu=*/true);
    std::cout << "  device: " << ctx.info().device_name
              << "  (HBM " << (ctx.info().global_mem_bytes >> 30) << " GB)\n";

    const int N     = 128;
    const int N_psi = 40;
    const int N_f   = N - N_psi;                 // 88 -- deliberately != N_psi

    // ------------------------------------------------------------------
    // (A) single forward step -- expected vs GPU
    // ------------------------------------------------------------------
    std::cout << "\n--- (A) GpuForwardStepper::step single iteration ---\n";
    {
        Eigen::MatrixXd W_inv = make_random_spd(N, /*seed=*/42);
        Eigen::MatrixXd R_prev = make_random_spd(N, /*seed=*/43);

        Eigen::MatrixXd R_expected = forward_reference(W_inv, R_prev);

        GpuForwardStepper fstep(ctx, N);
        fstep.init_R_prev_inv(R_prev);
        Eigen::MatrixXd R_gpu;
        fstep.step(W_inv, R_gpu);

        double scale = std::max(R_expected.cwiseAbs().maxCoeff(), 1e-30);
        double err   = (R_expected - R_gpu).cwiseAbs().maxCoeff();
        double rel   = err / scale;
        std::cout << "  ||R_ref||_max=" << std::scientific << std::setprecision(3) << scale
                  << "  ||ref-gpu||_max=" << err
                  << "  rel=" << rel << "\n";
        check(rel < 1e-11, "single forward step matches Eigen reference  (rel < 1e-11)");
    }

    // ------------------------------------------------------------------
    // (B) 5 recursive forward steps -- drift accumulation
    // ------------------------------------------------------------------
    std::cout << "\n--- (B) GpuForwardStepper::step recursive (5 iterations) ---\n";
    {
        const int n_iter = 5;
        std::vector<Eigen::MatrixXd> W_inv(n_iter);
        for (int k = 0; k < n_iter; ++k) {
            W_inv[k] = make_random_spd(N, /*seed=*/100 + k);
        }
        Eigen::MatrixXd R_prev_ref = make_random_spd(N, /*seed=*/99);
        Eigen::MatrixXd R_prev_gpu = R_prev_ref;     // exact copy

        GpuForwardStepper fstep(ctx, N);
        fstep.init_R_prev_inv(R_prev_gpu);

        double worst_rel = 0.0;
        for (int k = 0; k < n_iter; ++k) {
            R_prev_ref = forward_reference(W_inv[k], R_prev_ref);
            Eigen::MatrixXd R_gpu_out;
            fstep.step(W_inv[k], R_gpu_out);
            double scale = std::max(R_prev_ref.cwiseAbs().maxCoeff(), 1e-30);
            double rel = (R_prev_ref - R_gpu_out).cwiseAbs().maxCoeff() / scale;
            worst_rel = std::max(worst_rel, rel);
            std::cout << "  step " << (k + 1) << "/5  rel=" << std::scientific
                      << std::setprecision(3) << rel << "\n";
        }
        check(worst_rel < 1e-10, "5-step recursion drift < 1e-10");
    }

    // ------------------------------------------------------------------
    // (C) back step -- expected Y = W_inv * Rinv * Z
    // ------------------------------------------------------------------
    std::cout << "\n--- (C) GpuBackStepper::step psi+f vs Eigen reference ---\n";
    {
        Eigen::MatrixXd Rinv  = make_random_spd(N, /*seed=*/200);
        Eigen::MatrixXd W_inv = make_random_spd(N, /*seed=*/201);
        Eigen::MatrixXd Z     = Eigen::MatrixXd::Random(N, N_psi);

        Eigen::MatrixXd Z_next_ref, psi_ref, f_ref;
        back_reference(Rinv, W_inv, Z, N_psi, N_f, Z_next_ref, psi_ref, f_ref);

        GpuBackStepper bstep(ctx, N, N_psi, N_f);
        bstep.init_Z(Z);
        Eigen::MatrixXd psi_gpu, f_gpu;
        bstep.step(Rinv, W_inv, psi_gpu, &f_gpu, /*compute_f=*/true);

        double s_psi = std::max(psi_ref.cwiseAbs().maxCoeff(), 1e-30);
        double s_f   = std::max(f_ref.cwiseAbs().maxCoeff(),   1e-30);
        double e_psi = (psi_ref - psi_gpu).cwiseAbs().maxCoeff() / s_psi;
        double e_f   = (f_ref   - f_gpu  ).cwiseAbs().maxCoeff() / s_f;
        std::cout << "  psi: rel=" << std::scientific << std::setprecision(3) << e_psi << "\n";
        std::cout << "  f  : rel=" << std::scientific << std::setprecision(3) << e_f   << "\n";
        check(e_psi < 1e-12, "psi = top rows of W_inv * Rinv * Z  (rel < 1e-12)");
        check(e_f   < 1e-12, "f   = bottom rows of W_inv * Rinv * Z  (rel < 1e-12)");

        // Also verify d_Z on device now equals Rinv*Z (the swap keeps the
        // pre-W_inv intermediate).
        Eigen::MatrixXd Z_dev;
        bstep.get_Z(Z_dev);
        double e_Z = (Z_next_ref - Z_dev).cwiseAbs().maxCoeff() /
                     std::max(Z_next_ref.cwiseAbs().maxCoeff(), 1e-30);
        std::cout << "  Z(on-device) vs Rinv*Z: rel=" << std::scientific
                  << std::setprecision(3) << e_Z << "\n";
        check(e_Z < 1e-12, "post-step d_Z matches Rinv*Z");
    }

    // ------------------------------------------------------------------
    // (D) back step with compute_f=false: f buffer must stay untouched
    // ------------------------------------------------------------------
    std::cout << "\n--- (D) GpuBackStepper::step with compute_f = false ---\n";
    {
        Eigen::MatrixXd Rinv  = make_random_spd(N, /*seed=*/300);
        Eigen::MatrixXd W_inv = make_random_spd(N, /*seed=*/301);
        Eigen::MatrixXd Z     = Eigen::MatrixXd::Random(N, N_psi);

        Eigen::MatrixXd Z_next_ref, psi_ref, f_ref;
        back_reference(Rinv, W_inv, Z, N_psi, N_f, Z_next_ref, psi_ref, f_ref);

        GpuBackStepper bstep(ctx, N, N_psi, N_f);
        bstep.init_Z(Z);

        // Pre-fill f_gpu with a sentinel value; if the GPU touches it with
        // compute_f=false we'll catch that.
        Eigen::MatrixXd f_gpu = Eigen::MatrixXd::Constant(N_f, N_psi, -1234.5);
        Eigen::MatrixXd psi_gpu;
        bstep.step(Rinv, W_inv, psi_gpu, &f_gpu, /*compute_f=*/false);

        double e_psi = (psi_ref - psi_gpu).cwiseAbs().maxCoeff() /
                       std::max(psi_ref.cwiseAbs().maxCoeff(), 1e-30);
        check(e_psi < 1e-12, "psi still correct when compute_f = false");

        double touched = (f_gpu - Eigen::MatrixXd::Constant(N_f, N_psi, -1234.5))
                             .cwiseAbs().maxCoeff();
        check(touched == 0.0,
              "f buffer sentinel untouched  (compute_f=false actually skipped)");
    }

    std::cout << "\n" << (fails == 0 ? "PASS" : "FAIL")
              << " (" << fails << " failures)\n";
    return fails == 0 ? 0 : 1;
}
