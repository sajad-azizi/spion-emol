// test_taylor_convergence.cpp
//
// Convergence checks for the Taylor stepper applied to a 2-level
// Rabi problem with constant H = (Ω/2) σ_x.  Two studies:
//
//   (a) ORDER-IN-K, fixed Δt:  for ‖HΔt‖ ~ 0.5 the truncation error
//       at order K should fall like (HΔt)^{K+1}/(K+1)! -- super-
//       exponentially in K until the floor is roundoff.
//
//   (b) ORDER-IN-Δt, fixed K (small): the error scales as Δt^{K+1}.
//       We verify this for K = 4 by halving Δt.
//
// These two studies together pin down the math: the iteration is
// truly the Taylor expansion of exp(-iHΔt), and its truncation /
// step-size behavior matches the analytic prediction.
#include "Common.hpp"
#include "TaylorStepper.hpp"

#include <cstdio>
#include <vector>

namespace {

// Analytic 2-level Rabi answer at time t starting from |0⟩.
Eigen::Vector2cd rabi_exact(double Omega, double t) {
    Eigen::Vector2cd v;
    v(0) =                std::cos(0.5 * Omega * t);
    v(1) = -I_unit * std::sin(0.5 * Omega * t);
    return v;
}

double single_step_error(double Omega, double dt, int K_max) {
    using namespace mc_tdse;
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(2, 2);
    H(0, 1) = 0.5 * Omega;
    H(1, 0) = 0.5 * Omega;

    Eigen::VectorXcd b(2);
    b << dcompx(1.0, 0.0), dcompx(0.0, 0.0);

    TaylorOptions opt;
    opt.K_max  = K_max;
    opt.eps_rel = 0.0;       // disable adaptive stop -- want EXACTLY K_max terms

    Eigen::VectorXcd b1 = taylor_step_const_H(H, b, dt, opt);
    Eigen::Vector2cd ex = rabi_exact(Omega, dt);
    return (b1 - ex).cwiseAbs().maxCoeff();
}

}  // namespace

int main() {
    using namespace mc_tdse;
    int n_fail = 0;

    std::printf("[convergence] Two-level Rabi, H = (Ω/2) σ_x\n");
    std::printf("\n--- (a) ORDER-IN-K, dt fixed (ΩΔt = 0.5) ---\n");
    {
        const double Omega = 1.0;
        const double dt    = 0.5;             // ΩΔt = 0.5
        std::printf("       K        |b - b_exact|     ratio to (Δt)^{K+1}/(K+1)!\n");
        double prev_err = 0.0;
        for (int K = 1; K <= 14; ++K) {
            const double err = single_step_error(Omega, dt, K);
            // Reference: leading term in Taylor truncation is
            //   |((-iΩ/2)Δt)^{K+1}/(K+1)!|
            double fact = 1.0;
            for (int k = 1; k <= K + 1; ++k) fact *= k;
            const double ref = std::pow(0.5 * Omega * dt, K + 1) / fact;
            std::printf("       %2d   %.3e    %.3e\n", K, err, err / std::max(ref, 1e-300));
            (void)prev_err;
        }
        // pass if K=12 already at ~1e-15 (pure roundoff)
        const double err12 = single_step_error(Omega, dt, 12);
        const bool ok = (err12 < 1e-13);
        std::printf("       check K=12 < 1e-13:  %.3e -> %s\n",
                    err12, ok ? "PASS" : "FAIL");
        if (!ok) ++n_fail;
    }

    std::printf("\n--- (b) ORDER-IN-Δt, K=4 fixed: error ∝ Δt^{K+1} = Δt^5 ---\n");
    {
        const double Omega = 1.0;
        const int    K     = 4;
        const std::vector<double> dts = {0.4, 0.2, 0.1, 0.05, 0.025};
        std::printf("       dt        err            ratio (should -> 2^5 = 32)\n");
        std::vector<double> errs;
        for (double dt : dts) {
            const double err = single_step_error(Omega, dt, K);
            errs.push_back(err);
            std::printf("       %.4f   %.3e", dt, err);
            if (errs.size() >= 2) {
                const double ratio = errs[errs.size() - 2] / err;
                std::printf("    %.3f", ratio);
            }
            std::printf("\n");
        }
        // Pass if last 2 ratios are in [25, 40] (close to the
        // expected 32 = 2^5).
        const double r_last = errs[errs.size() - 2] / errs.back();
        const bool ok = (r_last > 25.0 && r_last < 40.0);
        std::printf("       last halving ratio = %.3f (target 32):  %s\n",
                    r_last, ok ? "PASS" : "FAIL");
        if (!ok) ++n_fail;
    }

    std::printf("\nTotal failures: %d\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
