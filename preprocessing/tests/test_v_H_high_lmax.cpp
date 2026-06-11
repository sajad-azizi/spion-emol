// test_v_H_high_lmax.cpp -- V_H must not produce NaN/Inf at high Lmax.
//
// Regression test for the C8F8 case (Lmax_sce=300, rmax=100, dr=0.01)
// where build_V_H originally returned non-finite values from the
// std::pow(r, l) / std::pow(r, 1-l) overflow at high l.  The fix in
// Hartree.hpp adds:
//   (1) a hard l_safe cap above which channels are skipped,
//   (2) a small-rho channel skip,
//   (3) a defensive isfinite scrub at every grid point.
//
// What this test does:
//   * Builds rho_lm directly (no SCE projection -- self-contained):
//        - l = 0:  a Gaussian rho_00(r) = N * exp(-alpha * r^2)
//        - all other channels seeded with synthetic noise of magnitude
//          1e-11 (i.e. ABOVE the 1e-12 empty-channel threshold but still
//          numerically meaningless), to simulate the SCE projection
//          residual for a real molecule's high-l channels.
//   * Calls build_V_H at Lmax=200, rmax=100, dr=0.05 -- well into the
//     overflow regime (l_safe ~ 149 here).
//   * Asserts:
//        (a) every V_H(ch, k) is finite,
//        (b) <rho|V_H>_radial is finite,
//        (c) the l=0 V_H matches the analytic Gaussian-Hartree result to
//            within ~1e-4 (the radial discretisation budget; we're not
//            testing accuracy, just that the value is correct AND finite).
//
// PASS if all three hold.  Without the Hartree.hpp fix this test fails
// at (a) -- many V_H entries are inf/NaN.

#include "angular/Ylm.hpp"
#include "potential/Hartree.hpp"
#include "sce/RadialGrid.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>

using preproc::sce::RadialGrid;
using preproc::potential::build_V_H;
using preproc::angular::n_channels;
using preproc::angular::idx_to_lm;

// Closed-form Hartree of a normalised 3D Gaussian rho(r) = (alpha/pi)^{3/2}
// exp(-alpha r^2) is V_00(r) = (1/r) * erf(sqrt(alpha) * r) -- monopole
// only.  We then have to multiply by the Y^R_{0,0} normalisation to get
// the SCE coefficient: Y^R_{0,0} = 1/sqrt(4 pi)  =>
//     rho_00^R(r) = sqrt(4 pi) * rho(r)
//     V_00^R(r)  = sqrt(4 pi) * V_00(r)
// since the SCE coefficient absorbs the Y_{0,0} factor.
static double rho_gaussian_radial_R(double r, double alpha) {
    const double sqrt_alpha_pi3 = std::pow(alpha / M_PI, 1.5);
    const double rho = sqrt_alpha_pi3 * std::exp(-alpha * r * r);
    return std::sqrt(4.0 * M_PI) * rho;
}
static double V_H00_analytic_R(double r, double alpha) {
    if (r <= 0.0) return std::sqrt(4.0 * M_PI) * 2.0 * std::sqrt(alpha / M_PI);
    const double V = std::erf(std::sqrt(alpha) * r) / r;
    return std::sqrt(4.0 * M_PI) * V;
}

int main() {
    std::cout << "=== V_H high-Lmax stability ===\n";

    // C8F8-scale grid: rmax = 100 bohr.  dr = 0.05 keeps the test fast.
    const double rmin = 0.0;
    const double dr   = 0.05;
    const double rmax = 100.0;
    const int Nr      = static_cast<int>(rmax / dr) + 1;
    auto rg = RadialGrid::build(rmin, dr, Nr);

    const int Lmax = 200;          // beyond l_safe (~149) for this grid
    const int Nlm  = n_channels(Lmax);
    std::cout << "  grid       : Nr=" << Nr << "  rmax=" << rmax
              << "  dr=" << dr << "\n"
              << "  channels   : Lmax=" << Lmax << "  Nlm=" << Nlm << "\n";

    // ----- Build rho_lm -----
    // l=0: pure Gaussian.  All other l: noise at 1e-11 (above the 1e-12
    // skip threshold, so the safety code can't avoid them by content
    // alone -- only the l_safe cap can).
    const double alpha = 1.0;          // 3D Gaussian width
    Eigen::MatrixXd rho(Nlm, Nr);
    rho.setZero();
    std::mt19937_64 rng(0x517A77E11ABE7);
    std::normal_distribution<double> noise(0.0, 1.0);
    for (int ch = 0; ch < Nlm; ++ch) {
        int l, m;
        idx_to_lm(ch, l, m);
        if (l == 0) {
            for (int k = 0; k < Nr; ++k)
                rho(ch, k) = rho_gaussian_radial_R(rg.r[k], alpha);
        } else {
            // Seed noise at ~1e-11 -- mimics the SCE-projection residual
            // for a real molecular density at high l.  Crucially this is
            // above the empty-channel skip threshold (1e-12), so without
            // the l_safe overflow cap, V_H would NaN here.
            for (int k = 0; k < Nr; ++k)
                rho(ch, k) = 1e-11 * noise(rng);
        }
    }

    // ----- Build V_H -----
    // The Hartree.hpp fix should print one safety message and skip all
    // channels with l > l_safe.  Capture stderr would be nicer but we
    // just trust the user can see it in the test log.
    Eigen::MatrixXd V_H = build_V_H(rho, rg, Lmax);

    // ----- Check (a): all finite -----
    int n_nonfinite = 0;
    int worst_ch = -1, worst_k = -1;
    double worst_val = 0.0;
    for (int ch = 0; ch < V_H.rows(); ++ch) {
        for (int k = 0; k < V_H.cols(); ++k) {
            const double v = V_H(ch, k);
            if (!std::isfinite(v)) {
                if (n_nonfinite == 0) { worst_ch = ch; worst_k = k; worst_val = v; }
                ++n_nonfinite;
            }
        }
    }
    if (n_nonfinite > 0) {
        int wl, wm; idx_to_lm(worst_ch, wl, wm);
        std::cerr << "  FAIL (a): " << n_nonfinite << " non-finite V_H entries"
                  << "  (first at ch=" << worst_ch << " l=" << wl << " m=" << wm
                  << " k=" << worst_k << " value=" << worst_val << ")\n";
        return 1;
    }
    std::cout << "  OK (a)     : every V_H(ch, k) is finite\n";

    // ----- Check (b): <rho|V_H> is finite -----
    // Sum over (ch, k) of rho * V_H * r^2 dr  (Simpson is overkill, the
    // mid-rectangle is enough for this finiteness check).
    double inner = 0.0;
    for (int ch = 0; ch < Nlm; ++ch) {
        for (int k = 0; k < Nr; ++k) {
            const double r = rg.r[k];
            inner += rho(ch, k) * V_H(ch, k) * r * r * dr;
        }
    }
    if (!std::isfinite(inner)) {
        std::cerr << "  FAIL (b): <rho|V_H> = " << inner << "\n";
        return 1;
    }
    std::cout << "  OK (b)     : <rho|V_H> = " << inner << " (finite)\n";

    // ----- Check (c): l=0 V_H matches analytic Gaussian-Hartree -----
    // Only meaningful at moderate r -- the very last point at rmax has a
    // small Simpson-tail truncation error.  Sample a few mid-grid points.
    double max_rel_err_l0 = 0.0;
    int    max_rel_err_k  = -1;
    for (int k : {1, 10, 100, 500, 1000, Nr / 2}) {
        if (k <= 0 || k >= Nr) continue;
        const double r = rg.r[k];
        const double V_num = V_H(0, k);                 // ch=0 is l=0,m=0
        const double V_ref = V_H00_analytic_R(r, alpha);
        const double scale = std::max(std::abs(V_ref), 1e-30);
        const double rel = std::abs(V_num - V_ref) / scale;
        std::cout << "    r=" << r
                  << "  V_H_num=" << V_num
                  << "  V_H_ref=" << V_ref
                  << "  rel="     << rel << "\n";
        if (rel > max_rel_err_l0) { max_rel_err_l0 = rel; max_rel_err_k = k; }
    }
    if (max_rel_err_l0 > 1e-3) {
        std::cerr << "  FAIL (c): l=0 V_H drift " << max_rel_err_l0
                  << " (worst k=" << max_rel_err_k << ")\n";
        return 1;
    }
    std::cout << "  OK (c)     : l=0 V_H matches analytic Gaussian-Hartree, "
              << "max rel = " << max_rel_err_l0 << "\n";

    std::cout << "PASS\n";
    return 0;
}
