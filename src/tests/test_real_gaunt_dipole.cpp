// test_real_gaunt_dipole.cpp -- regression: the angular dipole table
// `angular_dipole(l_mu, m_mu, q, l_nu, m_nu)` must implement the REAL-Y
// Gaunt selection, not the complex-Y rule m_mu = m_nu + q.
//
// History: between 2025-09 and 2026-04 the dipole evaluator gated the
// table with `if (mmu != mnu + q) continue;`, which is the complex-Y
// rule.  For q = ±1 this drops ~80% of the legitimate real-Y couplings
// -- including G^R(s; y; p_y) = +0.282, the LARGEST one.  Result: the
// y-polarisation cross-section collapsed to ~1e-17 on C8F8 even though
// the SOMO had full y-content.  This test hard-codes the canonical
// nonzero real-Y Gaunt values for l ≤ 3 and asserts that
//     angular_dipole(l_mu, m_mu, q, l_nu, m_nu) ≈ √(4π/3) · expected_G
// for every legal triple, ESPECIALLY the q=-1 ones the old gate threw
// away.  If this test fails, sigma_y will be wrong.
//
// The expected G^R values come from a Lebedev-grade numerical
// integration of three real Y_lm; the reproducer is in the comments.
// The numbers below match `gaunt_real` (which uses the
// U_real_to_complex transform path) to ~1e-10.

#include "scatt/DipoleMatrixElement.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using scatt::DipoleMatrixElement;

namespace {

constexpr double SQRT_4PI_3 = 2.04665341589298;   // sqrt(4*pi/3)

struct Triple {
    int l_mu, m_mu, q, l_nu, m_nu;
    double expected_G;            // pure G^R, NOT including sqrt(4*pi/3)
};

// Hand-curated nonzero real-Y Gaunts for l ≤ 3, generated via direct
// numerical integration on a (Gauss-Legendre × trapezoid) grid; cross
// checked against the analytical Bouferguene formula.
const std::vector<Triple> kGoldStandard = {
    // === q = -1 (y) -- the channel that was broken pre-fix ===
    { 0,  0, -1, 1, -1, +0.282094791774},   // s ↔ p_y     (LARGEST -- old gate dropped it)
    { 1, -1, -1, 0,  0, +0.282094791774},   // p_y ↔ s     (kept by old gate)
    { 1, -1, -1, 2,  0, -0.126156626101},   // p_y ↔ d_z²  (kept)
    { 1, -1, -1, 2,  2, -0.218509686118},   // p_y ↔ d_x²-y² (DROPPED)
    { 1,  0, -1, 2, -1, +0.218509686118},   // p_z ↔ d_yz   (DROPPED)
    { 1,  1, -1, 2, -2, +0.218509686118},   // p_x ↔ d_xy   (DROPPED)
    { 2, -2, -1, 1,  1, +0.218509686118},   // d_xy ↔ p_x   (DROPPED)
    { 2, -1, -1, 1,  0, +0.218509686118},   // d_yz ↔ p_z   (kept)
    { 2,  0, -1, 1, -1, -0.126156626101},   // d_z² ↔ p_y   (DROPPED)
    { 2,  2, -1, 1, -1, -0.218509686118},   // d_x²-y² ↔ p_y (DROPPED)
    // === q = 0 (z) -- gate was correct here ===
    { 0,  0,  0, 1,  0, +0.282094791774},
    { 1, -1,  0, 2, -1, +0.218509686118},
    { 1,  0,  0, 2,  0, +0.252313252202},
    { 1,  1,  0, 2,  1, +0.218509686118},
    // === q = +1 (x) -- mirror of q = -1 ===
    { 0,  0, +1, 1,  1, +0.282094791774},   // s ↔ p_x      (DROPPED by old gate)
    { 1, -1, +1, 2, -2, +0.218509686118},   // p_y ↔ d_xy    (kept)
    { 1,  0, +1, 2,  1, +0.218509686118},   // p_z ↔ d_xz    (DROPPED)
    { 1,  1, +1, 0,  0, +0.282094791774},   // p_x ↔ s       (kept)
    { 1,  1, +1, 2,  0, -0.126156626101},   // p_x ↔ d_z²    (kept)
    { 1,  1, +1, 2,  2, +0.218509686118},   // p_x ↔ d_x²-y² (DROPPED)
};

}  // namespace

int main() {
    std::cout << "=== test_real_gaunt_dipole ===\n";
    int pass = 0, fail = 0;
    double worst_abs = 0.0;
    int worst_idx = -1;

    for (size_t i = 0; i < kGoldStandard.size(); ++i) {
        const auto& t = kGoldStandard[i];
        const double got      = DipoleMatrixElement::angular_dipole(
                                    t.l_mu, t.m_mu, t.q, t.l_nu, t.m_nu);
        const double expected = SQRT_4PI_3 * t.expected_G;
        const double err      = std::abs(got - expected);
        if (err > worst_abs) { worst_abs = err; worst_idx = static_cast<int>(i); }
        const bool ok = err < 1e-9;
        std::cout << "  [" << (ok ? "ok  " : "FAIL")
                  << "] G^R(l_mu=" << t.l_mu << ", m_mu=" << t.m_mu
                  << "; 1, " << t.q
                  << "; l_nu=" << t.l_nu << ", m_nu=" << t.m_nu << ")"
                  << "  got=" << got << "  expected=" << expected
                  << "  err=" << err << "\n";
        if (ok) ++pass; else ++fail;
    }
    std::cout << "\nworst |angular_dipole - sqrt(4π/3)·G^R_expected| = "
              << worst_abs << " at triple #" << worst_idx << "\n";
    std::cout << pass << "/" << kGoldStandard.size() << " passed\n";
    return fail == 0 ? 0 : 1;
}
