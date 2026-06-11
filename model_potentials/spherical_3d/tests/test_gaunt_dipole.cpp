// test_gaunt_dipole.cpp -- a small set of canonical real-Y dipole
// gaunts G^R(l_a, m_a; 1, q; l_b, m_b) checked against hand-computed
// values from the parent project (src/tests/test_real_gaunt_dipole.cpp).
//
// If this passes, our local Angular.hpp is byte-compatible with the
// production scattering code's gaunt_real / U_real_to_complex.
#include "Angular.hpp"
#include <cstdio>

int main() {
    struct Triple { int la, ma, q, lb, mb; double expect; };
    // Subset of the parent project's gold standard, re-curated:
    const std::vector<Triple> golden = {
        // q = -1 (y), the channel that was broken pre-fix in the parent project.
        { 0,  0, -1, 1, -1, +0.282094791774},   // s ↔ p_y (largest)
        { 1, -1, -1, 0,  0, +0.282094791774},   // p_y ↔ s
        { 1, -1, -1, 2,  0, -0.126156626101},   // p_y ↔ d_z²
        { 1, -1, -1, 2,  2, -0.218509686118},   // p_y ↔ d_x²-y²
        { 1,  0, -1, 2, -1, +0.218509686118},   // p_z ↔ d_yz
        // q = 0 (z)
        { 0,  0,  0, 1,  0, +0.282094791774},   // s ↔ p_z
        { 1,  0,  0, 0,  0, +0.282094791774},
        { 1,  0,  0, 2,  0, +0.252313252202},   // p_z ↔ d_z²
        { 1,  1,  0, 2,  1, +0.218509686118},   // p_x ↔ d_xz
        // q = +1 (x)
        { 0,  0,  1, 1,  1, +0.282094791774},   // s ↔ p_x
        { 1,  1,  1, 0,  0, +0.282094791774},
        { 1,  1,  1, 2,  0, -0.126156626101},   // p_x ↔ d_z²
        { 1,  1,  1, 2,  2, +0.218509686118},   // p_x ↔ d_x²-y²
    };

    int n_fail = 0;
    double max_err = 0.0;
    for (const auto& t : golden) {
        const double g = ang3d::gaunt_real(t.la, t.ma, 1, t.q, t.lb, t.mb);
        const double err = std::fabs(g - t.expect);
        if (err > max_err) max_err = err;
        const bool ok = err < 1e-12;
        std::printf("%s G^R(%d,%+d; 1,%+d; %d,%+d) = %+.12f   "
                    "expect %+.12f   err=%+.2e\n",
                    ok ? " OK " : "FAIL",
                    t.la, t.ma, t.q, t.lb, t.mb, g, t.expect, err);
        if (!ok) ++n_fail;
    }
    std::printf("[gaunt_dipole] checked=%zu  max|err|=%.2e  fails=%d\n",
                golden.size(), max_err, n_fail);
    return (n_fail == 0) ? 0 : 1;
}
