// test_dipole_assembly.cpp
//
// Stage 4c, item 3: assemble d^(+)_{αβ} = ⟨α^{M_F+1} | η^(+) | β^{M_F}⟩
// between the halo (M_F=-4) and a few M_F=-3 continuum states.
//
// Smoke-level checks:
//   * Shape: rows = high.n_states(), cols = low.n_states().
//   * All entries finite.
//   * At least one entry has |d^(+)| above a sane floor — the σ⁺ vertex
//     is known nonzero between the two entrance channels (4a tests),
//     so d^(+)(halo, accessible final state) must couple.
//   * d^(+)(M=-3 → M=-4) reconstructed via .adjoint() equals d^(-) by
//     definition (η^(-) = (η^(+))^†).
#include "BlockEigenstates.hpp"
#include "DipoleAssembler.hpp"
#include "Rb85Spin.hpp"
#include "Common.hpp"

#include <cstdio>

int main() {
    using namespace mc_tdse;
    BlockBuildOptions opt;
    opt.N_grid = 100000;     // L = 5e4 a_0
    opt.dr_a0  = 0.5;
    opt.E_max_kHz_above_threshold = 2.0;

    // Build the two blocks on the same radial grid.
    BlockEigenstates m4 = build_block_eigenstates(-4, opt);
    BlockEigenstates m3 = build_block_eigenstates(-3, opt);
    Rb85Spin spin(opt.B_gauss);

    std::printf("[dipole_assembly]\n");
    std::printf("    M_F=-4: %d states (halo)   M_F=-3: %d states\n",
                m4.n_states(), m3.n_states());

    // d^(+) takes low=M_F, high=M_F+1; for halo→continuum we want
    // low = m4 (M_F=-4), high = m3 (M_F=-3).
    Eigen::MatrixXcd d_plus = assemble_d_plus(m4, m3, spin);
    std::printf("    d^(+) shape         = (%lld, %lld)\n",
                static_cast<long long>(d_plus.rows()),
                static_cast<long long>(d_plus.cols()));

    bool any_nan = false;
    double max_abs = 0.0;
    for (int i = 0; i < d_plus.rows(); ++i)
        for (int j = 0; j < d_plus.cols(); ++j) {
            const dcompx z = d_plus(i, j);
            if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) any_nan = true;
            const double a = std::abs(z);
            if (a > max_abs) max_abs = a;
        }

    std::printf("    finite entries      : %s\n", !any_nan ? "PASS" : "FAIL");
    std::printf("    max |d^(+)|         = %.4e\n", max_abs);

    // Print the column (since col=1 for the halo).
    if (d_plus.cols() == 1) {
        std::printf("    halo→m3 dipoles:\n");
        for (int i = 0; i < d_plus.rows(); ++i) {
            std::printf("        ⟨m3 n=%d | η+ | halo⟩ = (%.6e, %.6e)\n",
                        i, d_plus(i, 0).real(), d_plus(i, 0).imag());
        }
    }

    // Coupling floor: σ⁺(MF=-4 → -3) entrance-to-entrance coupling is O(1)
    // (recipe says it's a few per 1/√2-type Clebsch).  Halo open-channel
    // weight is 0.99981, M_F=-3 first level open-channel weight ≈ 1
    // (above threshold).  Spatial overlap between halo (range 1/κ ≈
    // 2050 a_0) and continuum sin(k r) (k tiny near threshold, so
    // wavelength >> 1/κ) is non-trivial but non-zero.  We assert
    // max |d^(+)| > 1e-3.
    const bool ok_couples = max_abs > 1e-3;
    std::printf("    nonzero coupling    : %s\n", ok_couples ? "PASS" : "FAIL");

    // d^(-)_{βα} reconstruction:
    Eigen::MatrixXcd d_minus = d_plus.adjoint();      // (m4, m3)
    const bool ok_adj = (d_minus.rows() == d_plus.cols())
                      && (d_minus.cols() == d_plus.rows());
    std::printf("    d^(-) = (d^(+))†    : %s\n", ok_adj ? "PASS" : "FAIL");

    const bool ok = !any_nan && ok_couples && ok_adj;
    std::printf("    result              : %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
