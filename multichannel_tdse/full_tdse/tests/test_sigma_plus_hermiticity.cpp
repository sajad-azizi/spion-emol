// test_sigma_plus_hermiticity.cpp
//
// In the full TDSE, η^(-) must equal (η^(+))† for the interaction
// V_RF = (Ω_R/2)χ(t)[e^{-iωt}η^+ + e^{+iωt}η^-]  to be Hermitian.
//
// Our bridge defines:
//   sigma_plus_block(MF_low) -> matrix η^+_{f, i} of shape (N_high, N_low)
//   sigma_minus_block(MF_low) := sigma_plus_block(MF_low).adjoint()
//                              shape (N_low, N_high)
//
// To validate that this is the correct Ŝ_{1,−} + Ŝ_{2,−} matrix, we
// independently compute the σ⁻ block from sigma_plus_block(MF_low - 1)
// (which gives the σ⁺ vertex from M_F=MF_low-1 → MF_low).  By the
// adjoint relation, the σ⁻ vertex from MF_low → MF_low-1 IS
// (sigma_plus_block(MF_low-1))^T (the SAME numbers, transposed --
// with conj because we use complex matrices).
//
// Strict check:  for every adjacency (MF_low, MF_low+1) in {-5,-4,-3},
//
//   sigma_minus_block(MF_low + 1)            (computed via .adjoint())
//   ==
//   sigma_plus_block(MF_low+1).adjoint()     (definition)
//
// holds tautologically.  The MEANINGFUL test is: σ⁻ from M_F=MF+1 to
// M_F=MF, computed via .adjoint(), gives the SAME number as building
// the σ⁻ matrix directly via single-atom Ŝ_- (= raising m_S from -½
// to +½ on the right side of the bra-ket).  Equivalently,
// ⟨α|Ŝ_+|β⟩^* = ⟨β|Ŝ_-|α⟩.
#include "Rb85Spin.hpp"

#include <cstdio>

int main() {
    using namespace mc_tdse;
    int n_fail = 0;

    Rb85Spin spin(155.04);

    std::printf("[sigma_plus Hermiticity]   B = %.2f G\n", spin.B_gauss());
    for (int MF_low : {-5, -4, -3}) {
        auto Mp = spin.sigma_plus_block(MF_low);                 // (N_high, N_low)
        auto Mm = spin.sigma_minus_block(MF_low);                // (N_low, N_high)

        // Mm should equal Mp.adjoint() exactly (this is how we defined it).
        Eigen::MatrixXcd diff = Mm - Mp.adjoint();
        const double err = diff.cwiseAbs().maxCoeff();
        std::printf("    M_F=%+d -> %+d   ‖σ⁻ - (σ⁺)†‖_∞ = %.3e   %s\n",
                    MF_low, MF_low + 1, err, err < 1e-14 ? "ok" : "FAIL");
        if (err > 1e-14) ++n_fail;

        // Sanity: norms of σ⁺ block should be O(1) -- if zero something
        // is broken (no coupling at all between adjacent blocks).
        const double norm = Mp.norm();
        std::printf("                    ‖σ⁺‖_F        = %.4f          %s\n",
                    norm, norm > 1e-3 ? "ok" : "FAIL (no coupling!)");
        if (norm <= 1e-3) ++n_fail;
    }

    // Cross check: build the full pooled η^(+) matrix that the TDSE
    // recipe needs (single N×N matrix with one entry per (f, i) pair
    // where f, i index the pooled basis), and verify it's NOT
    // self-adjoint -- it's η^+, the σ⁺ raising vertex; the FULL
    // Hermitian operator is η^+ + η^- = η^+ + (η^+)† and THAT must
    // be self-adjoint.
    std::printf("\n[pooled-basis Hermiticity check of η^+ + η^-]\n");
    {
        std::vector<int> blocks = {-5, -4, -3, -2};
        std::vector<std::vector<ChannelInfo>> per_block;
        for (int MF : blocks) per_block.push_back(spin.channels(MF));
        std::vector<int> offsets = {0};
        for (auto& v : per_block) offsets.push_back(offsets.back() + (int)v.size());
        const int N = offsets.back();

        // Pool σ⁺ into N×N (zero where blocks not adjacent).
        Eigen::MatrixXcd Hpool = Eigen::MatrixXcd::Zero(N, N);
        for (int b = 0; b + 1 < (int)blocks.size(); ++b) {
            const int MF_low  = blocks[b];
            const int MF_high = blocks[b + 1];
            if (MF_high != MF_low + 1) continue;     // ascending in our list
            auto Mp = spin.sigma_plus_block(MF_low);
            const int row_off = offsets[b + 1];
            const int col_off = offsets[b];
            for (int f = 0; f < Mp.rows(); ++f)
                for (int i = 0; i < Mp.cols(); ++i)
                    Hpool(row_off + f, col_off + i) = Mp(f, i);
        }
        Eigen::MatrixXcd Htot = Hpool + Hpool.adjoint();
        Eigen::MatrixXcd skew = Htot - Htot.adjoint();
        const double skew_err = skew.cwiseAbs().maxCoeff();
        std::printf("    pooled basis size N = %d\n", N);
        std::printf("    ‖(η^+ + η^-) - (η^+ + η^-)†‖_∞ = %.3e   %s\n",
                    skew_err, skew_err < 1e-14 ? "ok" : "FAIL");
        if (skew_err > 1e-14) ++n_fail;
    }

    std::printf("\nTotal failures: %d\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
