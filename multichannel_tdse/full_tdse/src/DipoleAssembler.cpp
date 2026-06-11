#include "DipoleAssembler.hpp"
#include "Rb85Spin.hpp"
#include "Common.hpp"

#include <stdexcept>

namespace mc_tdse {

Eigen::MatrixXcd assemble_d_plus(const BlockEigenstates& low,
                                 const BlockEigenstates& high,
                                 const Rb85Spin& spin) {
    if (high.M_F != low.M_F + 1)
        throw std::runtime_error("assemble_d_plus: high.M_F must equal low.M_F + 1");
    if (high.N_grid != low.N_grid || high.dr != low.dr)
        throw std::runtime_error("assemble_d_plus: BlockEigenstates must share the radial grid");

    // η⁺ vertex: shape (N_ch_high, N_ch_low).
    Eigen::MatrixXcd eta_full = spin.sigma_plus_block(low.M_F);
    if (eta_full.rows() < high.N_ch || eta_full.cols() < low.N_ch)
        throw std::runtime_error("assemble_d_plus: σ⁺ block too small for kept channel counts");
    // Truncate to (high.N_ch, low.N_ch) -- use a FRESH matrix to avoid the
    // self-assignment aliasing footgun (eta = eta.topLeftCorner(...)
    // resizes eta while the slice still references the old buffer ⇒ zeros).
    const Eigen::MatrixXcd eta = eta_full.topLeftCorner(high.N_ch, low.N_ch);

    const int Nb = low.n_states();
    const int Na = high.n_states();
    const int N_grid = low.N_grid;
    const double dr = low.dr;

    // Σ⁺ at recipe operating point (real B field) is REAL-valued; verify
    // and switch to a real fast path.  This is 2× faster (no complex
    // arithmetic) and Eigen vectorizes it cleanly.
    const double imag_norm = eta.imag().norm();
    if (imag_norm > 1e-12) {
        throw std::runtime_error(
            "assemble_d_plus: σ⁺ has nontrivial imaginary part; "
            "complex fast path not implemented (recipe assumes real B).");
    }
    const Eigen::MatrixXd eta_re = eta.real();
    const Eigen::MatrixXd eta_re_T = eta_re.transpose();   // (N_ch_low, N_ch_high)

    Eigen::MatrixXd d_plus_re = Eigen::MatrixXd::Zero(Na, Nb);

    // d_plus(α, β) = ∫dr w(ir) · u_α(ir,:) · η · u_β(ir,:)
    //
    // Inner GEMM:  v_β(ir, f) = Σ_i u_β(ir, i) · η^T_{i,f}  =  u_β · η^T
    //              shape (N_grid, N_ch_high) per β
    //
    // Then       d(α, β) = ∫dr w · u_α(ir,f) · v_β(ir,f)
    //                    = w_diag-weighted Frobenius product.
    //
    // Parallelism: outer α loop is OMP'd.  Each thread has its own v
    // workspace (no sharing).  Inner u_β · η^T is a small (N_grid × 2) ×
    // (2 × 2) GEMM that Eigen vectorizes.  Final scalar accumulation
    // is one Eigen array-prod-sum (vectorized SIMD).
    //
    // Cost: O(Na · Nb · N_grid · N_ch_high)  — single triple loop,
    // SIMD-friendly, OMP across α.  For (M_F=-5, M_F=-4) pair at
    // L=1e5 / dr=0.5 / N_grid=2e5: Na=84, Nb=4879 ⇒ ~3e11 ops, finishes
    // in O(seconds) with 8+ cores vs the OLD path's O(minutes).
    #pragma omp parallel
    {
        Eigen::MatrixXd v_workspace(N_grid, high.N_ch);

        #pragma omp for schedule(dynamic, 1)
        for (int alpha = 0; alpha < Na; ++alpha) {
            // Apply trapezoid endpoint weights to a pre-weighted u_α once.
            Eigen::MatrixXd u_a_w = high.u[alpha];   // copy (N_grid × N_ch_high)
            u_a_w.row(0)             *= 0.5;
            u_a_w.row(N_grid - 1)    *= 0.5;

            for (int beta = 0; beta < Nb; ++beta) {
                // v_β(ir, f) = u_β · η^T   (Eigen matmul; tiny inner dim → SIMD)
                v_workspace.noalias() = low.u[beta] * eta_re_T;
                // Frobenius product (Eigen .cwiseProduct().sum()).
                d_plus_re(alpha, beta) =
                    (u_a_w.array() * v_workspace.array()).sum() * dr;
            }
        }
    }
    return d_plus_re.cast<dcompx>();
}

}  // namespace mc_tdse
