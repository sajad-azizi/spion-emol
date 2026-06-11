// KMatrixExtractor.hpp -- extract K, S, eigenphases from the
// renormalized-Numerov R-matrix at the outer asymptotic boundary.
//
// Derivation (repeated here because the convention MATTERS):
//
//   Asymptotic ansatz (PDF eq. 22, regular A = I normalization):
//       ψ_n  =  J_n  +  N_n · K
//   where J_n, N_n are diagonal with Riccati-Bessel ĵ_{ℓμ}(k·r_n) and
//   Riccati-Neumann ŷ_{ℓμ}(k·r_n) respectively.
//
//   In the free-centrifugal asymptotic region (V is centrifugal-only to
//   leading order) the Numerov recurrence relates Z̃_n = W^free_n · ψ_n
//   at two consecutive ON-GRID points r_{n_M}, r_{n_M+1} via the R-matrix
//       Z̃_{n_M+1}  =  R_{n_M} · Z̃_{n_M}
//   where R_{n_M} is the FORWARD ratio at grid index n_M. Storage-wise
//   our ForwardRPropagator stores Rinv_n = R_n^{-1}; so
//       R_{n_M}  =  FRP.get(n_M).inverse().
//
//   Matching (substitute ψ = J + N·K and rearrange):
//     A_mat := W^free_{n_M+1} · J_{n_M+1}    −  S_schur · W^free_{n_M} · J_{n_M}
//     B_mat := S_schur · W^free_{n_M} · N_{n_M} − W^free_{n_M+1} · N_{n_M+1}
//     K      = B_mat^{-1} · A_mat
//
//   where S_schur is the N_ψ × N_ψ Schur complement of R on the ψ block:
//     S_schur = R_ψψ  −  R_ψf · R_ff^{-1} · R_fψ
//   (projects out closed exchange channels).
//
// OUTER-MATCHING INDEX CONVENTION (explicit, differs from version_0):
//   Default n_M = N_grid − 2, matching at the pair (N_grid−2, N_grid−1).
//   Both points are ON-grid, and R_{n_M} = FRP.get(N_grid−2).inverse().
//
//   version_0 uses FRP.rinv_final() = Rinv[N_grid−1], whose R relates
//   r_{N_grid−1} to the OFF-grid r_{N_grid}. Combined with J, N evaluated
//   at r_{N_grid−2}, r_{N_grid−1} this is an off-by-one that introduces
//   an O(h) finite-difference error in K.
//
// S-matrix convention (PDF eq. 25):
//   S = (I + iK) · (I − iK)^{-1}.
//   Real symmetric K ⇒ S is unitary (S · S† = I).
//
// Eigenphases:
//   δ_α = atan(κ_α),  κ_α = eigenvalues of K.

#pragma once

#include "scatt/ForwardRPropagator.hpp"
#include "scatt/SolverParams.hpp"

#include <Eigen/Dense>

#include <complex>
#include <string>
#include <vector>

namespace scatt {

struct ScatteringResult {
    Eigen::MatrixXd         K_matrix;     // (N_psi, N_psi), real symmetric
    Eigen::MatrixXcd        S_matrix;     // (N_psi, N_psi), unitary
    std::vector<double>     eigenphases;  // atan(eig K), N_psi values
    double                  unitarity_err = 0.0;   // ||S†S − I||_max
    double                  K_symmetry_err = 0.0;  // ||K − Kᵀ||_max
    int                     n_match = -1;
    double                  r_match_inner = 0.0;
    double                  r_match_outer = 0.0;
    bool                    schur_coupling_zero = false;  // f-block dropped
};

class KMatrixExtractor {
public:
    // n_match_in < 0 defaults to N_grid − 2. Matching pair is
    // (n_match_in, n_match_in + 1).  Passing a different n_match lets the
    // test probe matching-radius convergence.
    KMatrixExtractor(const SolverParams& sp,
                     ForwardRPropagator& frp,
                     int                 n_match_in = -1);

    ScatteringResult extract();

    int n_match() const { return n_match_; }

    // Build the (N_psi × N_psi) boundary value of ψ at the outermost grid
    // point  r_{N_grid-1}  using the regular-normalization ansatz
    //     ψ_{N_grid-1}  =  J_{N_grid-1}  +  N_{N_grid-1} · K
    // where J, N are diagonal Riccati-Bessel / Riccati-Neumann. Supplied as
    // a convenience so the caller doesn't have to re-build J, N.
    static Eigen::MatrixXd make_psi_boundary(const SolverParams&    sp,
                                             const Eigen::MatrixXd& K);

private:
    const SolverParams& sp_;
    ForwardRPropagator& frp_;
    int                 n_match_;
    std::vector<int>    l_psi_;

    void build_channel_info_();

    // Riccati-Bessel ĵ_l(x) = x·j_l(x),  Riccati-Neumann ŷ_l(x) = x·y_l(x).
    static double riccati_j_(int l, double x);
    static double riccati_y_(int l, double x);
};

}  // namespace scatt
