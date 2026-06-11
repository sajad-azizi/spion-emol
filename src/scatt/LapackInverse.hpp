// LapackInverse.hpp -- dense-matrix inverse via direct LAPACK, with an
// Eigen-native fallback.
//
// WHY NOT JUST Eigen::partialPivLu().inverse()?
//
//   When EIGEN_USE_MKL_ALL is defined at compile time, Eigen's LU already
//   dispatches to MKL's dgetrf/dgetri under the hood.  So for MKL builds
//   the two paths are functionally identical.  The payoff of going
//   directly through LAPACKE is:
//
//     * Explicit control over layout / workspace sizing (no Eigen wrapper
//       overhead for the once-per-step path in Sinv and Rinv).
//     * Future-proof: same interface lets us swap in dpotrf+dpotri (SPD
//       Cholesky, 2× faster) or dsytrf+dsytri (Bunch-Kaufman symmetric
//       indefinite) without touching callers.
//     * Bypasses Eigen's internal threshold heuristics -- for the
//       moderately-sized (N_psi × N_psi) matrices in the scattering loop,
//       Eigen sometimes chooses a suboptimal path.
//
//   When MKL is NOT linked (macOS dev box, no LAPACK), we transparently
//   fall back to Eigen's partialPivLu().inverse().  Bit-for-bit identical
//   numerics to what the code has been doing up to now.

#pragma once

#include <Eigen/Dense>

#if defined(SCATT_HAS_MKL)
// MKL_INT is `int` under -DMKL_LP64 (32-bit IPIV) and `long long int` under
// -DMKL_ILP64 (64-bit IPIV).  We use `MKL_INT` everywhere so this header
// adapts to either build flavour without changes:
//   * CPU build              : LP64    -> MKL_INT == int
//   * SYCL build (version_0  : ILP64   -> MKL_INT == long long int
//     recipe, -DMKL_ILP64)
// Do NOT replace MKL_INT with a fixed-width type -- mkl.h's prototypes
// for LAPACKE_dgetrf/dgetri are stamped with MKL_INT and you'll get
// silent ABI mismatches if the caller's int width disagrees.
#  include <mkl.h>        // LAPACKE_dgetrf, LAPACKE_dgetri, MKL_INT
#endif

#include <stdexcept>
#include <vector>

namespace scatt {

// Return  A^{-1}  for a real, square, INVERTIBLE matrix A.  Throws if the
// factorization reports a singular matrix.
//
// Both paths produce results within machine-precision of each other (they
// compute the same decomposition up to pivot-order permutations).  For the
// symmetrisation-dependent downstream users (SchurInverter, FRP), we leave
// the explicit  0.5*(M + M^T)  cleanup in the caller.
inline Eigen::MatrixXd inverse_general(const Eigen::MatrixXd& A)
{
    const Eigen::Index n = A.rows();
    if (A.cols() != n)
        throw std::runtime_error("inverse_general: matrix must be square");

#if defined(SCATT_HAS_MKL)
    // LAPACKE expects a modifiable buffer.  Use row-major layout to avoid
    // Eigen's default column-major <-> LAPACK dance.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        B(n, n);
    B.noalias() = A;

    std::vector<MKL_INT> ipiv(static_cast<std::size_t>(n));
    MKL_INT info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,
                                  static_cast<MKL_INT>(n),
                                  static_cast<MKL_INT>(n),
                                  B.data(),
                                  static_cast<MKL_INT>(n),
                                  ipiv.data());
    if (info != 0)
        throw std::runtime_error(
            "inverse_general: dgetrf failed (singular pivot), info = " +
            std::to_string(info));

    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR,
                          static_cast<MKL_INT>(n),
                          B.data(),
                          static_cast<MKL_INT>(n),
                          ipiv.data());
    if (info != 0)
        throw std::runtime_error(
            "inverse_general: dgetri failed, info = " +
            std::to_string(info));

    return B;  // Eigen converts row-major to column-major on assign-out.
#else
    return A.partialPivLu().inverse();
#endif
}

// Return  A^{-1}  for a real, square, SYMMETRIC INDEFINITE matrix A using
// Bunch-Kaufman LDLᵀ (LAPACKE_dsytrf + LAPACKE_dsytri).  At large N this is
// ~1.5-2x faster than inverse_general's general LU because:
//   * dsytrf does ~n³/3 flops vs dgetrf's ~n³ (same constant on dsytri vs
//     dgetri).
//   * Only the LOWER triangle of A is read.
//
// IMPORTANT: only the LOWER triangle of A is read.  The upper triangle is
// NEVER touched.  This is intentional: the SchurInverter caller builds S
// via a generic dgemm whose output has FP-rounding asymmetry of ~ε·||B||²
// ≈ 1e-13 relative -- the LOWER triangle is one valid representation of
// the mathematically symmetric matrix.  No pre-symmetrisation is needed.
//
// The returned matrix is FULL (both triangles populated by reflecting
// LOWER → UPPER), so downstream callers that read both triangles (e.g.
// BP's W^{-1} materialization) work unchanged.
//
// On singular pivot blocks (rare for invertible symmetric inputs; would
// require dsytrf to encounter a 2x2 zero block), throws -- same contract
// as inverse_general.  Caller can catch and fall back to the general LU
// path if needed.
//
// Eigen fallback: when MKL is not linked, the LAPACK symmetric path is not
// available -- we fall through to PartialPivLU, same as inverse_general.
// In that case the symmetric-indefinite call has identical numerics to the
// general call (just slower than dsytrf would have been).
inline Eigen::MatrixXd inverse_symmetric_indefinite(const Eigen::MatrixXd& A)
{
    const Eigen::Index n = A.rows();
    if (A.cols() != n)
        throw std::runtime_error(
            "inverse_symmetric_indefinite: matrix must be square");

#if defined(SCATT_HAS_MKL)
    // Eigen's MatrixXd is column-major by default.  Use COL_MAJOR + 'L'.
    // 'L' (LOWER) in column-major: dsytrf reads B(i,j) for i >= j.
    Eigen::MatrixXd B(n, n);
    B = A;   // deep copy; only lower triangle will be read

    std::vector<MKL_INT> ipiv(static_cast<std::size_t>(n));
    MKL_INT info = LAPACKE_dsytrf(LAPACK_COL_MAJOR, 'L',
                                  static_cast<MKL_INT>(n),
                                  B.data(),
                                  static_cast<MKL_INT>(n),
                                  ipiv.data());
    if (info != 0) {
        throw std::runtime_error(
            "inverse_symmetric_indefinite: dsytrf failed (singular pivot block), info = " +
            std::to_string(info));
    }

    info = LAPACKE_dsytri(LAPACK_COL_MAJOR, 'L',
                          static_cast<MKL_INT>(n),
                          B.data(),
                          static_cast<MKL_INT>(n),
                          ipiv.data());
    if (info != 0) {
        throw std::runtime_error(
            "inverse_symmetric_indefinite: dsytri failed, info = " +
            std::to_string(info));
    }

    // dsytri wrote only the LOWER triangle of A^{-1}.  Mirror LOWER -> UPPER
    // so callers get a fully populated, bit-symmetric matrix.  Diagonal is
    // already correct.
    for (Eigen::Index j = 1; j < n; ++j) {
        for (Eigen::Index i = 0; i < j; ++i) {
            B(i, j) = B(j, i);
        }
    }
    return B;
#else
    // No MKL linked: fall back to PartialPivLU, then explicitly symmetrise
    // to preserve the "returns bit-symmetric matrix" contract that the
    // MKL path provides via dsytri+mirror.  Same numerics as
    // inverse_general() + 0.5*(M+M^T) (i.e. the legacy path's exact
    // operations), so on the Mac/Eigen-only build the new symmetric-
    // indefinite path is BIT-EQUAL to the legacy general-LU + symmetrise
    // path.  We lose the speedup (still goes through general LU), but
    // bit-equality is preserved end-to-end on the test fixture.
    Eigen::MatrixXd Ainv = A.partialPivLu().inverse();
    Ainv = 0.5 * (Ainv + Ainv.transpose().eval());
    return Ainv;
#endif
}

}  // namespace scatt
