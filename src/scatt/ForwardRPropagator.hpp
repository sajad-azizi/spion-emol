// ForwardRPropagator.hpp -- Johnson's forward R-recursion for Numerov.
//
// Computes Rinv[n] = R_n^{-1} for every grid point n ∈ [0, N_grid) where
//
//     R_n = Z̃_{n+1} · Z̃_n^{-1},   Z̃_n = W_n · Y_n
//
// with the recurrence (derived from Y'' = -Q Y + Numerov):
//
//     Rinv_n = ( U_n − Rinv_{n-1} )^{-1},      U_n = 12·W_n^{-1} − 10·I.
//
// Serial in n (data-dependent); parallelism comes from MKL on each per-n
// LU inverse and gemm. Storage is reused PotentialStorage with
// channels = N_total. Supports the same MEMORY / DISK / AUTO modes and
// the same hybrid "build in RAM + dump to disk for reuse" checkpoint
// policy as SchurInverter.
//
// Near-origin region (n < n_start):
//   D_ff(r) = 1 − (h²/12) ℓ(ℓ+1)/r² becomes non-positive at small r,
//   breaking the Numerov recurrence. We cannot apply the numerical
//   recursion there. Instead we fill Rinv analytically, DIAGONAL only:
//
//     Rinv_n[μ, μ]   = M_n^ψ · ĵ_{ℓμ}(k·r_n) / (M_{n+1}^ψ · ĵ_{ℓμ}(k·r_{n+1}))
//     Rinv_n[f, f]   = M_n^f · r_n^{ℓσ+1}    / (M_{n+1}^f · r_{n+1}^{ℓσ+1})
//
//   where M_n^ψ = 1 + (h²/12)·Q_ψψ(n, μ, μ) = 1 + (h²/12)·(2E − 2V(n, μ, μ))
//         M_n^f = 1 + (h²/12)·Q_ff (diagonal, clamped same as SchurInverter).
//
//   The Riccati-Bessel ĵ_ℓ is the regular ψ solution at small r (PDF eq. 21);
//   r^{ℓ+1} is the regular f solution (PDF eq. 23). These serve as the
//   Cauchy initial condition handed to the numerical recurrence at n_start.
//
// Safe radius threshold (matches version_0):
//     r_crit = h · √(ℓ_max_f(ℓ_max_f+1) / 12)    (D_ff = 0 here)
//     n_start = ⌈2·r_crit / h⌉                    (factor 2 safety margin)
//
// JOHNSON STABILITY AT n_start ≤ n < stab_n_max:
//   The Sinv that WInverseOperator uses may have been shifted by
//   SchurInverter when A or S had small eigenvalues. That regularization
//   is part of the Johnson recipe; the gold-standard comparison in the
//   test skips any n where a shift actually fired.

#pragma once

#include "scatt/Potentials.hpp"
#include "scatt/PotentialStorage.hpp"
#include "scatt/SolverParams.hpp"
#include "scatt/WInverseOperator.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace scatt {

class ForwardRPropagator {
public:
    struct Config {
        StorageMode storage      = StorageMode::AUTO;
        std::string checkpoint_dir;
        int         chunk_size   = 50;
        bool        try_load_checkpoint = true;
        bool        save_checkpoint     = true;
        double      W_min        = 0.001;       // same floor as SchurInverter
        bool        verbose      = true;

        // --- GPU offload (SYCL / oneMKL on Intel GPUs) ------------------
        // When true AND the binary was compiled with SCATT_HAS_SYCL AND
        // a GPU device is visible, the per-step LU inverse runs on GPU
        // with R_prev_inv pinned on device between steps.  Gives 10–80×
        // speedup at N_total ≳ 5000; ~0 or small slowdown below that
        // (PCIe transfer dominates compute).  Falls back to CPU with a
        // warning when the runtime check fails.
        bool        use_gpu      = false;

        // Use Bunch-Kaufman LDLᵀ (dsytrf+dsytri) instead of general LU
        // (dgetrf+dgetri) for the per-step Rinv on the CPU path.
        //
        // R_current(n) = U(n) − Rinv_prev is mathematically symmetric:
        //   * U(n) = W^{-1}(n) is the inverse of the symmetric block matrix
        //     W = [A B; B^T D].  W is symmetric ⇒ W^{-1} is symmetric.
        //   * Rinv_prev is bit-symmetric (analytic-init seed is diagonal;
        //     subsequent steps were post-symmetrised in the legacy code).
        //   * Sum/difference of symmetric matrices stays symmetric.
        //
        // The GEMMs that build U leave at most ~ε·||·||² ≈ 1e-13 relative
        // FP-rounding asymmetry in the upper triangle.  dsytrf reads only
        // the LOWER triangle, so this asymmetry is irrelevant.  The
        // returned Rinv is bit-symmetric (mirror LOWER → UPPER).
        //
        // Affects ONLY the CPU path of FRP::run.  The GPU path
        // (oneapi::mkl::lapack::getrf/getri inside GpuForwardStepper) is
        // unchanged.  Default ON; flip to false for legacy bit-equal
        // dgetrf+dgetri+symmetrise behaviour.
        //
        // Saves ~0.55 s/iter on a 17461² inversion (CPU path).  Validated
        // bit-tight on H2O fixture by test_frp_symmetric_inverse.
        bool        use_symmetric_inverse = true;

        // ZERO-ACCURACY-LOSS optimisation for the on-disk Rinv format.
        // When true, on-disk chunk files store ONLY the lower triangle of
        // each Rinv(r) matrix.  Rinv is bit-symmetric: either the legacy
        // 0.5*(Rinv+Rinvᵀ.eval()) post-symmetrise OR the GPU-path
        // kernel_symmetrize_ writes both triangles from the same scalar.
        // At L=100 (N_total=17461), the on-disk byte count drops from
        // 2.27 GB to 1.22 GB per matrix -- ALSO drops below the Linux
        // pread 2 GB syscall cap so the short-read bug becomes
        // structurally impossible.  In-memory Rinv stays full N×N.
        // Default false preserves the legacy format byte-for-byte.
        // Validated bit-equivalent by test_storage_symmetric.cpp.
        bool        symmetric_storage = false;

        // ZERO-ACCURACY-LOSS optimisation for the on-disk write path.
        // When true, Rinv chunk writes use a multi-threaded
        // pwrite-at-distinct-offsets implementation with atomic temp +
        // rename + fsync.  Validated bit-equivalent by
        // test_storage_parallel_write.  Default false preserves the
        // legacy serial write byte-for-byte.
        bool        parallel_chunk_write = false;

        // Planner-controlled async chunk prefetch for the DISK Rinv
        // storage.  Mirrors SchurInverter::Config::enable_prefetch
        // (see there for the budget rationale).  Default false.
        bool        enable_prefetch = false;

        // Allow PotentialStorage to re-chunk on load.  See
        // SchurInverter::Config::allow_chunk_rechunk.  Default ON.
        bool        allow_chunk_rechunk = true;
    };

    // `pot` is needed for the analytic near-origin init (reads diagonal V).
    ForwardRPropagator(const SolverParams& sp,
                       Potentials&         pot,
                       WInverseOperator&   wi);

    void run(const Config& cfg);
    void run() { run(Config{}); }

    // Fetch Rinv(n).
    const Eigen::MatrixXd& get(std::size_t ir);

    // Rinv at the outer matching point (n = N_grid − 1). Cached during run().
    const Eigen::MatrixXd& rinv_final() const { return rinv_final_; }

    int         n_start() const { return n_start_; }
    std::size_t n_grid()  const { return sp_.n_grid; }
    int         shifts_applied() const { return shifts_applied_; }

    const PotentialStorage& storage() const { return storage_; }

    // Forward to PotentialStorage::start_prefetch.  Lets callers (e.g.
    // BackPropagator's backward sweep) launch an async read of the next
    // chunk while the current chunk is still being GEMM'd.  Bit-identical
    // to the legacy synchronous path -- only scheduling changes.
    void start_prefetch(int chunk_idx) { storage_.start_prefetch(chunk_idx); }
    int  chunk_size_disk() const { return storage_.chunk_size(); }
    int  num_chunks_disk() const { return storage_.num_chunks(); }

    // Per-stage wall-clock accumulators (nanoseconds). Populated during run().
    struct Stats {
        std::uint64_t t_pot_fetch_ns     = 0;  // pot_.get(n) inside U assembly
        std::uint64_t t_u_assemble_ns    = 0;  // U = h²/6·(E·I - V) + ...
        std::uint64_t t_linsolve_ns      = 0;  // partialPivLu().inverse() at each n
        std::uint64_t t_store_ns         = 0;  // storage_.store(n, ...)
        std::uint64_t n_steps            = 0;  // # n-iterations
    };
    const Stats& stats() const { return stats_; }

private:
    const SolverParams& sp_;
    Potentials&         pot_;
    WInverseOperator&   wi_;

    PotentialStorage    storage_;
    Eigen::MatrixXd     rinv_final_;
    int                 n_start_         = 0;
    int                 shifts_applied_  = 0;

    std::vector<int>    l_psi_;
    std::vector<int>    l_sigma_;

    Stats stats_;

    void build_channel_info_();
    int  compute_n_start_() const;
    void analytic_init_(int n, double W_min, Eigen::MatrixXd& Rinv) const;

    // Riccati–Bessel ĵ_ℓ(x) = x · j_ℓ(x), via GSL.
    static double riccati_besselJ_(int l, double x);
};

}  // namespace scatt
