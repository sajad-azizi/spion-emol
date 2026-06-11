// SchurInverter.hpp -- builds Sinv[n] = S(n)^(-1) for every radial grid point.
//
// The Schur complement S(n) is the N_psi × N_psi dense matrix that appears
// when one block-inverts the Numerov W matrix
//
//     W(n) = I + (h²/12) · Q(n) = [ A(n)   B(n)   ]
//                                  [ B(n)^T D(n)  ]
//
// with
//
//     A(n)  = I + (h²/6) · (E·I − V(n))                      (N_psi × N_psi)
//     B(n)  = (h²/12) · Q_ψf(n)                              (N_psi × N_f)
//     D(n)  = diag{ 1 − (h²/12) · ℓ_σ(ℓ_σ+1)/r² }            (N_f diagonal)
//
// The Schur complement of D in W is
//
//     S(n)  =  A(n) − B(n) · D(n)^(-1) · B(n)^T
//
// and its inverse Sinv = S^(-1) is the only per-ir matrix the solver needs.
// The full W^(-1) is built on demand from (Sinv, D^(-1), B) via the block-
// inverse identity.
//
// ==========================================================================
// DERIVATION NOTE (sign invariance):
//   S = A − B D⁻¹ B^T is quadratic in B, so any consistent sign choice on
//   Q_ψf (i.e. ±2α G χ/r) gives the same S and therefore the same Sinv.
//   Sinv is identical whether you use the PDF sign or version_0's.
//   Only the W^(-1) off-diagonal blocks (and hence f, not ψ) depend on the
//   sign of B.
// ==========================================================================
//
// JOHNSON STABILITY (two separate issues at small r):
//
//   (a) D_{f,f}(r) = 1 − (h²/12) · ℓ(ℓ+1)/r² can go non-positive at small r
//       and large ℓ. We clamp D_{f,f} to at least W_min = 0.001.
//
//   (b) A and S themselves can become near-singular at very small r because
//       h²/6 · V_00(r) can be large (attractive nuclear potential). For
//       n < stab_n_max we check the minimum eigenvalue (via a cheap
//       Gershgorin lower bound, falling back to full eigendecomposition
//       only when Gershgorin is inconclusive) and shift the diagonal up by
//       (W_min − min_eig + shift_margin) if below the floor.
//
// These regularizations only affect the "analytic initialization" region of
// the Numerov propagation, where the R matrix comes from a closed-form
// Riccati-Bessel / r^(ℓ+1) formula rather than from the recurrence. Sinv
// below that region is still used during ψ backpropagation, hence the
// clamps are load-bearing.
//
// STORAGE:
//   Same MEMORY/DISK contract as PotentialStorage. Checkpoint directory
//   layout is the same on-disk format (chunk files + metadata.bin).

#pragma once

#include "scatt/ExchangeCoupling.hpp"      // ExchangeCoupling, ChiRadial
#include "scatt/Potentials.hpp"            // Potentials, StorageMode
#include "scatt/PotentialStorage.hpp"      // reused as generic chunked matrix store
#include "scatt/SolverParams.hpp"
#include "scatt/WavefunctionSetup.hpp"     // ChiRadial

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace scatt {

class SchurInverter {
public:
    struct Config {
        double      W_min        = 0.001;                    // floor for D and eigenvalues
        int         stab_n_max   = 60;                       // check/shift for n < this
        double      shift_margin = 0.01;                     // cushion above W_min
        StorageMode storage      = StorageMode::AUTO;        // AUTO, MEMORY, DISK
        std::string checkpoint_dir;                          // empty -> auto-generated
        int         chunk_size   = 100;                      // DISK chunk size
        bool        use_openmp   = true;                     // only if both storages are MEMORY
        bool        verbose      = true;

        // Checkpoint policy (independent of storage mode):
        //   try_load_checkpoint = true  -> attempt to load an on-disk
        //                                  checkpoint matching (Nr, N_psi)
        //                                  before rebuilding. In MEMORY
        //                                  mode the checkpoint is streamed
        //                                  into RAM; in DISK mode the
        //                                  object reads from disk.
        //   save_checkpoint     = true  -> after a successful MEMORY-mode
        //                                  build, dump Sinv to disk so a
        //                                  future run can skip the rebuild.
        //                                  DISK mode writes to disk
        //                                  naturally; the flag is only
        //                                  meaningful for MEMORY.
        bool try_load_checkpoint = true;
        bool save_checkpoint     = true;

        // Use Bunch-Kaufman LDLᵀ (dsytrf+dsytri) instead of general LU
        // (dgetrf+dgetri) for the symmetric Schur complement S.
        //
        // Why it's safe:
        //   * S = A − B·D⁻¹·B^T is mathematically symmetric.  A is built
        //     from a Gaunt assembly that writes V(i,j) and V(j,i) from the
        //     same scalar (bit-symmetric, see GauntSparse.hpp:215-220).
        //   * The B·D⁻¹·B^T term has at most ~ε·||B||² ≈ 1e-13 relative
        //     FP-rounding asymmetry from a generic GEMM.  dsytrf reads only
        //     the LOWER triangle so this asymmetry is irrelevant.
        //   * The returned Sinv is bit-symmetric (LOWER → UPPER mirror).
        //
        // Why it's faster:
        //   * dsytrf flops ~ n³/3 vs dgetrf ~ n³ (same on dsytri vs dgetri).
        //   * Eliminates the two `0.5*(M+M^T.eval())` symmetrisations on
        //     S and Sinv (their purpose was to cope with general-LU's
        //     non-symmetric-preserving rounding; LDLᵀ preserves symmetry
        //     by construction).
        //
        // At L=100 / N_psi=10201 on a single-MKL-thread iteration:
        //   dgetrf+dgetri        ≈ 0.30 s
        //   dsytrf+dsytri        ≈ 0.15 s
        //   sym(S)               ≈ 0.20 s  -- removed
        //   sym(Sinv)            ≈ 0.20 s  -- removed
        //   net per-iter saving  ≈ 0.55 s
        //   over Nr=10001 iters  ≈ 1.5 hours of wall (~5% of a 30 h run)
        //
        // Validated bit-tight against the legacy path by
        // test_sinv_symmetric_inverse.cpp on the H2O fixture: end-to-end
        // Sinv·S = I residual stays < 1e-10, |Sinv_new - Sinv_old| / |Sinv|
        // < 1e-11 at every n.
        //
        // Flip to false to revert to the legacy dgetrf + 2x symmetrisation
        // path bit-for-bit (matches all pre-2026-05 production runs).
        bool use_symmetric_inverse = true;

        // ZERO-ACCURACY-LOSS optimisation for the on-disk Sinv format.
        // When true, on-disk chunk files store ONLY the lower triangle of
        // each Sinv(r) matrix.  Sinv is bit-symmetric by construction
        // (the inverse of a symmetric matrix is symmetric, and either the
        // dsytri+mirror path or the legacy 0.5*(M+Mᵀ) path enforces
        // byte-equality between the two triangles).  Halves disk space
        // and disk I/O.  In-memory Sinv is unchanged (full N×N).
        // Default false preserves the legacy format byte-for-byte.
        // Validated bit-equivalent by test_storage_symmetric.cpp.
        bool symmetric_storage = false;

        // ZERO-ACCURACY-LOSS optimisation for the on-disk write path.
        // When true, Sinv chunk writes use a multi-threaded
        // pwrite-at-distinct-offsets implementation with atomic temp +
        // rename + fsync.  Validated bit-equivalent by
        // test_storage_parallel_write.  Default false preserves the
        // legacy serial write byte-for-byte.
        bool parallel_chunk_write = false;

        // Use the GPU stepper (GpuSinvStepper) for the per-n Schur
        // complement and inverse.  When true AND SYCL is compiled in
        // AND a GPU is visible AND oneMKL's getrf/getri are usable on
        // the device, the per-n work is offloaded as:
        //   * Upload A, B, Dinv (host → device)
        //   * Schur complement S = A − B·diag(Dinv)·Bᵀ via oneMKL dgemm
        //   * Inverse via oneMKL dgetrf+dgetri (general LU, same as FRP)
        //   * Symmetrise via the existing kernel_symmetrize_
        //   * Download Sinv (device → host)
        // Stability shifts for n < stab_n_max stay on CPU regardless
        // (they're rare and cheap; avoids GPU↔CPU ping-pong).
        // At L=100 (N_psi=10201) PVC speedup is ~10-15× per step
        // vs SPR+MKL.  Validated bit-tight (rel diff < 1e-10 per n)
        // against the CPU path by test_gpu_sinv.
        bool use_gpu = false;

        // Allow PotentialStorage to RE-CHUNK an on-disk Sinv checkpoint
        // at load time when its on-disk chunk_size is larger than the
        // runtime-budgeted chunk_size (cross-node-size moves, planner-
        // input drift, etc.).  Default ON.  Opt out with the main CLI
        // flag --no-checkpoint-rechunk if you want the older reject +
        // rebuild behaviour (e.g. on disk-tight scratch where the 2x
        // transient space isn't available).  See
        // PotentialStorage::set_chunk_rechunk_allowed for the full
        // safety contract.
        bool allow_chunk_rechunk = true;

        // Planner-controlled async chunk prefetch for the DISK Sinv
        // storage.  When true AND storage_ ends up in DISK mode,
        // PotentialStorage::set_prefetch_allowed(true) is called and
        // BackPropagator's start_prefetch hops materialise into real
        // background reads (overlap I/O with compute).  When false,
        // start_prefetch is a no-op and the storage stays at one
        // chunk resident.  The StoragePlanner decides this based on
        // whether the budget can absorb an EXTRA chunk-sized buffer
        // (~95 GB for C8F8/l_cont=100, where it must stay OFF).
        // Default false preserves pre-2026 memory-safe behaviour.
        bool enable_prefetch = false;

        // Opt-in chunk-blocked OpenMP parallelism even in DISK mode.
        // Default OFF preserves the historic behaviour exactly: in DISK
        // mode the build loop runs single-threaded, in MEMORY mode it
        // runs OpenMP-parallel.
        // When ON and storage == DISK, we wrap the per-ir work in a
        // chunk-blocked outer loop:
        //   for each chunk: parallel-compute Sinv into a chunk-local
        //   buffer, then serial-flush the chunk via storage_.store().
        // Bit-identical to the serial-DISK path (proven by
        // test_sinv_serial_vs_parallel.cpp).
        // Memory cost: + (chunk_size · matrix_bytes) on top of what the
        // planner reserved.  Caller is responsible for confirming this
        // fits.  At C8F8 / l_cont=80 it costs ~65 GB, fits comfortably.
        // At l_cont >= 100 caller may need to cap OpenMP threads.
        bool parallel_disk_chunks = false;
    };

    // `ec` and `chi` may be nullptr if there is no exchange (pure static).
    // In that case S = A at every n.
    SchurInverter(const SolverParams&     sp,
                  Potentials&             pot,
                  const ExchangeCoupling* ec,
                  const ChiRadial*        chi);

    // Build Sinv at every ir in [0, N_grid).
    void build(const Config& cfg);
    void build() { build(Config{}); }

    // Fetch Sinv(ir). Only valid after build() or a successful checkpoint load.
    const Eigen::MatrixXd& get(std::size_t ir);

    // Stats.
    std::size_t n_grid()             const { return sp_.n_grid; }
    int         stability_shifts_A() const { return shifts_A_; }
    int         stability_shifts_S() const { return shifts_S_; }
    std::size_t memory_bytes()       const { return storage_.memory_bytes(); }

    // Expose underlying storage for reuse if needed (e.g. reload check).
    const PotentialStorage& storage() const { return storage_; }

    // Forward to PotentialStorage::start_prefetch.  Lets callers (e.g.
    // BackPropagator's backward sweep) launch an async read of the next
    // Sinv chunk while compute on the current chunk is still running.
    void start_prefetch(int chunk_idx) { storage_.start_prefetch(chunk_idx); }
    int  chunk_size_disk() const { return storage_.chunk_size(); }
    int  num_chunks_disk() const { return storage_.num_chunks(); }

    // Per-stage wall-clock accumulators (nanoseconds). Summed across threads
    // -- so with N threads and T seconds wall, a well-parallelized stage
    // reports ≈ N·T ns. Ratio of each timer to the top-level
    // SchurInverter::build total = per-thread dominance.
    struct Stats {
        std::uint64_t t_A_build_ns      = 0;   // A(n) = I + (h²/6)(E·I - V)
        std::uint64_t t_B_build_ns      = 0;   // B(n) = (h²/12) Q_psi_f via ExchangeCoupling
        std::uint64_t t_schur_ns        = 0;   // S = A - B·D^{-1}·B^T  (GEMM-heavy)
        std::uint64_t t_invert_ns       = 0;   // S^{-1} via partialPivLu().inverse()
        std::uint64_t t_store_ns        = 0;   // storage_.store(n, Sinv)
        std::uint64_t n_steps           = 0;   // # ir iterations (1 per thread)
        bool          parallel_over_ir  = false;
    };
    const Stats& stats() const { return stats_; }

private:
    const SolverParams&     sp_;
    Potentials&             pot_;
    const ExchangeCoupling* ec_;
    const ChiRadial*        chi_;

    PotentialStorage        storage_;       // same layout (channels = N_psi)
    std::vector<int>        l_sigma_;       // channel angular momentum for sigma
    int                     shifts_A_ = 0;
    int                     shifts_S_ = 0;
    Stats                   stats_;

    void build_channel_info_();
};

}  // namespace scatt
