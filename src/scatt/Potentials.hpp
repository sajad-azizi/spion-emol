// Potentials.hpp -- static-exchange-HF scattering potential matrix.
//
// Port of version_0/src/Potentials.{hpp,cpp}. Same algorithm, same
// conventions, same sparse-Gaunt optimization. Differences are purely
// structural / I/O:
//
//   1. The V_ee array (called V_lm_ee in version_0) is now loaded from
//      our preprocessing HDF5 under /potential/V_H. Version_0 used raw
//      pread binaries; we use the HDF5 Reader. The math is identical.
//
//   2. The V_en multipole expansion is computed on-the-fly every radial
//      point, exactly like version_0. This is the cheaper path (O(Natoms
//      * Lmax) per r) and keeps the V_en information localized to one
//      routine that is easy to audit.
//
//   3. The Gaunt contraction V_{mu',mu}(r) = sum_sigma V_sigma(r) *
//      G^R(mu', sigma, mu) is done via GauntSparseMatrix (inlined
//      version of the version_0 "potential_sparse_matrix.hpp"), which
//      at Lmax=100 is the difference between ~seconds per r and
//      ~infeasible.
//
//   4. The potential matrix V(r) is stored in a flexible container
//      PotentialStorage which can hold in memory (small runs) or chunk
//      to disk (C8F8-scale).
//
// Interface:
//
//     Potentials pot(params);
//     pot.build(reader);                // assemble from preprocessing HDF5
//     const auto& V_at_r = pot.get(ir); // channels x channels dense matrix
//
// The Parameters struct must already be validate()d.

#pragma once

#include "angular/GauntSparse.hpp"
#include "io/HDF5Reader.hpp"
#include "scatt/Parameters.hpp"
#include "scatt/PotentialStorage.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scatt {

// Memory policy for V(r) storage.
//
//   MEMORY      All V(r) kept in RAM. OpenMP parallel over radial points
//               during build() is SAFE because each thread writes a
//               distinct ir slot. Fast reads.
//
//   DISK        V(r) streamed to chunk files on disk as we build. Build
//               loop is SERIAL over r (the write_buffer is sequential
//               state, multi-thread would race). Read phase uses a
//               single-chunk cache; serial by contract.
//
//   ON_DEMAND   No storage. Every V_at(ir) call recomputes by sparse
//               matvec. Safe under parallel reads (each call allocates
//               its own dense matrix). Used for tests / when the
//               Gaunt matrix fits in RAM but Nr * channels^2 does not
//               AND disk I/O is undesirable.
//
//   AUTO        Pick MEMORY if the total V-cube is <= memory_budget_bytes
//               (from Parameters), else DISK. The default for top-level
//               driver code.
enum class StorageMode { AUTO, MEMORY, DISK, ON_DEMAND };

class Potentials {
public:
    explicit Potentials(const Parameters& params);

    // Assemble V(r). With AUTO/MEMORY/DISK we run the full per-r loop
    // (parallel for MEMORY, serial for DISK) and cache to storage_. With
    // ON_DEMAND we only build the sparse Gaunt matrix and borrow `data`
    // so that V_at(ir) can rebuild lazily; `data` must outlive this
    // Potentials object in that case.
    //
    // checkpoint_dir is used by DISK mode as the directory for chunk
    // files; a valid existing checkpoint there is reloaded without
    // recomputing. Leave empty to auto-generate a parameter-encoded name.
    //
    // Checkpoint policy (same manifest + SUCCESS pattern as SchurInverter,
    // ForwardRPropagator, BackPropagator):
    //   try_load_checkpoint: if a matching finalized checkpoint exists at
    //                        `checkpoint_dir`, load it and skip rebuild.
    //   save_checkpoint    : after a MEMORY build, also dump to disk with
    //                        manifest for future reloads. DISK mode writes
    //                        during the build regardless.
    //
    // V depends only on the molecule + grid + l_cont + Lmax_sce (NOT on
    // energy), so this checkpoint can be reused across all energies of a
    // scan.
    // symmetric_storage:
    //   ZERO-ACCURACY-LOSS optimisation for the on-disk checkpoint format.
    //   When true, on-disk chunk files store ONLY the lower triangle of
    //   each V(r) matrix (V is bit-symmetric by Gaunt-write construction;
    //   see GauntSparse.hpp:215-220).  Halves disk space and disk I/O.
    //   In-memory representation is unchanged (full N×N).
    //   Default false preserves the legacy format byte-for-byte.
    //   Backward-compatible: pre-existing v1 (non-prefixed) checkpoints
    //   load correctly under the new code regardless of this flag.
    // parallel_chunk_write:
    //   ZERO-ACCURACY-LOSS optimisation for the on-disk write path.
    //   When true, write_chunk dispatches to a multi-threaded
    //   pwrite-at-distinct-offsets implementation with atomic temp +
    //   rename + fsync.  Validated bit-equivalent by
    //   test_storage_parallel_write.  Default false preserves the
    //   legacy serial single-threaded write byte-for-byte.
    void build(const io::PreprocData& data,
               StorageMode        mode              = StorageMode::AUTO,
               const std::string& checkpoint_dir    = "",
               bool               verbose           = true,
               bool               try_load_checkpoint = true,
               bool               save_checkpoint     = true,
               bool               symmetric_storage   = false,
               bool               parallel_chunk_write = false);

    // Three ways to access V(r_ir). Pick by cost:
    //
    //               MEMORY          DISK                ON_DEMAND
    //   get()       ref, FREE       ref to chunk-cache  THROWS
    //                               (same-chunk: free;
    //                                new chunk: one I/O
    //                                + memcpy)
    //   V_at()      by-val COPY     by-val COPY         BUILD from Gaunt
    //               (one matrix)    + possible I/O      + matvec (slow)
    //   apply_V()   y = V x, uses get() internally in MEMORY/DISK and
    //               V_at() only in ON_DEMAND. Never heap-allocates the
    //               full matrix in the caller's space in ON_DEMAND
    //               because it immediately matvec-contracts.
    //
    // Numerov / any hot loop should use get() or apply_V(). V_at() is a
    // convenience for code that needs to keep the matrix alive beyond
    // the next get() call (in DISK mode the reference becomes stale
    // after another get on a different chunk).
    //
    // In MEMORY mode get(ir) is the literal vector slot --
    // `pot_storage_.memory_storage_[ir]` -- exactly analogous to
    // version_0's `pot_component[ir]`. No recomputation.

    // Const reference. Throws in ON_DEMAND (no storage to refer to).
    const Eigen::MatrixXd& get(std::size_t ir) const;

    // Always returns a new Eigen matrix by value. Works in every mode.
    Eigen::MatrixXd V_at(std::size_t ir) const;

    // Fused y = V(r_ir) * x. Allocates nothing in the caller; y is
    // resized if necessary. Safe in any mode.
    void apply_V(std::size_t ir, const Eigen::VectorXd& x,
                 Eigen::VectorXd& y) const;

    // Diagnostics after build().
    double max_symmetry_deviation(std::size_t n_samples = 0) const;    // max ||V - V^T||_inf
    void   check_cubic_symmetry() const;                              // prints p_x/p_y/p_z diag

    // Read-only references for later milestones (Wavefunctions, Dipole).
    const Parameters&      parameters()   const { return params_; }
    const PotentialStorage& storage()     const { return pot_storage_; }
    StorageMode            storage_mode() const { return mode_; }

    // Free the underlying chunk read cache (DISK mode only).  After this,
    // a subsequent get(ir) will lazily re-read the requested chunk from
    // disk -- correctness is preserved.  Used by run_one_energy() to
    // reclaim ~155 GB at L=100 between FRP::run and BP::run, since pot is
    // not touched at all by BP, K-extraction, AsymptoticAmplitudes, or
    // DipoleMatrixElement (the BP outer-boundary V_N is pre-cached by the
    // caller).  No-op in MEMORY mode.
    void release_read_buffer() { pot_storage_.release_read_buffer(); }

    // Sparse-Gaunt introspection for benchmarks.
    std::size_t gaunt_nonzeros()    const { return gaunt_.nonzeros();    }
    std::size_t gaunt_memory_bytes() const { return gaunt_.memory_bytes(); }

    // Per-phase wall-clock accumulators (nanoseconds). Summed across threads
    // — in MEMORY-parallel mode, ratio (sum / top-level Potentials::build)
    // reveals effective thread count for the assembly loop.
    struct Stats {
        std::uint64_t t_gaunt_build_ns = 0;    // sparse Gaunt table construction (once)
        std::uint64_t t_v_sigma_ns     = 0;    // compute_V_sigma_total (per ir)
        std::uint64_t t_gaunt_matvec_ns = 0;   // gaunt_.compute_V (per ir)
        std::uint64_t t_v_pol_ns        = 0;   // compute_U_polarization_at_r (per ir)
        std::uint64_t t_store_ns        = 0;   // pot_storage_.store (per ir)
        std::uint64_t n_steps           = 0;   // # ir iterations processed
        bool          parallel_over_ir  = false;
    };
    const Stats& stats() const { return stats_; }

private:
    // Compute the (l_exp_max+1)^2-sized V_sigma vector at one radial r.
    // Combines:
    //   - V_ee from preprocessing HDF5 (uniform-grid column at index ir),
    //   - V_en from on-the-fly multipole expansion over nuclei (Ri, Zi).
    Eigen::VectorXd compute_V_sigma_total(
        double r, std::size_t ir,
        const io::PreprocData& data) const;

    // Polarization model (version_0 returned zero; we mirror that here
    // and expose a hook for later refinement using the full alpha tensor).
    Eigen::MatrixXd compute_U_polarization_at_r(double r,
                                                const io::PreprocData& data) const;

    const Parameters&          params_;
    angular::GauntSparseMatrix gaunt_;
    PotentialStorage           pot_storage_;       // MEMORY or DISK
    StorageMode                mode_ = StorageMode::AUTO;
    const io::PreprocData*     data_ptr_ = nullptr; // borrowed; valid iff ON_DEMAND
    Stats                      stats_;
};

}  // namespace scatt
