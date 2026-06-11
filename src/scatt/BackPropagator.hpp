// BackPropagator.hpp -- reconstruct ψ(r) and (optionally) f(r) on the
// radial grid by backward application of the stored R-matrix.
//
// Algorithm (derived from Y'' = -Q Y  +  Numerov):
//
//     Z̃_n  =  Rinv_n · Z̃_{n+1}          (one backward step, one matmul)
//     Y_n  =  W_n^{-1} · Z̃_n             (apply WInverseOperator)
//     ψ_n  =  Y_n.top(N_ψ),   f_n = Y_n.bottom(N_f)
//
// Boundary condition at n = N_grid − 1:
//     Y_N = [ ψ_N ; 0 ]                    (closed f-channels ⇒ f_N = 0)
//     Z̃_N = W_N · Y_N = [ A_N · ψ_N ;  B_Nᵀ · ψ_N ]
//         (block expansion of W_N Y_N; see cpp for details)
//
// The boundary ψ_N is supplied by the caller. The natural choice from the
// extracted K-matrix is
//     ψ_N  =  J_N + N_N · K             (regular normalization A = I, B = K)
// available via KMatrixExtractor::make_psi_boundary.
//
// ---- MEMORY OPTIMIZATION (user request) --------------------------------
// For the dipole overlap  ⟨ψ_final | r | ψ_init⟩  we only need ψ on the
// range where ψ_init has nonzero amplitude (up to some r_overlap_max),
// since the initial state decays exponentially. The asymptotic region is
// not needed for the overlap (we already extracted K from it in Step 5).
//
// The Config fields n_keep_lo / n_keep_hi select the [lo, hi] inclusive
// grid range to STORE. The backprop still propagates Z̃ from N_grid-1
// down to 0 (the recurrence is coupled, can't be skipped), but only the
// kept range is materialized in ψ_storage / f_storage. The transient Z̃
// in the discarded range is freed after use.
//
// psi is square (N_ψ × N_ψ), so it reuses PotentialStorage (which can
// be MEMORY or DISK). f is rectangular (N_f × N_ψ), so for now it's
// held only in MEMORY (vector<MatrixXd>); rectangular-chunked-disk is
// a later extension.

#pragma once

#include "scatt/ForwardRPropagator.hpp"
#include "scatt/Potentials.hpp"             // for StorageMode
#include "scatt/PotentialStorage.hpp"
#include "scatt/SolverParams.hpp"
#include "scatt/WInverseOperator.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace scatt {

class BackPropagator {
public:
    struct Config {
        // Main ψ keep-range.  ψ (and optional f) are stored for n ∈
        // [n_keep_lo, n_keep_hi] (inclusive).  Default = full grid.
        int    n_keep_lo = 0;
        int    n_keep_hi = -1;               // -1 -> N_grid − 1

        // "Asymptotic buffer" trick (borrowed from version_0).  In addition
        // to the main keep-range, if n_asym > 0 we also hold the last n_asym
        // matrices in a tiny IN-MEMORY vector (psi_asym_).  The asymptotic
        // fit in AsymptoticAmplitudes reads ONLY from this buffer; the main
        // store therefore does NOT need to extend to r_max — it can be
        // truncated to wherever χ_init support ends (≈ n_transition).  Big
        // disk/memory win on large-r_max production runs.
        //
        // Contract:
        //   * The asymptotic buffer covers ir ∈ [N − n_asym + 1, N].
        //   * Default location is MEMORY.  At small L (1–30 MB total) that
        //     is free.  At production L=100 (N_ψ ≈ 10201, n_asym=300) the
        //     buffer is ~250 GB and DOES NOT fit alongside the four big
        //     storages on a 503 GB node; the caller (main.cpp) must then
        //     set psi_asym_storage = DISK and the buffer is chunked on disk
        //     just like the main psi store.  Either way the contract below
        //     is identical: get_psi(n) for n ∈ [asym_offset_, N] returns the
        //     correct ψ_n.
        //   * If n_keep_hi + 1 < N − n_asym + 1, there is a GAP between the
        //     main store and the asym buffer.  get_psi() in that gap throws.
        //     Callers that know they don't need the middle range (dipole
        //     uses orbital-support; asymptotic fit uses outer points) are
        //     safe.
        //   * n_asym = 0 (default) disables the trick: psi_asym_ stays empty,
        //     get_psi() always goes through the main storage.  Back-compat.
        int    n_asym    = 0;

        // Where to hold the asymptotic-buffer tail.  MEMORY = legacy
        // std::vector<MatrixXd> path (fine when bytes_psi_asym fits).
        // DISK = chunked PotentialStorage backed by a separate directory
        // (psi_asym_checkpoint_dir / psi_asym_chunk_size).  Validated bit-
        // identical to MEMORY by test_back_propagator (round-trip through
        // get_psi() returns the same bytes either way).
        StorageMode psi_asym_storage      = StorageMode::MEMORY;
        int         psi_asym_chunk_size   = 20;
        std::string psi_asym_checkpoint_dir;  // empty -> "{checkpoint_dir}/asym"

        bool   compute_f = false;
        StorageMode psi_storage = StorageMode::MEMORY;   // MEMORY or DISK
        std::string checkpoint_dir;                       // "" -> auto-generated
        int    chunk_size = 50;
        bool   verbose    = true;

        // Checkpoint policy (same pattern as SchurInverter / FRP).
        // When true, try to load a matching finalized checkpoint (with
        // manifest + SUCCESS marker) before running. If false or no valid
        // checkpoint exists, fall through to rebuild.
        bool try_load_checkpoint = true;
        // When true AND we built fresh AND we're in MEMORY psi storage,
        // dump the ψ output to disk (with manifest) so next run can reload.
        // For DISK psi_storage this is a no-op (it was already written).
        bool save_checkpoint = true;

        // Allow PotentialStorage to re-chunk an on-disk ψ checkpoint at
        // load time when its on-disk chunk_size > runtime budget.  Default
        // ON.  See SchurInverter::Config::allow_chunk_rechunk.
        bool allow_chunk_rechunk = true;

        // --- GPU offload ------------------------------------------------
        // When true AND compiled with SCATT_HAS_SYCL: each step uploads
        // Rinv + W_inv to the device, runs two GEMMs on GPU, and downloads
        // ψ (and optionally f).  Z is pinned on device between steps.
        // Same "fail loudly if no GPU visible" semantics as the forward
        // stepper.  For small problems the host→device bandwidth dominates
        // and you get a small slowdown; for N_total ≳ 5000 the device
        // GEMM wins by 10–50×.
        bool use_gpu = false;

        // ZERO-ACCURACY-LOSS optimisation for the on-disk psi write path.
        // When true, psi chunk writes use the multi-threaded
        // pwrite-at-distinct-offsets implementation with atomic temp +
        // rename + fsync.  ψ is the LARGEST single-stage storage at
        // production scales (~2 TB at L=100 / N_psi=10201 over
        // n_transition=2761), so parallel writes here are typically the
        // single biggest write-time win.  ψ is NOT symmetric, so the
        // separate symmetric_storage flag does NOT apply -- this flag is
        // independent.  Validated bit-equivalent by
        // test_storage_parallel_write.  Default false preserves the
        // legacy serial single-threaded write byte-for-byte.
        bool parallel_chunk_write = false;

        // Optional pre-cached V at the outer boundary n = N_grid − 1.
        // BP touches the potential exactly once: in
        // compute_Z_at_outer_boundary_(), as `pot_.get(N) * psi_boundary`.
        // If the caller has already fetched V_N (and possibly released the
        // chunk read cache to save ~155 GB at L=100 — pot is unused after
        // FRP::run if --check-residual is off), it can pass a pointer here
        // and BP will use it instead of pot_.get(N).
        // Bit-identical to the default path: same matrix bytes feeding the
        // same Eigen GEMM expression, so the resulting Z̃_N (and every
        // downstream ψ_n) matches byte-for-byte.
        // nullptr (default) preserves the original behaviour.
        const Eigen::MatrixXd* cached_V_outer = nullptr;
    };

    BackPropagator(const SolverParams& sp,
                   Potentials&         pot,
                   ForwardRPropagator& frp,
                   WInverseOperator&   wi);

    // psi_boundary: (N_ψ × N_ψ) value of ψ at n = N_grid − 1.
    // After run(), ψ (and optionally f) on [n_keep_lo, n_keep_hi] is
    // available through get_psi / get_f.
    void run(const Eigen::MatrixXd& psi_boundary, const Config& cfg);

    // Access ψ_n. Throws if n ∉ [n_keep_lo_, n_keep_hi_]. In DISK mode
    // this may stream a chunk.
    const Eigen::MatrixXd& get_psi(std::size_t n);

    // Access f_n (only valid if cfg.compute_f was true). Throws otherwise.
    const Eigen::MatrixXd& get_f(std::size_t n) const;

    int  n_keep_lo() const { return n_keep_lo_; }
    int  n_keep_hi() const { return n_keep_hi_; }
    bool has_f()     const { return has_f_; }

    // Returns true iff the main psi store is in MEMORY mode.  Consumers
    // that want to call get_psi(ir) from multiple threads concurrently
    // must check this first: in MEMORY mode get_psi returns a reference
    // into psi_memory_[idx] (always thread-safe for reads from distinct
    // ir).  In DISK mode get_psi hits PotentialStorage's internal chunk
    // cache and is NOT thread-safe.
    bool psi_in_memory() const { return !psi_on_disk_; }

    // Outer ψ window usable by asymptotic-fit consumers. Returns
    // [asym_offset, N] when n_asym > 0 (the in-memory tail buffer); else
    // falls back to the main-store upper range [n_keep_lo, n_keep_hi].
    // Consumers that need ψ in the asymptotic region (AsymptoticAmplitudes)
    // should clamp their fit window against this, NOT against n_keep_*.
    int  outer_window_lo() const {
        return (n_asym_ > 0) ? asym_offset_ : n_keep_lo_;
    }
    int  outer_window_hi() const {
        return (n_asym_ > 0)
            ? (asym_offset_ + n_asym_ - 1)
            : n_keep_hi_;
    }

    // Diagnostic: the outer-BC we impose forces f_{ℓ=0}(r_N) = 0. The
    // physically-correct ℓ=0 component has  f(r → ∞)  →  const  =  b_phys.
    // The difference between "our f" and the physical f is a homogeneous
    // line  (b_phys / r_N) · r  (ℓ=0 homogeneous sol'n = a·r).
    //
    // b_phys can be READ OFF from our own backprop output by fitting our f
    // in the asymptotic region (past r_fit_min, beyond χ support). In that
    // region our f is a straight line with slope a and intercept -a·r_N,
    // so  b_phys = -a · r_N = slope_magnitude · r_N.
    //
    // Returns a matrix of shape (n_occ, N_psi) giving b_phys[i, j] for
    // the σ = 0 (ℓ=0) channel of orbital i and scattering column j.
    // Requires compute_f = true and both r_fit_min, r_N in the kept range.
    //
    // For a sanity check: if max |b_phys| / |ψ|_max < 1e-2, the
    // uncorrected dipole overlap is trusted at the 1 % level. Above that,
    // consider self-consistent iteration (option 3 in the earlier note).
    Eigen::MatrixXd b_phys_monopole(double r_fit_min = -1.0) const;

    // Per-stage wall-clock accumulators (nanoseconds). Populated during run(),
    // reset at start of each run(). See BackPropagator.cpp for semantics.
    struct Stats {
        std::uint64_t t_rinv_fetch_ns = 0;   // frp_.get(n) -- storage fetch
        std::uint64_t t_gemm_z_ns     = 0;   // Z_next = Rinv_n * Z (dense GEMM)
        std::uint64_t t_wi_apply_ns   = 0;   // W^{-1} · Z  via WInverseOperator::apply
        std::uint64_t t_store_ns      = 0;   // store psi (DISK I/O or MEMORY copy)
        std::uint64_t n_steps         = 0;   // # iterations of inner loop
    };
    const Stats& stats() const { return stats_; }

private:
    const SolverParams& sp_;
    Potentials&         pot_;
    ForwardRPropagator& frp_;
    WInverseOperator&   wi_;

    // Kept range.
    int n_keep_lo_ = 0;
    int n_keep_hi_ = -1;

    // ψ storage (square N_ψ × N_ψ, MEMORY or DISK via PotentialStorage).
    PotentialStorage             psi_disk_;
    std::vector<Eigen::MatrixXd> psi_memory_;
    bool                         psi_on_disk_ = false;

    // Asymptotic-buffer trick (see Config::n_asym). Either MEMORY or DISK
    // depending on Config::psi_asym_storage.  Valid for n ∈ [asym_offset_, N]
    // with size n_asym_.  Empty when n_asym = 0.
    //
    // MEMORY: psi_asym_memory_[asym_idx] holds the matrix, asym_idx = n - asym_offset_.
    // DISK:   psi_asym_disk_  holds chunks; the disk store index is
    //         asym_disk_idx = (N_grid - 1) - n  (so we WRITE in monotonically
    //         increasing order as the backprop loop walks n = N downward;
    //         required by PotentialStorage's DISK write contract).
    std::vector<Eigen::MatrixXd> psi_asym_memory_;
    PotentialStorage             psi_asym_disk_;
    bool                         psi_asym_on_disk_ = false;
    int                          asym_offset_ = -1;   // first n covered
    int                          n_asym_      = 0;
    // Temporary buffer returned by get_psi() when the asym path is DISK.
    mutable Eigen::MatrixXd      psi_asym_read_cache_;

    // f storage (N_f × N_ψ, MEMORY only).
    std::vector<Eigen::MatrixXd> f_memory_;
    bool                         has_f_ = false;

    // Temporary buffer returned by get_psi() in DISK mode (wraps the
    // chunk-cached matrix into a persistent reference for the caller).
    mutable Eigen::MatrixXd psi_read_cache_;

    // Channel metadata for the boundary-Z̃ construction.
    std::vector<int> l_sigma_;
    void build_channel_info_();

    // Internal per-stage timers -- populated during run().
    Stats stats_;

    // Build Z̃_N = W_N · Y_N with Y_N = [ ψ_boundary ; 0 ].
    // V_outer_opt: optional pointer to a pre-fetched V_N (== pot_.get(N) bytes).
    //              If non-null, used instead of calling pot_.get(N).  Always
    //              bit-identical: same matrix data feeds the same GEMM.
    Eigen::MatrixXd compute_Z_at_outer_boundary_(
        const Eigen::MatrixXd&  psi_boundary,
        double                  W_min,
        const Eigen::MatrixXd*  V_outer_opt = nullptr) const;

    void store_(int n, const Eigen::MatrixXd& psi, const Eigen::MatrixXd& f);
};

}  // namespace scatt
