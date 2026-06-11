#include "scatt/BackPropagator.hpp"

#include "angular/Gaunt.hpp"
#include "scatt/GpuPropagate.hpp"
#include "scatt/MklThreadGuard.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace scatt {

namespace {

// Asymptotic-buffer checkpoint format (sidecar file `psi_asym.bin` next
// to the main psi chunks). The buffer is small (n_asym × N_psi² × 8 B,
// typically <1 GB) so a single binary file is fine.  Header is 32 bytes
// + n_asym matrices stored contiguously in column-major (Eigen native).
//
//   magic        : 8 B  = 'PSI_ASYM'
//   version      : 4 B  = 1
//   n_asym       : 4 B  = number of stored matrices
//   asym_offset  : 4 B  = first grid index covered (= N - n_asym + 1)
//   N_psi        : 4 B  = matrix dimension
//   reserved     : 8 B
constexpr char kAsymMagic[8] = {'P','S','I','_','A','S','Y','M'};
constexpr std::int32_t kAsymVersion = 1;

bool save_psi_asym(const std::string& dir,
                   int n_asym, int asym_offset, int N_psi,
                   const std::vector<Eigen::MatrixXd>& psi_asym)
{
    if (n_asym <= 0) return true;  // nothing to save -- not an error
    const std::string path = dir + "/psi_asym.bin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(kAsymMagic, 8);
    const std::int32_t ver = kAsymVersion;
    const std::int32_t na  = n_asym;
    const std::int32_t ao  = asym_offset;
    const std::int32_t Np  = N_psi;
    const std::int64_t pad = 0;
    f.write(reinterpret_cast<const char*>(&ver), 4);
    f.write(reinterpret_cast<const char*>(&na),  4);
    f.write(reinterpret_cast<const char*>(&ao),  4);
    f.write(reinterpret_cast<const char*>(&Np),  4);
    f.write(reinterpret_cast<const char*>(&pad), 8);
    const std::streamsize matrix_bytes =
        static_cast<std::streamsize>(N_psi) * N_psi * sizeof(double);
    for (int i = 0; i < n_asym; ++i) {
        if (psi_asym[i].rows() != N_psi || psi_asym[i].cols() != N_psi) return false;
        f.write(reinterpret_cast<const char*>(psi_asym[i].data()), matrix_bytes);
    }
    return f.good();
}

bool load_psi_asym(const std::string& dir,
                   int expect_n_asym, int expect_asym_offset, int expect_N_psi,
                   std::vector<Eigen::MatrixXd>& psi_asym)
{
    if (expect_n_asym <= 0) return true;
    const std::string path = dir + "/psi_asym.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8] = {};
    f.read(magic, 8);
    if (!f || std::memcmp(magic, kAsymMagic, 8) != 0) return false;
    std::int32_t ver = 0, na = 0, ao = 0, Np = 0;
    std::int64_t pad = 0;
    f.read(reinterpret_cast<char*>(&ver), 4);
    f.read(reinterpret_cast<char*>(&na),  4);
    f.read(reinterpret_cast<char*>(&ao),  4);
    f.read(reinterpret_cast<char*>(&Np),  4);
    f.read(reinterpret_cast<char*>(&pad), 8);
    if (!f) return false;
    if (ver != kAsymVersion)            return false;
    if (na  != expect_n_asym)           return false;
    if (ao  != expect_asym_offset)      return false;
    if (Np  != expect_N_psi)            return false;
    psi_asym.assign(static_cast<std::size_t>(na),
                    Eigen::MatrixXd::Zero(Np, Np));
    const std::streamsize matrix_bytes =
        static_cast<std::streamsize>(Np) * Np * sizeof(double);
    for (int i = 0; i < na; ++i) {
        f.read(reinterpret_cast<char*>(psi_asym[i].data()), matrix_bytes);
        if (!f) return false;
    }
    return true;
}

}  // namespace

BackPropagator::BackPropagator(const SolverParams& sp,
                               Potentials&         pot,
                               ForwardRPropagator& frp,
                               WInverseOperator&   wi)
    : sp_(sp), pot_(pot), frp_(frp), wi_(wi)
{
    build_channel_info_();
}

void BackPropagator::build_channel_info_() {
    l_sigma_.resize(sp_.n_sigma);
    for (int s = 0; s < sp_.n_sigma; ++s) {
        int l, m;
        angular::idx_to_lm(s, l, m);
        l_sigma_[s] = l;
    }
}

// Build Z̃_N at the outer boundary.  If V_outer_opt is non-null, use it
// instead of pot_.get(N).  The two paths are bit-identical (same matrix
// bytes feeding the same Eigen GEMM); the optional pointer just lets the
// caller release pot's chunk cache before BP starts.
Eigen::MatrixXd
BackPropagator::compute_Z_at_outer_boundary_(
    const Eigen::MatrixXd&  psi_boundary,
    double                  /*W_min*/,
    const Eigen::MatrixXd*  V_outer_opt) const
{
    // Build Z̃_N = W_N · Y_N with Y_N = [ ψ_N ; f_N ] chosen SO THAT
    //     Z̃_f_N  ≡  B_N^T · ψ_N + D_N · f_N  =  0.
    // This is the outer BC consistent with the Schur reduction used in
    // K-matrix extraction (which assumed Z̃_f = 0 at the outer matching
    // point to eliminate closed f-channels). The consistent f_N is:
    //
    //     f_N  =  −D_N^{-1} · B_N^T · ψ_N
    //
    // For molecules where χ(r_N) ≈ 0 (orbital fully decayed), B_N ≈ 0 and
    // f_N ≈ 0 to numerical precision. For r_N inside the orbital tail,
    // this correction is important.
    //
    // Block expansion of Z̃_N = W_N · Y_N:
    //   Z̃_ψ,N  =  A_N · ψ_N + B_N · f_N
    //   Z̃_f,N  =  B_N^T · ψ_N + D_N · f_N   =   0   by construction.
    //
    // A_N = I + (h²/6)(E·I − V_N),  B_N = (h²/12) Q_ψf(N),
    // D_N = diag{ max(1 − (h²/12) ℓ(ℓ+1)/r², W_min) }.

    const int    N     = static_cast<int>(sp_.n_grid) - 1;
    const int    N_psi = sp_.n_mu;
    const int    N_f   = sp_.n_occ * sp_.n_sigma;
    const int    N_tot = N_psi + N_f;
    const double h2_6  = sp_.dr * sp_.dr / 6.0;

    if (psi_boundary.rows() != N_psi || psi_boundary.cols() != N_psi) {
        throw std::runtime_error(
            "BackPropagator: psi_boundary must be (N_psi × N_psi)");
    }

    // Fetch B_N and D_N^{-1} directly from the WInverseOperator.
    Eigen::MatrixXd B_N(N_psi, N_f);
    Eigen::VectorXd Dinv_N(N_f);
    auto ws = wi_.make_workspace();
    wi_.load_B_Dinv(N, B_N, Dinv_N, ws);

    // Outer BC for f:  f_N = −D_N^{-1} · B_N^T · ψ_N
    //                      = −(Dinv_N) .* (B_N^T · ψ_N)   (diagonal scale).
    Eigen::MatrixXd Bt_psi = B_N.transpose() * psi_boundary;    // (N_f × N_psi)
    Eigen::MatrixXd f_N    = Dinv_N.asDiagonal() * Bt_psi;
    f_N = -f_N;

    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(N_tot, N_psi);

    // Z̃_ψ,N  =  A_N · ψ_N + B_N · f_N
    //        =  ψ_N + (h²/6)(E·ψ_N − V_N · ψ_N) + B_N · f_N
    // V_N: use pre-cached copy if the caller provided one (so they could
    // free pot's chunk read cache before BP started); otherwise fetch
    // from pot.  The two are bit-identical: V_outer_opt == &(deep copy of
    // pot_.get(N)) by contract.
    const Eigen::MatrixXd& V_N_at_boundary = V_outer_opt
        ? *V_outer_opt
        : pot_.get(static_cast<std::size_t>(N));

    Z.topRows(N_psi).noalias()  = psi_boundary;
    Z.topRows(N_psi).noalias() += h2_6 * sp_.energy * psi_boundary;
    Z.topRows(N_psi).noalias() -= h2_6 * V_N_at_boundary * psi_boundary;
    Z.topRows(N_psi).noalias() += B_N * f_N;

    // Z̃_f,N by construction is 0 — leave as zero (already set).
    // (Could fill it in from B_N^T · ψ_N + D_N · f_N; it would be exactly
    // zero up to rounding.  Explicitly zeroing avoids any LU drift.)

    return Z;
}

void BackPropagator::run(const Eigen::MatrixXd& psi_boundary,
                         const Config&          cfg)
{
    // Force the calling (= main) thread's MKL pool to the full OMP thread
    // count for the duration of this serial-outer loop.  Globally MKL runs
    // with mkl_set_dynamic(1) to avoid oversubscription inside the
    // SchurInverter / Potentials OMP-parallel regions; the local override
    // here lets the per-step Sinv·B / Bᵀ·Y GEMMs inside wi_.apply (and
    // the materialize_into used on the GPU path) saturate all 112 cores
    // instead of dropping to a single thread under MKL's nest heuristic.
    // Bit-equivalent to the legacy code up to MKL summation order
    // (same tolerance the existing tests already accept).
#if defined(_OPENMP)
    MklThreadGuard _mkl_local(omp_get_max_threads());
#else
    MklThreadGuard _mkl_local(1);
#endif

    const int N    = static_cast<int>(sp_.n_grid) - 1;
    const int N_psi = sp_.n_mu;
    const int N_f   = sp_.n_occ * sp_.n_sigma;
    const int N_tot = N_psi + N_f;

    // Resolve keep range.
    n_keep_lo_ = std::max(0, cfg.n_keep_lo);
    n_keep_hi_ = (cfg.n_keep_hi < 0) ? N : cfg.n_keep_hi;
    if (n_keep_hi_ > N) n_keep_hi_ = N;
    if (n_keep_lo_ > n_keep_hi_) {
        throw std::runtime_error(
            "BackPropagator: n_keep_lo > n_keep_hi (got " +
            std::to_string(n_keep_lo_) + " > " + std::to_string(n_keep_hi_) + ")");
    }
    const int  n_keep = n_keep_hi_ - n_keep_lo_ + 1;
    has_f_            = cfg.compute_f;

    // ---- Asymptotic-buffer setup (trick from version_0) ------------------
    // psi_asym covers grid indices [asym_offset_, N].  At small L this is
    // 1–30 MB and can stay in std::vector<MatrixXd>; at production L=100
    // (~250 GB) it MUST go to DISK or we OOM the BP stage.  Decision is
    // controlled by cfg.psi_asym_storage.
    //
    // Storage layout:
    //   MEMORY: psi_asym_memory_[asym_idx] with asym_idx = n - asym_offset_.
    //   DISK:   psi_asym_disk_.store(asym_disk_idx, ψ) with
    //           asym_disk_idx = N - n  (monotonically increasing as the
    //           backprop loop visits n from N downward; required by the
    //           PotentialStorage DISK write contract).
    n_asym_           = std::max(0, cfg.n_asym);
    asym_offset_      = (n_asym_ > 0) ? (N - n_asym_ + 1) : -1;
    psi_asym_on_disk_ = (n_asym_ > 0 && cfg.psi_asym_storage == StorageMode::DISK);
    psi_asym_memory_.clear();
    psi_asym_disk_.clear();
    // NB: a gap between main store [n_keep_lo, n_keep_hi] and the asym
    // buffer [asym_offset_, N] is ALLOWED (the whole point of the trick).
    // get_psi() will throw if any consumer asks for an index in the gap.
    if (n_asym_ > 0 && asym_offset_ < 0) asym_offset_ = 0;

    // ---- Storage setup ----
    psi_memory_.clear();
    f_memory_.clear();

    // Compose parameter-encoded dir and manifest (checkpoint integrity).
    auto fmt_E = [](double e) {
        char buf[32]; std::snprintf(buf, sizeof(buf), "%.6f", e); return std::string(buf);
    };
    const std::string manifest =
        "kind=psi"
        " energy=" + fmt_E(sp_.energy) +
        " Nr="     + std::to_string(sp_.n_grid) +
        " dr="     + fmt_E(sp_.dr) +
        " Npsi="   + std::to_string(N_psi) +
        " Nf="     + std::to_string(N_f) +
        " l_cont=" + std::to_string(sp_.l_max_continuum) +
        " l_exch=" + std::to_string(sp_.l_max_exchange) +
        " n_occ="  + std::to_string(sp_.n_occ) +
        " n_trans="+ std::to_string(sp_.n_transition) +
        " keep="   + std::to_string(n_keep_lo_) + ".." + std::to_string(n_keep_hi_);

    const std::string dir = cfg.checkpoint_dir.empty()
        ? ("./checkpoints/psi_Nr"   + std::to_string(sp_.n_grid)
           + "_Npsi"  + std::to_string(N_psi)
           + "_E"     + fmt_E(sp_.energy)
           + "_lc"    + std::to_string(sp_.l_max_continuum)
           + "_keep"  + std::to_string(n_keep_lo_) + "-" + std::to_string(n_keep_hi_))
        : cfg.checkpoint_dir;

    // Helper: try to (re)hydrate the asymptotic buffer from a prior run.
    // Returns true when the asym data is ready (or none is needed).
    //
    // Two on-disk formats exist (the writer chose one based on the WRITING
    // node's memory budget):
    //   * single-file   `<dir>/psi_asym.bin`           — written when the
    //                                                    writer's psi_asym
    //                                                    fit in MEMORY mode.
    //   * chunked dir   `<asym_dir_pl>/pot_chunk_*.bin` — written when the
    //                                                    writer routed
    //                                                    psi_asym to DISK.
    //
    // The reading node's memory budget is DIFFERENT in general (e.g. a
    // dipole-only recovery on a low-memory node, or vice versa), so we
    // auto-detect by checking what is actually on disk rather than using
    // `psi_asym_on_disk_` (which was set from cfg.psi_asym_storage at
    // line 227 and reflects only the READING node's preference).
    //
    // After a successful load we UPDATE psi_asym_on_disk_ to match where
    // the data ended up (psi_asym_memory_ vs psi_asym_disk_), so subsequent
    // reads via get_psi_asym()/etc. take the right branch.
    namespace fs = std::filesystem;
    const std::string asym_dir_pl = cfg.psi_asym_checkpoint_dir.empty()
        ? (dir + "/asym")
        : cfg.psi_asym_checkpoint_dir;
    const std::string asym_single_path = dir + "/psi_asym.bin";
    auto load_asym = [&]() -> bool {
        if (n_asym_ <= 0) return true;

        const bool single_present  = fs::exists(asym_single_path);
        const bool chunked_present = fs::is_directory(asym_dir_pl)
                                  && fs::exists(asym_dir_pl + "/__SUCCESS__");

        // Prefer single-file when present (cheaper to load + verify), then
        // try the chunked subdir.  If neither is on disk, the loader fails
        // and the caller will (typically) recompute from scratch.
        if (single_present) {
            if (load_psi_asym(dir, n_asym_, asym_offset_,
                              N_psi, psi_asym_memory_)) {
                psi_asym_on_disk_ = false;  // data is in psi_asym_memory_
                psi_asym_disk_.clear();
                if (cfg.verbose) {
                    std::cout << "[BackPropagator] asym buffer loaded from "
                              << asym_single_path << " (single-file format)\n";
                }
                return true;
            }
        }
        if (chunked_present) {
            psi_asym_disk_.set_manifest(
                "kind=psi_asym energy=" + fmt_E(sp_.energy) +
                " Nr=" + std::to_string(sp_.n_grid) +
                " dr=" + fmt_E(sp_.dr) +
                " Npsi=" + std::to_string(N_psi) +
                " n_asym=" + std::to_string(n_asym_) +
                " offset=" + std::to_string(asym_offset_));
            if (psi_asym_disk_.initialize_from_checkpoint(
                    static_cast<std::size_t>(n_asym_), N_psi,
                    asym_dir_pl,
                    std::max(1, cfg.psi_asym_chunk_size))) {
                psi_asym_on_disk_ = true;   // data is in psi_asym_disk_
                psi_asym_memory_.clear();
                if (cfg.verbose) {
                    std::cout << "[BackPropagator] asym buffer loaded from "
                              << asym_dir_pl << " (chunked-dir format)\n";
                }
                return true;
            }
        }
        return false;
    };

    // Try hot-load if requested.
    if (cfg.try_load_checkpoint) {
        psi_disk_.set_manifest(manifest);
        // Forward the chunk-rechunk policy.  Must be set BEFORE the
        // checkpoint-load attempt; PotentialStorage::initialize_from_
        // checkpoint reads chunk_rechunk_allowed_ inside that call.
        psi_disk_.set_chunk_rechunk_allowed(cfg.allow_chunk_rechunk);
        if (cfg.psi_storage == StorageMode::MEMORY) {
            if (psi_disk_.try_load_into_memory(
                    static_cast<std::size_t>(n_keep), N_psi,
                    dir, cfg.chunk_size))
            {
                // Load succeeded. Populate psi_memory_ from psi_disk_ and
                // zero-fill f_memory_ (f is never checkpointed yet).
                psi_memory_.resize(static_cast<std::size_t>(n_keep));
                for (int i = 0; i < n_keep; ++i) {
                    psi_memory_[i] = psi_disk_.get(static_cast<std::size_t>(i));
                }
                psi_on_disk_ = false;
                // Rebuild f_memory_ from backprop IF compute_f was requested
                // (f is cheap; just no way to re-derive from ψ alone). Warn.
                if (has_f_) {
                    if (cfg.verbose) {
                        std::cout << "[BackPropagator] ψ loaded from "
                                     "checkpoint, but f was requested; f is "
                                     "not checkpointed so we fall through to "
                                     "a rebuild.\n";
                    }
                    psi_memory_.clear();
                } else if (!load_asym()) {
                    if (cfg.verbose) {
                        std::cout << "[BackPropagator] ψ loaded from "
                                     "checkpoint, but psi_asym missing/"
                                     "mismatched -- falling through to "
                                     "rebuild (asymptotic buffer required "
                                     "for AsymptoticAmplitudes fit).\n";
                    }
                    psi_memory_.clear();
                    psi_asym_disk_.clear();
                    psi_asym_memory_.clear();
                } else {
                    if (cfg.verbose) {
                        std::cout << "[BackPropagator] ψ loaded from "
                                     "checkpoint: " << dir << "\n";
                    }
                    return;
                }
            }
        } else { // DISK
            if (psi_disk_.initialize_from_checkpoint(
                    static_cast<std::size_t>(n_keep), N_psi,
                    dir, cfg.chunk_size))
            {
                psi_on_disk_ = true;
                if (has_f_) {
                    if (cfg.verbose) {
                        std::cout << "[BackPropagator] ψ DISK-checkpoint "
                                     "found but f was requested; falling "
                                     "through to rebuild.\n";
                    }
                    psi_disk_.clear();
                } else if (!load_asym()) {
                    if (cfg.verbose) {
                        std::cout << "[BackPropagator] ψ DISK-checkpoint "
                                     "found, but psi_asym missing/"
                                     "mismatched -- falling through to "
                                     "rebuild.\n";
                    }
                    psi_disk_.clear();
                    psi_asym_disk_.clear();
                    psi_asym_memory_.clear();
                } else {
                    if (cfg.verbose) {
                        std::cout << "[BackPropagator] ψ loaded from "
                                     "DISK checkpoint: " << dir << "\n";
                    }
                    return;
                }
            }
        }
    }

    // Fresh build: set up asym write storage (MEMORY vector or DISK chunks).
    // Manifest is independent from the main psi manifest (different shape and
    // grid range) so the main store and asym store can be checkpointed
    // separately without colliding.
    const std::string asym_dir = cfg.psi_asym_checkpoint_dir.empty()
        ? (dir + "/asym")
        : cfg.psi_asym_checkpoint_dir;
    const std::string asym_manifest =
        "kind=psi_asym"
        " energy=" + fmt_E(sp_.energy) +
        " Nr="     + std::to_string(sp_.n_grid) +
        " dr="     + fmt_E(sp_.dr) +
        " Npsi="   + std::to_string(N_psi) +
        " n_asym=" + std::to_string(n_asym_) +
        " offset=" + std::to_string(asym_offset_);
    if (n_asym_ > 0) {
        if (psi_asym_on_disk_) {
            psi_asym_disk_.set_manifest(asym_manifest);
            psi_asym_disk_.initialize_for_write(
                static_cast<std::size_t>(n_asym_), N_psi,
                PotentialStorage::Mode::DISK,
                asym_dir,
                std::max(1, cfg.psi_asym_chunk_size),
                /*symmetric_storage=*/false,           // ψ is NOT symmetric
                /*parallel_chunk_write=*/cfg.parallel_chunk_write);
        } else {
            psi_asym_memory_.assign(static_cast<std::size_t>(n_asym_),
                                    Eigen::MatrixXd::Zero(N_psi, N_psi));
        }
    }

    // Fresh build: set up write storage.
    if (cfg.psi_storage == StorageMode::DISK) {
        psi_disk_.set_manifest(manifest);
        psi_disk_.initialize_for_write(
            static_cast<std::size_t>(n_keep), N_psi,
            PotentialStorage::Mode::DISK,
            dir, cfg.chunk_size,
            /*symmetric_storage=*/false,           // ψ is NOT symmetric
            /*parallel_chunk_write=*/cfg.parallel_chunk_write);
        psi_on_disk_ = true;
    } else {
        psi_memory_.resize(static_cast<std::size_t>(n_keep));
        for (int i = 0; i < n_keep; ++i) {
            psi_memory_[i] = Eigen::MatrixXd::Zero(N_psi, N_psi);
        }
        psi_on_disk_ = false;
        psi_disk_.set_manifest(manifest);   // for save_to_disk below
    }
    if (has_f_) {
        f_memory_.resize(static_cast<std::size_t>(n_keep));
        for (int i = 0; i < n_keep; ++i) {
            f_memory_[i] = Eigen::MatrixXd::Zero(N_f, N_psi);
        }
    }

    // For DISK-mode ψ, we must store in MONOTONICALLY INCREASING order of
    // the local store index (0..n_keep-1). Our backward loop visits n in
    // DECREASING grid order, so the mapping from grid index to store
    // index is reversed:
    //     store_idx  =  n_keep_hi_ − n   (for n ∈ [n_keep_lo_, n_keep_hi_])
    // which is itself increasing as n decreases. ✓ matches DISK contract.

    auto store_pair = [&](int n, const Eigen::MatrixXd& psi,
                          const Eigen::MatrixXd* f) {
        // Asymptotic tail: independent of n_keep_lo/hi.
        // MEMORY: asym_idx = n - asym_offset_  (in [0, n_asym_)).
        // DISK:   asym_disk_idx = N - n        (monotonically increasing
        //         across the backward loop -- required by PotentialStorage
        //         DISK write contract).
        if (n_asym_ > 0 && n >= asym_offset_ && n <= N) {
            if (psi_asym_on_disk_) {
                const int asym_disk_idx = N - n;
                psi_asym_disk_.store(static_cast<std::size_t>(asym_disk_idx),
                                     psi);
            } else {
                const int asym_idx = n - asym_offset_;
                psi_asym_memory_[static_cast<std::size_t>(asym_idx)] = psi;
            }
        }

        // Main keep-range.
        if (n < n_keep_lo_ || n > n_keep_hi_) return;
        const int local_idx = n_keep_hi_ - n;    // 0 at the first store
        if (psi_on_disk_) {
            psi_disk_.store(static_cast<std::size_t>(local_idx), psi);
        } else {
            psi_memory_[static_cast<std::size_t>(local_idx)] = psi;
        }
        if (has_f_ && f != nullptr) {
            f_memory_[static_cast<std::size_t>(local_idx)] = *f;
        }
    };

    // ---- Step 1: boundary Y_N, store at n = N if requested ----
    auto t0 = std::chrono::steady_clock::now();

    // Build Z̃_N using the consistent outer BC (Z̃_f_N = 0) AND recover
    // the associated f_N = −D_N^{-1} · B_N^T · ψ_N for storage.
    Eigen::MatrixXd B_N_loc(N_psi, N_f);
    Eigen::VectorXd Dinv_N_loc(N_f);
    auto ws_bc = wi_.make_workspace();
    wi_.load_B_Dinv(N, B_N_loc, Dinv_N_loc, ws_bc);
    Eigen::MatrixXd Y_N_psi = psi_boundary;
    Eigen::MatrixXd Y_N_f   = Dinv_N_loc.asDiagonal() *
                              (B_N_loc.transpose() * psi_boundary);
    Y_N_f = -Y_N_f;
    store_pair(N, Y_N_psi, has_f_ ? &Y_N_f : nullptr);

    // Z̃_N from Y_N.  Pass the optional pre-cached V_N from cfg if set.
    Eigen::MatrixXd Z = compute_Z_at_outer_boundary_(psi_boundary,
                                                     /*W_min*/ 0.001,
                                                     cfg.cached_V_outer);

    // ---- Step 2: backward loop n = N − 1 ... 0 ----
    Eigen::MatrixXd Z_next(N_tot, N_psi);
    Eigen::MatrixXd Y_n   (N_tot, N_psi);
    auto ws = wi_.make_workspace();

    // ---- optional GPU stepper ------------------------------------------
    std::unique_ptr<GpuContext>    gpu_ctx;
    std::unique_ptr<GpuBackStepper> gpu_step;
    Eigen::MatrixXd W_inv_host;     // only used on GPU path
    Eigen::MatrixXd psi_gpu, f_gpu; // download buffers for GPU path

    if (cfg.use_gpu) {
        if (!GpuContext::gpu_available()) {
            throw std::runtime_error(
                "BackPropagator: use_gpu=true but no SYCL GPU is visible "
                "(rebuild with -DSCATT_WITH_SYCL=ON using icpx, and run on a "
                "GPU node).");
        }
        gpu_ctx  = std::make_unique<GpuContext>(/*prefer_gpu*/ true);
        gpu_step = std::make_unique<GpuBackStepper>(*gpu_ctx, N_tot, N_psi, N_f);
        gpu_step->init_Z(Z);
        if (cfg.verbose) {
            std::cout << "[BackPropagator] GPU offload enabled: "
                      << gpu_ctx->info().device_name
                      << "  (HBM "
                      << (gpu_ctx->info().global_mem_bytes >> 30) << " GB)\n";
        }
    }

    // Reset per-run timing counters. Per-iteration deltas via std::chrono
    // are cheap enough here (~60 ns/call) relative to the matrix work they
    // bracket (µs–ms per step).
    stats_ = Stats{};
    using clk = std::chrono::steady_clock;

    // Backward-prefetch setup.  Both Rinv (frp_) and Sinv (si_) are stored
    // chunked-on-disk; the backward sweep reads them in DECREASING n order.
    // As soon as we touch chunk c on either store, we know the NEXT chunk
    // we'll need is c-1.  Issuing an async prefetch immediately overlaps
    // the disk read of chunk c-1 with the dense GEMM compute happening on
    // chunk c.  Bit-identical to the synchronous path: the bytes brought
    // into RAM are the SAME pread() output, just landed earlier.
    //
    // We detect chunk transitions via PotentialStorage::chunk_size()
    // (0 in MEMORY mode -> prefetch is a no-op and we skip the bookkeeping).
    const int frp_cs = frp_.chunk_size_disk();
    const int si_cs  = wi_.si_chunk_size_disk();
    int last_frp_chunk = -1;
    int last_si_chunk  = -1;

    for (int n = N - 1; n >= 0; --n) {
        ++stats_.n_steps;

        // Async backward prefetch on Rinv (frp_): as soon as we enter a
        // new chunk on the descending sweep, kick off the read of the
        // chunk we'll need next (current_chunk - 1).  Skipped in MEMORY
        // mode (chunk_size_disk() == 0) and at chunk 0 (no predecessor).
        if (frp_cs > 0) {
            const int cur_chunk = n / frp_cs;
            if (cur_chunk != last_frp_chunk) {
                last_frp_chunk = cur_chunk;
                if (cur_chunk - 1 >= 0)
                    frp_.start_prefetch(cur_chunk - 1);
            }
        }

        auto ts0 = clk::now();
        const Eigen::MatrixXd& Rinv_n = frp_.get(static_cast<std::size_t>(n));
        auto ts1 = clk::now();
        stats_.t_rinv_fetch_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(ts1 - ts0).count());

        // Async backward prefetch on Sinv (via wi_'s underlying SI):
        // mirror Rinv.  Both the GPU and CPU paths consume Sinv this step
        // (GPU via wi_.materialize_into; CPU via wi_.apply), so kick the
        // prefetch off before the if/else.
        if (si_cs > 0) {
            const int cur_chunk = n / si_cs;
            if (cur_chunk != last_si_chunk) {
                last_si_chunk = cur_chunk;
                if (cur_chunk - 1 >= 0)
                    wi_.start_prefetch(cur_chunk - 1);
            }
        }

        if (gpu_step) {
            // GPU path: materialize W_inv on host, upload + GEMMs on device.
            wi_.materialize_into(n, W_inv_host, ws);
            auto ts2_h = clk::now();
            stats_.t_wi_apply_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(ts2_h - ts1).count());

            gpu_step->step(Rinv_n, W_inv_host, psi_gpu,
                           has_f_ ? &f_gpu : nullptr, has_f_);
            auto ts3_h = clk::now();
            stats_.t_gemm_z_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(ts3_h - ts2_h).count());

            if (has_f_) {
                store_pair(n, psi_gpu, &f_gpu);
            } else {
                store_pair(n, psi_gpu, nullptr);
            }
            auto ts4_h = clk::now();
            stats_.t_store_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(ts4_h - ts3_h).count());
            if(n%100==0){
                std::cerr << "n=" << n << "  t_rinv_fetch=" << stats_.t_rinv_fetch_ns / stats_.n_steps << " ns"
                          << "  t_gemm_z=" << stats_.t_gemm_z_ns / stats_.n_steps << " ns"
                          << "  t_wi_apply=" << stats_.t_wi_apply_ns / stats_.n_steps << " ns"
                          << "  t_store=" << stats_.t_store_ns / stats_.n_steps << " ns\n";
            }
            
        } else {
            //  Z̃_n  =  Rinv_n · Z̃_{n+1}    (dense GEMM)
            Z_next.noalias() = Rinv_n * Z;
            Z.swap(Z_next);
            auto ts2 = clk::now();
            stats_.t_gemm_z_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(ts2 - ts1).count());

            //  Y_n  =  W_n^{-1} · Z̃_n       (triangular solve inside WInverseOperator)
            wi_.apply(n, Z, Y_n, ws);
            auto ts3 = clk::now();
            stats_.t_wi_apply_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(ts3 - ts2).count());

            Eigen::MatrixXd psi_n = Y_n.topRows(N_psi);
            if (has_f_) {
                Eigen::MatrixXd f_n = Y_n.bottomRows(N_f);
                store_pair(n, psi_n, &f_n);
            } else {
                store_pair(n, psi_n, nullptr);
            }
            auto ts4 = clk::now();
            stats_.t_store_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(ts4 - ts3).count());


            if(n%100==0){
                std::cerr << "n=" << n << "  t_rinv_fetch=" << stats_.t_rinv_fetch_ns / stats_.n_steps << " ns"
                          << "  t_gemm_z=" << stats_.t_gemm_z_ns / stats_.n_steps << " ns"
                          << "  t_wi_apply=" << stats_.t_wi_apply_ns / stats_.n_steps << " ns"
                          << "  t_store=" << stats_.t_store_ns / stats_.n_steps << " ns\n";
            }
        }

        if (cfg.verbose && n > 0 && n % 500 == 0) {
            std::cout << "  [BackPropagator] n=" << n << "\n";
        }
    }

    // Finalize the asym DISK store first (no-op when MEMORY).  Done before
    // the main psi checkpoint write so an abort here leaves the asym blob
    // either fully present (its own SUCCESS marker) or absent -- never half-
    // written.  The MEMORY-asym path keeps its legacy single-file format
    // (save_psi_asym below).
    if (n_asym_ > 0 && psi_asym_on_disk_) {
        psi_asym_disk_.finalize_write();
    }

    if (psi_on_disk_) {
        psi_disk_.finalize_write();   // writes manifest + SUCCESS
        if (n_asym_ > 0 && !psi_asym_on_disk_ &&
            !save_psi_asym(dir, n_asym_, asym_offset_,
                           N_psi, psi_asym_memory_)) {
            std::cerr << "[BackPropagator] WARNING: failed to write "
                         "psi_asym.bin to " << dir
                      << " -- subsequent --try-load-checkpoint runs will "
                         "rebuild BP from scratch.\n";
        }
    } else if (cfg.save_checkpoint && cfg.psi_storage == StorageMode::MEMORY) {
        // Dump MEMORY build to disk as a checkpoint.
        // Need a finalized PotentialStorage in MEMORY mode first.
        PotentialStorage tmp_store;
        tmp_store.initialize_for_write(
            static_cast<std::size_t>(n_keep), N_psi,
            PotentialStorage::Mode::MEMORY, "", cfg.chunk_size);
        for (int i = 0; i < n_keep; ++i) {
            tmp_store.store(static_cast<std::size_t>(i), psi_memory_[i]);
        }
        tmp_store.finalize_write();
        tmp_store.set_manifest(manifest);
        tmp_store.save_to_disk(dir, cfg.chunk_size,
                               /*symmetric_storage=*/false,
                               /*parallel_chunk_write=*/cfg.parallel_chunk_write);
        if (n_asym_ > 0 && !psi_asym_on_disk_ &&
            !save_psi_asym(dir, n_asym_, asym_offset_,
                           N_psi, psi_asym_memory_)) {
            std::cerr << "[BackPropagator] WARNING: failed to write "
                         "psi_asym.bin to " << dir
                      << " -- subsequent --try-load-checkpoint runs will "
                         "rebuild BP from scratch.\n";
        }
    }

    const double dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    if (cfg.verbose) {
        std::cout << "[BackPropagator] done in " << dt << " s ("
                  << (static_cast<double>(N + 1) / dt) << " pts/s, "
                  << (dt / (N + 1) * 1e3) << " ms/pt)"
                  << "  keep=[" << n_keep_lo_ << "," << n_keep_hi_ << "]"
                  << "  f=" << (has_f_ ? "yes" : "no") << "\n";
    }
}

const Eigen::MatrixXd& BackPropagator::get_psi(std::size_t n) {
    const int ni = static_cast<int>(n);
    const int N  = static_cast<int>(sp_.n_grid) - 1;

    // 1) Asymptotic-buffer fast path.  Preferred when the requested n falls
    //    in the tail (avoids DISK reads on the main store, which is the
    //    whole point of the asym trick).
    //    MEMORY: read directly from the vector.
    //    DISK:   asym_disk_idx = N - ni  (write-order indexing, see
    //            store_pair lambda in run()).
    if (n_asym_ > 0 && ni >= asym_offset_) {
        const int asym_idx = ni - asym_offset_;
        if (asym_idx < n_asym_) {
            if (psi_asym_on_disk_) {
                const int asym_disk_idx = N - ni;
                psi_asym_read_cache_ = psi_asym_disk_.get(
                    static_cast<std::size_t>(asym_disk_idx));
                return psi_asym_read_cache_;
            }
            return psi_asym_memory_[static_cast<std::size_t>(asym_idx)];
        }
        // ni > N: fall through to raise below.
    }

    // 2) Main store.
    if (ni >= n_keep_lo_ && ni <= n_keep_hi_) {
        const std::size_t local_idx = static_cast<std::size_t>(n_keep_hi_ - ni);
        if (psi_on_disk_) {
            return psi_disk_.get(local_idx);
        }
        return psi_memory_[local_idx];
    }

    // 3) Gap or out-of-range.
    throw std::runtime_error(
        "BackPropagator::get_psi: n=" + std::to_string(n) +
        " outside kept range [" + std::to_string(n_keep_lo_) + "," +
        std::to_string(n_keep_hi_) + "]" +
        (n_asym_ > 0
            ? (" and asym buffer [" + std::to_string(asym_offset_) +
               ", " + std::to_string(asym_offset_ + n_asym_ - 1) + "]")
            : std::string{}));
}

Eigen::MatrixXd
BackPropagator::b_phys_monopole(double r_fit_min) const
{
    if (!has_f_) {
        throw std::runtime_error(
            "b_phys_monopole requires compute_f = true");
    }
    const int N   = static_cast<int>(sp_.n_grid) - 1;
    const int N_psi = sp_.n_mu;
    const int n_occ = sp_.n_occ;
    const int n_sigma = sp_.n_sigma;
    const double r_N = sp_.r_min + N * sp_.dr;

    // Default fit window: the outer 10 % of the grid. If the user supplies
    // r_fit_min we use that, but clamp so that we have at least 50 points
    // and so that r_fit_min ≥ r_N − 10 au (in practice well past any
    // reasonable χ range for valence orbitals).
    const double r_fit = (r_fit_min > 0.0)
        ? r_fit_min
        : (r_N - std::min(0.1 * (r_N - sp_.r_min), 10.0));

    // Map r_fit onto a grid index (rounded up).
    int n_lo = static_cast<int>(std::ceil((r_fit - sp_.r_min) / sp_.dr));
    n_lo = std::max(n_keep_lo_, std::min(n_lo, N - 10));
    const int n_hi = std::min(N, n_keep_hi_);
    if (n_hi - n_lo < 10) {
        throw std::runtime_error(
            "b_phys_monopole: kept range [" + std::to_string(n_keep_lo_) +
            ", " + std::to_string(n_keep_hi_) +
            "] doesn't have enough points past r_fit = " +
            std::to_string(r_fit));
    }

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_occ, N_psi);

    // For each (i, j): least-squares fit f_{σ=0}^i(r_n, j)  =  a·r_n + b
    // over n ∈ [n_lo, n_hi]. Then b_phys = -a · r_N.
    // (Under exact arithmetic and our BC, b should equal -a·r_N as well,
    //  which gives another consistency check.)
    const int Npts = n_hi - n_lo + 1;
    Eigen::VectorXd r_vec(Npts);
    for (int k = 0; k < Npts; ++k) {
        r_vec(k) = sp_.r_min + (n_lo + k) * sp_.dr;
    }
    const double r_mean  = r_vec.mean();
    const double r_var   = (r_vec.array() - r_mean).square().sum();

    for (int i = 0; i < n_occ; ++i) {
        const int f_idx = i * n_sigma + 0;  // σ = 0 (ℓ=0, m=0)
        for (int j = 0; j < N_psi; ++j) {
            Eigen::VectorXd y(Npts);
            // This is a non-const copy-back because get_f is const and
            // returns a ref owned by f_memory_ / a cache slot.
            for (int k = 0; k < Npts; ++k) {
                y(k) = f_memory_[static_cast<std::size_t>(
                    n_keep_hi_ - (n_lo + k))](f_idx, j);
            }
            const double y_mean = y.mean();
            const double num = ((r_vec.array() - r_mean) *
                                (y.array()     - y_mean)).sum();
            const double slope = num / r_var;
            result(i, j) = -slope * r_N;
        }
    }
    return result;
}

const Eigen::MatrixXd& BackPropagator::get_f(std::size_t n) const {
    if (!has_f_) {
        throw std::runtime_error("BackPropagator::get_f: compute_f was false");
    }
    const int ni = static_cast<int>(n);
    if (ni < n_keep_lo_ || ni > n_keep_hi_) {
        throw std::runtime_error(
            "BackPropagator::get_f: n=" + std::to_string(n) +
            " outside keep range [" + std::to_string(n_keep_lo_) + "," +
            std::to_string(n_keep_hi_) + "]");
    }
    const std::size_t local_idx = static_cast<std::size_t>(n_keep_hi_ - ni);
    return f_memory_[local_idx];
}

}  // namespace scatt
