#include "scatt/SchurInverter.hpp"
#include "scatt/SystemMemory.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "angular/Gaunt.hpp"   // idx_to_lm
#include "scatt/GpuPropagate.hpp"   // GpuContext, GpuSinvStepper
#include "scatt/LapackInverse.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
#include <omp.h>
#endif

namespace scatt {

namespace {

// Cheap O(N²) Gershgorin lower bound on the smallest eigenvalue of a
// symmetric matrix.
double gershgorin_min(const Eigen::MatrixXd& M) {
    const int n = static_cast<int>(M.rows());
    double lb = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        double off_sum = 0.0;
        for (int j = 0; j < n; ++j) if (i != j) off_sum += std::abs(M(i, j));
        lb = std::min(lb, M(i, i) - off_sum);
    }
    return lb;
}

// Returns min eigenvalue. Uses Gershgorin first; falls back to full
// SelfAdjointEigenSolver only when the Gershgorin bound is inconclusive
// (within 0.05 of the W_min threshold).
double min_eig_fast(const Eigen::MatrixXd& M, double threshold) {
    double gersh = gershgorin_min(M);
    if (gersh > threshold + 0.05) return gersh;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M, Eigen::EigenvaluesOnly);
    return es.eigenvalues()(0);
}

// Shift M.diagonal() so that min(eig) ≥ W_min + margin. No-op if already so.
// Returns true if a shift was applied.
bool stability_shift(Eigen::MatrixXd& M, double W_min, double margin) {
    const double me = min_eig_fast(M, W_min);
    if (me >= W_min) return false;
    const double delta = W_min - me + margin;
    M.diagonal().array() += delta;
    return true;
}

}  // namespace

SchurInverter::SchurInverter(const SolverParams&     sp,
                             Potentials&             pot,
                             const ExchangeCoupling* ec,
                             const ChiRadial*        chi)
    : sp_(sp), pot_(pot), ec_(ec), chi_(chi)
{
    if (ec_ && !chi_) {
        throw std::runtime_error(
            "SchurInverter: chi must be provided when ExchangeCoupling is non-null");
    }
    build_channel_info_();
}

void SchurInverter::build_channel_info_() {
    l_sigma_.resize(sp_.n_sigma);
    for (int sigma = 0; sigma < sp_.n_sigma; ++sigma) {
        int l, m;
        angular::idx_to_lm(sigma, l, m);
        l_sigma_[sigma] = l;
    }
}

void SchurInverter::build(const Config& cfg) {
    const int    N_psi   = sp_.n_mu;
    const int    N_f     = sp_.n_occ * sp_.n_sigma;
    const std::size_t Nr = sp_.n_grid;
    const double h       = sp_.dr;
    const double h2_12   = h * h / 12.0;
    const double h2_6    = h * h / 6.0;

    // ---- storage mode -----------------------------------------------------
    const std::size_t bytes_total = Nr * N_psi * N_psi * sizeof(double);
    PotentialStorage::Mode st_mode = PotentialStorage::Mode::MEMORY;
    if (cfg.storage == StorageMode::DISK) {
        st_mode = PotentialStorage::Mode::DISK;
    } else if (cfg.storage == StorageMode::AUTO &&
               bytes_total > 10ULL * 1024ULL * 1024ULL * 1024ULL)
    {
        st_mode = PotentialStorage::Mode::DISK;
    }

    // Parameter-encoded checkpoint dir: prevents loading a checkpoint
    // built with different (E, l_cont, exchange, grid) when the user runs
    // at a new energy in the same working directory.
    // Manifest describes the run so a checkpoint from a different setup
    // at the same directory is rejected on reload.
    auto fmt_E = [](double e) {
        char buf[32]; std::snprintf(buf, sizeof(buf), "%.6f", e); return std::string(buf);
    };
    const std::string manifest =
        "kind=sinv"
        " energy=" + fmt_E(sp_.energy) +
        " Nr="     + std::to_string(Nr) +
        " dr="     + fmt_E(sp_.dr) +
        " Npsi="   + std::to_string(N_psi) +
        " Nf="     + std::to_string(N_f) +
        " l_cont=" + std::to_string(sp_.l_max_continuum) +
        " l_exch=" + std::to_string(sp_.l_max_exchange) +
        " n_occ="  + std::to_string(sp_.n_occ) +
        " n_trans="+ std::to_string(sp_.n_transition) +
        " W_min="  + fmt_E(cfg.W_min);

    const std::string dir = cfg.checkpoint_dir.empty()
        ? ("./checkpoints/sinv_Nr"     + std::to_string(Nr)
           + "_Npsi"  + std::to_string(N_psi)
           + "_E"     + fmt_E(sp_.energy)
           + "_lc"    + std::to_string(sp_.l_max_continuum)
           + "_le"    + std::to_string(sp_.l_max_exchange))
        : cfg.checkpoint_dir;
    storage_.set_manifest(manifest);
    // Forward the planner's prefetch decision now -- before any early
    // return path (checkpoint hot-load) so the flag is correct on the
    // PotentialStorage that BackPropagator will later poke via
    // wi_.start_prefetch().  In MEMORY mode the flag is benign
    // (start_prefetch short-circuits on mode_ != DISK).
    storage_.set_prefetch_allowed(cfg.enable_prefetch);
    // Forward the chunk-rechunk policy too: PotentialStorage::
    // initialize_from_checkpoint reads it inside the checkpoint-load
    // call below, so it must be set first.
    storage_.set_chunk_rechunk_allowed(cfg.allow_chunk_rechunk);

    // ---- try to hot-load a checkpoint (regardless of target storage) -----
    //   MEMORY target: stream on-disk chunks into memory_storage_.
    //   DISK   target: point the store at the on-disk chunks for lazy reads.
    if (cfg.try_load_checkpoint) {
        if (st_mode == PotentialStorage::Mode::MEMORY) {
            if (storage_.try_load_into_memory(Nr, N_psi, dir, cfg.chunk_size)) {
                if (cfg.verbose) {
                    std::cout << "[SchurInverter] loaded checkpoint into MEMORY: "
                              << dir << "\n";
                }
                return;
            }
        } else {
            if (storage_.initialize_from_checkpoint(Nr, N_psi, dir, cfg.chunk_size)) {
                if (cfg.verbose) {
                    std::cout << "[SchurInverter] loaded checkpoint (DISK): "
                              << dir << "\n";
                }
                return;
            }
        }
    }

    storage_.initialize_for_write(Nr, N_psi, st_mode, dir, cfg.chunk_size,
                                  cfg.symmetric_storage,
                                  cfg.parallel_chunk_write);

    if (cfg.verbose) {
        std::cout << "[SchurInverter] build start: N_psi=" << N_psi
                  << "  N_f=" << N_f
                  << "  Nr=" << Nr
                  << "  mode="
                  << (st_mode == PotentialStorage::Mode::MEMORY ? "MEMORY" : "DISK")
                  << "  footprint=" << (bytes_total >> 20) << " MB\n";
    }

    auto t0 = std::chrono::steady_clock::now();
    shifts_A_ = 0;
    shifts_S_ = 0;
    stats_ = Stats{};

    // ---- optional GPU stepper -------------------------------------------
    // Set up before the per-n loop so its persistent device buffers (A,
    // B, Bscaled, Dinv, temp, ipiv, getrf/getri scratchpads) are
    // allocated once.  Bit-tight equivalence to the CPU path is
    // verified by test_gpu_sinv on the H2O fixture.
    std::unique_ptr<GpuContext>    gpu_ctx;
    std::unique_ptr<GpuSinvStepper> gpu_step;
    if (cfg.use_gpu) {
        if (!GpuContext::gpu_available()) {
            throw std::runtime_error(
                "SchurInverter::build: cfg.use_gpu=true but no SYCL GPU is "
                "visible (rebuild with -DSCATT_WITH_SYCL=ON using icpx, and "
                "run on a GPU node).");
        }
        gpu_ctx  = std::make_unique<GpuContext>(/*prefer_gpu*/ true);
        gpu_step = std::make_unique<GpuSinvStepper>(*gpu_ctx, N_psi, N_f);
        if (cfg.verbose) {
            std::cout << "[SchurInverter] GPU offload enabled: "
                      << gpu_ctx->info().device_name
                      << "  (HBM " << (gpu_ctx->info().global_mem_bytes >> 30)
                      << " GB)\n";
            std::cout << "[SchurInverter] per-n Schur + LU inversion will "
                         "run on device for n >= stab_n_max ("
                      << cfg.stab_n_max << "); stability-region n < "
                      << cfg.stab_n_max << " stays on CPU.\n";
        }
    }

    // ---- thread-local workspace -------------------------------------------
    struct WS {
        Eigen::MatrixXd A;
        Eigen::MatrixXd S;
        Eigen::MatrixXd B;
        Eigen::MatrixXd Sinv;
        Eigen::MatrixXd Q;
        Eigen::MatrixXd B_scaled;
        Eigen::VectorXd Dinv;
        ExchangeCouplingWorkspace ec_ws;
    };

    auto make_ws = [&]() {
        WS w;
        w.A.resize(N_psi, N_psi);
        w.S.resize(N_psi, N_psi);
        w.B.resize(N_psi, N_f);
        w.Sinv.resize(N_psi, N_psi);
        w.Q.resize(N_psi, N_f);
        w.B_scaled.resize(N_psi, N_f);
        w.Dinv.resize(N_f);
        if (ec_) w.ec_ws = ec_->make_workspace();
        return w;
    };

    // ---------- compute Sinv at one ir, given V(n) externally ------------
    // This is the per-ir computation shared by:
    //   * default work_one (which fetches V from pot_.get(n))
    //   * the opt-in parallel-DISK chunk-blocked path (which reads V from
    //     a thread-safe local pre-copied buffer)
    // The function writes the result to w.Sinv and updates the per-stage
    // timing accumulators (A, B, schur, invert).  It does NOT call
    // storage_.store and does NOT update t_store_ns or n_steps -- those
    // are handled by the caller.
    auto compute_sinv_at = [&](int n, const Eigen::MatrixXd* V_n_ptr, WS& w) {
        using clk = std::chrono::steady_clock;
        const double r  = sp_.r_min + static_cast<double>(n) * sp_.dr;
        const double r2 = r * r;

        auto tA = clk::now();
        // -------- A(n) = I + (h²/6)·(E·I − V(n)) --------
        w.A.setIdentity();
        w.A.diagonal().array() += h2_6 * sp_.energy;
        if (n < static_cast<int>(sp_.n_grid)) {
            const Eigen::MatrixXd& V_n = V_n_ptr
                ? *V_n_ptr
                : pot_.get(static_cast<std::size_t>(n));
            w.A.noalias() -= h2_6 * V_n;
        }

        // Stability shift on A for small n.
        if (n < cfg.stab_n_max) {
            w.A = 0.5 * (w.A + w.A.transpose().eval());
            if (stability_shift(w.A, cfg.W_min, cfg.shift_margin)) {
                #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
                #pragma omp atomic
                #endif
                ++shifts_A_;
            }
        }

        const bool exchange_on = ec_ && (n < sp_.n_transition);

        // ========================================================== //
        // OPT-IN GPU dispatch:
        // For n >= cfg.stab_n_max no S-side stability shift is needed,
        // so the heavy Schur GEMM + dense inverse can be offloaded to
        // the device.  We still build A, D⁻¹ and B on the CPU (cheap +
        // already there), upload them, get Sinv back, and skip the
        // entire CPU Schur + inverse path.  Bit-tight equivalence
        // proven by test_gpu_sinv.
        // ========================================================== //
        if (gpu_step && n >= cfg.stab_n_max) {
            auto tB0_gpu = clk::now();
            if (exchange_on) {
                // CPU: build D⁻¹ and B as the GPU step expects.
                for (int f_idx = 0; f_idx < N_f; ++f_idx) {
                    const int l = l_sigma_[f_idx % sp_.n_sigma];
                    const double centrif =
                        (r2 > 1e-30) ? (static_cast<double>(l) * (l + 1)) / r2 : 0.0;
                    double Df = 1.0 - h2_12 * centrif;
                    if (Df < cfg.W_min) Df = cfg.W_min;
                    w.Dinv(f_idx) = 1.0 / Df;
                }
                ec_->compute_into(n, (*chi_)[static_cast<std::size_t>(n)],
                                  w.ec_ws, w.Q);
                w.B.noalias() = h2_12 * w.Q;
                auto tInv_gpu_start = clk::now();
                // GPU: S = A − B·D⁻¹·Bᵀ followed by getrf+getri+symmetrise.
                gpu_step->step(w.A, w.B, w.Dinv, w.Sinv);
                auto tInv_gpu_end = clk::now();
                const std::uint64_t dB =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(tInv_gpu_start - tB0_gpu).count();
                const std::uint64_t dInv =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(tInv_gpu_end - tInv_gpu_start).count();
                stats_.t_B_build_ns += dB;
                stats_.t_invert_ns  += dInv;
            } else {
                // S = A: only invert on device.
                auto tInv_gpu_start = clk::now();
                gpu_step->step_inverse_only(w.A, w.Sinv);
                auto tInv_gpu_end = clk::now();
                const std::uint64_t dInv =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(tInv_gpu_end - tInv_gpu_start).count();
                stats_.t_invert_ns += dInv;
            }
            const std::uint64_t dA =
                std::chrono::duration_cast<std::chrono::nanoseconds>(tB0_gpu - tA).count();
            stats_.t_A_build_ns += dA;
            return;     // Skip the CPU Schur/invert path below.
        }

        auto tB0 = clk::now();

        if (!exchange_on) {
            // S = A (no exchange coupling at this point).
            w.S = w.A;
        } else {
            // -------- D⁻¹ diagonal, with Johnson clamp --------
            for (int f_idx = 0; f_idx < N_f; ++f_idx) {
                const int l  = l_sigma_[f_idx % sp_.n_sigma];
                const double centrif =
                    (r2 > 1e-30) ? (static_cast<double>(l) * (l + 1)) / r2 : 0.0;
                double Df = 1.0 - h2_12 * centrif;
                if (Df < cfg.W_min) Df = cfg.W_min;
                w.Dinv(f_idx) = 1.0 / Df;
            }

            // -------- B(n) = (h²/12) Q_ψf(n) --------
            ec_->compute_into(n, (*chi_)[static_cast<std::size_t>(n)], w.ec_ws, w.Q);
            w.B.noalias() = h2_12 * w.Q;
            auto tB1 = clk::now();

            // -------- S = A − B · D⁻¹ · B^T --------
            for (int f_idx = 0; f_idx < N_f; ++f_idx) {
                w.B_scaled.col(f_idx) = w.B.col(f_idx) * w.Dinv(f_idx);
            }
            w.S.noalias()  = w.A;
            w.S.noalias() -= w.B_scaled * w.B.transpose();

            // Symmetrise S only on the LEGACY (general-LU) path: dgetrf
            // doesn't preserve symmetry, and the legacy code symmetrised
            // S to bit-equality before inversion to keep Sinv consistent.
            // The new symmetric-LDLᵀ path (dsytrf, LOWER-only) doesn't
            // need this -- it reads the LOWER triangle directly, treating
            // any FP-rounding asymmetry in the upper triangle as a
            // discardable detail of the generic GEMM (~1e-13 relative).
            if (!cfg.use_symmetric_inverse) {
                w.S = 0.5 * (w.S + w.S.transpose().eval());
            }

            if (n < cfg.stab_n_max) {
                if (stability_shift(w.S, cfg.W_min, cfg.shift_margin)) {
                    #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
                    #pragma omp atomic
                    #endif
                    ++shifts_S_;
                }
            }

            const std::uint64_t dB = std::chrono::duration_cast<std::chrono::nanoseconds>(tB1 - tB0).count();
            const std::uint64_t dS = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - tB1).count();
            #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
            #pragma omp atomic
            #endif
            stats_.t_B_build_ns += dB;
            #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
            #pragma omp atomic
            #endif
            stats_.t_schur_ns += dS;
        }

        auto tInvStart = clk::now();
        // -------- Sinv = S^(-1) --------
        // New path (default, cfg.use_symmetric_inverse = true):
        //   Bunch-Kaufman LDLᵀ via LAPACKE_dsytrf+dsytri.  Reads only the
        //   LOWER triangle of S, returns a fully populated bit-symmetric
        //   inverse.  ~1.5-2× faster than dgetrf+dgetri at large N because
        //   of fewer flops AND because it lets us skip both 0.5*(M+M^T)
        //   symmetrisations.  Numerically equivalent to the legacy path
        //   to ~1e-13 relative; validated by test_sinv_symmetric_inverse.
        // Legacy path (cfg.use_symmetric_inverse = false):
        //   dgetrf+dgetri followed by an explicit 0.5*(Sinv+Sinv^T)
        //   symmetrisation to clean up dgetri's row/column-op asymmetry.
        if (cfg.use_symmetric_inverse) {
            w.Sinv = inverse_symmetric_indefinite(w.S);
        } else {
            w.Sinv.noalias() = inverse_general(w.S);
            w.Sinv = 0.5 * (w.Sinv + w.Sinv.transpose().eval());
        }
        auto tInvEnd = clk::now();

        const std::uint64_t dA     = std::chrono::duration_cast<std::chrono::nanoseconds>(tB0 - tA).count();
        const std::uint64_t dInv   = std::chrono::duration_cast<std::chrono::nanoseconds>(tInvEnd - tInvStart).count();
        #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        #pragma omp atomic
        #endif
        stats_.t_A_build_ns += dA;
        #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        #pragma omp atomic
        #endif
        stats_.t_invert_ns  += dInv;
    };

    // Thin wrapper used by the default (serial-DISK + MEMORY-parallel) paths:
    // compute Sinv (using pot.get() inline) and immediately store it.
    auto work_one = [&](int n, WS& w) {
        using clk = std::chrono::steady_clock;
        compute_sinv_at(n, /*V_n_ptr=*/nullptr, w);
        auto tStore0 = clk::now();
        storage_.store(static_cast<std::size_t>(n), w.Sinv);
        auto tStoreEnd = clk::now();
        const std::uint64_t dStore = std::chrono::duration_cast<std::chrono::nanoseconds>(tStoreEnd - tStore0).count();
        #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        #pragma omp atomic
        #endif
        stats_.t_store_ns += dStore;
        #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
        #pragma omp atomic
        #endif
        ++stats_.n_steps;
    };

    // ---- drive the loop ---------------------------------------------------
    // Parallelism is safe ONLY when both storages are MEMORY:
    //   - pot.get() is ref-to-vector-slot in MEMORY mode (thread-safe read).
    //   - storage_.store() is thread-safe for distinct ir in MEMORY mode.
    // GPU mode is single-threaded: one sycl::queue, one set of device
    // buffers; concurrent host threads would race.  Force serial loop
    // when cfg.use_gpu (parallelism already comes from the device).
    const bool can_parallel =
        cfg.use_openmp
        && pot_.storage_mode() == StorageMode::MEMORY
        && st_mode == PotentialStorage::Mode::MEMORY
        && !cfg.use_gpu;

    // Opt-in: chunk-blocked OpenMP parallelism in DISK mode.
    // Bit-identical to the serial-DISK output (proven by
    // test_sinv_serial_vs_parallel.cpp + per-ir compute determinism).
    // Triggers only when:
    //   - cfg.parallel_disk_chunks == true (caller explicitly enables it)
    //   - cfg.use_openmp == true (otherwise just a single OMP thread)
    //   - storage mode is DISK (no benefit/safety guarantee otherwise:
    //     MEMORY mode already parallelises directly).
    const bool use_chunked_disk_parallel =
        cfg.parallel_disk_chunks
        && cfg.use_openmp
        && st_mode == PotentialStorage::Mode::DISK
        && !cfg.use_gpu;     // GPU dispatch is serial-only

    stats_.parallel_over_ir = can_parallel || use_chunked_disk_parallel;
    #if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
    if (can_parallel) {
        #pragma omp parallel
        {
            WS w = make_ws();
            #pragma omp for schedule(dynamic, 64)
            for (int n = 0; n < static_cast<int>(Nr); ++n) {
                work_one(n, w);
            }
        }
    } else if (use_chunked_disk_parallel) {
        // Chunk-blocked parallel-DISK pattern.
        //
        // Memory: per-thread workspace is large (~ 5·N_psi² + 3·N_psi·N_f
        // doubles).  We compute a SAFE thread cap from the system RAM
        // budget so other code phases still see the full thread pool --
        // we use num_threads(T_cap) on the parallel region, NOT a global
        // omp_set_num_threads.
        //
        // Pot pre-fetch: we exploit the StoragePlanner contract --
        //   pot.matrix_bytes == sinv.matrix_bytes  (both N_ch²·8) so
        //   the planner always sets pot.chunk_size == sinv.chunk_size,
        //   and chunks of the same index cover the same ir range.
        // We touch pot.get(n_lo) once before each parallel region; this
        // forces pot's read_buffer to load the chunk that covers
        // [n_lo, n_hi).  Inside the parallel region every thread's
        // pot.get(n) call hits the loaded buffer (no reload) -- pot
        // becomes a read-only shared resource for the duration.
        //
        // Per chunk:
        //   1. pot.get(n_lo)              -> single-thread pot chunk load
        //   2. parallel for n in chunk:   compute Sinv into chunk_sinv[n - n_lo]
        //   3. for n in chunk:            storage_.store(n, chunk_sinv[i])
        //                                 (sequential flush satisfies the
        //                                  PotentialStorage DISK contract)
        const int chunk = std::max(1, cfg.chunk_size);
        std::vector<Eigen::MatrixXd> chunk_sinv(chunk);
        for (auto& m : chunk_sinv) m.resize(N_psi, N_psi);

        // ---- thread cap ---------------------------------------------------
        // Estimate memory available right now for thread workspaces.
        // Conservative: budget = 90% of detected RAM, minus what we
        // already know is allocated:
        //   - pot resident: chunk_size · matrix_bytes(sinv) (same size class)
        //   - sinv write_buffer: chunk_size · matrix_bytes
        //   - chunk_sinv (just allocated above): chunk_size · matrix_bytes
        // Per-thread WS = 5·N_psi²·8 + 3·N_psi·N_f·8 (the WS struct).
        const std::size_t mat_bytes_psi   = static_cast<std::size_t>(N_psi)
                                          * static_cast<std::size_t>(N_psi)
                                          * sizeof(double);
        const std::size_t mat_bytes_psi_f = static_cast<std::size_t>(N_psi)
                                          * static_cast<std::size_t>(N_f)
                                          * sizeof(double);
        const std::size_t per_thread_ws   = 5 * mat_bytes_psi
                                          + 3 * mat_bytes_psi_f;
        const std::size_t total_ram       = scatt::detect_total_ram_bytes();
        const std::size_t ram_budget      = static_cast<std::size_t>(
            static_cast<double>(total_ram) * 0.90);
        const std::size_t known_buffers   = 3 * static_cast<std::size_t>(chunk)
                                          * mat_bytes_psi;     // pot+sinv resid + chunk_sinv
        std::size_t avail_for_threads = (ram_budget > known_buffers)
                                      ? ram_budget - known_buffers : 0;
        const int T_user = omp_get_max_threads();
        const int T_safe_unbounded = (per_thread_ws > 0)
            ? static_cast<int>(avail_for_threads / per_thread_ws) : 1;
        int T_cap = std::min(T_user, T_safe_unbounded);
        if (T_cap < 1) T_cap = 1;

        const int n_chunks = (static_cast<int>(Nr) + chunk - 1) / chunk;
        if (cfg.verbose) {
            std::cout << "[SchurInverter] PARALLEL-DISK chunk-blocked: "
                      << n_chunks << " chunks of size " << chunk
                      << "; auto-thread cap = " << T_cap
                      << " of " << T_user
                      << "  (per-thread WS ~ "
                      << (per_thread_ws >> 20) << " MB; "
                      << "RAM budget = " << (ram_budget >> 30) << " GB)\n";
            if (T_cap < T_user) {
                std::cout << "[SchurInverter] threads capped from " << T_user
                          << " to " << T_cap
                          << " to fit memory; other code phases keep the full "
                          << T_user << "\n";
            }
        }
        if (T_cap < 2) {
            // Parallel-DISK with one thread is just a slower serial-DISK
            // (extra chunk_sinv copy + flush loop).  Fall back to the
            // existing serial path inside this branch.
            if (cfg.verbose) {
                std::cout << "[SchurInverter] memory tight; falling back to "
                             "serial DISK build for this run\n";
            }
            WS w = make_ws();
            for (int n = 0; n < static_cast<int>(Nr); ++n) {
                work_one(n, w);
            }
        } else {
            for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
                const int n_lo = chunk_idx * chunk;
                const int n_hi = std::min(n_lo + chunk, static_cast<int>(Nr));

                // 1. Force pot's chunk to be loaded into its read_buffer.
                //    For DISK pot this triggers a single chunk read; for
                //    MEMORY pot this is a cheap reference return.
                //    Crucially: subsequent parallel pot.get() calls for
                //    n in [n_lo, n_hi) will hit the loaded chunk and
                //    return refs without triggering a reload (planner
                //    guarantees chunk alignment between pot and sinv).
                (void) pot_.get(static_cast<std::size_t>(n_lo));

                // 2. Parallel compute, capped to T_cap threads.
                #pragma omp parallel num_threads(T_cap)
                {
                    WS w_local = make_ws();
                    #pragma omp for schedule(dynamic, 16)
                    for (int n = n_lo; n < n_hi; ++n) {
                        compute_sinv_at(n, /*V_n_ptr=*/nullptr, w_local);
                        chunk_sinv[n - n_lo] = w_local.Sinv;
                    }
                }

                // 3. Single-thread sequential flush.
                for (int n = n_lo; n < n_hi; ++n) {
                    using clk = std::chrono::steady_clock;
                    auto tStore0 = clk::now();
                    storage_.store(static_cast<std::size_t>(n),
                                   chunk_sinv[n - n_lo]);
                    auto tStoreEnd = clk::now();
                    stats_.t_store_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tStoreEnd - tStore0).count();
                    ++stats_.n_steps;
                }
            }
        }
    } else
    #endif
    {
        WS w = make_ws();
        for (int n = 0; n < static_cast<int>(Nr); ++n) {
            work_one(n, w);
            if (cfg.verbose && (n % 500 == 0 || n == static_cast<int>(Nr) - 1)) {
                const double progress = 100.0 * static_cast<double>(n + 1) / static_cast<double>(Nr);
                std::cout << "\r[SchurInverter] building: " << progress << "% (ir "
                          << n + 1 << "/" << Nr << ")"
                          << std::flush;
            }
        }
    }

    storage_.finalize_write();

    // Persist a MEMORY build to disk so a future run can skip the rebuild.
    // DISK mode already writes to disk during finalize_write.
    if (st_mode == PotentialStorage::Mode::MEMORY && cfg.save_checkpoint) {
        storage_.save_to_disk(dir, cfg.chunk_size, cfg.symmetric_storage,
                              cfg.parallel_chunk_write);
    }

    const double dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    if (cfg.verbose) {
        std::cout << "[SchurInverter] build done in " << dt << " s ("
                  << (static_cast<double>(Nr) / dt) << " pts/s)"
                  << "  shifts A/S: " << shifts_A_ << "/" << shifts_S_
                  << "  parallel: " << (can_parallel ? "yes" : "no") << "\n";
    }
}

const Eigen::MatrixXd& SchurInverter::get(std::size_t ir) {
    return storage_.get(ir);
}

}  // namespace scatt
