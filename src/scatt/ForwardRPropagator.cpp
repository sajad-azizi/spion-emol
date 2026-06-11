#include "scatt/ForwardRPropagator.hpp"

#include "angular/Gaunt.hpp"
#include "scatt/GpuPropagate.hpp"
#include "scatt/LapackInverse.hpp"
#include "scatt/MklThreadGuard.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace scatt {

ForwardRPropagator::ForwardRPropagator(const SolverParams& sp,
                                       Potentials&         pot,
                                       WInverseOperator&   wi)
    : sp_(sp), pot_(pot), wi_(wi)
{
    build_channel_info_();
    n_start_ = compute_n_start_();
}

void ForwardRPropagator::build_channel_info_() {
    l_psi_.resize(sp_.n_mu);
    for (int mu = 0; mu < sp_.n_mu; ++mu) {
        int l, m;
        angular::idx_to_lm(mu, l, m);
        l_psi_[mu] = l;
    }
    l_sigma_.resize(sp_.n_sigma);
    for (int s = 0; s < sp_.n_sigma; ++s) {
        int l, m;
        angular::idx_to_lm(s, l, m);
        l_sigma_[s] = l;
    }
}

int ForwardRPropagator::compute_n_start_() const {
    int l_max_f = 0;
    for (int s = 0; s < sp_.n_sigma; ++s) {
        l_max_f = std::max(l_max_f, l_sigma_[s]);
    }
    // r_crit = h · √(ℓ_max(ℓ_max+1)/12).  Factor-2 safety margin matches
    // version_0.
    const double r_crit = sp_.dr *
        std::sqrt(static_cast<double>(l_max_f * (l_max_f + 1)) / 12.0);
    const double r_start = 2.0 * r_crit;
    int ns = std::max(1, static_cast<int>(std::ceil(r_start / sp_.dr)));
    ns = std::min(ns, static_cast<int>(sp_.n_grid) - 1);
    return ns;
}

double ForwardRPropagator::riccati_besselJ_(int l, double x) {
    // GSL's gsl_sf_bessel_jl returns the spherical Bessel j_l(x) (regular).
    // Riccati-Bessel ĵ_l(x) = x · j_l(x).
    if (x < 1e-300) return 0.0;
    gsl_sf_result res;
    const int status = gsl_sf_bessel_jl_e(l, x, &res);
    if (status != GSL_SUCCESS && status != GSL_EUNDRFLW) {
        throw std::runtime_error(std::string("gsl_sf_bessel_jl_e failed: ") +
                                 gsl_strerror(status));
    }
    return x * res.val;
}

void
ForwardRPropagator::analytic_init_(int n, double W_min,
                                   Eigen::MatrixXd& Rinv) const
{
    // Diagonal-only. Rows/cols already zeroed by caller.
    if (n <= 0) return;   // formula undefined at n=0 (r=0 or r=r_min;
                          // ratio isn't numerically meaningful).

    const int N_psi = sp_.n_mu;
    const int N_f   = sp_.n_occ * sp_.n_sigma;
    const double h     = sp_.dr;
    const double h2_12 = h * h / 12.0;
    const double h2_6  = h * h / 6.0;
    const double k     = std::sqrt(2.0 * sp_.energy);   // k real for E > 0
    const double r_n   = sp_.r_min + n       * h;
    const double r_np1 = sp_.r_min + (n + 1) * h;

    // --- ψ channels ---
    // M_n^ψ(μ, μ) = 1 + (h²/12)·Q_ψψ(n, μ, μ)
    //            = 1 + (h²/6)·(E − V(n, μ, μ))    (diag only)
    // Since V here is the FULL potential (centrifugal + nuclear + polarisation)
    // from Potentials::get, M includes the right Numerov correction.
    const Eigen::MatrixXd& V_n   = pot_.get(static_cast<std::size_t>(n));
    const Eigen::MatrixXd& V_np1 = pot_.get(static_cast<std::size_t>(n + 1));

    for (int mu = 0; mu < N_psi; ++mu) {
        const double M_n   = 1.0 + h2_6 * (sp_.energy - V_n(mu, mu));
        const double M_np1 = 1.0 + h2_6 * (sp_.energy - V_np1(mu, mu));
        const double y_n   = riccati_besselJ_(l_psi_[mu], k * r_n);
        const double y_np1 = riccati_besselJ_(l_psi_[mu], k * r_np1);
        const double denom = M_np1 * y_np1;
        if (std::abs(denom) > 1e-300) {
            Rinv(mu, mu) = (M_n * y_n) / denom;
        }
        // else: leave zero (should be rare; j_l(kr_np1) = 0 would be a node)
    }

    // --- f channels ---
    // M_n^f diag = 1 + (h²/12)·Q_ff(n, f, f) = 1 - (h²/12)·ℓ(ℓ+1)/r_n²,
    // clamped to W_min (same policy as SchurInverter).
    const double r_n2   = r_n   * r_n;
    const double r_np12 = r_np1 * r_np1;
    for (int f_idx = 0; f_idx < N_f; ++f_idx) {
        const int l = l_sigma_[f_idx % sp_.n_sigma];
        const double centrif_n   = (r_n2   > 1e-30) ? double(l * (l + 1)) / r_n2   : 0.0;
        const double centrif_np1 = (r_np12 > 1e-30) ? double(l * (l + 1)) / r_np12 : 0.0;
        double M_n   = 1.0 - h2_12 * centrif_n;
        double M_np1 = 1.0 - h2_12 * centrif_np1;
        if (M_n   < W_min) M_n   = W_min;
        if (M_np1 < W_min) M_np1 = W_min;
        const double y_n   = std::pow(r_n,   static_cast<double>(l + 1));
        const double y_np1 = std::pow(r_np1, static_cast<double>(l + 1));
        const double denom = M_np1 * y_np1;
        if (std::abs(denom) > 1e-300) {
            Rinv(N_psi + f_idx, N_psi + f_idx) = (M_n * y_n) / denom;
        }
    }
}

void ForwardRPropagator::run(const Config& cfg) {
    // Force the main thread's MKL pool to the full OMP thread count for
    // the serial-outer per-n loop.  Without this, MKL's dynamic-adjustment
    // heuristic (mkl_set_dynamic(1) set globally to protect against
    // oversubscription inside SchurInverter's OMP-parallel region) can
    // drop the GEMM/LU thread count for the per-step wi_.materialize_into
    // even though we're in a serial section.  At C8F8 / l_cont=100 this
    // had been costing ~21 hours of CPU time per energy point.
    // Bit-equivalent to the legacy code up to MKL summation order (same
    // tolerance the existing test suite already accepts).
#if defined(_OPENMP)
    MklThreadGuard _mkl_local(omp_get_max_threads());
#else
    MklThreadGuard _mkl_local(1);
#endif

    const int         N_psi = sp_.n_mu;
    const int         N_f   = sp_.n_occ * sp_.n_sigma;
    const int         N_tot = N_psi + N_f;
    const std::size_t Nr    = sp_.n_grid;

    // ---- storage mode ----------------------------------------------------
    const std::size_t bytes_total =
        Nr * static_cast<std::size_t>(N_tot) * static_cast<std::size_t>(N_tot) * sizeof(double);
    PotentialStorage::Mode st_mode = PotentialStorage::Mode::MEMORY;
    if (cfg.storage == StorageMode::DISK) {
        st_mode = PotentialStorage::Mode::DISK;
    } else if (cfg.storage == StorageMode::AUTO &&
               bytes_total > 10ULL * 1024ULL * 1024ULL * 1024ULL)
    {
        st_mode = PotentialStorage::Mode::DISK;
    }

    // Parameter-encoded dir + manifest (same pattern as SchurInverter).
    auto fmt_E = [](double e) {
        char buf[32]; std::snprintf(buf, sizeof(buf), "%.6f", e); return std::string(buf);
    };
    const std::string manifest =
        "kind=rinv"
        " energy=" + fmt_E(sp_.energy) +
        " Nr="     + std::to_string(Nr) +
        " dr="     + fmt_E(sp_.dr) +
        " Ntot="   + std::to_string(N_tot) +
        " Npsi="   + std::to_string(N_psi) +
        " Nf="     + std::to_string(N_f) +
        " l_cont=" + std::to_string(sp_.l_max_continuum) +
        " l_exch=" + std::to_string(sp_.l_max_exchange) +
        " n_occ="  + std::to_string(sp_.n_occ) +
        " n_trans="+ std::to_string(sp_.n_transition) +
        " W_min="  + fmt_E(cfg.W_min);

    const std::string dir = cfg.checkpoint_dir.empty()
        ? ("./checkpoints/rinv_Nr"     + std::to_string(Nr)
           + "_Ntot"  + std::to_string(N_tot)
           + "_E"     + fmt_E(sp_.energy)
           + "_lc"    + std::to_string(sp_.l_max_continuum)
           + "_le"    + std::to_string(sp_.l_max_exchange))
        : cfg.checkpoint_dir;
    storage_.set_manifest(manifest);
    // Forward the planner's prefetch decision now -- before any early
    // return path (checkpoint hot-load) so the flag is correct on the
    // PotentialStorage that BackPropagator will later poke via
    // frp_.start_prefetch().  In MEMORY mode the flag is benign
    // (start_prefetch short-circuits on mode_ != DISK).
    storage_.set_prefetch_allowed(cfg.enable_prefetch);
    // Forward the chunk-rechunk policy.  Must be set BEFORE the
    // checkpoint-load attempt below.
    storage_.set_chunk_rechunk_allowed(cfg.allow_chunk_rechunk);

    // ---- try checkpoint load --------------------------------------------
    if (cfg.try_load_checkpoint) {
        if (st_mode == PotentialStorage::Mode::MEMORY) {
            if (storage_.try_load_into_memory(Nr, N_tot, dir, cfg.chunk_size)) {
                rinv_final_ = storage_.get(Nr - 1);
                if (cfg.verbose) {
                    std::cout << "[ForwardRPropagator] loaded checkpoint into MEMORY: "
                              << dir << "\n";
                }
                return;
            }
        } else {
            if (storage_.initialize_from_checkpoint(Nr, N_tot, dir, cfg.chunk_size)) {
                rinv_final_ = storage_.get(Nr - 1);
                if (cfg.verbose) {
                    std::cout << "[ForwardRPropagator] loaded checkpoint (DISK): "
                              << dir << "\n";
                }
                return;
            }
        }
    }

    storage_.initialize_for_write(Nr, N_tot, st_mode, dir, cfg.chunk_size,
                                  cfg.symmetric_storage,
                                  cfg.parallel_chunk_write);

    if (cfg.verbose) {
        std::cout << "[ForwardRPropagator] build start: N_total=" << N_tot
                  << "  Nr=" << Nr
                  << "  n_start=" << n_start_
                  << "  mode="
                  << (st_mode == PotentialStorage::Mode::MEMORY ? "MEMORY" : "DISK")
                  << "  footprint=" << (bytes_total >> 20) << " MB\n";
    }

    auto t0 = std::chrono::steady_clock::now();

    // ---- allocate working matrices --------------------------------------
    Eigen::MatrixXd I_total   = Eigen::MatrixXd::Identity(N_tot, N_tot);
    Eigen::MatrixXd Rinv_prev = Eigen::MatrixXd::Zero(N_tot, N_tot);
    Eigen::MatrixXd U         = Eigen::MatrixXd::Zero(N_tot, N_tot);
    Eigen::MatrixXd R_current = Eigen::MatrixXd::Zero(N_tot, N_tot);
    Eigen::MatrixXd Rinv_n    = Eigen::MatrixXd::Zero(N_tot, N_tot);
    auto ws = wi_.make_workspace();

    // ---- analytic init for n ∈ [0, n_start) ------------------------------
    for (int n = 0; n < n_start_; ++n) {
        Eigen::MatrixXd Rinv_analytic = Eigen::MatrixXd::Zero(N_tot, N_tot);
        analytic_init_(n, cfg.W_min, Rinv_analytic);   // no-op at n = 0
        storage_.store(static_cast<std::size_t>(n), Rinv_analytic);
        if (n == n_start_ - 1) Rinv_prev = Rinv_analytic;
    }

    // ---- optional GPU stepper (persistent buffers, pinned R_prev_inv) ---
    std::unique_ptr<GpuContext>        gpu_ctx;
    std::unique_ptr<GpuForwardStepper> gpu_step;
    if (cfg.use_gpu) {
        if (!GpuContext::gpu_available()) {
            throw std::runtime_error(
                "ForwardRPropagator: use_gpu=true but no SYCL GPU is visible "
                "(rebuild with -DSCATT_WITH_SYCL=ON using icpx, and run on a "
                "GPU node).");
        }
        gpu_ctx  = std::make_unique<GpuContext>(/*prefer_gpu*/ true);
        gpu_step = std::make_unique<GpuForwardStepper>(*gpu_ctx, N_tot);
        gpu_step->init_R_prev_inv(Rinv_prev);
        // Request the symmetric-LDLᵀ inversion path on the GPU.  Honoured
        // only if oneMKL on this device supports sytrf/sytri (checked at
        // construction); otherwise a silent fallback to getrf/getri is used.
        gpu_step->enable_symmetric_inverse(cfg.use_symmetric_inverse);
        if (cfg.verbose) {
            std::cout << "[ForwardRPropagator] GPU offload enabled: "
                      << gpu_ctx->info().device_name
                      << "  (HBM "
                      << (gpu_ctx->info().global_mem_bytes >> 30) << " GB)\n";
            std::cout << "[ForwardRPropagator] GPU inversion path: "
                      << (gpu_step->is_symmetric_inverse_active()
                          ? "dsytrf+dsytri (LDLᵀ on LOWER, mirror to UPPER)"
                          : (cfg.use_symmetric_inverse
                             ? "dgetrf+dgetri (sytrf NOT supported on this device; legacy fallback)"
                             : "dgetrf+dgetri (legacy; user disabled symmetric path)"))
                      << "\n";
        }
    }

    // ---- numerical recursion for n ∈ [n_start, N_grid) -------------------
    stats_ = Stats{};
    using clk = std::chrono::steady_clock;

    // Workspace matrix for the GPU path: W_inv materialized once per step
    // on host, uploaded to device.  Unused on CPU path.
    Eigen::MatrixXd W_inv_host;

    for (int n = n_start_; n < static_cast<int>(Nr); ++n) {
        ++stats_.n_steps;

        auto tA = clk::now();

        if (gpu_step) {
            // GPU path: materialize W_inv(n) once on host, upload and let
            // the device do U_combine + LU inverse.
            wi_.materialize_into(n, W_inv_host, ws);
            auto tB = clk::now();
            stats_.t_u_assemble_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(tB - tA).count());

            gpu_step->step(W_inv_host, Rinv_n);
            auto tC = clk::now();
            stats_.t_linsolve_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(tC - tB).count());

            storage_.store(static_cast<std::size_t>(n), Rinv_n);
            // Rinv_prev is already pinned on device; we only need the host
            // copy for the (rare) verbose symmetry print below.
            auto tD = clk::now();
            stats_.t_store_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(tD - tC).count());

            if(n%100==0){
                std::cerr << "n=" << n << "  t_u_assemble=" << stats_.t_u_assemble_ns/1e9 << " s  "
                          << "t_linsolve=" << stats_.t_linsolve_ns/1e9 << " s  "
                          << "t_store=" << stats_.t_store_ns/1e9 << " s\n";
            }

        } else {
            // CPU path (unchanged).
            wi_.apply_U(n, I_total, U, ws);
            R_current.noalias() = U;
            R_current.noalias() -= Rinv_prev;
            auto tB = clk::now();
            stats_.t_u_assemble_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(tB - tA).count());

            // CPU inversion of R_current = U − Rinv_prev.
            //
            // PRE-2026-05-08 BEHAVIOUR (kept after Phase 1 revert):
            //   dgetrf+dgetri followed by an explicit 0.5*(M+M^T.eval())
            //   symmetrise.  Mathematically R_current is symmetric (proof
            //   in Config::use_symmetric_inverse); a Bunch-Kaufman LDLᵀ
            //   path was tried in Phase 1 but rolled back because the
            //   3000-step Numerov recursion compounds the per-step
            //   ε·κ rounding noise to ~3e-8 max relative diff vs the
            //   legacy LU path on the H2O fixture.  Both algorithms are
            //   equally close to true Rinv (~ε·κ each), but they pick
            //   different rounding paths and their bytes diverge.  For
            //   the user's paranoid-zero-accuracy mandate we keep the
            //   legacy path verbatim here.  cfg.use_symmetric_inverse
            //   is retained as a Config field for the GPU path (where it
            //   gates oneMKL's sytrf+sytri inside GpuForwardStepper); on
            //   the CPU it is currently a no-op.
            (void)cfg.use_symmetric_inverse;
            Rinv_n.noalias() = inverse_general(R_current);
            Rinv_n = 0.5 * (Rinv_n + Rinv_n.transpose().eval());
            auto tC = clk::now();
            stats_.t_linsolve_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(tC - tB).count());

            storage_.store(static_cast<std::size_t>(n), Rinv_n);
            Rinv_prev = Rinv_n;
            auto tD = clk::now();
            stats_.t_store_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(tD - tC).count());


            if(n%100==0){
                std::cerr << "n=" << n << "  t_u_assemble=" << stats_.t_u_assemble_ns/1e9 << " s  "
                          << "t_linsolve=" << stats_.t_linsolve_ns/1e9 << " s  "
                          << "t_store=" << stats_.t_store_ns/1e9 << " s\n";
            }
        }

        if (cfg.verbose && n % 500 == 0 && n > 0) {
            const double sym_err =
                (Rinv_n - Rinv_n.transpose()).cwiseAbs().maxCoeff();
            std::cout << "  n=" << n <<std::scientific<<std::setprecision(10)<< "  sym_err=" << sym_err << "\n";
        }
    }

    // On the GPU path we never updated Rinv_prev each step; fetch the final
    // one from the last Rinv_n we downloaded.
    rinv_final_ = gpu_step ? Rinv_n : Rinv_prev;
    storage_.finalize_write();

    if (st_mode == PotentialStorage::Mode::MEMORY && cfg.save_checkpoint) {
        storage_.save_to_disk(dir, cfg.chunk_size, cfg.symmetric_storage,
                              cfg.parallel_chunk_write);
    }

    const double dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    if (cfg.verbose) {
        std::cout << "[ForwardRPropagator] build done in " << dt << " s ("
                  << (static_cast<double>(Nr) / dt) << " pts/s, "
                  << (dt / Nr * 1e3) << " ms/pt)\n";
    }
}

const Eigen::MatrixXd& ForwardRPropagator::get(std::size_t ir) {
    return storage_.get(ir);
}

}  // namespace scatt
