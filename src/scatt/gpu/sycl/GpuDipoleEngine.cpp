// GpuDipoleEngine.cpp -- SYCL / oneMKL implementation of the DGEMM-per-ir
// dipole bottleneck.  Structured exactly like GpuPropagate.cpp:
//
//   1. When SCATT_HAS_SYCL is defined:
//        - GpuDipoleEngine::Impl owns three USM-device buffers (V, psi, result)
//          and pulls the sycl::queue from the GpuContext passed at construction.
//        - step() does up-load, ONE oneapi::mkl::blas DGEMM, down-load, all
//          .wait()ed.
//
//   2. When SCATT_HAS_SYCL is NOT defined:
//        - Construction throws std::runtime_error ("GPU path not compiled in").
//        - step() throws too, so we still link cleanly into a CPU-only build.
//
// Why a separate translation unit (not folded into GpuPropagate.cpp):
//   * Single-responsibility: DME's GEMM shape (n_slots × N_psi) · (N_psi × N_psi)
//     is unrelated to FRP/BP's (N_total × N_total) GEMMs.  Different ID,
//     different buffer life-cycle.
//   * Matches the existing src/scatt/gpu/{sycl,cuda}/ layout (one file per
//     class so the CUDA port can mirror it 1-1).
//
// THIS FILE IS THE SYCL-SIDE GPU IMPLEMENTATION.  The corresponding CUDA
// implementation lives in src/scatt/gpu/cuda/GpuDipoleEngine.cu and is
// gated on SCATT_WITH_CUDA.  CMake selects exactly one .cpp / .cu pair via
// the SCATT_GPU_BACKEND_SRC list.

#include "scatt/GpuDipoleEngine.hpp"
#include "scatt/GpuPropagate.hpp"      // GpuContext

#include <chrono>
#include <stdexcept>
#include <string>

#ifdef SCATT_HAS_SYCL
  #if __has_include(<sycl/sycl.hpp>)
    #include <sycl/sycl.hpp>
  #else
    #include <CL/sycl.hpp>
  #endif
  #include <oneapi/mkl.hpp>
  #include <oneapi/mkl/blas.hpp>
#endif

namespace scatt {

// ===========================================================================
// SYCL BRANCH
// ===========================================================================
#ifdef SCATT_HAS_SYCL

namespace {

inline sycl::queue& as_queue(void* opaque) {
    return *static_cast<sycl::queue*>(opaque);
}

inline std::uint64_t ns_since(const std::chrono::steady_clock::time_point& t0) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count());
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// GpuDipoleEngine::Impl
// ---------------------------------------------------------------------------
struct GpuDipoleEngine::Impl {
    sycl::queue&       q;
    const std::int64_t Npsi;
    const std::int64_t Ns;        // n_slots

    double* d_V      = nullptr;   // n_slots × N_psi   (column-major)
    double* d_psi    = nullptr;   // N_psi   × N_psi   (column-major)
    double* d_result = nullptr;   // n_slots × N_psi   (column-major)

    Impl(sycl::queue& q_, std::int64_t Np, std::int64_t Ns_)
        : q(q_), Npsi(Np), Ns(Ns_)
    {
        const std::size_t V_sz   = static_cast<std::size_t>(Ns)   * Np;
        const std::size_t Psi_sz = static_cast<std::size_t>(Np)   * Np;
        const std::size_t Res_sz = static_cast<std::size_t>(Ns)   * Np;

        d_V      = sycl::malloc_device<double>(V_sz,   q);
        d_psi    = sycl::malloc_device<double>(Psi_sz, q);
        d_result = sycl::malloc_device<double>(Res_sz, q);

        if (!d_V || !d_psi || !d_result) {
            throw std::runtime_error(
                "GpuDipoleEngine: GPU malloc_device returned null "
                "(insufficient HBM for N_psi=" + std::to_string(Np) +
                ", n_slots=" + std::to_string(Ns_) + "?)");
        }
    }

    ~Impl() {
        if (d_V)      sycl::free(d_V,      q);
        if (d_psi)    sycl::free(d_psi,    q);
        if (d_result) sycl::free(d_result, q);
    }
};

GpuDipoleEngine::GpuDipoleEngine(GpuContext& ctx, int N_psi, int n_slots)
    : ctx_(ctx), N_psi_(N_psi), n_slots_(n_slots)
{
    if (N_psi <= 0 || n_slots <= 0) {
        throw std::runtime_error(
            "GpuDipoleEngine: N_psi and n_slots must be positive (got N_psi=" +
            std::to_string(N_psi) + ", n_slots=" + std::to_string(n_slots) + ")");
    }
    impl_ = std::make_unique<Impl>(as_queue(ctx_.raw_queue()),
                                   static_cast<std::int64_t>(N_psi),
                                   static_cast<std::int64_t>(n_slots));
}

GpuDipoleEngine::~GpuDipoleEngine() = default;

void GpuDipoleEngine::reset_stats() { stats_ = Stats{}; }

void GpuDipoleEngine::step(const Eigen::MatrixXd& V,
                            const Eigen::MatrixXd& psi_ir,
                            Eigen::MatrixXd&       result_out)
{
    if (V.rows() != n_slots_ || V.cols() != N_psi_) {
        throw std::runtime_error(
            "GpuDipoleEngine::step: V wrong shape (got " +
            std::to_string(V.rows()) + "×" + std::to_string(V.cols()) +
            ", expected " + std::to_string(n_slots_) + "×" + std::to_string(N_psi_) + ")");
    }
    if (psi_ir.rows() != N_psi_ || psi_ir.cols() != N_psi_) {
        throw std::runtime_error(
            "GpuDipoleEngine::step: psi_ir wrong shape (got " +
            std::to_string(psi_ir.rows()) + "×" + std::to_string(psi_ir.cols()) +
            ", expected " + std::to_string(N_psi_) + "²)");
    }
    if (result_out.rows() != n_slots_ || result_out.cols() != N_psi_) {
        result_out.resize(n_slots_, N_psi_);
    }

    auto& q = as_queue(ctx_.raw_queue());
    const std::int64_t Np = N_psi_;
    const std::int64_t Ns = n_slots_;
    const std::size_t  V_sz   = static_cast<std::size_t>(Ns) * Np;
    const std::size_t  Psi_sz = static_cast<std::size_t>(Np) * Np;
    const std::size_t  Res_sz = static_cast<std::size_t>(Ns) * Np;

    // (1) Upload V and psi_ir.  Two memcpys on the same in-order queue:
    //     order matters less than the .wait() bookend but kept consistent
    //     with the upload-first convention of GpuPropagate.cpp.
    auto t0 = std::chrono::steady_clock::now();
    q.memcpy(impl_->d_V,   V.data(),      V_sz   * sizeof(double));
    q.memcpy(impl_->d_psi, psi_ir.data(), Psi_sz * sizeof(double)).wait();
    stats_.t_upload_ns += ns_since(t0);

    // (2) DGEMM: result = V · psi_ir   (column-major, no transpose).
    //     V is (Ns × Np) column-major; psi_ir is (Np × Np) column-major;
    //     result is (Ns × Np) column-major.  Standard GEMM dims:
    //         M = Ns, N = Np, K = Np
    //         lda = Ns (V leading dim), ldb = Np (psi leading dim),
    //         ldc = Ns (result leading dim).
    //     Bit-equivalent to the per-row CPU GEMVs up to the GEMM-vs-GEMV
    //     summation-order rounding (ε_mach × N -- the standard floor of
    //     all oneMKL / MKL GEMMs at the same precision).
    auto t1 = std::chrono::steady_clock::now();
    try {
        oneapi::mkl::blas::column_major::gemm(
            q,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            Ns, Np, Np,
            1.0, impl_->d_V,      Ns,
                 impl_->d_psi,    Np,
            0.0, impl_->d_result, Ns).wait();
    } catch (const oneapi::mkl::exception& e) {
        throw std::runtime_error(
            std::string("GpuDipoleEngine::step: oneMKL DGEMM failed: ") + e.what());
    }
    stats_.t_gemm_ns += ns_since(t1);

    // (3) Download result.
    auto t2 = std::chrono::steady_clock::now();
    q.memcpy(result_out.data(), impl_->d_result, Res_sz * sizeof(double)).wait();
    stats_.t_download_ns += ns_since(t2);

    ++stats_.n_steps;
}

// ===========================================================================
// NON-SYCL BRANCH -- same API, all stubs throw.
// ===========================================================================
#else  // !SCATT_HAS_SYCL

namespace {
[[noreturn]] void gpu_not_compiled_() {
    throw std::runtime_error(
        "GpuDipoleEngine: GPU path not compiled in. Rebuild with "
        "-DSCATT_WITH_SYCL=ON (Intel oneAPI DPC++) or -DSCATT_WITH_CUDA=ON.");
}
}

struct GpuDipoleEngine::Impl {};

GpuDipoleEngine::GpuDipoleEngine(GpuContext& ctx, int N_psi, int n_slots)
    : ctx_(ctx), N_psi_(N_psi), n_slots_(n_slots)
{
    (void)ctx_;        // silences -Wunused-private-field in the no-SYCL stub
    gpu_not_compiled_();
}

GpuDipoleEngine::~GpuDipoleEngine() = default;

void GpuDipoleEngine::reset_stats() { stats_ = Stats{}; }

void GpuDipoleEngine::step(const Eigen::MatrixXd&,
                            const Eigen::MatrixXd&,
                            Eigen::MatrixXd&)
{
    gpu_not_compiled_();
}

#endif  // SCATT_HAS_SYCL

}  // namespace scatt
