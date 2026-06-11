// GpuDipoleEngine.cu -- CUDA / cuBLAS implementation of the DGEMM-per-ir
// dipole bottleneck.  Mirrors src/scatt/gpu/sycl/GpuDipoleEngine.cpp 1-1
// at the public-API level so build-time backend selection (CMake
// SCATT_GPU_BACKEND_SRC) can pick this file for NVIDIA nodes (H100, A100,
// L40, ...) without touching call sites in DipoleMatrixElement.cpp.
//
// VALIDATION STATUS (2026-05-23)
// ------------------------------
// CODE WRITTEN, NOT YET COMPILED OR RUN.  This Mac dev box has no CUDA
// toolkit (cuda_runtime.h, cublas_v2.h, cuBLAS .so) — the file is
// code-review only.  First-compile is expected to surface small bugs:
// header paths, launch-config arithmetic, leading dimensions on
// cublasDgemm (column-major like the SYCL path here; cuBLAS is
// column-major by default).  Validate on an NVIDIA node by running
//   cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release \
//       -DSCATT_WITH_CUDA=ON -DSCATT_WITH_SYCL=OFF
//   cmake --build build_cuda -j
//   cd build_cuda && ctest -R 'gpu_dme' --output-on-failure
//
// GpuContext (from GpuPropagate.{hpp,cu}) is shared with the FRP / BP /
// Sinv steppers, so this file reuses its CudaContextBlob via the opaque
// pointer just like the SYCL path reuses the sycl::queue.

#include "scatt/GpuDipoleEngine.hpp"
#include "scatt/GpuPropagate.hpp"      // GpuContext (CUDA branch via CudaContextBlob)

#include <chrono>
#include <stdexcept>
#include <string>

#if defined(SCATT_HAS_CUDA) && SCATT_HAS_CUDA
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
#endif

namespace scatt {

// ===========================================================================
// CUDA BRANCH
// ===========================================================================
#if defined(SCATT_HAS_CUDA) && SCATT_HAS_CUDA

namespace {

// Forward declaration of the CUDA-side context blob.  The actual layout
// is defined in GpuPropagate.cu; we only need the field names to access
// the cuBLAS handle and the stream.
struct CudaContextBlob {
    cudaStream_t        stream;
    cublasHandle_t      cublas;
    void*               cusolver;     // opaque to this TU
};

inline CudaContextBlob& as_blob(void* opaque) {
    return *static_cast<CudaContextBlob*>(opaque);
}

#define CUDA_CHECK(expr)                                                 \
    do {                                                                  \
        cudaError_t _e = (expr);                                          \
        if (_e != cudaSuccess) {                                          \
            throw std::runtime_error(                                     \
                std::string("CUDA error: ") + cudaGetErrorString(_e));    \
        }                                                                 \
    } while (0)

#define CUBLAS_CHECK(expr)                                                \
    do {                                                                  \
        cublasStatus_t _s = (expr);                                       \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                \
            throw std::runtime_error(                                     \
                "cuBLAS error: status=" + std::to_string(int(_s)));       \
        }                                                                 \
    } while (0)

inline std::uint64_t ns_since(const std::chrono::steady_clock::time_point& t0) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count());
}

inline void stream_sync(cudaStream_t s) { CUDA_CHECK(cudaStreamSynchronize(s)); }

}  // anonymous namespace

// ---------------------------------------------------------------------------
// GpuDipoleEngine::Impl  (CUDA)
// ---------------------------------------------------------------------------
struct GpuDipoleEngine::Impl {
    CudaContextBlob&   ctx;
    const long long    Npsi;
    const long long    Ns;

    double* d_V      = nullptr;   // n_slots × N_psi   (column-major)
    double* d_psi    = nullptr;   // N_psi   × N_psi   (column-major)
    double* d_result = nullptr;   // n_slots × N_psi   (column-major)

    Impl(CudaContextBlob& c, long long Np, long long Ns_)
        : ctx(c), Npsi(Np), Ns(Ns_)
    {
        const std::size_t V_sz   = static_cast<std::size_t>(Ns)   * Np * sizeof(double);
        const std::size_t Psi_sz = static_cast<std::size_t>(Np)   * Np * sizeof(double);
        const std::size_t Res_sz = static_cast<std::size_t>(Ns)   * Np * sizeof(double);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_V),      V_sz));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_psi),    Psi_sz));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_result), Res_sz));
    }

    ~Impl() {
        if (d_V)      cudaFree(d_V);
        if (d_psi)    cudaFree(d_psi);
        if (d_result) cudaFree(d_result);
    }
};

GpuDipoleEngine::GpuDipoleEngine(GpuContext& ctx, int N_psi, int n_slots)
    : ctx_(ctx), N_psi_(N_psi), n_slots_(n_slots)
{
    if (N_psi <= 0 || n_slots <= 0) {
        throw std::runtime_error("GpuDipoleEngine: N_psi and n_slots must be positive");
    }
    impl_ = std::make_unique<Impl>(
        as_blob(ctx_.raw_queue()),
        static_cast<long long>(N_psi),
        static_cast<long long>(n_slots));
}

GpuDipoleEngine::~GpuDipoleEngine() = default;

void GpuDipoleEngine::reset_stats() { stats_ = Stats{}; }

void GpuDipoleEngine::step(const Eigen::MatrixXd& V,
                            const Eigen::MatrixXd& psi_ir,
                            Eigen::MatrixXd&       result_out)
{
    if (V.rows() != n_slots_ || V.cols() != N_psi_) {
        throw std::runtime_error("GpuDipoleEngine::step: V wrong shape");
    }
    if (psi_ir.rows() != N_psi_ || psi_ir.cols() != N_psi_) {
        throw std::runtime_error("GpuDipoleEngine::step: psi_ir wrong shape");
    }
    if (result_out.rows() != n_slots_ || result_out.cols() != N_psi_) {
        result_out.resize(n_slots_, N_psi_);
    }

    auto& blob = impl_->ctx;
    const long long Np = N_psi_;
    const long long Ns = n_slots_;
    const std::size_t V_sz   = static_cast<std::size_t>(Ns) * Np * sizeof(double);
    const std::size_t Psi_sz = static_cast<std::size_t>(Np) * Np * sizeof(double);
    const std::size_t Res_sz = static_cast<std::size_t>(Ns) * Np * sizeof(double);

    // (1) Upload V and psi_ir.
    auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_V,   V.data(),      V_sz,   cudaMemcpyHostToDevice, blob.stream));
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_psi, psi_ir.data(), Psi_sz, cudaMemcpyHostToDevice, blob.stream));
    stream_sync(blob.stream);
    stats_.t_upload_ns += ns_since(t0);

    // (2) DGEMM:  result = V · psi_ir   (column-major, no transpose).
    //     M = Ns, N = Np, K = Np
    //     lda = Ns, ldb = Np, ldc = Ns.
    const double alpha = 1.0, beta = 0.0;
    auto t1 = std::chrono::steady_clock::now();
    CUBLAS_CHECK(cublasSetStream(blob.cublas, blob.stream));
    CUBLAS_CHECK(cublasDgemm(
        blob.cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(Ns), static_cast<int>(Np), static_cast<int>(Np),
        &alpha,
        impl_->d_V,      static_cast<int>(Ns),
        impl_->d_psi,    static_cast<int>(Np),
        &beta,
        impl_->d_result, static_cast<int>(Ns)));
    stream_sync(blob.stream);
    stats_.t_gemm_ns += ns_since(t1);

    // (3) Download result.
    auto t2 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(result_out.data(), impl_->d_result, Res_sz,
                                cudaMemcpyDeviceToHost, blob.stream));
    stream_sync(blob.stream);
    stats_.t_download_ns += ns_since(t2);
    ++stats_.n_steps;
}

#endif  // SCATT_HAS_CUDA

}  // namespace scatt
