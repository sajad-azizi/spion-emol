// GpuPropagate.cu -- NVIDIA / cuBLAS / cuSOLVER backend for the
// scattering pipeline's GPU steppers.  Mirrors src/scatt/gpu/sycl/GpuPropagate.cpp
// public symbols (struct GpuContext::Impl, GpuForwardStepper::Impl,
// GpuSinvStepper::Impl, GpuBackStepper::Impl) so that the public-class
// signatures in src/scatt/GpuPropagate.hpp -- shared with the SYCL
// backend -- stay identical for FRP / BP / BackPropagator / SchurInverter
// call sites.
//
// VALIDATION STATUS (2026-05-21)
//   * Code-review only.  This file has NOT been compiled or run on a
//     CUDA-equipped node yet -- write-up was done on a Mac dev box
//     without nvcc.  First-build is expected to surface minor API
//     details (cuBLAS handle scoping, exact cusolver workspace
//     sizes for the device's compute capability).  Each section
//     below is structured to fail cleanly with a clear runtime error
//     if anything misbehaves at execution time.
//   * Bit-equivalence gate: when this builds on an H100 / A100 node,
//     `test_gpu_propagate`, `test_gpu_steppers`, and `test_gpu_sinv`
//     must pass at the same ε_mach × N tolerance the SYCL tests
//     already meet.  Cross-vendor comparison (SYCL vs CUDA on a node
//     with both) is the multi-vendor validation.
//
// API MAPPINGS  (SYCL → CUDA)
//   sycl::malloc_device          → cudaMalloc
//   sycl::free                   → cudaFree
//   q.memcpy(...).wait()         → cudaMemcpy (sync)
//   q.memset(...).wait()         → cudaMemset (sync)
//   q.parallel_for(...)          → custom __global__ kernel
//   oneapi::mkl::blas::gemm      → cublasDgemm
//   oneapi::mkl::lapack::getrf   → cusolverDnDgetrf
//   oneapi::mkl::lapack::getri   → NOT IN cuSOLVER; replaced by
//                                  cusolverDnDgetrs(A=LU, B=I) so
//                                  X solves A·X = I  ⇒  X = A⁻¹.
//   oneapi::mkl::lapack::sytrf   → cusolverDnDsytrf (different API)
//   oneapi::mkl::lapack::sytri   → NOT IN cuSOLVER; symmetric-inverse
//                                  path is permanently OFF here
//                                  (symmetric_inverse_supported()
//                                   returns false), matching the
//                                  SYCL build without sytri.
//
// MEMORY MODEL
//   GpuContext::raw_queue() returns void* into a CudaContextBlob
//   (stream + cublasHandle_t + cusolverDnHandle_t).  Each Impl stores
//   a reference to that blob; per-stepper buffers are owned by Impl.

#include "scatt/GpuPropagate.hpp"

#if defined(SCATT_HAS_CUDA) && SCATT_HAS_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace scatt {

// ---------------------------------------------------------------------------
// Error-check helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            std::ostringstream _os; \
            _os << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " in " << #call << " -- " \
                << cudaGetErrorName(_e) << ": " \
                << cudaGetErrorString(_e); \
            throw std::runtime_error(_os.str()); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t _s = (call); \
        if (_s != CUBLAS_STATUS_SUCCESS) { \
            std::ostringstream _os; \
            _os << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                << " in " << #call << " -- status " << static_cast<int>(_s); \
            throw std::runtime_error(_os.str()); \
        } \
    } while (0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t _s = (call); \
        if (_s != CUSOLVER_STATUS_SUCCESS) { \
            std::ostringstream _os; \
            _os << "cuSOLVER error at " << __FILE__ << ":" << __LINE__ \
                << " in " << #call << " -- status " << static_cast<int>(_s); \
            throw std::runtime_error(_os.str()); \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// gpu_available(): static compile-time + runtime device presence check.
// ---------------------------------------------------------------------------
bool GpuContext::gpu_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess) && (count > 0);
}

// ---------------------------------------------------------------------------
// CudaContextBlob: bundles the per-context CUDA state.  Pointed-to by the
// public class's `void* queue_opaque_` member.  Lifetime is tied to
// GpuContext (created in ctor, destroyed in dtor).
// ---------------------------------------------------------------------------
namespace {

struct CudaContextBlob {
    cudaStream_t       stream    = nullptr;
    cublasHandle_t     cublas    = nullptr;
    cusolverDnHandle_t cusolver  = nullptr;
};

inline CudaContextBlob& as_blob(void* opaque) {
    return *static_cast<CudaContextBlob*>(opaque);
}

inline std::uint64_t ns_since(const std::chrono::steady_clock::time_point& t0) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count());
}

// Wait for all work on the stream to finish.  Used to make the SYCL-style
// "q.something().wait()" semantics translate cleanly: every cuBLAS / kernel
// launch below is followed by a stream-sync so observed-from-host timing
// matches the SYCL backend's behaviour.
inline void stream_sync(cudaStream_t s) {
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ---------------------------------------------------------------------------
// Custom kernels -- direct ports of the SYCL parallel_for lambdas in
// src/scatt/gpu/sycl/GpuPropagate.cpp.  Block sizes are a default 16×16;
// the kernels are O(N²) per launch and small relative to the GEMM cost,
// so block geometry isn't a hot-path tuning target.
// ---------------------------------------------------------------------------
__global__ void cuda_k_U_minus_10I(const double* __restrict__ d_W,
                                    double* __restrict__ d_U,
                                    std::int64_t n)
{
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::int64_t j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        const std::int64_t k = j * n + i;
        double v = 12.0 * d_W[k];
        if (i == j) v -= 10.0;
        d_U[k] = v;
    }
}

__global__ void cuda_k_sub(const double* __restrict__ d_U,
                            const double* __restrict__ d_R,
                            double* __restrict__ d_out,
                            std::int64_t nn)
{
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < nn) d_out[k] = d_U[k] - d_R[k];
}

__global__ void cuda_k_symmetrize(double* __restrict__ d_A, std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::int64_t j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && j < n && i <= j) {
        const std::int64_t ij = j * n + i;
        const std::int64_t ji = i * n + j;
        const double v = 0.5 * (d_A[ij] + d_A[ji]);
        d_A[ij] = v;
        d_A[ji] = v;
    }
}

__global__ void cuda_k_extract_psi(const double* __restrict__ d_Y,
                                    double* __restrict__ d_psi,
                                    std::int64_t N_total,
                                    std::int64_t N_psi)
{
    const std::int64_t r = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::int64_t c = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r < N_psi && c < N_psi) {
        d_psi[c * N_psi + r] = d_Y[c * N_total + r];
    }
}

__global__ void cuda_k_extract_f(const double* __restrict__ d_Y,
                                  double* __restrict__ d_f,
                                  std::int64_t N_total,
                                  std::int64_t N_psi,
                                  std::int64_t N_f)
{
    const std::int64_t r = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::int64_t c = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r < N_f && c < N_psi) {
        d_f[c * N_f + r] = d_Y[c * N_total + N_psi + r];
    }
}

// Make d_I = I_n (column-major identity).  Used as the RHS of getrs to
// compute A⁻¹ = A_LU \ I (cuSOLVER has no direct getri).
__global__ void cuda_k_set_identity(double* __restrict__ d_I, std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::int64_t j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        d_I[j * n + i] = (i == j) ? 1.0 : 0.0;
    }
}

inline dim3 grid_2d(std::int64_t Nx, std::int64_t Ny, int bx = 16, int by = 16) {
    return dim3(static_cast<unsigned>((Nx + bx - 1) / bx),
                static_cast<unsigned>((Ny + by - 1) / by),
                1);
}

inline dim3 grid_1d(std::int64_t N, int b = 256) {
    return dim3(static_cast<unsigned>((N + b - 1) / b), 1, 1);
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// GpuContext (CUDA).
// ---------------------------------------------------------------------------
GpuContext::GpuContext(bool prefer_gpu) {
    if (!prefer_gpu) {
        // Host fallback -- nothing to allocate, no CUDA touched.
        info_ = Info{};
        info_.device_name = "host (CUDA build, no GPU requested)";
        queue_opaque_ = nullptr;
        return;
    }

    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        throw std::runtime_error(
            "GpuContext: no CUDA device visible.  Ensure nvcc-compatible "
            "hardware and that CUDA_VISIBLE_DEVICES isn't masking all GPUs.");
    }
    // Pick the device with the largest global memory (mirrors the SYCL
    // backend's "largest HBM wins" tie-break).
    int best = 0;
    std::size_t best_mem = 0;
    for (int d = 0; d < count; ++d) {
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, d));
        if (p.totalGlobalMem > best_mem) {
            best = d;
            best_mem = p.totalGlobalMem;
        }
    }
    CUDA_CHECK(cudaSetDevice(best));

    auto* blob = new CudaContextBlob{};
    try {
        CUDA_CHECK(cudaStreamCreate(&blob->stream));
        CUBLAS_CHECK(cublasCreate(&blob->cublas));
        CUSOLVER_CHECK(cusolverDnCreate(&blob->cusolver));
        CUBLAS_CHECK(cublasSetStream(blob->cublas, blob->stream));
        CUSOLVER_CHECK(cusolverDnSetStream(blob->cusolver, blob->stream));
    } catch (...) {
        if (blob->cusolver) cusolverDnDestroy(blob->cusolver);
        if (blob->cublas)   cublasDestroy(blob->cublas);
        if (blob->stream)   cudaStreamDestroy(blob->stream);
        delete blob;
        throw;
    }
    queue_opaque_ = blob;

    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, best));
    info_.device_name      = p.name;
    info_.platform_name    = "NVIDIA CUDA Runtime";
    info_.is_gpu           = true;
    info_.global_mem_bytes = p.totalGlobalMem;
}

GpuContext::~GpuContext() {
    if (queue_opaque_) {
        auto* blob = static_cast<CudaContextBlob*>(queue_opaque_);
        if (blob->cusolver) cusolverDnDestroy(blob->cusolver);
        if (blob->cublas)   cublasDestroy(blob->cublas);
        if (blob->stream)   cudaStreamDestroy(blob->stream);
        delete blob;
        queue_opaque_ = nullptr;
    }
}

// ===========================================================================
// GpuBackStepper (CUDA): persistent device buffers + two cublasDgemms per
// step + extract kernel.  Mirrors SYCL backend lines 658-803.
// ===========================================================================
struct GpuBackStepper::Impl {
    CudaContextBlob&   ctx;
    const std::int64_t N_total;
    const std::int64_t N_psi;
    const std::int64_t N_f;

    double* d_Z       = nullptr;   // N_total × N_psi (pinned across steps)
    double* d_Ztmp    = nullptr;   // same shape (GEMM dest then swap)
    double* d_Y       = nullptr;   // same shape
    double* d_Rinv    = nullptr;   // N_total × N_total
    double* d_Winv    = nullptr;   // N_total × N_total
    double* d_psi_out = nullptr;   // N_psi × N_psi
    double* d_f_out   = nullptr;   // N_f × N_psi  (nullable when N_f=0)

    Impl(CudaContextBlob& c, std::int64_t Nt, std::int64_t Np, std::int64_t Nf)
        : ctx(c), N_total(Nt), N_psi(Np), N_f(Nf)
    {
        const std::size_t Z_sz   = static_cast<std::size_t>(Nt) * Np;
        const std::size_t Mat_sz = static_cast<std::size_t>(Nt) * Nt;
        const std::size_t psi_sz = static_cast<std::size_t>(Np) * Np;
        const std::size_t f_sz   = static_cast<std::size_t>(Nf) * Np;

        CUDA_CHECK(cudaMalloc(&d_Z,    Z_sz   * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Ztmp, Z_sz   * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Y,    Z_sz   * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Rinv, Mat_sz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Winv, Mat_sz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_psi_out, psi_sz * sizeof(double)));
        if (Nf > 0) {
            CUDA_CHECK(cudaMalloc(&d_f_out, f_sz * sizeof(double)));
        }
    }

    ~Impl() {
        if (d_Z)       cudaFree(d_Z);
        if (d_Ztmp)    cudaFree(d_Ztmp);
        if (d_Y)       cudaFree(d_Y);
        if (d_Rinv)    cudaFree(d_Rinv);
        if (d_Winv)    cudaFree(d_Winv);
        if (d_psi_out) cudaFree(d_psi_out);
        if (d_f_out)   cudaFree(d_f_out);
    }
};

GpuBackStepper::GpuBackStepper(GpuContext& ctx, int N_total, int N_psi, int N_f)
    : ctx_(ctx), N_total_(N_total), N_psi_(N_psi), N_f_(N_f)
{
    impl_ = std::make_unique<Impl>(
        as_blob(ctx_.raw_queue()),
        static_cast<std::int64_t>(N_total),
        static_cast<std::int64_t>(N_psi),
        static_cast<std::int64_t>(N_f));
}

GpuBackStepper::~GpuBackStepper() = default;

void GpuBackStepper::init_Z(const Eigen::MatrixXd& Z) {
    if (Z.rows() != N_total_ || Z.cols() != N_psi_) {
        throw std::runtime_error("GpuBackStepper::init_Z: wrong shape");
    }
    auto& c = as_blob(ctx_.raw_queue());
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_Z, Z.data(),
                                static_cast<std::size_t>(N_total_) * N_psi_ * sizeof(double),
                                cudaMemcpyHostToDevice, c.stream));
    stream_sync(c.stream);
}

void GpuBackStepper::step(const Eigen::MatrixXd& Rinv,
                          const Eigen::MatrixXd& W_inv,
                          Eigen::MatrixXd&       psi_out,
                          Eigen::MatrixXd*       f_out,
                          bool                   compute_f)
{
    if (Rinv.rows()  != N_total_ || Rinv.cols()  != N_total_) {
        throw std::runtime_error("GpuBackStepper::step: Rinv wrong shape");
    }
    if (W_inv.rows() != N_total_ || W_inv.cols() != N_total_) {
        throw std::runtime_error("GpuBackStepper::step: W_inv wrong shape");
    }
    auto& c = as_blob(ctx_.raw_queue());
    const std::int64_t Nt = N_total_;
    const std::int64_t Np = N_psi_;
    const std::int64_t Nf = N_f_;
    const std::size_t  Mat_sz = static_cast<std::size_t>(Nt) * Nt;
    const double alpha = 1.0, beta = 0.0;

    // (1) Upload Rinv + W_inv to GPU.
    auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_Rinv, Rinv.data(),  Mat_sz * sizeof(double),
                                cudaMemcpyHostToDevice, c.stream));
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_Winv, W_inv.data(), Mat_sz * sizeof(double),
                                cudaMemcpyHostToDevice, c.stream));
    stream_sync(c.stream);
    stats_.t_upload_ns += ns_since(t0);

    // (2) Z_tmp = Rinv · Z ; swap so d_Z is the new one.
    auto t1 = std::chrono::steady_clock::now();
    CUBLAS_CHECK(cublasDgemm(c.cublas,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              Nt, Np, Nt,
                              &alpha,
                              impl_->d_Rinv, Nt,
                              impl_->d_Z,    Nt,
                              &beta,
                              impl_->d_Ztmp, Nt));
    stream_sync(c.stream);
    std::swap(impl_->d_Z, impl_->d_Ztmp);
    stats_.t_gemm_z_ns += ns_since(t1);

    // (3) Y = W_inv · Z.
    auto t2 = std::chrono::steady_clock::now();
    CUBLAS_CHECK(cublasDgemm(c.cublas,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              Nt, Np, Nt,
                              &alpha,
                              impl_->d_Winv, Nt,
                              impl_->d_Z,    Nt,
                              &beta,
                              impl_->d_Y,    Nt));
    stream_sync(c.stream);
    stats_.t_gemm_y_ns += ns_since(t2);

    // (4) Extract psi (top N_psi rows of Y) [and f (bottom N_f rows)] and
    //     download to host.
    auto t3 = std::chrono::steady_clock::now();
    {
        dim3 block(16, 16, 1);
        dim3 grid = grid_2d(Np, Np);
        cuda_k_extract_psi<<<grid, block, 0, c.stream>>>(
            impl_->d_Y, impl_->d_psi_out, Nt, Np);
        CUDA_CHECK(cudaGetLastError());
    }
    if (psi_out.rows() != Np || psi_out.cols() != Np) psi_out.resize(Np, Np);
    CUDA_CHECK(cudaMemcpyAsync(psi_out.data(), impl_->d_psi_out,
                                static_cast<std::size_t>(Np) * Np * sizeof(double),
                                cudaMemcpyDeviceToHost, c.stream));

    if (compute_f && f_out != nullptr && Nf > 0) {
        dim3 block(16, 16, 1);
        dim3 grid = grid_2d(Np, Nf);
        cuda_k_extract_f<<<grid, block, 0, c.stream>>>(
            impl_->d_Y, impl_->d_f_out, Nt, Np, Nf);
        CUDA_CHECK(cudaGetLastError());
        if (f_out->rows() != Nf || f_out->cols() != Np) f_out->resize(Nf, Np);
        CUDA_CHECK(cudaMemcpyAsync(f_out->data(), impl_->d_f_out,
                                    static_cast<std::size_t>(Nf) * Np * sizeof(double),
                                    cudaMemcpyDeviceToHost, c.stream));
    }
    stream_sync(c.stream);
    stats_.t_extract_ns += ns_since(t3);
    ++stats_.n_steps;
}

void GpuBackStepper::get_Z(Eigen::MatrixXd& Z_out) const {
    if (Z_out.rows() != N_total_ || Z_out.cols() != N_psi_) {
        Z_out.resize(N_total_, N_psi_);
    }
    auto& c = as_blob(ctx_.raw_queue());
    CUDA_CHECK(cudaMemcpyAsync(Z_out.data(), impl_->d_Z,
                                static_cast<std::size_t>(N_total_) * N_psi_ * sizeof(double),
                                cudaMemcpyDeviceToHost, c.stream));
    stream_sync(c.stream);
}

// ===========================================================================
// GpuForwardStepper (CUDA): U-build + invert(R_current).  The symmetric-LDLᵀ
// path (sytrf/sytri) is NOT supported -- cuSOLVER lacks dsytri, mirroring
// the SYCL behaviour when SCATT_HAS_ONEMKL_SYTRI is undefined.  The legacy
// LU path is always taken: cusolverDnDgetrf factorises, cusolverDnDgetrs
// solves A·X = I to produce A⁻¹ (cuSOLVER has no direct getri equivalent).
// ===========================================================================
struct GpuForwardStepper::Impl {
    CudaContextBlob&   ctx;
    const std::int64_t n;

    // Per-step buffers.
    double* d_Winv  = nullptr;   // (n,n) input W⁻¹
    double* d_U     = nullptr;   // (n,n) scratch for 12·W⁻¹ − 10·I
    double* d_R     = nullptr;   // (n,n) R_current = U − R_prev
    double* d_Rprev = nullptr;   // (n,n) pinned across steps
    double* d_LU    = nullptr;   // (n,n) in-place LU target
    double* d_inv   = nullptr;   // (n,n) inverse result (no in-place getri on cuSOLVER)
    int*    d_ipiv  = nullptr;   // (n,)  pivot vector
    int*    d_info  = nullptr;   // (1,)  cuSOLVER info code

    // cusolverDnDgetrf workspace (size queried at construction).
    double* d_workspace      = nullptr;
    int     workspace_lwork  = 0;

    // user flag (set by enable_symmetric_inverse()); kept for API parity
    // but step() ignores it -- the CUDA backend doesn't implement the
    // symmetric path.
    bool    use_symmetric_requested = false;

    Impl(CudaContextBlob& c, std::int64_t n_) : ctx(c), n(n_) {
        const std::size_t nn = static_cast<std::size_t>(n_) * n_;
        CUDA_CHECK(cudaMalloc(&d_Winv,  nn * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_U,     nn * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_R,     nn * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Rprev, nn * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_LU,    nn * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_inv,   nn * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ipiv,  n_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_info,  sizeof(int)));

        // Query getrf workspace size and allocate.
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(
            c.cusolver, static_cast<int>(n_), static_cast<int>(n_),
            d_LU, static_cast<int>(n_), &lwork));
        workspace_lwork = lwork;
        if (lwork > 0) {
            CUDA_CHECK(cudaMalloc(&d_workspace, lwork * sizeof(double)));
        }

        // Zero-init d_Rprev so the first step (if caller forgets init)
        // is well-defined.
        CUDA_CHECK(cudaMemsetAsync(d_Rprev, 0, nn * sizeof(double), c.stream));
        stream_sync(c.stream);
    }

    ~Impl() {
        if (d_Winv)      cudaFree(d_Winv);
        if (d_U)         cudaFree(d_U);
        if (d_R)         cudaFree(d_R);
        if (d_Rprev)     cudaFree(d_Rprev);
        if (d_LU)        cudaFree(d_LU);
        if (d_inv)       cudaFree(d_inv);
        if (d_ipiv)      cudaFree(d_ipiv);
        if (d_info)      cudaFree(d_info);
        if (d_workspace) cudaFree(d_workspace);
    }
};

GpuForwardStepper::GpuForwardStepper(GpuContext& ctx, int N_total)
    : ctx_(ctx), N_(N_total)
{
    impl_ = std::make_unique<Impl>(as_blob(ctx_.raw_queue()),
                                    static_cast<std::int64_t>(N_total));
}

GpuForwardStepper::~GpuForwardStepper() = default;

void GpuForwardStepper::enable_symmetric_inverse(bool enable) {
    impl_->use_symmetric_requested = enable;
}

// cuSOLVER has no dsytri.  The symmetric path is permanently disabled on
// the CUDA backend, matching the SYCL build's behaviour when
// SCATT_HAS_ONEMKL_SYTRI is undefined.
bool GpuForwardStepper::is_symmetric_inverse_active() const { return false; }
bool GpuForwardStepper::symmetric_inverse_supported() const { return false; }

void GpuForwardStepper::init_R_prev_inv(const Eigen::MatrixXd& R) {
    if (R.rows() != N_ || R.cols() != N_) {
        throw std::runtime_error(
            "GpuForwardStepper::init_R_prev_inv: wrong shape (got " +
            std::to_string(R.rows()) + "×" + std::to_string(R.cols()) +
            ", expected " + std::to_string(N_) + "²)");
    }
    auto& c = as_blob(ctx_.raw_queue());
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_Rprev, R.data(),
                                static_cast<std::size_t>(N_) * N_ * sizeof(double),
                                cudaMemcpyHostToDevice, c.stream));
    stream_sync(c.stream);
}

double GpuForwardStepper::step(const Eigen::MatrixXd& W_inv,
                               Eigen::MatrixXd&       Rinv_out)
{
    if (W_inv.rows() != N_ || W_inv.cols() != N_) {
        throw std::runtime_error("GpuForwardStepper::step: W_inv wrong shape");
    }
    auto& c = as_blob(ctx_.raw_queue());
    const std::int64_t n  = N_;
    const std::size_t  nn = static_cast<std::size_t>(n) * n;

    // (1) Upload W_inv to GPU.
    auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_Winv, W_inv.data(), nn * sizeof(double),
                                cudaMemcpyHostToDevice, c.stream));
    stream_sync(c.stream);
    stats_.t_upload_ns += ns_since(t0);

    // (2) U = 12·W⁻¹ − 10·I ; R_current = U − R_prev.  Copy R into d_LU
    //     (cuSOLVER getrf overwrites its input).
    auto t1 = std::chrono::steady_clock::now();
    {
        dim3 block(16, 16, 1);
        dim3 grid = grid_2d(n, n);
        cuda_k_U_minus_10I<<<grid, block, 0, c.stream>>>(
            impl_->d_Winv, impl_->d_U, n);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 block(256, 1, 1);
        dim3 grid = grid_1d(static_cast<std::int64_t>(nn));
        cuda_k_sub<<<grid, block, 0, c.stream>>>(
            impl_->d_U, impl_->d_Rprev, impl_->d_R, static_cast<std::int64_t>(nn));
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_LU, impl_->d_R, nn * sizeof(double),
                                cudaMemcpyDeviceToDevice, c.stream));
    stream_sync(c.stream);
    stats_.t_u_combine_ns += ns_since(t1);

    // (3) Invert R via LU factorise + solve-vs-identity.
    auto t2 = std::chrono::steady_clock::now();
    CUSOLVER_CHECK(cusolverDnDgetrf(
        c.cusolver, static_cast<int>(n), static_cast<int>(n),
        impl_->d_LU, static_cast<int>(n),
        impl_->d_workspace,
        impl_->d_ipiv,
        impl_->d_info));
    // Build identity on device, then solve LU · X = I  ⇒  X = A⁻¹.
    {
        dim3 block(16, 16, 1);
        dim3 grid = grid_2d(n, n);
        cuda_k_set_identity<<<grid, block, 0, c.stream>>>(impl_->d_inv, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUSOLVER_CHECK(cusolverDnDgetrs(
        c.cusolver, CUBLAS_OP_N,
        static_cast<int>(n), static_cast<int>(n),     // n RHS columns
        impl_->d_LU, static_cast<int>(n),
        impl_->d_ipiv,
        impl_->d_inv, static_cast<int>(n),
        impl_->d_info));
    stream_sync(c.stream);
    // Quick info-code check (non-zero ⇒ singular matrix or invalid arg).
    {
        int info_host = 0;
        CUDA_CHECK(cudaMemcpy(&info_host, impl_->d_info, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (info_host != 0) {
            throw std::runtime_error(
                "GpuForwardStepper::step: cuSOLVER getrf/getrs returned info=" +
                std::to_string(info_host) + " (singular or invalid input)");
        }
    }
    stats_.t_inverse_ns += ns_since(t2);

    // (4) Copy d_inv -> d_Rprev (for next step) and download to host.
    auto t3 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_Rprev, impl_->d_inv, nn * sizeof(double),
                                cudaMemcpyDeviceToDevice, c.stream));
    if (Rinv_out.rows() != N_ || Rinv_out.cols() != N_) Rinv_out.resize(N_, N_);
    CUDA_CHECK(cudaMemcpyAsync(Rinv_out.data(), impl_->d_inv, nn * sizeof(double),
                                cudaMemcpyDeviceToHost, c.stream));
    stream_sync(c.stream);
    stats_.t_download_ns += ns_since(t3);
    ++stats_.n_steps;

    // Symmetry diagnostic (cheap host-side; matches the SYCL return).
    double sym_err = 0.0;
    for (int j = 0; j < N_; ++j) {
        for (int i = 0; i < j; ++i) {
            const double v = std::abs(Rinv_out(i, j) - Rinv_out(j, i));
            if (v > sym_err) sym_err = v;
        }
    }
    return sym_err;
}

// ===========================================================================
// GpuSinvStepper (CUDA): Schur complement S = A − B·D⁻¹·Bᵀ, then S⁻¹.
// Mirrors SYCL backend lines around 453-650 (the version that does
// schur+invert+symmetrise per ir).
// ===========================================================================
namespace {

// Custom kernel: per-column scale -- d_out[:, j] = d_in[:, j] * Dinv[j].
// Used in the Schur complement: B_scaled[:, j] = B[:, j] * Dinv[j], then
// S = A - B_scaled · Bᵀ.
__global__ void cuda_k_col_scale(const double* __restrict__ d_in,
                                  const double* __restrict__ d_Dinv,
                                  double* __restrict__ d_out,
                                  std::int64_t n_rows,
                                  std::int64_t n_cols)
{
    const std::int64_t r = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::int64_t c = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r < n_rows && c < n_cols) {
        d_out[c * n_rows + r] = d_in[c * n_rows + r] * d_Dinv[c];
    }
}

}  // anonymous namespace

struct GpuSinvStepper::Impl {
    CudaContextBlob&   ctx;
    const std::int64_t N_psi;
    const std::int64_t N_f;

    // Per-step buffers.
    double* d_A        = nullptr;  // (N_psi × N_psi) input
    double* d_B        = nullptr;  // (N_psi × N_f)   input
    double* d_Bscaled  = nullptr;  // (N_psi × N_f)   B · diag(Dinv)
    double* d_Dinv     = nullptr;  // (N_f)           Dinv vector
    double* d_S        = nullptr;  // (N_psi × N_psi) Schur complement (overwrites d_A pattern)
    double* d_LU       = nullptr;  // (N_psi × N_psi) LU factors
    double* d_inv      = nullptr;  // (N_psi × N_psi) inverse result
    int*    d_ipiv     = nullptr;
    int*    d_info     = nullptr;

    double* d_workspace     = nullptr;
    int     workspace_lwork = 0;

    Impl(CudaContextBlob& c, std::int64_t Np, std::int64_t Nf)
        : ctx(c), N_psi(Np), N_f(Nf)
    {
        const std::size_t S_sz  = static_cast<std::size_t>(Np) * Np;
        const std::size_t B_sz  = static_cast<std::size_t>(Np) * Nf;
        CUDA_CHECK(cudaMalloc(&d_A,       S_sz * sizeof(double)));
        if (Nf > 0) {
            CUDA_CHECK(cudaMalloc(&d_B,       B_sz * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_Bscaled, B_sz * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_Dinv,    Nf   * sizeof(double)));
        }
        CUDA_CHECK(cudaMalloc(&d_S,    S_sz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_LU,   S_sz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_inv,  S_sz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ipiv, Np   * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(
            c.cusolver, static_cast<int>(Np), static_cast<int>(Np),
            d_LU, static_cast<int>(Np), &lwork));
        workspace_lwork = lwork;
        if (lwork > 0) {
            CUDA_CHECK(cudaMalloc(&d_workspace, lwork * sizeof(double)));
        }
    }

    ~Impl() {
        if (d_A)         cudaFree(d_A);
        if (d_B)         cudaFree(d_B);
        if (d_Bscaled)   cudaFree(d_Bscaled);
        if (d_Dinv)      cudaFree(d_Dinv);
        if (d_S)         cudaFree(d_S);
        if (d_LU)        cudaFree(d_LU);
        if (d_inv)       cudaFree(d_inv);
        if (d_ipiv)      cudaFree(d_ipiv);
        if (d_info)      cudaFree(d_info);
        if (d_workspace) cudaFree(d_workspace);
    }
};

GpuSinvStepper::GpuSinvStepper(GpuContext& ctx, int N_psi, int N_f)
    : ctx_(ctx), N_psi_(N_psi), N_f_(N_f)
{
    impl_ = std::make_unique<Impl>(as_blob(ctx_.raw_queue()),
                                    static_cast<std::int64_t>(N_psi),
                                    static_cast<std::int64_t>(N_f));
}

GpuSinvStepper::~GpuSinvStepper() = default;

double GpuSinvStepper::step(const Eigen::MatrixXd& A,
                            const Eigen::MatrixXd& B,
                            const Eigen::VectorXd& Dinv,
                            Eigen::MatrixXd&       Sinv_out)
{
    if (A.rows() != N_psi_ || A.cols() != N_psi_) {
        throw std::runtime_error("GpuSinvStepper::step: A wrong shape");
    }
    if (B.rows() != N_psi_ || B.cols() != N_f_) {
        throw std::runtime_error("GpuSinvStepper::step: B wrong shape");
    }
    if (Dinv.size() != N_f_) {
        throw std::runtime_error("GpuSinvStepper::step: Dinv wrong shape");
    }
    auto& c  = as_blob(ctx_.raw_queue());
    const std::int64_t Np = N_psi_;
    const std::int64_t Nf = N_f_;
    const std::size_t  S_sz = static_cast<std::size_t>(Np) * Np;
    const std::size_t  B_sz = static_cast<std::size_t>(Np) * Nf;

    // (1) Upload A, B, Dinv.
    auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_A, A.data(), S_sz * sizeof(double),
                                cudaMemcpyHostToDevice, c.stream));
    if (Nf > 0) {
        CUDA_CHECK(cudaMemcpyAsync(impl_->d_B, B.data(), B_sz * sizeof(double),
                                    cudaMemcpyHostToDevice, c.stream));
        CUDA_CHECK(cudaMemcpyAsync(impl_->d_Dinv, Dinv.data(),
                                    Nf * sizeof(double),
                                    cudaMemcpyHostToDevice, c.stream));
    }
    stream_sync(c.stream);
    stats_.t_upload_ns += ns_since(t0);

    // (2) S = A − B · diag(Dinv) · Bᵀ.
    //     Implemented as: Bscaled = B · diag(Dinv);   S = A − Bscaled · Bᵀ.
    auto t1 = std::chrono::steady_clock::now();
    // Start S := A.
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_S, impl_->d_A, S_sz * sizeof(double),
                                cudaMemcpyDeviceToDevice, c.stream));
    if (Nf > 0) {
        // Bscaled[:, j] = B[:, j] * Dinv[j].
        dim3 block(16, 16, 1);
        dim3 grid = grid_2d(Nf, Np);
        cuda_k_col_scale<<<grid, block, 0, c.stream>>>(
            impl_->d_B, impl_->d_Dinv, impl_->d_Bscaled, Np, Nf);
        CUDA_CHECK(cudaGetLastError());
        // S = -1.0 · Bscaled · Bᵀ + 1.0 · S    (overwrites S in-place).
        const double m_one = -1.0;
        const double p_one =  1.0;
        CUBLAS_CHECK(cublasDgemm(c.cublas,
                                  CUBLAS_OP_N, CUBLAS_OP_T,
                                  Np, Np, Nf,
                                  &m_one,
                                  impl_->d_Bscaled, Np,
                                  impl_->d_B,       Np,
                                  &p_one,
                                  impl_->d_S,       Np));
    }
    stream_sync(c.stream);
    stats_.t_schur_ns += ns_since(t1);

    // (3) Invert S via LU + solve-vs-identity, then symmetrise.
    auto t2 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_LU, impl_->d_S, S_sz * sizeof(double),
                                cudaMemcpyDeviceToDevice, c.stream));
    CUSOLVER_CHECK(cusolverDnDgetrf(
        c.cusolver, static_cast<int>(Np), static_cast<int>(Np),
        impl_->d_LU, static_cast<int>(Np),
        impl_->d_workspace, impl_->d_ipiv, impl_->d_info));
    {
        dim3 block(16, 16, 1);
        dim3 grid = grid_2d(Np, Np);
        cuda_k_set_identity<<<grid, block, 0, c.stream>>>(impl_->d_inv, Np);
        CUDA_CHECK(cudaGetLastError());
    }
    CUSOLVER_CHECK(cusolverDnDgetrs(
        c.cusolver, CUBLAS_OP_N,
        static_cast<int>(Np), static_cast<int>(Np),
        impl_->d_LU, static_cast<int>(Np),
        impl_->d_ipiv,
        impl_->d_inv, static_cast<int>(Np),
        impl_->d_info));
    // Symmetrise: A = 0.5 (A + Aᵀ).
    {
        dim3 block(16, 16, 1);
        dim3 grid = grid_2d(Np, Np);
        cuda_k_symmetrize<<<grid, block, 0, c.stream>>>(impl_->d_inv, Np);
        CUDA_CHECK(cudaGetLastError());
    }
    stream_sync(c.stream);
    {
        int info_host = 0;
        CUDA_CHECK(cudaMemcpy(&info_host, impl_->d_info, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (info_host != 0) {
            throw std::runtime_error(
                "GpuSinvStepper::step: cuSOLVER getrf/getrs returned info=" +
                std::to_string(info_host));
        }
    }
    stats_.t_inverse_ns += ns_since(t2);

    // (4) Download.
    auto t3 = std::chrono::steady_clock::now();
    if (Sinv_out.rows() != Np || Sinv_out.cols() != Np) Sinv_out.resize(Np, Np);
    CUDA_CHECK(cudaMemcpyAsync(Sinv_out.data(), impl_->d_inv,
                                S_sz * sizeof(double),
                                cudaMemcpyDeviceToHost, c.stream));
    stream_sync(c.stream);
    stats_.t_download_ns += ns_since(t3);
    ++stats_.n_steps;

    // Symmetry-residual diagnostic.  After our explicit symmetrise the
    // returned matrix is bit-symmetric so this is essentially 0.
    double sym_err = 0.0;
    for (int j = 0; j < N_psi_; ++j) {
        for (int i = 0; i < j; ++i) {
            const double v = std::abs(Sinv_out(i, j) - Sinv_out(j, i));
            if (v > sym_err) sym_err = v;
        }
    }
    return sym_err;
}

double GpuSinvStepper::step_inverse_only(const Eigen::MatrixXd& A,
                                         Eigen::MatrixXd&       Sinv_out)
{
    // No B contribution -- just invert A directly.  Mirror step() but skip
    // the Schur GEMM.
    if (A.rows() != N_psi_ || A.cols() != N_psi_) {
        throw std::runtime_error("GpuSinvStepper::step_inverse_only: A wrong shape");
    }
    auto& c = as_blob(ctx_.raw_queue());
    const std::int64_t Np = N_psi_;
    const std::size_t  S_sz = static_cast<std::size_t>(Np) * Np;

    auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(impl_->d_LU, A.data(), S_sz * sizeof(double),
                                cudaMemcpyHostToDevice, c.stream));
    stream_sync(c.stream);
    stats_.t_upload_ns += ns_since(t0);

    auto t2 = std::chrono::steady_clock::now();
    CUSOLVER_CHECK(cusolverDnDgetrf(
        c.cusolver, static_cast<int>(Np), static_cast<int>(Np),
        impl_->d_LU, static_cast<int>(Np),
        impl_->d_workspace, impl_->d_ipiv, impl_->d_info));
    {
        dim3 block(16, 16, 1);
        dim3 grid = grid_2d(Np, Np);
        cuda_k_set_identity<<<grid, block, 0, c.stream>>>(impl_->d_inv, Np);
        CUDA_CHECK(cudaGetLastError());
    }
    CUSOLVER_CHECK(cusolverDnDgetrs(
        c.cusolver, CUBLAS_OP_N,
        static_cast<int>(Np), static_cast<int>(Np),
        impl_->d_LU, static_cast<int>(Np),
        impl_->d_ipiv,
        impl_->d_inv, static_cast<int>(Np),
        impl_->d_info));
    {
        dim3 block(16, 16, 1);
        dim3 grid = grid_2d(Np, Np);
        cuda_k_symmetrize<<<grid, block, 0, c.stream>>>(impl_->d_inv, Np);
        CUDA_CHECK(cudaGetLastError());
    }
    stream_sync(c.stream);
    stats_.t_inverse_ns += ns_since(t2);

    auto t3 = std::chrono::steady_clock::now();
    if (Sinv_out.rows() != Np || Sinv_out.cols() != Np) Sinv_out.resize(Np, Np);
    CUDA_CHECK(cudaMemcpyAsync(Sinv_out.data(), impl_->d_inv,
                                S_sz * sizeof(double),
                                cudaMemcpyDeviceToHost, c.stream));
    stream_sync(c.stream);
    stats_.t_download_ns += ns_since(t3);
    ++stats_.n_steps;

    double sym_err = 0.0;
    for (int j = 0; j < N_psi_; ++j) {
        for (int i = 0; i < j; ++i) {
            const double v = std::abs(Sinv_out(i, j) - Sinv_out(j, i));
            if (v > sym_err) sym_err = v;
        }
    }
    return sym_err;
}

}  // namespace scatt

#endif  // SCATT_HAS_CUDA
