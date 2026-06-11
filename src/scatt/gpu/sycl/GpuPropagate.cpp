// GpuPropagate.cpp -- SYCL/oneMKL implementation of the forward + back
// per-step kernels.  Structured in two halves:
//
//   1. When SCATT_HAS_SYCL is defined:
//        - GpuContext owns a sycl::queue pinning the first GPU device.
//        - GpuForwardStepper::Impl holds the USM-device buffers for the
//          six working N×N matrices + LAPACK scratchpad.
//        - GpuBackStepper::Impl holds (Z, Z_temp, Y, Rinv, W_inv, psi, f)
//          buffers.  All GEMMs go through oneapi::mkl::blas.
//
//   2. When SCATT_HAS_SYCL is NOT defined:
//        - gpu_available() returns false.
//        - Construction with prefer_gpu=true throws std::runtime_error.
//        - All step() calls throw ("GPU path compiled out").
//
// This layout means CPU-only platforms (macOS dev, GCC-only LRZ partitions)
// can still build the full scattering code; the GPU path is opt-in at
// configure time and opt-in at runtime via Config::use_gpu.

#include "scatt/GpuPropagate.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef SCATT_HAS_SYCL
  // Intel DPC++ provides either <sycl/sycl.hpp> (new) or <CL/sycl.hpp> (old).
  #if __has_include(<sycl/sycl.hpp>)
    #include <sycl/sycl.hpp>
  #else
    #include <CL/sycl.hpp>
  #endif
  #include <oneapi/mkl.hpp>
  #include <oneapi/mkl/blas.hpp>
  #include <oneapi/mkl/lapack.hpp>
#endif

namespace scatt {

// ---------------------------------------------------------------------------
// gpu_available(): static compile-time + runtime device presence check.
// ---------------------------------------------------------------------------
bool GpuContext::gpu_available() {
#ifdef SCATT_HAS_SYCL
    try {
        auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
        return !devs.empty();
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

// ===========================================================================
// SYCL BRANCH
// ===========================================================================
#ifdef SCATT_HAS_SYCL

namespace {

inline sycl::queue& as_queue(void* opaque) {
    return *static_cast<sycl::queue*>(opaque);
}

// Pick the first GPU device (prefer Intel Level-Zero to OpenCL for PVC).
sycl::device pick_gpu_device_() {
    try {
        auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (gpus.empty()) {
            throw std::runtime_error(
                "GpuContext: no SYCL GPU device visible. "
                "On LRZ SuperMUC-NG Phase 2 load the oneAPI module and run "
                "on a GPU node.");
        }
        // Prefer devices reporting the most global memory (skips tiny iGPUs
        // if there are multiple).
        sycl::device best = gpus.front();
        std::size_t best_mem = best.get_info<sycl::info::device::global_mem_size>();
        for (const auto& d : gpus) {
            std::size_t m = d.get_info<sycl::info::device::global_mem_size>();
            if (m > best_mem) { best = d; best_mem = m; }
        }
        return best;
    } catch (const sycl::exception& e) {
        throw std::runtime_error(std::string("GpuContext SYCL error: ") + e.what());
    }
}

// Kernel: U = 12*W_inv − 10*I   (column-major, in-place writing d_U).
inline void kernel_U_minus_10I_(sycl::queue& q,
                                const double* d_Winv, double* d_U,
                                std::int64_t n)
{
    q.parallel_for(sycl::range<2>(n, n), [=](sycl::id<2> idx) {
        const std::int64_t i = idx[0], j = idx[1];
        const std::int64_t k = j * n + i;    // column-major linear index
        double v = 12.0 * d_Winv[k];
        if (i == j) v -= 10.0;
        d_U[k] = v;
    }).wait();
}

// Kernel: R = U − R_prev   (elementwise over n^2 entries).
inline void kernel_sub_(sycl::queue& q,
                        const double* d_U, const double* d_Rprev,
                        double* d_R, std::int64_t nn)
{
    q.parallel_for(sycl::range<1>(nn), [=](sycl::id<1> idx) {
        d_R[idx] = d_U[idx] - d_Rprev[idx];
    }).wait();
}

// Kernel: A = 0.5*(A + A^T)   (column-major, N×N, upper-triangle writes lower).
inline void kernel_symmetrize_(sycl::queue& q, double* d_A, std::int64_t n) {
    q.parallel_for(sycl::range<2>(n, n), [=](sycl::id<2> idx) {
        const std::int64_t i = idx[0], j = idx[1];
        if (i <= j) {
            const std::int64_t ij = j * n + i;
            const std::int64_t ji = i * n + j;
            const double v = 0.5 * (d_A[ij] + d_A[ji]);
            d_A[ij] = v;
            d_A[ji] = v;
        }
    }).wait();
}

// Kernel: mirror LOWER -> UPPER  (column-major, N×N).
//
// After dsytri with uplo=lower, A^{-1} is in the LOWER triangle (i.e.
// A(i,j) for i >= j) and the UPPER triangle is untouched.  This kernel
// copies LOWER to UPPER so the matrix is bit-symmetric for downstream
// consumers that read both triangles.
//
// In column-major layout, A(i,j) is at d_A[j*n + i].  We want, for i < j,
//   d_A[j*n + i]  :=  d_A[i*n + j]    // upper element receives lower
// I.e., A(i,j) := A(j,i).
inline void kernel_mirror_lower_to_upper_(sycl::queue& q, double* d_A,
                                          std::int64_t n)
{
    q.parallel_for(sycl::range<2>(n, n), [=](sycl::id<2> idx) {
        const std::int64_t i = idx[0], j = idx[1];
        if (i < j) {
            d_A[j * n + i] = d_A[i * n + j];
        }
    }).wait();
}

// Kernel: extract top N_psi rows of Y (N_total × N_psi) into psi (N_psi × N_psi).
inline void kernel_extract_psi_(sycl::queue& q,
                                const double* d_Y, double* d_psi,
                                std::int64_t N_total, std::int64_t N_psi)
{
    q.parallel_for(sycl::range<2>(N_psi, N_psi), [=](sycl::id<2> idx) {
        const std::int64_t r = idx[0], c = idx[1];
        d_psi[c * N_psi + r] = d_Y[c * N_total + r];
    }).wait();
}

// Kernel: extract bottom N_f rows of Y into f (N_f × N_psi).
inline void kernel_extract_f_(sycl::queue& q,
                              const double* d_Y, double* d_f,
                              std::int64_t N_total, std::int64_t N_psi,
                              std::int64_t N_f)
{
    q.parallel_for(sycl::range<2>(N_f, N_psi), [=](sycl::id<2> idx) {
        const std::int64_t r = idx[0], c = idx[1];
        d_f[c * N_f + r] = d_Y[c * N_total + N_psi + r];
    }).wait();
}

inline std::uint64_t ns_since(const std::chrono::steady_clock::time_point& t0) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count());
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// GpuContext (SYCL).
// ---------------------------------------------------------------------------
GpuContext::GpuContext(bool prefer_gpu) {
    sycl::device dev = prefer_gpu ? pick_gpu_device_() : sycl::device(sycl::cpu_selector_v);
    auto* q = new sycl::queue(
        dev,
        sycl::property_list{sycl::property::queue::in_order{}});
    queue_opaque_ = static_cast<void*>(q);

    info_.device_name      = dev.get_info<sycl::info::device::name>();
    info_.platform_name    = dev.get_platform().get_info<sycl::info::platform::name>();
    info_.is_gpu           = dev.is_gpu();
    info_.global_mem_bytes = dev.get_info<sycl::info::device::global_mem_size>();
}

GpuContext::~GpuContext() {
    if (queue_opaque_) {
        delete static_cast<sycl::queue*>(queue_opaque_);
        queue_opaque_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// GpuForwardStepper::Impl.
// ---------------------------------------------------------------------------
struct GpuForwardStepper::Impl {
    sycl::queue&          q;
    const std::int64_t    n;

    double*               d_Winv   = nullptr;
    double*               d_U      = nullptr;
    double*               d_R      = nullptr;   // R_current
    double*               d_Rprev  = nullptr;   // pinned Rinv_{n-1}
    double*               d_temp   = nullptr;   // in-place LU/LDLᵀ target
    std::int64_t*         d_ipiv   = nullptr;

    // getrf+getri scratchpads (LEGACY general-LU path; always allocated).
    double*               d_scratch_rf = nullptr;
    double*               d_scratch_ri = nullptr;
    std::int64_t          scratch_rf_size = 0;
    std::int64_t          scratch_ri_size = 0;

    // sytrf+sytri scratchpads (NEW symmetric-LDLᵀ path; allocated only if
    // oneMKL supports them on this device.  If the scratchpad query throws
    // -- typically when oneMKL on the installed driver hasn't compiled the
    // GPU sytrf kernel -- we leave d_scratch_sf/si null and set
    // sytrf_supported = false, and step() falls back to the legacy path).
    double*               d_scratch_sf = nullptr;
    double*               d_scratch_si = nullptr;
    std::int64_t          scratch_sf_size = 0;
    std::int64_t          scratch_si_size = 0;
    bool                  sytrf_supported = false;

    // User flag (set by enable_symmetric_inverse()).  step() uses the
    // symmetric path iff `use_symmetric_requested && sytrf_supported`.
    bool                  use_symmetric_requested = false;

    Impl(sycl::queue& q_, std::int64_t n_) : q(q_), n(n_) {
        const std::size_t nn = static_cast<std::size_t>(n_) * n_;
        d_Winv   = sycl::malloc_device<double>(nn, q);
        d_U      = sycl::malloc_device<double>(nn, q);
        d_R      = sycl::malloc_device<double>(nn, q);
        d_Rprev  = sycl::malloc_device<double>(nn, q);
        d_temp   = sycl::malloc_device<double>(nn, q);
        d_ipiv   = sycl::malloc_device<std::int64_t>(n, q);

        scratch_rf_size = oneapi::mkl::lapack::getrf_scratchpad_size<double>(q, n, n, n);
        scratch_ri_size = oneapi::mkl::lapack::getri_scratchpad_size<double>(q, n, n);
        d_scratch_rf = sycl::malloc_device<double>(scratch_rf_size, q);
        d_scratch_ri = sycl::malloc_device<double>(scratch_ri_size, q);

        if (!d_Winv || !d_U || !d_R || !d_Rprev || !d_temp ||
            !d_ipiv || !d_scratch_rf || !d_scratch_ri)
        {
            throw std::runtime_error(
                "GpuForwardStepper: GPU malloc_device returned null "
                "(insufficient HBM for N=" + std::to_string(n_) + "?)");
        }

        // Try to allocate sytrf/sytri scratchpads.  Both APIs are needed:
        // sytrf factorises (LDLᵀ on lower triangle), sytri inverts from
        // those factors.  Some Intel oneMKL versions ship sytrf but not
        // sytri for the GPU LAPACK API; the build-time probe
        // SCATT_HAS_ONEMKL_SYTRI in CMakeLists.txt detects this and
        // permanently disables the symmetric path when sytri is missing.
        // When the macro IS defined, we still wrap the runtime allocation
        // in try/catch -- the GPU device or driver may still refuse the
        // call at runtime even when the symbol resolves at link time.
#ifdef SCATT_HAS_ONEMKL_SYTRI
        try {
            scratch_sf_size = oneapi::mkl::lapack::sytrf_scratchpad_size<double>(
                q, oneapi::mkl::uplo::lower, n, n);
            scratch_si_size = oneapi::mkl::lapack::sytri_scratchpad_size<double>(
                q, oneapi::mkl::uplo::lower, n, n);
            d_scratch_sf = sycl::malloc_device<double>(
                static_cast<std::size_t>(scratch_sf_size), q);
            d_scratch_si = sycl::malloc_device<double>(
                static_cast<std::size_t>(scratch_si_size), q);
            sytrf_supported = (d_scratch_sf != nullptr && d_scratch_si != nullptr);
        } catch (const oneapi::mkl::lapack::exception&) {
            sytrf_supported = false;
        } catch (const oneapi::mkl::exception&) {
            sytrf_supported = false;
        } catch (const std::exception&) {
            sytrf_supported = false;
        } catch (...) {
            sytrf_supported = false;
        }
        if (!sytrf_supported) {
            if (d_scratch_sf) { sycl::free(d_scratch_sf, q); d_scratch_sf = nullptr; }
            if (d_scratch_si) { sycl::free(d_scratch_si, q); d_scratch_si = nullptr; }
            scratch_sf_size = 0;
            scratch_si_size = 0;
        }
#else
        // Build-time probe didn't find oneapi::mkl::lapack::sytri.
        // step() will always take the legacy getrf+getri path.
        sytrf_supported = false;
#endif

        // Zero-init Rprev so the first step (if caller forgets init) is
        // at least well-defined.
        q.memset(d_Rprev, 0, nn * sizeof(double)).wait();
    }

    ~Impl() {
        if (d_Winv)        sycl::free(d_Winv,        q);
        if (d_U)           sycl::free(d_U,           q);
        if (d_R)           sycl::free(d_R,           q);
        if (d_Rprev)       sycl::free(d_Rprev,       q);
        if (d_temp)        sycl::free(d_temp,        q);
        if (d_ipiv)        sycl::free(d_ipiv,        q);
        if (d_scratch_rf)  sycl::free(d_scratch_rf,  q);
        if (d_scratch_ri)  sycl::free(d_scratch_ri,  q);
        if (d_scratch_sf)  sycl::free(d_scratch_sf,  q);
        if (d_scratch_si)  sycl::free(d_scratch_si,  q);
    }
};

GpuForwardStepper::GpuForwardStepper(GpuContext& ctx, int N_total)
    : ctx_(ctx), N_(N_total)
{
    impl_ = std::make_unique<Impl>(as_queue(ctx_.raw_queue()),
                                   static_cast<std::int64_t>(N_total));
}

GpuForwardStepper::~GpuForwardStepper() = default;

void GpuForwardStepper::enable_symmetric_inverse(bool enable) {
    impl_->use_symmetric_requested = enable;
}

bool GpuForwardStepper::is_symmetric_inverse_active() const {
    return impl_->use_symmetric_requested && impl_->sytrf_supported;
}

bool GpuForwardStepper::symmetric_inverse_supported() const {
    return impl_->sytrf_supported;
}

void GpuForwardStepper::init_R_prev_inv(const Eigen::MatrixXd& R) {
    if (R.rows() != N_ || R.cols() != N_) {
        throw std::runtime_error(
            "GpuForwardStepper::init_R_prev_inv: wrong shape (got " +
            std::to_string(R.rows()) + "×" + std::to_string(R.cols()) +
            ", expected " + std::to_string(N_) + "²)");
    }
    auto& q = as_queue(ctx_.raw_queue());
    q.memcpy(impl_->d_Rprev, R.data(),
             static_cast<std::size_t>(N_) * N_ * sizeof(double)).wait();
}

double GpuForwardStepper::step(const Eigen::MatrixXd& W_inv,
                               Eigen::MatrixXd&       Rinv_out)
{
    if (W_inv.rows() != N_ || W_inv.cols() != N_) {
        throw std::runtime_error("GpuForwardStepper::step: W_inv wrong shape");
    }
    auto& q = as_queue(ctx_.raw_queue());
    const std::int64_t n  = N_;
    const std::size_t  nn = static_cast<std::size_t>(n) * n;

    // (1) Upload W_inv to GPU.
    auto t0 = std::chrono::steady_clock::now();
    q.memcpy(impl_->d_Winv, W_inv.data(), nn * sizeof(double)).wait();
    stats_.t_upload_ns += ns_since(t0);

    // (2) U = 12 W_inv − 10 I ;  R_current = U − R_prev_inv.
    auto t1 = std::chrono::steady_clock::now();
    kernel_U_minus_10I_(q, impl_->d_Winv, impl_->d_U, n);
    kernel_sub_(q, impl_->d_U, impl_->d_Rprev, impl_->d_R, static_cast<std::int64_t>(nn));
    // Copy R into temp for in-place inversion.
    q.memcpy(impl_->d_temp, impl_->d_R, nn * sizeof(double)).wait();
    stats_.t_u_combine_ns += ns_since(t1);

    // (3) Invert R_current into d_temp.  Two paths:
    //   (a) NEW path: dsytrf + dsytri (LOWER triangle, ~½ flops of LU)
    //                 followed by mirror LOWER -> UPPER.  Active iff
    //                 the user requested it AND oneMKL supported it at
    //                 construction.
    //   (b) LEGACY path: dgetrf + dgetri + symmetrise.  Always available.
    auto t2 = std::chrono::steady_clock::now();
    // sytrf_supported is permanently false when SCATT_HAS_ONEMKL_SYTRI is
    // undefined (build-time probe didn't find sytri); see Impl ctor above.
    // So `use_sym` can only be true when the macro IS defined, and the
    // sytri call below is unreachable when it is not.  Still, guard the
    // call with #ifdef so the source compiles even on builds where the
    // symbol doesn't exist in oneMKL.
    const bool use_sym = impl_->use_symmetric_requested && impl_->sytrf_supported;
    try {
        if (use_sym) {
#ifdef SCATT_HAS_ONEMKL_SYTRI
            oneapi::mkl::lapack::sytrf(
                q, oneapi::mkl::uplo::lower, n,
                impl_->d_temp, n, impl_->d_ipiv,
                impl_->d_scratch_sf, impl_->scratch_sf_size).wait();
            oneapi::mkl::lapack::sytri(
                q, oneapi::mkl::uplo::lower, n,
                impl_->d_temp, n, impl_->d_ipiv,
                impl_->d_scratch_si, impl_->scratch_si_size).wait();
#else
            // unreachable: sytrf_supported=false when the macro is undefined.
            throw std::runtime_error(
                "GpuForwardStepper: SCATT_HAS_ONEMKL_SYTRI not defined but "
                "use_sym=true (logic bug in Impl ctor).");
#endif
        } else {
            oneapi::mkl::lapack::getrf(q, n, n, impl_->d_temp, n, impl_->d_ipiv,
                                       impl_->d_scratch_rf, impl_->scratch_rf_size).wait();
            oneapi::mkl::lapack::getri(q, n, impl_->d_temp, n, impl_->d_ipiv,
                                       impl_->d_scratch_ri, impl_->scratch_ri_size).wait();
        }
    } catch (const oneapi::mkl::lapack::exception& e) {
        throw std::runtime_error(
            std::string("GpuForwardStepper: oneMKL inversion failed (")
            + (use_sym ? "sytrf/sytri" : "getrf/getri")
            + "): " + e.what());
    }
    if (use_sym) {
        // dsytri wrote only the LOWER triangle.  Mirror to UPPER so the
        // matrix is bit-symmetric for downstream consumers.
        kernel_mirror_lower_to_upper_(q, impl_->d_temp, n);
    } else {
        // dgetri produces a non-symmetric result (rounding asymmetry from
        // row/column ops).  Symmetrise to bit-equality.
        kernel_symmetrize_(q, impl_->d_temp, n);
    }
    // Make d_temp the new R_prev_inv (swap pointers to avoid a copy).
    std::swap(impl_->d_Rprev, impl_->d_temp);
    stats_.t_inverse_ns += ns_since(t2);

    // (4) Download Rinv_out for checkpointing.  If callers never need a
    // checkpoint they could skip this; for now we mirror version_0.
    auto t3 = std::chrono::steady_clock::now();
    if (Rinv_out.rows() != N_ || Rinv_out.cols() != N_) Rinv_out.resize(N_, N_);
    q.memcpy(Rinv_out.data(), impl_->d_Rprev, nn * sizeof(double)).wait();
    stats_.t_download_ns += ns_since(t3);

    ++stats_.n_steps;

    // Host-side symmetry error (cheap).
    return (Rinv_out - Rinv_out.transpose()).cwiseAbs().maxCoeff();
}

// ===========================================================================
// GpuSinvStepper -- per-radial-grid Schur complement + LU inverse on GPU.
// ===========================================================================

namespace {
// Kernel: column-scale  Bscaled[:, j] = B[:, j] * Dinv[j]   (N_psi × N_f).
inline void kernel_col_scale_(sycl::queue& q,
                              const double* d_B,
                              const double* d_Dinv,
                              double*       d_Bscaled,
                              std::int64_t  N_psi,
                              std::int64_t  N_f)
{
    q.parallel_for(sycl::range<2>(N_psi, N_f), [=](sycl::id<2> idx) {
        const std::int64_t i = idx[0];
        const std::int64_t j = idx[1];
        const std::int64_t k = j * N_psi + i;   // column-major
        d_Bscaled[k] = d_B[k] * d_Dinv[j];
    }).wait();
}
}  // anonymous namespace

struct GpuSinvStepper::Impl {
    sycl::queue&        q;
    const std::int64_t  N_psi;
    const std::int64_t  N_f;

    // Persistent device buffers, allocated once at construction.
    double*             d_A        = nullptr;  // N_psi × N_psi  (input A, also Schur target)
    double*             d_B        = nullptr;  // N_psi × N_f    (input B)
    double*             d_Bscaled  = nullptr;  // N_psi × N_f    (B · diag(Dinv))
    double*             d_Dinv     = nullptr;  // N_f vector
    double*             d_temp     = nullptr;  // N_psi × N_psi  (in-place inverse target)
    std::int64_t*       d_ipiv     = nullptr;  // N_psi
    double*             d_scratch_rf = nullptr;
    double*             d_scratch_ri = nullptr;
    std::int64_t        scratch_rf_size = 0;
    std::int64_t        scratch_ri_size = 0;

    Impl(sycl::queue& q_, std::int64_t Np, std::int64_t Nf) : q(q_), N_psi(Np), N_f(Nf)
    {
        const std::size_t pp = static_cast<std::size_t>(Np) * Np;
        const std::size_t pf = static_cast<std::size_t>(Np) * Nf;

        d_A        = sycl::malloc_device<double>(pp, q);
        d_B        = sycl::malloc_device<double>(pf, q);
        d_Bscaled  = sycl::malloc_device<double>(pf, q);
        d_Dinv     = sycl::malloc_device<double>(static_cast<std::size_t>(Nf), q);
        d_temp     = sycl::malloc_device<double>(pp, q);
        d_ipiv     = sycl::malloc_device<std::int64_t>(static_cast<std::size_t>(Np), q);

        scratch_rf_size = oneapi::mkl::lapack::getrf_scratchpad_size<double>(q, Np, Np, Np);
        scratch_ri_size = oneapi::mkl::lapack::getri_scratchpad_size<double>(q, Np, Np);
        d_scratch_rf = sycl::malloc_device<double>(
            static_cast<std::size_t>(scratch_rf_size), q);
        d_scratch_ri = sycl::malloc_device<double>(
            static_cast<std::size_t>(scratch_ri_size), q);

        if (!d_A || !d_B || !d_Bscaled || !d_Dinv || !d_temp || !d_ipiv ||
            !d_scratch_rf || !d_scratch_ri)
        {
            throw std::runtime_error(
                "GpuSinvStepper: GPU malloc_device returned null "
                "(insufficient HBM for N_psi=" + std::to_string(Np) +
                ", N_f=" + std::to_string(Nf) + "?)");
        }
    }

    ~Impl() {
        if (d_A)            sycl::free(d_A,            q);
        if (d_B)            sycl::free(d_B,            q);
        if (d_Bscaled)      sycl::free(d_Bscaled,      q);
        if (d_Dinv)         sycl::free(d_Dinv,         q);
        if (d_temp)         sycl::free(d_temp,         q);
        if (d_ipiv)         sycl::free(d_ipiv,         q);
        if (d_scratch_rf)   sycl::free(d_scratch_rf,   q);
        if (d_scratch_ri)   sycl::free(d_scratch_ri,   q);
    }
};

GpuSinvStepper::GpuSinvStepper(GpuContext& ctx, int N_psi, int N_f)
    : ctx_(ctx), N_psi_(N_psi), N_f_(N_f)
{
    impl_ = std::make_unique<Impl>(as_queue(ctx_.raw_queue()),
                                   static_cast<std::int64_t>(N_psi),
                                   static_cast<std::int64_t>(N_f));
}

GpuSinvStepper::~GpuSinvStepper() = default;

double GpuSinvStepper::step(const Eigen::MatrixXd& A_host,
                            const Eigen::MatrixXd& B_host,
                            const Eigen::VectorXd& Dinv_host,
                            Eigen::MatrixXd&       Sinv_out)
{
    if (A_host.rows() != N_psi_ || A_host.cols() != N_psi_) {
        throw std::runtime_error("GpuSinvStepper::step: A has wrong shape");
    }
    if (B_host.rows() != N_psi_ || B_host.cols() != N_f_) {
        throw std::runtime_error("GpuSinvStepper::step: B has wrong shape");
    }
    if (Dinv_host.size() != N_f_) {
        throw std::runtime_error("GpuSinvStepper::step: Dinv has wrong size");
    }

    auto& q = as_queue(ctx_.raw_queue());
    const std::int64_t Np = N_psi_;
    const std::int64_t Nf = N_f_;
    const std::size_t  pp = static_cast<std::size_t>(Np) * Np;
    const std::size_t  pf = static_cast<std::size_t>(Np) * Nf;

    // (1) Upload A, B, Dinv to device.
    auto t0 = std::chrono::steady_clock::now();
    q.memcpy(impl_->d_A,    A_host.data(),    pp * sizeof(double)).wait();
    q.memcpy(impl_->d_B,    B_host.data(),    pf * sizeof(double)).wait();
    q.memcpy(impl_->d_Dinv, Dinv_host.data(), static_cast<std::size_t>(Nf) * sizeof(double)).wait();
    stats_.t_upload_ns += ns_since(t0);

    // (2) Compute Schur complement on device: S = A − B · diag(Dinv) · Bᵀ.
    //     Step (a): Bscaled[i,j] = B[i,j] * Dinv[j]    (col-scale kernel)
    //     Step (b): d_A := d_A − d_Bscaled · d_Bᵀ      (dgemm into d_A)
    auto t1 = std::chrono::steady_clock::now();
    kernel_col_scale_(q, impl_->d_B, impl_->d_Dinv, impl_->d_Bscaled, Np, Nf);
    oneapi::mkl::blas::column_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans,
        Np, Np, Nf,
        -1.0,  impl_->d_Bscaled, Np,
               impl_->d_B,       Np,
         1.0,  impl_->d_A,       Np
    ).wait();
    // (2c) Symmetrise S to bit-equality.  Matches the CPU symmetrise path.
    kernel_symmetrize_(q, impl_->d_A, Np);
    stats_.t_schur_ns += ns_since(t1);

    // (3) Invert: copy S into d_temp, run dgetrf+dgetri in-place.
    auto t2 = std::chrono::steady_clock::now();
    q.memcpy(impl_->d_temp, impl_->d_A, pp * sizeof(double)).wait();
    try {
        oneapi::mkl::lapack::getrf(
            q, Np, Np, impl_->d_temp, Np, impl_->d_ipiv,
            impl_->d_scratch_rf, impl_->scratch_rf_size).wait();
        oneapi::mkl::lapack::getri(
            q, Np, impl_->d_temp, Np, impl_->d_ipiv,
            impl_->d_scratch_ri, impl_->scratch_ri_size).wait();
    } catch (const oneapi::mkl::lapack::exception& e) {
        throw std::runtime_error(
            std::string("GpuSinvStepper: oneMKL inversion failed: ") + e.what());
    }
    // Symmetrise Sinv to bit-equality.
    kernel_symmetrize_(q, impl_->d_temp, Np);
    stats_.t_inverse_ns += ns_since(t2);

    // (4) Download Sinv to host.
    auto t3 = std::chrono::steady_clock::now();
    if (Sinv_out.rows() != Np || Sinv_out.cols() != Np) {
        Sinv_out.resize(Np, Np);
    }
    q.memcpy(Sinv_out.data(), impl_->d_temp, pp * sizeof(double)).wait();
    stats_.t_download_ns += ns_since(t3);

    ++stats_.n_steps;

    // Cheap host-side symmetry check.
    return (Sinv_out - Sinv_out.transpose()).cwiseAbs().maxCoeff();
}

double GpuSinvStepper::step_inverse_only(const Eigen::MatrixXd& A_host,
                                         Eigen::MatrixXd&       Sinv_out)
{
    if (A_host.rows() != N_psi_ || A_host.cols() != N_psi_) {
        throw std::runtime_error("GpuSinvStepper::step_inverse_only: A has wrong shape");
    }
    auto& q = as_queue(ctx_.raw_queue());
    const std::int64_t Np = N_psi_;
    const std::size_t  pp = static_cast<std::size_t>(Np) * Np;

    auto t0 = std::chrono::steady_clock::now();
    q.memcpy(impl_->d_temp, A_host.data(), pp * sizeof(double)).wait();
    stats_.t_upload_ns += ns_since(t0);

    auto t2 = std::chrono::steady_clock::now();
    try {
        oneapi::mkl::lapack::getrf(
            q, Np, Np, impl_->d_temp, Np, impl_->d_ipiv,
            impl_->d_scratch_rf, impl_->scratch_rf_size).wait();
        oneapi::mkl::lapack::getri(
            q, Np, impl_->d_temp, Np, impl_->d_ipiv,
            impl_->d_scratch_ri, impl_->scratch_ri_size).wait();
    } catch (const oneapi::mkl::lapack::exception& e) {
        throw std::runtime_error(
            std::string("GpuSinvStepper::step_inverse_only: oneMKL inversion failed: ") + e.what());
    }
    kernel_symmetrize_(q, impl_->d_temp, Np);
    stats_.t_inverse_ns += ns_since(t2);

    auto t3 = std::chrono::steady_clock::now();
    if (Sinv_out.rows() != Np || Sinv_out.cols() != Np) {
        Sinv_out.resize(Np, Np);
    }
    q.memcpy(Sinv_out.data(), impl_->d_temp, pp * sizeof(double)).wait();
    stats_.t_download_ns += ns_since(t3);

    ++stats_.n_steps;
    return (Sinv_out - Sinv_out.transpose()).cwiseAbs().maxCoeff();
}

// ---------------------------------------------------------------------------
// GpuBackStepper::Impl.
// ---------------------------------------------------------------------------
struct GpuBackStepper::Impl {
    sycl::queue&     q;
    const std::int64_t N_total;
    const std::int64_t N_psi;
    const std::int64_t N_f;

    double* d_Z       = nullptr;   // N_total × N_psi (pinned across steps)
    double* d_Ztmp    = nullptr;   // same shape (GEMM dest)
    double* d_Y       = nullptr;   // same shape
    double* d_Rinv    = nullptr;   // N_total × N_total
    double* d_Winv    = nullptr;   // N_total × N_total
    double* d_psi_out = nullptr;   // N_psi × N_psi
    double* d_f_out   = nullptr;   // N_f × N_psi   (nullable when N_f=0)

    Impl(sycl::queue& q_, std::int64_t Nt, std::int64_t Np, std::int64_t Nf)
        : q(q_), N_total(Nt), N_psi(Np), N_f(Nf)
    {
        const std::size_t Z_sz    = static_cast<std::size_t>(Nt) * Np;
        const std::size_t Mat_sz  = static_cast<std::size_t>(Nt) * Nt;
        const std::size_t psi_sz  = static_cast<std::size_t>(Np) * Np;
        const std::size_t f_sz    = static_cast<std::size_t>(Nf) * Np;

        d_Z    = sycl::malloc_device<double>(Z_sz,   q);
        d_Ztmp = sycl::malloc_device<double>(Z_sz,   q);
        d_Y    = sycl::malloc_device<double>(Z_sz,   q);
        d_Rinv = sycl::malloc_device<double>(Mat_sz, q);
        d_Winv = sycl::malloc_device<double>(Mat_sz, q);
        d_psi_out = sycl::malloc_device<double>(psi_sz, q);
        d_f_out   = (Nf > 0) ? sycl::malloc_device<double>(f_sz, q) : nullptr;

        if (!d_Z || !d_Ztmp || !d_Y || !d_Rinv || !d_Winv || !d_psi_out ||
            (Nf > 0 && !d_f_out))
        {
            throw std::runtime_error(
                "GpuBackStepper: GPU malloc_device returned null "
                "(insufficient HBM for N_total=" + std::to_string(Nt) + "?)");
        }
    }

    ~Impl() {
        if (d_Z)       sycl::free(d_Z,       q);
        if (d_Ztmp)    sycl::free(d_Ztmp,    q);
        if (d_Y)       sycl::free(d_Y,       q);
        if (d_Rinv)    sycl::free(d_Rinv,    q);
        if (d_Winv)    sycl::free(d_Winv,    q);
        if (d_psi_out) sycl::free(d_psi_out, q);
        if (d_f_out)   sycl::free(d_f_out,   q);
    }
};

GpuBackStepper::GpuBackStepper(GpuContext& ctx, int N_total, int N_psi, int N_f)
    : ctx_(ctx), N_total_(N_total), N_psi_(N_psi), N_f_(N_f)
{
    impl_ = std::make_unique<Impl>(
        as_queue(ctx_.raw_queue()),
        static_cast<std::int64_t>(N_total),
        static_cast<std::int64_t>(N_psi),
        static_cast<std::int64_t>(N_f));
}

GpuBackStepper::~GpuBackStepper() = default;

void GpuBackStepper::init_Z(const Eigen::MatrixXd& Z) {
    if (Z.rows() != N_total_ || Z.cols() != N_psi_) {
        throw std::runtime_error("GpuBackStepper::init_Z: wrong shape");
    }
    auto& q = as_queue(ctx_.raw_queue());
    q.memcpy(impl_->d_Z, Z.data(),
             static_cast<std::size_t>(N_total_) * N_psi_ * sizeof(double)).wait();
}

void GpuBackStepper::step(const Eigen::MatrixXd& Rinv,
                          const Eigen::MatrixXd& W_inv,
                          Eigen::MatrixXd&       psi_out,
                          Eigen::MatrixXd*       f_out,
                          bool                   compute_f)
{
    if (Rinv.rows() != N_total_ || Rinv.cols() != N_total_) {
        throw std::runtime_error("GpuBackStepper::step: Rinv wrong shape");
    }
    if (W_inv.rows() != N_total_ || W_inv.cols() != N_total_) {
        throw std::runtime_error("GpuBackStepper::step: W_inv wrong shape");
    }

    auto& q = as_queue(ctx_.raw_queue());
    const std::int64_t Nt = N_total_;
    const std::int64_t Np = N_psi_;
    const std::int64_t Nf = N_f_;
    const std::size_t  Mat_sz = static_cast<std::size_t>(Nt) * Nt;

    // (1) Upload Rinv + W_inv.
    auto t0 = std::chrono::steady_clock::now();
    q.memcpy(impl_->d_Rinv, Rinv.data(),  Mat_sz * sizeof(double)).wait();
    q.memcpy(impl_->d_Winv, W_inv.data(), Mat_sz * sizeof(double)).wait();
    stats_.t_upload_ns += ns_since(t0);

    // (2) Z_tmp = Rinv · Z ; swap so Z is the new one.
    auto t1 = std::chrono::steady_clock::now();
    oneapi::mkl::blas::column_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        Nt, Np, Nt,
        1.0,  impl_->d_Rinv, Nt,
              impl_->d_Z,    Nt,
        0.0,  impl_->d_Ztmp, Nt).wait();
    std::swap(impl_->d_Z, impl_->d_Ztmp);
    stats_.t_gemm_z_ns += ns_since(t1);

    // (3) Y = W_inv · Z.
    auto t2 = std::chrono::steady_clock::now();
    oneapi::mkl::blas::column_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        Nt, Np, Nt,
        1.0,  impl_->d_Winv, Nt,
              impl_->d_Z,    Nt,
        0.0,  impl_->d_Y,    Nt).wait();
    stats_.t_gemm_y_ns += ns_since(t2);

    // (4) Extract psi (top N_psi rows) [and f (bottom N_f rows)] and download.
    auto t3 = std::chrono::steady_clock::now();
    kernel_extract_psi_(q, impl_->d_Y, impl_->d_psi_out, Nt, Np);
    if (psi_out.rows() != Np || psi_out.cols() != Np) psi_out.resize(Np, Np);
    q.memcpy(psi_out.data(), impl_->d_psi_out,
             static_cast<std::size_t>(Np) * Np * sizeof(double)).wait();

    if (compute_f && f_out != nullptr && Nf > 0) {
        kernel_extract_f_(q, impl_->d_Y, impl_->d_f_out, Nt, Np, Nf);
        if (f_out->rows() != Nf || f_out->cols() != Np) f_out->resize(Nf, Np);
        q.memcpy(f_out->data(), impl_->d_f_out,
                 static_cast<std::size_t>(Nf) * Np * sizeof(double)).wait();
    }
    stats_.t_extract_ns += ns_since(t3);
    ++stats_.n_steps;
}

void GpuBackStepper::get_Z(Eigen::MatrixXd& Z_out) const {
    if (Z_out.rows() != N_total_ || Z_out.cols() != N_psi_) {
        Z_out.resize(N_total_, N_psi_);
    }
    auto& q = as_queue(ctx_.raw_queue());
    q.memcpy(Z_out.data(), impl_->d_Z,
             static_cast<std::size_t>(N_total_) * N_psi_ * sizeof(double)).wait();
}

// ===========================================================================
// NON-SYCL BRANCH -- same API, all stubs throw.
// ===========================================================================
#else  // !SCATT_HAS_SYCL

namespace {
[[noreturn]] void gpu_not_compiled_() {
    throw std::runtime_error(
        "GPU path not compiled in. Rebuild with -DSCATT_WITH_SYCL=ON "
        "using Intel oneAPI DPC++ (icpx).");
}
}

GpuContext::GpuContext(bool prefer_gpu) {
    if (prefer_gpu) gpu_not_compiled_();
    info_ = Info{};        // all-zeros, is_gpu=false
    info_.device_name = "host (SYCL not compiled)";
}
GpuContext::~GpuContext() = default;

struct GpuForwardStepper::Impl {};
GpuForwardStepper::GpuForwardStepper(GpuContext& ctx, int N_total) : ctx_(ctx), N_(N_total) {
    gpu_not_compiled_();
}
GpuForwardStepper::~GpuForwardStepper() = default;
void GpuForwardStepper::init_R_prev_inv(const Eigen::MatrixXd&) { gpu_not_compiled_(); }
double GpuForwardStepper::step(const Eigen::MatrixXd&, Eigen::MatrixXd&) { gpu_not_compiled_(); }
void GpuForwardStepper::enable_symmetric_inverse(bool) { /* SYCL not compiled in: silent no-op */ }
bool GpuForwardStepper::is_symmetric_inverse_active() const { return false; }
bool GpuForwardStepper::symmetric_inverse_supported() const { return false; }

struct GpuSinvStepper::Impl {};
GpuSinvStepper::GpuSinvStepper(GpuContext& ctx, int Np, int Nf)
    : ctx_(ctx), N_psi_(Np), N_f_(Nf)
{
    gpu_not_compiled_();
}
GpuSinvStepper::~GpuSinvStepper() = default;
double GpuSinvStepper::step(const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                            const Eigen::VectorXd&, Eigen::MatrixXd&) {
    gpu_not_compiled_();
}
double GpuSinvStepper::step_inverse_only(const Eigen::MatrixXd&, Eigen::MatrixXd&) {
    gpu_not_compiled_();
}

struct GpuBackStepper::Impl {};
GpuBackStepper::GpuBackStepper(GpuContext& ctx, int Nt, int Np, int Nf)
    : ctx_(ctx), N_total_(Nt), N_psi_(Np), N_f_(Nf)
{
    gpu_not_compiled_();
}
GpuBackStepper::~GpuBackStepper() = default;
void GpuBackStepper::init_Z(const Eigen::MatrixXd&) { gpu_not_compiled_(); }
void GpuBackStepper::step(const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                          Eigen::MatrixXd&, Eigen::MatrixXd*, bool) {
    gpu_not_compiled_();
}
void GpuBackStepper::get_Z(Eigen::MatrixXd&) const { gpu_not_compiled_(); }

#endif  // SCATT_HAS_SYCL

}  // namespace scatt
