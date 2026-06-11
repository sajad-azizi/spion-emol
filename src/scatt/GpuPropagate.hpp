// GpuPropagate.hpp -- SYCL / oneMKL offload of the per-step Numerov work.
//
// The forward R-recursion and the backward Y = W^{-1}·(Rinv·Z) sweep are
// both dominated by per-step O(N^3) dense LAPACK/BLAS on an (N_total ×
// N_total) matrix.  At C₈F₈ scale (N_total ~10^4) one CPU step is several
// seconds; the same step on a PVC GPU is tens of milliseconds plus an
// O(N^2) PCIe upload of W_inv.  This header exposes two GPU steppers that
// mirror the two CPU hot spots without changing call sites on the CPU:
//
//     GpuForwardStepper   -- one call per n replaces
//                              wi_.apply_U(n, I, U)  +  inverse_general(U − Rinv_prev)
//                            Keeps R_prev_inv pinned on device across steps.
//
//     GpuBackStepper      -- one call per n replaces
//                              Z = Rinv_n · Z;  Y = W_inv_n · Z;
//                              (extract psi + f)
//                            Keeps Z pinned on device across steps.
//
// Both classes allocate persistent GPU buffers at construction, so the
// only per-step transfer is the upload of W_inv (and Rinv for the back
// sweep, which already lives on host via ForwardRPropagator::get).
//
// Build-time gated on SCATT_HAS_SYCL.  When SYCL is not available, the
// header still compiles (everything is just declared; callers check the
// static bool `gpu_available()`).  All heavy lifting lives in
// GpuPropagate.cpp so the SYCL headers don't leak into the rest of the
// codebase.

#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace scatt {

// ---------------------------------------------------------------------------
// Device selection & queue ownership.
// ---------------------------------------------------------------------------
// A GpuContext owns one sycl::queue.  All GpuForwardStepper /
// GpuBackStepper objects take a reference so GPU memory and kernels share
// the same device.  The queue is `in_order` to keep the code easy to
// reason about (each kernel/blas call .wait()s internally).
//
// Construction picks the first GPU device if `prefer_gpu = true`, or the
// host CPU otherwise.  Throws std::runtime_error when SCATT_HAS_SYCL is
// not defined and prefer_gpu = true (so misconfigured builds fail loudly
// rather than silently fall back to CPU).
class GpuContext {
public:
    struct Info {
        std::string device_name;
        std::string platform_name;
        bool        is_gpu = false;
        std::size_t global_mem_bytes = 0;
    };

    // prefer_gpu=true -> fail if no SYCL GPU is visible.
    // prefer_gpu=false -> succeed even without SCATT_HAS_SYCL (host fallback).
    explicit GpuContext(bool prefer_gpu = true);
    ~GpuContext();

    GpuContext(const GpuContext&)            = delete;
    GpuContext& operator=(const GpuContext&) = delete;

    const Info& info() const { return info_; }

    // Opaque pointer to the underlying sycl::queue.  Only GpuPropagate.cpp
    // reinterprets it (avoids exposing SYCL headers here).
    void* raw_queue() const { return queue_opaque_; }

    static bool gpu_available();      // compile-time SCATT_HAS_SYCL AND a GPU exists

private:
    Info  info_{};
    void* queue_opaque_ = nullptr;    // sycl::queue*
};

// ---------------------------------------------------------------------------
// Forward stepper:   Rinv_n = ( 12 W_n^{-1} − 10 I  − Rinv_{n-1} )^{-1}
// ---------------------------------------------------------------------------
// Persistent GPU buffers: W_inv, U, R_current, R_prev_inv, temp (inversion
// target), LAPACK scratchpad, pivot array.  Total ≈ (4–5) N² × 8 B + scratch.
// At N=20000 that's ~16 GB, well within PVC's 128 GB HBM.
class GpuForwardStepper {
public:
    GpuForwardStepper(GpuContext& ctx, int N_total);
    ~GpuForwardStepper();

    GpuForwardStepper(const GpuForwardStepper&)            = delete;
    GpuForwardStepper& operator=(const GpuForwardStepper&) = delete;

    // Upload initial Rinv (e.g. analytic Rinv at n_start − 1).
    void init_R_prev_inv(const Eigen::MatrixXd& R);

    // One forward step.  W_inv is uploaded each call (the only O(N²) H→D
    // traffic per step).  On return, Rinv_out contains the new R_prev_inv
    // and the GPU copy is ready for the next step.
    //
    // Returns max |Rinv − Rinv^T| for diagnostics (cheap on host).
    double step(const Eigen::MatrixXd& W_inv, Eigen::MatrixXd& Rinv_out);

    // Request the symmetric-LDLᵀ inversion path (oneMKL sytrf+sytri on the
    // LOWER triangle, then a custom mirror kernel; ~½ flops of getrf+getri,
    // and removes the explicit symmetrise kernel).
    //
    // Argument is only a REQUEST.  The actual path chosen depends on
    // whether oneMKL on this device supported the sytrf/sytri scratchpad
    // queries at construction.  If `enable=true` and sytrf is supported,
    // subsequent step() calls take the symmetric path.  If `enable=false`
    // OR sytrf is unsupported, step() falls back to the legacy
    // getrf+getri+symmetrise path bit-for-bit.
    //
    // The actual active state is queryable via is_symmetric_inverse_active.
    void enable_symmetric_inverse(bool enable);

    // Returns true iff:
    //   * enable_symmetric_inverse(true) was called (the request), AND
    //   * oneMKL successfully allocated sytrf/sytri scratchpads at
    //     construction (the device support).
    // step() takes the symmetric path iff this returns true.
    bool is_symmetric_inverse_active() const;

    // Runtime check at construction: was oneMKL's sytrf/sytri scratchpad
    // available on this device?  (Independent of enable_symmetric_inverse.)
    bool symmetric_inverse_supported() const;

    int N_total() const { return N_; }

    // Accumulated per-phase timings (nanoseconds), cleared by reset_stats().
    struct Stats {
        std::uint64_t t_upload_ns   = 0;    // W_inv H→D
        std::uint64_t t_u_combine_ns= 0;    // U = 12 Winv − 10 I  +  R = U − Rprev
        std::uint64_t t_inverse_ns  = 0;    // getrf + getri + symmetrize
        std::uint64_t t_download_ns = 0;    // Rinv D→H
        std::uint64_t n_steps       = 0;
    };
    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

private:
    [[maybe_unused]] GpuContext& ctx_;
    int         N_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Stats stats_{};
};

// ---------------------------------------------------------------------------
// Sinv stepper: per-radial-grid Schur-complement inverse on GPU.
// ---------------------------------------------------------------------------
// One call per ir replaces the CPU side of compute_sinv_at:
//      S = A − B · D⁻¹ · Bᵀ       (Schur complement, GEMM)
//      Sinv = S⁻¹                 (LAPACK dgetrf + dgetri)
//      Sinv = ½ (Sinv + Sinvᵀ)    (symmetrise, GPU kernel)
//
// Caller (SchurInverter on the host) is responsible for:
//   * Building A = I + (h²/6)(E·I − V_n)              (small per-step cost)
//   * Building B = (h²/12)·Q_ψf(n)                    (ExchangeCoupling kernel)
//   * Filling Dinv with the clamped 1 − (h²/12)·ℓ(ℓ+1)/r²  (cheap)
//   * Optional CPU-side stability shifts on A and S for small n
//     (see SchurInverter::Config::stab_n_max).  The GPU stepper does
//     NOT perform any stability check; the caller decides whether to
//     dispatch a given n to the GPU stepper or to the CPU path.
//
// Memory: persistent device buffers for d_A, d_B, d_Bscaled, d_S, d_temp
// (in-place LU target), d_ipiv, getrf/getri scratchpads, d_Dinv.
// At L=100 (N_psi=10201, N_f=7260) total HBM usage is ~5–6 GB.
//
// Numerical equivalence: bit-equivalent to running dgemm + dgetrf +
// dgetri + symmetrise on CPU MKL, up to FP-roundoff (~ε·κ(S) per call).
// Sinv is computed INDEPENDENTLY per ir -- no recursion, no error
// accumulation between steps.  Validated bit-tight against the CPU
// path by test_gpu_sinv on the H2O fixture.
class GpuSinvStepper {
public:
    GpuSinvStepper(GpuContext& ctx, int N_psi, int N_f);
    ~GpuSinvStepper();

    GpuSinvStepper(const GpuSinvStepper&)            = delete;
    GpuSinvStepper& operator=(const GpuSinvStepper&) = delete;

    // Per-step Sinv computation:  Sinv = (A − B·diag(Dinv)·Bᵀ)⁻¹
    //
    // Inputs (host memory):
    //   A    (N_psi × N_psi, column-major double)
    //   B    (N_psi × N_f,   column-major double)
    //   Dinv (N_f vector)
    // Output (host memory):
    //   Sinv_out (N_psi × N_psi, column-major double; resized if needed)
    //
    // Returns max |Sinv − Sinvᵀ|_inf as a cheap diagnostic; should be 0
    // since the GPU symmetrise kernel produces bit-symmetric output.
    double step(const Eigen::MatrixXd& A,
                const Eigen::MatrixXd& B,
                const Eigen::VectorXd& Dinv,
                Eigen::MatrixXd&       Sinv_out);

    // Inverse-only variant for the exchange-off region (n ≥ n_transition,
    // where S = A and there's no B contribution).  Skips the Schur GEMM.
    //   Sinv = A⁻¹
    double step_inverse_only(const Eigen::MatrixXd& A,
                             Eigen::MatrixXd&       Sinv_out);

    int N_psi() const { return N_psi_; }
    int N_f()   const { return N_f_; }

    struct Stats {
        std::uint64_t t_upload_ns   = 0;   // A,B,Dinv H→D
        std::uint64_t t_schur_ns    = 0;   // S = A − B·D⁻¹·Bᵀ
        std::uint64_t t_inverse_ns  = 0;   // getrf + getri + symmetrise
        std::uint64_t t_download_ns = 0;   // Sinv D→H
        std::uint64_t n_steps       = 0;
    };
    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

private:
    [[maybe_unused]] GpuContext& ctx_;
    int         N_psi_, N_f_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Stats stats_{};
};

// ---------------------------------------------------------------------------
// Back stepper:   Z := Rinv_n · Z;   Y := W_inv_n · Z;   split psi/f
// ---------------------------------------------------------------------------
// Persistent GPU buffers: Z_current (N_total × N_psi), Z_temp, Y (same),
// Rinv (N_total × N_total), W_inv (N_total × N_total), psi_out (N_psi^2),
// f_out (N_f × N_psi).  Z_current is pinned across steps.
class GpuBackStepper {
public:
    GpuBackStepper(GpuContext& ctx, int N_total, int N_psi, int N_f);
    ~GpuBackStepper();

    GpuBackStepper(const GpuBackStepper&)            = delete;
    GpuBackStepper& operator=(const GpuBackStepper&) = delete;

    void init_Z(const Eigen::MatrixXd& Z);

    // One backward step.  Uploads Rinv and W_inv (each O(N²)), runs two
    // GEMMs on device, extracts psi (and optionally f) and downloads them.
    //
    // psi_out shape: (N_psi × N_psi).  f_out shape: (N_f × N_psi); only
    // written if compute_f = true and f_out != nullptr.
    void step(const Eigen::MatrixXd& Rinv,
              const Eigen::MatrixXd& W_inv,
              Eigen::MatrixXd&       psi_out,
              Eigen::MatrixXd*       f_out,
              bool                   compute_f);

    // For tests / debug.
    void get_Z(Eigen::MatrixXd& Z_out) const;

    int N_total() const { return N_total_; }
    int N_psi()   const { return N_psi_; }
    int N_f()     const { return N_f_; }

    struct Stats {
        std::uint64_t t_upload_ns   = 0;    // Rinv + W_inv H→D
        std::uint64_t t_gemm_z_ns   = 0;    // Z := Rinv · Z
        std::uint64_t t_gemm_y_ns   = 0;    // Y := W_inv · Z
        std::uint64_t t_extract_ns  = 0;    // split rows + D→H
        std::uint64_t n_steps       = 0;
    };
    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

private:
    [[maybe_unused]] GpuContext& ctx_;
    int         N_total_, N_psi_, N_f_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Stats stats_{};
};

}  // namespace scatt
