// GpuDipoleEngine.hpp -- SYCL / CUDA offload of the DipoleMatrixElement
// per-ir GEMV bottleneck.
//
// CONTEXT
// -------
// DipoleMatrixElement::compute_six dominates a single-energy C8F8 / l_cont=100
// run after BP (11,118 s of `Dip::integrand` out of 60,623 s total wall in
// the May-2026 production benchmark).  Inside the ir-loop, per ir we do
//
//     for slot = 0..5:                 (gauge × pol, 6 dipole rows)
//         row_result_β = Ξ_slot(row, :) · ψ_ir(:, β)             // GEMV
//     for α = 0..n_occ-1:              (bound-orbital overlaps, 60 at C8F8)
//         row_result_β = φ_α(ir, :μhi) · ψ_ir(:μhi, β)            // GEMV
//
// i.e. (6 + n_occ) row-vector-by-matrix products against the SAME N_psi × N_psi
// matrix ψ_ir.  Each one is an MKL DGEMV.  Stacking the (6 + n_occ) row
// vectors into a single (n_slots × N_psi) matrix V turns the whole per-ir
// work into a single GEMM
//
//     result(n_slots × N_psi) = V(n_slots × N_psi) · ψ_ir(N_psi × N_psi)
//
// Mathematically every result element is the same dot product as the CPU
// GEMV; FP-wise, GEMM and GEMV summation orders differ inside MKL /
// oneMKL / cuBLAS so the result is NOT bit-identical to the CPU path.
// The tolerance is the same ε_mach × N floor the existing GPU code paths
// (GpuForwardStepper, GpuSinvStepper, GpuBackStepper) already meet (~1e-13
// relative on DGEMM output).  Validated by `test_gpu_dme`.
//
// DESIGN
// ------
//   * GpuDipoleEngine holds persistent device buffers:
//       d_V        :  n_slots × N_psi      (uploaded per ir)
//       d_psi      :  N_psi    × N_psi     (uploaded per ir)
//       d_result   :  n_slots × N_psi      (downloaded per ir)
//     Total HBM ≈ (2·n_slots + N_psi) · N_psi · 8 B
//                ≈ 832 MB at C8F8 / l_cont=100 / n_slots=66 -- comfortably
//                  inside PVC's 64 GB HBM.
//   * Reuses the existing `GpuContext` from `GpuPropagate.hpp` (same queue,
//     same device).  No second GPU context allocated.
//   * Pimpl pattern: no SYCL / CUDA headers leak into this header (callers
//     just see `std::unique_ptr<Impl>`).  Same trick used by GpuPropagate.
//   * Strict in-order queue (taken from GpuContext) → naturally serial across
//     ir steps → no races with the DISK-mode ψ chunk cache (the chunk cache
//     itself is single-threaded; we never read more than one chunk at a
//     time).
//   * `step()` is a one-shot up-load → GEMM → down-load with internal
//     .wait() bookends so callers don't need to manage the queue.
//
// ACCURACY CONTRACT
// -----------------
// GEMM / GEMV are mathematically equivalent for each output element but
// differ FP-wise inside the BLAS implementation (MKL vs oneMKL vs cuBLAS).
// Per-call output discrepancy vs the CPU GEMV path is ≤ ε_mach × N ≈ 1e-13
// relative on every element of `result`.  This propagates linearly through
// the Simpson sum (n_pts ≈ 2761 for C8F8) and the final `Σ |D_μ|²` so the
// observable `partial_sigma` deviates by ≤ ~1e-12 absolute -- 8 orders of
// magnitude below the BP fit residual already accepted in production
// (4e-4).  `test_gpu_dme` asserts this bound.
//
// BUILD-TIME / RUNTIME GATES
// --------------------------
//   * Compile-time: `SCATT_HAS_SYCL` (defined when CMake's
//     `SCATT_WITH_SYCL=ON` finds the IntelSYCL package).  When NOT defined
//     the header still compiles; `gpu_available()` returns false; every
//     non-trivial method throws "GPU path not compiled in".
//   * Runtime: `GpuContext::gpu_available()` returns false when no SYCL
//     GPU device is visible (login nodes, CPU-only test runs).  Callers
//     check this BEFORE constructing a GpuDipoleEngine.

#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace scatt {

class GpuContext;   // forward decl; defined in GpuPropagate.hpp

// ---------------------------------------------------------------------------
// GpuDipoleEngine: one-call-per-ir GEMM offload.
// ---------------------------------------------------------------------------
class GpuDipoleEngine {
public:
    // ctx       : shared device + queue (the same one used by FRP / BP / Sinv)
    // N_psi     : asymptotic channel count  (= n_mu)
    // n_slots   : 6 (dipole rows for gauge × pol) + n_occ (bound-orbital
    //             overlap rows) ; arbitrary, fixed across all ir steps of
    //             one compute_six invocation.
    //
    // Allocates persistent device buffers:
    //     d_V       : n_slots × N_psi  (column-major)
    //     d_psi     : N_psi   × N_psi  (column-major)
    //     d_result  : n_slots × N_psi  (column-major)
    //
    // Throws std::runtime_error when SYCL is not compiled in or when a
    // device allocation fails.
    GpuDipoleEngine(GpuContext& ctx, int N_psi, int n_slots);
    ~GpuDipoleEngine();

    GpuDipoleEngine(const GpuDipoleEngine&)            = delete;
    GpuDipoleEngine& operator=(const GpuDipoleEngine&) = delete;

    // One ir step:
    //     V(n_slots × N_psi)        host (already filled by caller)
    //     psi_ir(N_psi × N_psi)     host (returned by bp_.get_psi(ir))
    //   →
    //     result_out(n_slots × N_psi)   host (resized if needed)
    //
    // Internally:
    //     1. memcpy(host → device) for V and psi_ir   (≤ 838 MB at C8F8)
    //     2. ONE DGEMM   result = V · psi_ir          (oneMKL / cuBLAS)
    //     3. memcpy(device → host) for result          (≤ 5.4 MB at C8F8)
    //   Each step .wait()s on the in-order queue before returning so the
    //   caller can immediately read `result_out` and reuse `V`, `psi_ir`.
    //
    // Strict serial usage: caller must never invoke step() concurrently
    // from multiple threads against the same engine (the device queue is
    // shared with FRP / BP / Sinv).  This matches the DISK-ψ contract of
    // DipoleMatrixElement::compute_six -- the ir loop is serial there too.
    void step(const Eigen::MatrixXd& V,
              const Eigen::MatrixXd& psi_ir,
              Eigen::MatrixXd&       result_out);

    int N_psi()    const { return N_psi_;   }
    int n_slots()  const { return n_slots_; }

    // Per-call timing accumulators (ns).  Reset by reset_stats() so
    // callers can attribute upload / gemm / download cost to phases.
    struct Stats {
        std::uint64_t t_upload_ns   = 0;
        std::uint64_t t_gemm_ns     = 0;
        std::uint64_t t_download_ns = 0;
        std::uint64_t n_steps       = 0;
    };
    const Stats& stats() const { return stats_; }
    void reset_stats();

private:
    GpuContext& ctx_;
    int         N_psi_   = 0;
    int         n_slots_ = 0;

    struct Impl;
    std::unique_ptr<Impl> impl_;
    Stats                 stats_{};
};

}  // namespace scatt
