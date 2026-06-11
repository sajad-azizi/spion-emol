// MklThreadGuard.hpp -- per-thread RAII override of MKL's thread count.
//
// PROBLEM
//   In production we initialise MKL with:
//       mkl_set_num_threads(N)    // tell MKL the cap is N (e.g. 112)
//       mkl_set_dynamic(1)        // let MKL auto-adjust when nested in OMP
//
//   `dynamic(1)` is intentional protection against oversubscription INSIDE
//   the SchurInverter / Potentials `#pragma omp parallel for` regions:
//   each OMP thread's nested MKL call auto-drops to 1 thread, so we
//   never see N·N software threads on N cores.
//
//   The side effect: when MKL is called from a SERIAL hot path
//   (BackPropagator::run / ForwardRPropagator::run main loop -- where
//   we WANT all N MKL threads for the big GEMM / LU each step),
//   MKL's heuristic sometimes drops thread count anyway (the OMP
//   runtime can leave residual nest markers; MKL is overly cautious).
//   On the LRZ PVC node this manifested as `wi_.materialize_into`
//   eating ~75 000 s (~21 h) of CPU per energy point at l_cont=100.
//
// FIX
//   At the entry of a known-serial hot section, install this guard:
//
//       {
//           scatt::MklThreadGuard _mt(omp_get_max_threads());
//           // ... serial loop calling MKL/Eigen GEMMs ...
//       }
//
//   `mkl_set_num_threads_local(N)` overrides BOTH the global setting AND
//   the dynamic-adjustment heuristic FOR THE CALLING THREAD ONLY.  Other
//   OMP threads inside any nested SchurInverter region keep their own
//   thread-local state (= 0, meaning "use the global default with
//   dynamic adjustment") -- so the nesting protection still works.
//
// ACCURACY
//   Forcing a fixed thread count changes MKL's INTERNAL tile-reduction
//   order for the dense GEMM / LU, which is non-deterministic between
//   thread counts at the level of FP roundoff (~ε_mach × N).  This is
//   the SAME tolerance the existing test suite already accepts -- the
//   cc_dipole_bruteforce / dipole_gauge / asymptotic_amplitudes tests
//   are sized to ε_mach × condition-number not "bit-identical between
//   runs", because MKL has never been bit-deterministic across thread
//   counts.  The guard does NOT introduce any new tolerance; it merely
//   makes the production run use the SAME thread count consistently
//   (which actually IMPROVES run-to-run reproducibility).
//
// NO-OP without MKL
//   When SCATT_HAS_MKL is not defined (CPU-only dev build on macOS,
//   plain Eigen), the guard collapses to an empty struct.
#pragma once

#if defined(SCATT_HAS_MKL) && SCATT_HAS_MKL
#  include <mkl.h>
#endif

namespace scatt {

class MklThreadGuard {
public:
    explicit MklThreadGuard(int n_threads) noexcept {
#if defined(SCATT_HAS_MKL) && SCATT_HAS_MKL
        // Save the prior local setting so a nested guard or a caller
        // outside the hot section restores cleanly.
        prev_ = mkl_set_num_threads_local(n_threads);
#else
        (void)n_threads;
#endif
    }

    ~MklThreadGuard() noexcept {
#if defined(SCATT_HAS_MKL) && SCATT_HAS_MKL
        // Restore.  Passing 0 means "fall back to the global default
        // (mkl_set_num_threads + mkl_set_dynamic)".  Restoring the
        // PRIOR local value is the strict right thing -- supports
        // nested guards without leaking state.
        mkl_set_num_threads_local(prev_);
#endif
    }

    MklThreadGuard(const MklThreadGuard&)            = delete;
    MklThreadGuard& operator=(const MklThreadGuard&) = delete;
    MklThreadGuard(MklThreadGuard&&)                 = delete;
    MklThreadGuard& operator=(MklThreadGuard&&)      = delete;

private:
#if defined(SCATT_HAS_MKL) && SCATT_HAS_MKL
    int prev_ = 0;
#endif
};

}  // namespace scatt
