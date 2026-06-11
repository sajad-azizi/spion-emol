// StoragePlanner.hpp -- decide MEMORY vs DISK for the four big scattering
// objects (pot, sinv, rinv, psi) based on the available memory budget, and
// compute the per-stage DISK chunk size.
//
// Policy (agreed 2026-04-23):
//
//   * Priority order: [pot, sinv, rinv, psi].
//     pot is placed first because it is ENERGY-INDEPENDENT and is touched at
//     every ik in the scan -- keeping it resident gives the biggest reuse
//     savings. sinv is next because it is densely accessed by both rinv
//     build and psi back-prop within one energy. rinv is used only during
//     psi back-prop. psi is placed last -- written once, read once per ik.
//
//   * Reserve 10% of the budget for OS, Eigen temporaries, and anything
//     outside the four stages.
//
//   * Greedy fill: walk the priority list; stage goes to MEMORY if it fits
//     in the remaining budget, else DISK.
//
//   * Peak concurrency is during back-propagation: all four stages are
//     simultaneously live. The budget check uses full-size per stage so
//     the sum actually fits.
//
//   * For DISK stages, the chunk size is
//         chunk = clamp(disk_budget_per_stage / matrix_bytes, 4, 200)
//     where disk_budget_per_stage = (budget_free) / n_disk_stages, and
//     budget_free is what remains after the MEMORY choices were subtracted.
//     Chunk controls the resident working set -- one chunk per DISK stage.

#pragma once

#include "scatt/Potentials.hpp"   // StorageMode

#include <cstddef>
#include <string>

namespace scatt {

enum class StageKind { Pot, Sinv, Rinv, Psi };

inline const char* stage_name(StageKind s) {
    switch (s) { case StageKind::Pot:  return "pot";
                 case StageKind::Sinv: return "sinv";
                 case StageKind::Rinv: return "rinv";
                 case StageKind::Psi:  return "psi"; }
    return "?";
}

struct StagePlan {
    StageKind    kind;
    StorageMode  mode;               // MEMORY or DISK
    std::size_t  matrix_bytes  = 0;  // one matrix (ir-slice)
    std::size_t  full_bytes    = 0;  // matrix_bytes * n_slices
    int          chunk_size    = 100;// only used when mode == DISK
    std::size_t  resident_bytes = 0; // matrix_bytes * (n_slices if MEMORY else chunk_size)
    // Async prefetch decision (see prefetch_request_mask on plan_storage).
    // True ⇒ caller may invoke PotentialStorage::start_prefetch on this
    // stage; an EXTRA chunk-sized buffer (resident_bytes) was reserved
    // out of the budget for it.  False ⇒ start_prefetch is a no-op
    // (planner ran out of budget OR mode is MEMORY OR caller didn't
    // request it).  Default false preserves pre-prefetch behaviour.
    bool         enable_prefetch = false;
};

// Bit flags for plan_storage's `prefetch_request_mask`.  OR these
// together for stages the caller intends to call start_prefetch on
// (today: BackPropagator prefetches sinv via WInverseOperator and
// rinv via ForwardRPropagator inside its descending sweep).
constexpr int kPrefetchRequestSinv = 1 << 0;
constexpr int kPrefetchRequestRinv = 1 << 1;

struct StoragePlan {
    std::size_t   system_ram_bytes      = 0;
    std::size_t   user_cap_bytes        = 0;
    std::size_t   non_storage_bytes     = 0;  // psi_lm + chi + occ.phi + sparse + ...
    std::size_t   fixed_runtime_bytes   = 0;  // OS + MKL + SYCL/GPU + glibc fragment
    std::size_t   budget_bytes          = 0;  // (raw - runtime - non_storage) * (1 - reserve)
    double        reserve_fraction      = 0.10;
    StagePlan     pot, sinv, rinv, psi;

    // Formatted multi-line report for logging.
    std::string   report() const;
};

// Inputs:
//   n_grid         : number of radial points (pot/sinv/rinv have this many slices)
//   N_ch           : channel basis size, N_ch = (l_max_continuum+1)^2
//                    Used for pot (V_total) and sinv (Schur inverse).
//   N_beta         : number of scattering solution columns in psi
//   n_keep         : #radial points in the kept psi window (usually n_grid)
//   system_ram_b   : detected physical RAM (0 => treat user_cap as budget)
//   user_cap_b     : user-supplied cap (0 => no cap)
//   N_total        : full Johnson-block dimension N_psi + N_f for rinv.
//                    Pre-2026 versions defaulted this to N_ch which
//                    under-allocated rinv by a factor (1+n_occ)^2 and
//                    routed it to MEMORY when it should have been DISK.
//                    Pass the correct (1 + n_occ) * N_ch from main().
// Production-safe default for the runtime overhead the OS+libraries hold
// resident on top of our four chunked storages.  Empirically 20-35 GB on
// Sapphire Rapids + PVC Max nodes (OS kernel, glibc heap fragmentation,
// MKL workspaces, SYCL/oneAPI runtime, GPU host-shared buffers).  Conservative
// round-up.  Production callers (main.cpp) should pass this; tests can pass 0
// to keep the synthetic-RAM scenarios policy-free.
inline constexpr std::size_t kDefaultRuntimeOverheadBytes = 30ull << 30;

StoragePlan plan_storage(std::size_t n_grid,
                         int         N_ch,
                         int         N_beta,
                         std::size_t n_keep,
                         std::size_t system_ram_b,
                         std::size_t user_cap_b,
                         double      reserve_fraction       = 0.10,
                         int         N_total                = -1,    // -1 => N_ch (legacy)
                         std::size_t non_storage_bytes      = 0,     // psi_lm + chi + occ.phi + ...
                         std::size_t fixed_runtime_bytes    = 0,     // production: pass kDefaultRuntimeOverheadBytes
                         int         pinned_chunk_pot       = 0,     // 0 = let planner decide; >0 = on-disk chunk_size from existing checkpoint
                         int         prefetch_request_mask  = 0);    // OR of kPrefetchRequest* for stages the caller wants async prefetch on; planner approves only if budget permits

}  // namespace scatt
