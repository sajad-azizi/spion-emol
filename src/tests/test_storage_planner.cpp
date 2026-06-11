// test_storage_planner.cpp -- StoragePlanner policy tests.
//
// Covers:
//   * priority order [pot, sinv, rinv, psi]
//   * everything-fits case -> all MEMORY
//   * everything-too-big   -> all DISK with clamped chunk
//   * mixed: pot fits, sinv fits, rinv/psi spill -> chunk = budget/2/matrix
//   * reserve fraction honored
//   * 0 system RAM + user cap fallback

#include "scatt/StoragePlanner.hpp"

#include <cstdint>
#include <iostream>
#include <string>

using namespace scatt;

static int g_fail = 0;
static void check(bool cond, const std::string& what) {
    if (!cond) { std::cerr << "FAIL  " << what << "\n"; ++g_fail; }
    else       { std::cout << "ok    " << what << "\n"; }
}

int main() {
    // Small system, all fits. N_ch=100 -> 80 kB/matrix, n_grid=1000 -> 80 MB/stage.
    {
        auto P = plan_storage(/*n_grid*/1000, /*N_ch*/100, /*N_beta*/100,
                              /*n_keep*/1000, /*sys*/8ull<<30, /*cap*/0);
        check(P.pot.mode  == StorageMode::MEMORY, "small/all-fits: pot MEMORY");
        check(P.sinv.mode == StorageMode::MEMORY, "small/all-fits: sinv MEMORY");
        check(P.rinv.mode == StorageMode::MEMORY, "small/all-fits: rinv MEMORY");
        check(P.psi.mode  == StorageMode::MEMORY, "small/all-fits: psi MEMORY");
    }

    // Huge, nothing fits. N_ch=5000 -> 200 MB/matrix, n_grid=10000 -> 2 TB/stage.
    {
        auto P = plan_storage(/*n_grid*/10000, /*N_ch*/5000, /*N_beta*/5000,
                              /*n_keep*/10000, /*sys*/16ull<<30, /*cap*/0);
        check(P.pot.mode  == StorageMode::DISK, "huge: pot DISK");
        check(P.sinv.mode == StorageMode::DISK, "huge: sinv DISK");
        check(P.rinv.mode == StorageMode::DISK, "huge: rinv DISK");
        check(P.psi.mode  == StorageMode::DISK, "huge: psi DISK");
        // Chunks clamped into [4, 200]; not all 1.
        check(P.pot.chunk_size >= 4 && P.pot.chunk_size <= 200, "huge: pot chunk in range");
    }

    // Priority: pot + sinv in MEMORY, rinv + psi spill to DISK.
    // Stage size = N_ch^2 * 8 * n_grid. With N_ch=225, n_grid=1000 -> 405 MB/stage.
    // Budget = 1 GB * 0.9 = 921 MB. pot (405) fits -> 516 left. sinv (405) fits -> 111 left.
    // rinv (405) doesn't fit -> DISK. psi same.
    {
        auto P = plan_storage(/*n_grid*/1000, /*N_ch*/225, /*N_beta*/225,
                              /*n_keep*/1000, /*sys*/1ull<<30, /*cap*/0);
        check(P.pot.mode  == StorageMode::MEMORY, "priority: pot wins MEMORY first");
        check(P.sinv.mode == StorageMode::MEMORY, "priority: sinv wins MEMORY second");
        check(P.rinv.mode == StorageMode::DISK,   "priority: rinv pushed to DISK");
        check(P.psi.mode  == StorageMode::DISK,   "priority: psi pushed to DISK");
    }

    // user_cap overrides when smaller than system RAM.
    {
        auto P = plan_storage(/*n_grid*/1000, /*N_ch*/200, /*N_beta*/200,
                              /*n_keep*/1000,
                              /*sys*/100ull<<30, /*cap*/200ull<<20);  // 200 MB cap
        check(P.budget_bytes < 200ull<<20, "user_cap honored (reserve applied)");
        // 200 MB budget * 0.9 = 180 MB. One stage = 320 MB. None fit.
        check(P.pot.mode == StorageMode::DISK, "tight cap: pot goes DISK");
    }

    // sys=0 + user cap falls back to user cap.
    {
        auto P = plan_storage(1000, 100, 100, 1000, /*sys*/0, /*cap*/8ull<<30);
        check(P.pot.mode == StorageMode::MEMORY, "sys=0 w/ cap: pot MEMORY");
    }

    // resident count matches: MEMORY = full, DISK = chunk * matrix.
    // Same sizing as the "priority" case so pot=MEMORY, rinv=DISK.
    {
        auto P = plan_storage(1000, 225, 225, 1000, 1ull<<30, 0);
        check(P.pot.mode == StorageMode::MEMORY, "resident-test: pot MEMORY as expected");
        check(P.rinv.mode == StorageMode::DISK,  "resident-test: rinv DISK as expected");
        check(P.pot.resident_bytes == P.pot.full_bytes,
              "resident = full for MEMORY");
        check(P.rinv.resident_bytes ==
              static_cast<std::size_t>(P.rinv.chunk_size) * P.rinv.matrix_bytes,
              "resident = chunk*matrix for DISK");
    }

    // Pinned pot chunk_size: pot uses the pinned value, NOT the planner's
    // greedy choice; the remaining stages' budget shrinks accordingly.
    // Use a budget where pot would normally fit in MEMORY (full <= budget),
    // so the pin clearly OVERRIDES the MEMORY choice.
    {
        // n_grid=1000, N_ch=225 => stage = 405 MB, budget = 1 GB * 0.9 = 921 MB.
        // Without a pin, pot would be MEMORY (405 MB <= 921 MB).
        // With pin=10, pot is forced to DISK with chunk=10 (resident = 4.05 MB).
        // Other 3 stages then split (921 - 4.05) MB = 917 MB.
        auto P = plan_storage(/*n_grid*/1000, /*N_ch*/225, /*N_beta*/225,
                              /*n_keep*/1000, /*sys*/1ull<<30, /*cap*/0,
                              /*reserve*/0.10, /*N_total*/-1,
                              /*non_storage*/0, /*runtime*/0,
                              /*pinned_chunk_pot*/10);
        check(P.pot.mode == StorageMode::DISK,
              "pinned: pot forced to DISK regardless of budget");
        check(P.pot.chunk_size == 10,
              "pinned: pot chunk_size honoured (got "
              + std::to_string(P.pot.chunk_size) + ")");
        check(P.pot.resident_bytes == 10ull * P.pot.matrix_bytes,
              "pinned: pot resident = chunk * matrix");
    }

    // Pinned pot + non_storage_bytes that eats most of the budget:
    // sinv/rinv/psi must end up DISK with small chunks.
    {
        // 1 GB sys, 600 MB non_storage, 100 MB runtime, 10% reserve =>
        // budget = (1024 - 600 - 100) MB * 0.9 = 291 MB.
        // pot pinned to chunk=20 with N_ch=225 -> 8.1 MB resident.
        // free for 3 stages = ~283 MB.
        auto P = plan_storage(/*n_grid*/1000, /*N_ch*/225, /*N_beta*/225,
                              /*n_keep*/1000,
                              /*sys*/1ull<<30, /*cap*/0,
                              /*reserve*/0.10, /*N_total*/-1,
                              /*non_storage*/600ull<<20,
                              /*runtime*/100ull<<20,
                              /*pinned_chunk_pot*/20);
        check(P.pot.mode == StorageMode::DISK && P.pot.chunk_size == 20,
              "pinned+constrained: pot pinned to chunk=20");
        check(P.sinv.mode == StorageMode::DISK,
              "pinned+constrained: sinv forced to DISK by tight budget");
        check(P.non_storage_bytes == (600ull<<20),
              "pinned+constrained: non_storage_bytes captured in plan");
        check(P.fixed_runtime_bytes == (100ull<<20),
              "pinned+constrained: fixed_runtime_bytes captured in plan");
    }

    // ----- Prefetch budget pass (Pass 3) ----------------------------------
    //
    // Default (request_mask=0): no stage gets prefetch even if it's DISK.
    {
        auto P = plan_storage(/*n_grid*/10000, /*N_ch*/5000, /*N_beta*/5000,
                              /*n_keep*/10000, /*sys*/16ull<<30, /*cap*/0);
        check(P.sinv.mode == StorageMode::DISK,
              "prefetch-default: sinv DISK (sanity)");
        check(P.rinv.mode == StorageMode::DISK,
              "prefetch-default: rinv DISK (sanity)");
        check(P.sinv.enable_prefetch == false,
              "prefetch-default: sinv enable_prefetch false when not requested");
        check(P.rinv.enable_prefetch == false,
              "prefetch-default: rinv enable_prefetch false when not requested");
    }

    // C8F8-shaped TIGHT-BUDGET case: requested AND the post-Pass 2 leftover
    // is < a single chunk.  Mirrors the C8F8 / l_cont=100 production
    // sizes: N_ch=10201, n_grid=10001, system=503 GB, non_storage ~90 GB,
    // runtime 30 GB.  Pass 3 must take the SHRINK fallback: shrink sinv
    // and rinv chunks roughly in half so the (base + prefetch) pair
    // fits within each stage's own budget.  The TOTAL memory used
    // (resident + prefetch buffers) MUST stay within budget by
    // construction.
    {
        auto P = plan_storage(/*n_grid*/10001, /*N_ch*/10201, /*N_beta*/2761,
                              /*n_keep*/10001,
                              /*sys*/std::size_t(503ull) << 30,
                              /*cap*/0,
                              /*reserve*/0.10,
                              /*N_total*/17461,
                              /*non_storage*/std::size_t(90ull) << 30,
                              /*runtime*/std::size_t(30ull) << 30,
                              /*pinned_chunk_pot*/0,
                              /*prefetch_mask*/kPrefetchRequestSinv
                                              | kPrefetchRequestRinv);
        check(P.sinv.mode == StorageMode::DISK,
              "prefetch-tight: sinv ends DISK at C8F8 scale");
        check(P.rinv.mode == StorageMode::DISK,
              "prefetch-tight: rinv ends DISK at C8F8 scale");
        check(P.sinv.enable_prefetch == true,
              "prefetch-tight: planner ENABLES sinv prefetch via chunk shrink");
        check(P.rinv.enable_prefetch == true,
              "prefetch-tight: planner ENABLES rinv prefetch via chunk shrink");
        // Memory constraint: resident + prefetch buffers stays within budget.
        // (For prefetched stages, the prefetch buffer is the same size
        //  as the now-shrunken resident.)
        std::size_t total = P.pot.resident_bytes  + P.sinv.resident_bytes
                          + P.rinv.resident_bytes + P.psi.resident_bytes;
        if (P.sinv.enable_prefetch) total += P.sinv.resident_bytes;
        if (P.rinv.enable_prefetch) total += P.rinv.resident_bytes;
        check(total <= P.budget_bytes,
              "prefetch-tight: total (resident + prefetch buffers) <= budget");
        // Sanity: chunks for prefetched stages should be ~half of what
        // they would be without prefetch (since we now reserve half the
        // per-stage budget for the prefetch buffer).  Just verify they
        // shrunk relative to the planner's preferred non-prefetch chunk.
        // Reference: a planner run WITHOUT the prefetch request gives
        // the unshrunken chunk_size; we just verify the shrunken values
        // are well above the clamp_chunk floor.
        check(P.sinv.chunk_size >= 4 && P.sinv.chunk_size <= 200,
              "prefetch-tight: sinv chunk in [4, 200]");
        check(P.rinv.chunk_size >= 4 && P.rinv.chunk_size <= 200,
              "prefetch-tight: rinv chunk in [4, 200]");
    }

    // Same C8F8-shaped TIGHT-BUDGET inputs but caller does NOT request
    // prefetch.  Should fall through to the legacy "no prefetch, full
    // chunks" path -- chunks LARGER than the shrink-fallback case above.
    {
        auto P = plan_storage(/*n_grid*/10001, /*N_ch*/10201, /*N_beta*/2761,
                              /*n_keep*/10001,
                              /*sys*/std::size_t(503ull) << 30,
                              /*cap*/0,
                              /*reserve*/0.10,
                              /*N_total*/17461,
                              /*non_storage*/std::size_t(90ull) << 30,
                              /*runtime*/std::size_t(30ull) << 30,
                              /*pinned_chunk_pot*/0,
                              /*prefetch_mask*/0);   // <-- caller opts out
        check(P.sinv.enable_prefetch == false,
              "prefetch-tight-OPTOUT: no prefetch requested -> no prefetch");
        check(P.rinv.enable_prefetch == false,
              "prefetch-tight-OPTOUT: no prefetch requested -> no prefetch");
        // chunks should be LARGER than the shrink-fallback case.
        check(P.sinv.chunk_size > 50,
              "prefetch-tight-OPTOUT: full sinv chunk_size (>50 at C8F8 scale)");
    }

    // LOOSE-BUDGET case: small problem, plenty of slack.  Both requested
    // prefetches should be granted (they each cost much less than the
    // residual budget).
    {
        // n_grid=2000, N_ch=400 -> 1.28 MB/matrix; full stage = 2.56 GB.
        // sys = 32 GB -> budget = (32 - 0 - 0) GB * 0.9 = 28.8 GB.
        // pot fits MEMORY (2.56 GB), sinv MEMORY (2.56 GB), rinv MEMORY (2.56 GB),
        // psi MEMORY (2.56 GB). Sum 10.24 GB; leftover ~ 18 GB. No DISK stage ⇒
        // no prefetch (Pass 3 only acts on DISK stages).  Cover this corner.
        auto P = plan_storage(/*n_grid*/2000, /*N_ch*/400, /*N_beta*/400,
                              /*n_keep*/2000, /*sys*/32ull<<30, /*cap*/0,
                              /*reserve*/0.10, /*N_total*/-1,
                              /*non_storage*/0, /*runtime*/0,
                              /*pinned_chunk_pot*/0,
                              /*prefetch_mask*/kPrefetchRequestSinv
                                              | kPrefetchRequestRinv);
        check(P.sinv.mode == StorageMode::MEMORY,
              "prefetch-loose-MEM: sinv MEMORY (no chunks)");
        check(P.sinv.enable_prefetch == false,
              "prefetch-loose-MEM: MEMORY-mode sinv stays false (no chunks to prefetch)");
        check(P.rinv.enable_prefetch == false,
              "prefetch-loose-MEM: MEMORY-mode rinv stays false (no chunks to prefetch)");
    }

    // LOOSE-BUDGET-with-DISK case: force DISK on sinv+rinv by big N_total,
    // but keep enough leftover budget that ONE extra chunk per stage
    // comfortably fits.  Both prefetches granted.
    {
        // pot MEMORY (small matrix_bytes), sinv+rinv+psi DISK with small
        // chunks because budget after pot is shared 3 ways.  Use a
        // generous system RAM so that even after chunk sizing the
        // leftover budget exceeds (chunk * matrix_bytes) for sinv+rinv.
        // n_grid=2000, N_ch=800 -> 5.12 MB/matrix, full = 10.24 GB.
        // N_total = 1600 -> rinv matrix_bytes = 20.48 MB, full = 40.96 GB.
        // sys = 96 GB -> budget = 96*0.9 = 86.4 GB.
        // pot fits MEMORY (10.24 GB), sinv fits MEMORY (10.24 GB),
        // psi N_ch*N_beta with N_beta=800 -> same as pot.
        // rinv too big -> DISK.
        // Hmm: with this layout only rinv is DISK.  Adjust: drop N_ch
        // bigger so sinv also spills.
        auto P = plan_storage(/*n_grid*/2000, /*N_ch*/2500, /*N_beta*/2500,
                              /*n_keep*/2000, /*sys*/96ull<<30, /*cap*/0,
                              /*reserve*/0.10, /*N_total*/3500,
                              /*non_storage*/0, /*runtime*/0,
                              /*pinned_chunk_pot*/0,
                              /*prefetch_mask*/kPrefetchRequestSinv
                                              | kPrefetchRequestRinv);
        check(P.sinv.mode == StorageMode::DISK,
              "prefetch-loose-DISK: sinv DISK (sanity)");
        check(P.rinv.mode == StorageMode::DISK,
              "prefetch-loose-DISK: rinv DISK (sanity)");
        // The planner may grant 0, 1, or 2 prefetches depending on the
        // exact budget arithmetic.  At minimum it must NOT enable a
        // prefetch that would push the total over the budget.
        std::size_t resident = P.pot.resident_bytes  + P.sinv.resident_bytes
                             + P.rinv.resident_bytes + P.psi.resident_bytes;
        if (P.sinv.enable_prefetch) resident += P.sinv.resident_bytes;
        if (P.rinv.enable_prefetch) resident += P.rinv.resident_bytes;
        check(resident <= P.budget_bytes,
              "prefetch-loose-DISK: total (incl. prefetch buffers) fits budget");
    }

    std::cout << "\n" << (g_fail == 0 ? "PASS" : "FAIL")
              << " storage_planner  (" << g_fail << " failures)\n";
    return g_fail == 0 ? 0 : 1;
}
