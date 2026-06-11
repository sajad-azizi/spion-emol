#include "scatt/StoragePlanner.hpp"

#include <algorithm>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>

namespace scatt {

namespace {

std::string format_bytes(std::size_t b) {
    const double G = b / double(1ull << 30);
    const double M = b / double(1ull << 20);
    std::ostringstream o;
    o << std::fixed << std::setprecision(5);
    if (G >= 1.0) o << G << " GB";
    else          o << M << " MB";
    return o.str();
}

int clamp_chunk(std::size_t chunk) {
    if (chunk < 4)   return 4;
    if (chunk > 200) return 200;
    return static_cast<int>(chunk);
}

}  // namespace

StoragePlan plan_storage(std::size_t n_grid,
                         int         N_ch,
                         int         N_beta,
                         std::size_t n_keep,
                         std::size_t system_ram_b,
                         std::size_t user_cap_b,
                         double      reserve_fraction,
                         int         N_total,
                         std::size_t non_storage_bytes,
                         std::size_t fixed_runtime_bytes,
                         int         pinned_chunk_pot,
                         int         prefetch_request_mask)
{
    // Backward-compat: pre-2026 callers passed only N_ch; legacy value
    // -1 means "use N_ch for everything" (the old, wrong sizing).
    if (N_total < 0) N_total = N_ch;
    StoragePlan P;
    P.system_ram_bytes    = system_ram_b;
    P.user_cap_bytes      = user_cap_b;
    P.non_storage_bytes   = non_storage_bytes;
    P.fixed_runtime_bytes = fixed_runtime_bytes;
    P.reserve_fraction    = reserve_fraction;

    // Effective raw = min(system, user_cap), if either is nonzero.
    std::size_t raw = 0;
    if (system_ram_b > 0 && user_cap_b > 0) raw = std::min(system_ram_b, user_cap_b);
    else if (system_ram_b > 0)              raw = system_ram_b;
    else                                    raw = user_cap_b;
    if (raw == 0) raw = 8ull << 30;  // 8 GB last-resort default

    // Subtract:
    //   1. fixed_runtime_bytes -- OS + MKL + SYCL + GPU host + glibc fragment
    //   2. non_storage_bytes   -- psi_lm + chi + occ.phi + chi_init + BP scratch
    //                             + sparse coeffs (everything resident during
    //                             BP that's not one of the four chunked storages)
    // ... then apply the reserve_fraction to the remainder.
    std::size_t avail = raw;
    if (avail > fixed_runtime_bytes) avail -= fixed_runtime_bytes;
    else                             avail = 0;
    if (avail > non_storage_bytes)   avail -= non_storage_bytes;
    else                             avail = 0;

    P.budget_bytes = static_cast<std::size_t>(
        static_cast<double>(avail) * (1.0 - reserve_fraction));

    // Sizes per stage.  pot and sinv hold (N_ch x N_ch) matrices;
    // rinv holds (N_total x N_total) in the joined ψ⊕f space; psi
    // holds (N_ch x N_beta).  See header doc for the full picture.
    const std::size_t cxc    = static_cast<std::size_t>(N_ch) * static_cast<std::size_t>(N_ch);
    const std::size_t txt    = static_cast<std::size_t>(N_total) * static_cast<std::size_t>(N_total);
    const std::size_t cxb    = static_cast<std::size_t>(N_ch) * static_cast<std::size_t>(N_beta);
    const std::size_t bpr    = 8ull;  // double

    P.pot  = { StageKind::Pot,  StorageMode::MEMORY, cxc * bpr, cxc * bpr * n_grid, 100, 0 };
    P.sinv = { StageKind::Sinv, StorageMode::MEMORY, cxc * bpr, cxc * bpr * n_grid, 100, 0 };
    P.rinv = { StageKind::Rinv, StorageMode::MEMORY, txt * bpr, txt * bpr * n_grid, 100, 0 };
    P.psi  = { StageKind::Psi,  StorageMode::MEMORY, cxb * bpr, cxb * bpr * n_keep, 100, 0 };

    std::size_t free_bytes = P.budget_bytes;

    // Honour an existing pot checkpoint's on-disk chunk_size.  Cross-node
    // case: pot built on a 1 TB node with chunk=200 cannot be loaded on a
    // 503 GB node with chunk=109 -- the read_buffer would exceed RAM at
    // first read.  Pin pot to the on-disk value and reserve its resident
    // before greedy-filling the other three stages.
    if (pinned_chunk_pot > 0) {
        P.pot.mode           = StorageMode::DISK;
        P.pot.chunk_size     = pinned_chunk_pot;
        P.pot.resident_bytes = static_cast<std::size_t>(pinned_chunk_pot) * P.pot.matrix_bytes;
        free_bytes = (free_bytes > P.pot.resident_bytes)
                   ? (free_bytes - P.pot.resident_bytes) : 0;
    }

    // Greedy fill in priority order.  Skip pot if it was pinned above.
    StagePlan* order_all[4]   = { &P.pot, &P.sinv, &P.rinv, &P.psi };
    StagePlan* order_unpin[3] = { &P.sinv, &P.rinv, &P.psi };
    StagePlan** order = (pinned_chunk_pot > 0) ? order_unpin : order_all;
    const int   n_order = (pinned_chunk_pot > 0) ? 3 : 4;

    // Pass 1: place stages into MEMORY if they fit.
    for (int i = 0; i < n_order; ++i) {
        StagePlan* s = order[i];
        if (s->full_bytes <= free_bytes) {
            s->mode           = StorageMode::MEMORY;
            s->resident_bytes = s->full_bytes;
            free_bytes       -= s->full_bytes;
        } else {
            s->mode = StorageMode::DISK;
        }
    }

    // Pass 2: size chunks for DISK stages.  Pinned pot is excluded by the
    // `order` selection above; its chunk_size and resident_bytes were set
    // earlier and must not be overwritten here.
    int n_disk = 0;
    for (int i = 0; i < n_order; ++i)
        if (order[i]->mode == StorageMode::DISK) ++n_disk;
    if (n_disk > 0) {
        const std::size_t per = free_bytes / static_cast<std::size_t>(n_disk);
        for (int i = 0; i < n_order; ++i) {
            StagePlan* s = order[i];
            if (s->mode != StorageMode::DISK) continue;
            std::size_t chunk = (s->matrix_bytes > 0)
                              ? (per / s->matrix_bytes) : 100;
            if (chunk == 0) chunk = 1;
            s->chunk_size     = clamp_chunk(chunk);
            s->resident_bytes = static_cast<std::size_t>(s->chunk_size) * s->matrix_bytes;
        }
    }

    // Pass 3: prefetch budget reservation, with chunk-shrink fallback.
    //
    // PotentialStorage::start_prefetch lazily allocates a SECOND
    // chunk-sized buffer (matrix_bytes * chunk_size) so that the next
    // chunk can be read off disk while the current one is being
    // consumed.  Pass 1+2 did NOT reserve that extra buffer.  Two
    // strategies, tried in order, for each requested stage:
    //
    //   Case A.  The prefetch buffer fits in the LEFTOVER budget that
    //   Pass 2 didn't use (typical for small problems where chunks
    //   were clamped to 200 by clamp_chunk).  Enable prefetch as-is;
    //   the stage's chunk_size is unchanged.
    //
    //   Case B.  Leftover is too small.  SHRINK this stage's chunk_size
    //   so that base + prefetch ( = 2 × chunk × matrix_bytes ) fit
    //   inside (old_resident + leftover) -- i.e. inside the stage's own
    //   share of the per-stage budget plus whatever leftover the
    //   planner had.  Each prefetched stage pays for its own prefetch
    //   from its own budget; non-prefetched stages keep their full
    //   chunk_size.  This is the path that ENABLES prefetch at C8F8 /
    //   l_cont=100 (Sinv chunk_size 93 → 46, Rinv 31 → 15) and unlocks
    //   the I/O / compute overlap for BP::rinv_fetch + BP::wi_apply.
    //
    //   Case C.  Even at the planner's minimum chunk_size (4 by
    //   clamp_chunk), base + prefetch don't fit.  enable_prefetch
    //   stays false; the stage runs without prefetch (memory-safe).
    //
    // All three cases preserve the BUDGET CONTRAINT:  sum of
    // (resident_bytes + prefetch_bytes_if_enabled) across all stages
    // ≤ P.budget_bytes.  The planner never over-subscribes RAM.
    //
    // Bit-identity contract: enable_prefetch = true is a runtime
    // OPTIMISATION ONLY -- the bytes a PotentialStorage::get(ir) returns
    // are byte-identical whether prefetch fires or not.  Gated by
    // test_storage_prefetch_bit_equivalence.  Smaller chunks → more
    // pread() syscalls but identical bytes on the page-cache side.
    {
        const std::size_t used_resident =
              P.pot.resident_bytes  + P.sinv.resident_bytes
            + P.rinv.resident_bytes + P.psi.resident_bytes;
        std::size_t leftover = (P.budget_bytes > used_resident)
                             ? (P.budget_bytes - used_resident) : 0;

        auto try_enable = [&](StagePlan& s, int bit) {
            if (!(prefetch_request_mask & bit)) return;
            if (s.mode != StorageMode::DISK)    return;
            if (s.matrix_bytes == 0)            return;

            // Case A: prefetch fits in leftover as-is.
            if (s.resident_bytes != 0 && s.resident_bytes <= leftover) {
                s.enable_prefetch = true;
                leftover -= s.resident_bytes;
                return;
            }

            // Case B: shrink this stage's chunk_size so that
            //   2 * new_chunk * matrix_bytes <= s.resident_bytes + leftover
            // i.e. we reclaim this stage's base budget for a smaller
            // base + a same-sized prefetch buffer.  Anything we don't
            // consume goes back into leftover for subsequent calls.
            const std::size_t available = s.resident_bytes + leftover;
            const std::size_t pair_bytes = available / 2;
            const std::size_t new_chunk_raw =
                pair_bytes / s.matrix_bytes;
            const int new_chunk = clamp_chunk(new_chunk_raw);
            // clamp_chunk's floor is 4; if even pair_bytes/matrix_bytes < 4,
            // the clamp lifts it back to 4 -- which would now NOT fit
            // (would over-subscribe).  Detect that case explicitly.
            const std::size_t need_for_pair =
                2 * static_cast<std::size_t>(new_chunk) * s.matrix_bytes;
            if (new_chunk_raw < 4 && need_for_pair > available) {
                // Case C: cannot fit prefetch even at minimum chunk_size.
                // Leave the stage as-is (no prefetch, original chunk_size).
                return;
            }
            if (new_chunk >= s.chunk_size) {
                // No need to shrink (rare; only when matrix_bytes is huge
                // and the original chunk was already at the floor).  Just
                // turn on prefetch and consume from leftover.
                if (s.resident_bytes <= leftover) {
                    s.enable_prefetch = true;
                    leftover -= s.resident_bytes;
                }
                return;
            }
            const std::size_t old_resident = s.resident_bytes;
            s.chunk_size      = new_chunk;
            s.resident_bytes  = static_cast<std::size_t>(new_chunk) * s.matrix_bytes;
            s.enable_prefetch = true;
            // Budget accounting: we released old_resident from the
            // stage's base, consumed s.resident_bytes for base, and
            // consumed s.resident_bytes for prefetch.  Net effect on
            // leftover:
            //   leftover_new = leftover + old_resident - 2 * s.resident_bytes
            // (this can be 0 or slightly positive; never negative by
            // construction since 2 * s.resident_bytes <= available).
            leftover = leftover + old_resident
                     - 2 * s.resident_bytes;
        };
        try_enable(P.sinv, kPrefetchRequestSinv);
        try_enable(P.rinv, kPrefetchRequestRinv);
    }
    return P;
}

std::string StoragePlan::report() const {
    std::ostringstream o;
    o << std::fixed << std::setprecision(5);
    o << "[storage-plan]\n";
    o << "  system RAM   : " << format_bytes(system_ram_bytes)
      << "    user cap : " << (user_cap_bytes ? format_bytes(user_cap_bytes) : std::string("unset"))
      << "\n";
    o << "  runtime ovh  : " << format_bytes(fixed_runtime_bytes)
      << "  (OS + MKL + SYCL/GPU host + glibc fragment)\n";
    o << "  non-storage  : " << format_bytes(non_storage_bytes)
      << "  (psi_lm + chi + occ.phi + chi_init + BP scratch + sparse coeffs)\n";
    o << "  reserve      : " << int(reserve_fraction * 100) << "%"
      << "    budget   : " << format_bytes(budget_bytes)
      << "  (= (RAM - runtime - non-storage) * (1 - reserve))\n";

    auto row = [&](const StagePlan& s) {
        o << "  " << std::setw(5) << std::left << stage_name(s.kind) << " : "
          << (s.mode == StorageMode::MEMORY ? "MEMORY" : "DISK  ")
          << "  full=" << std::setw(9) << std::right << format_bytes(s.full_bytes);
        if (s.mode == StorageMode::DISK) {
            o << "  chunk=" << s.chunk_size
              << "  resident=" << format_bytes(s.resident_bytes);
            if (s.enable_prefetch) {
                o << "  prefetch=ON (+" << format_bytes(s.resident_bytes) << ")";
            }
        } else {
            o << "  resident=" << format_bytes(s.resident_bytes);
        }
        o << "\n";
    };
    row(pot); row(sinv); row(rinv); row(psi);

    std::size_t total_resident = pot.resident_bytes + sinv.resident_bytes
                               + rinv.resident_bytes + psi.resident_bytes;
    std::size_t prefetch_extra = 0;
    if (pot.enable_prefetch)  prefetch_extra += pot.resident_bytes;
    if (sinv.enable_prefetch) prefetch_extra += sinv.resident_bytes;
    if (rinv.enable_prefetch) prefetch_extra += rinv.resident_bytes;
    if (psi.enable_prefetch)  prefetch_extra += psi.resident_bytes;
    o << "  peak resident (all four live): " << format_bytes(total_resident);
    if (prefetch_extra > 0) {
        o << "  + prefetch buffers " << format_bytes(prefetch_extra)
          << "  = " << format_bytes(total_resident + prefetch_extra);
    }
    o << "\n";
    return o.str();
}

}  // namespace scatt
