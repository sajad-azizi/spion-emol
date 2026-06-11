// test_storage_prefetch_bit_equivalence.cpp
//
// Direct byte-for-byte gate for the async chunk prefetch in
// PotentialStorage.
//
// The async prefetch path is supposed to be bit-identical to the
// synchronous path: the bytes brought into RAM by start_prefetch +
// later read_chunk (zero-copy swap) must equal the bytes a plain
// synchronous read_chunk would produce.  Only the SCHEDULING differs.
//
// This test exercises that contract directly:
//   1. Write a small DISK-mode PotentialStorage with KNOWN matrix
//      contents (deterministic PRNG seed).
//   2. Read it back in two ways:
//      A. WITHOUT prefetch: a fresh PotentialStorage, walk get(ir)
//         from ir=0..Nr-1 sequentially.  Record every matrix.
//      B. WITH prefetch: another fresh PotentialStorage, walk get(ir)
//         from ir=0..Nr-1, but BEFORE each chunk crossing call
//         start_prefetch(next_chunk).  Record every matrix.
//   3. Assert A[ir] == B[ir] byte-for-byte (memcmp of the raw double
//      data on every element).  Zero tolerance.
//
// This is the explicit gate that backs the "100% accuracy preserved"
// claim for the prefetch optimization.

#include "scatt/PotentialStorage.hpp"

#include <Eigen/Dense>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int n_fail = 0;
static void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++n_fail;
}

// Equal IFF every byte of every element matches.  We don't accept any
// tolerance -- this is bit-equivalence.
static bool eq_bytes(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    const std::size_t n_bytes =
        static_cast<std::size_t>(a.size()) * sizeof(double);
    return std::memcmp(a.data(), b.data(), n_bytes) == 0;
}

// Walk DISK storage 0..Nr-1, optionally triggering an async prefetch
// of the NEXT chunk just before each chunk transition.  Returns the
// full matrix sequence so the caller can compare two walks.
static std::vector<Eigen::MatrixXd>
walk(scatt::PotentialStorage& store, std::size_t Nr, int chunk_size,
     bool with_prefetch)
{
    std::vector<Eigen::MatrixXd> out(Nr);
    int last_chunk = -1;
    for (std::size_t ir = 0; ir < Nr; ++ir) {
        if (with_prefetch) {
            const int cur_chunk = static_cast<int>(ir) / chunk_size;
            // Just BEFORE we cross into a new chunk's first ir, ask
            // for a prefetch of THIS chunk.  start_prefetch on the
            // current chunk is a no-op (already resident) -- we want
            // to prefetch the chunk we're about to need.
            //
            // To exercise the full prefetch-hit code path, kick the
            // next chunk off whenever we land on the first ir of any
            // chunk (we already have chunk N-1 read once we arrive at
            // ir = cur_chunk * chunk_size; so prefetch cur_chunk + 1).
            if (cur_chunk != last_chunk) {
                last_chunk = cur_chunk;
                store.start_prefetch(cur_chunk + 1);
            }
        }
        out[ir] = store.get(ir);   // DEEP copy out of read_buffer_
    }
    return out;
}

int main(int argc, char** argv) {
    const fs::path dir = (argc > 1)
        ? fs::path(argv[1])
        : fs::path("./test_storage_prefetch_dir");
    fs::remove_all(dir);

    // Small enough for ctest, big enough to span multiple chunks.
    const std::size_t Nr       = 47;
    const int         channels = 8;
    const int         chunk_sz = 11;     // 5 chunks: 11, 11, 11, 11, 3

    // -------- write a DISK checkpoint with deterministic contents --------
    std::cout << "--- prefetch bit-equivalence test ---\n";
    std::cout << "  Nr=" << Nr << "  channels=" << channels
              << "  chunk_size=" << chunk_sz << "\n";
    {
        scatt::PotentialStorage w;
        w.set_manifest("test_storage_prefetch_bit_equivalence");
        w.initialize_for_write(Nr, channels,
                               scatt::PotentialStorage::Mode::DISK,
                               dir.string(), chunk_sz);
        std::mt19937_64 rng(0xDEADBEEFULL);
        std::uniform_real_distribution<double> dist(-3.0, 3.0);
        for (std::size_t ir = 0; ir < Nr; ++ir) {
            Eigen::MatrixXd M(channels, channels);
            for (int i = 0; i < channels; ++i)
                for (int j = 0; j < channels; ++j)
                    M(i, j) = dist(rng);
            w.store(ir, M);
        }
        w.finalize_write();
    }

    // -------- read back WITHOUT prefetch --------
    std::vector<Eigen::MatrixXd> a;
    {
        scatt::PotentialStorage r;
        r.set_manifest("test_storage_prefetch_bit_equivalence");
        const bool ok = r.initialize_from_checkpoint(Nr, channels,
                                                      dir.string(), chunk_sz);
        check(ok, "load checkpoint (no prefetch)");
        a = walk(r, Nr, chunk_sz, /*with_prefetch=*/false);
    }

    // -------- read back WITH prefetch --------
    // set_prefetch_allowed(true) is REQUIRED to exercise the async path:
    // the planner-controlled gate defaults to OFF (see
    // StoragePlanner Pass 3 + PotentialStorage::set_prefetch_allowed),
    // so without this call start_prefetch would be a no-op and walk B
    // would silently degrade to a synchronous read identical to walk A.
    // We want to ACTUALLY exercise the prefetch path here.
    std::vector<Eigen::MatrixXd> b;
    {
        scatt::PotentialStorage r;
        r.set_manifest("test_storage_prefetch_bit_equivalence");
        const bool ok = r.initialize_from_checkpoint(Nr, channels,
                                                      dir.string(), chunk_sz);
        check(ok, "load checkpoint (with prefetch)");
        r.set_prefetch_allowed(true);
        b = walk(r, Nr, chunk_sz, /*with_prefetch=*/true);
    }

    // -------- byte-for-byte compare every matrix --------
    int n_bad = 0;
    for (std::size_t ir = 0; ir < Nr; ++ir) {
        if (!eq_bytes(a[ir], b[ir])) ++n_bad;
    }
    std::printf("  walked %zu ir, %d mismatches\n", Nr, n_bad);
    check(n_bad == 0, "every (sync read) == (prefetch + swap) byte-for-byte");

    // -------- also verify the same instance can be used both ways --------
    // (Prefetch a chunk, then call get() for an UNRELATED ir that's in a
    //  different chunk -- the prefetched chunk must not corrupt anything.)
    {
        scatt::PotentialStorage r;
        r.set_manifest("test_storage_prefetch_bit_equivalence");
        r.initialize_from_checkpoint(Nr, channels, dir.string(), chunk_sz);
        r.set_prefetch_allowed(true);   // see comment in walk-B block above
        // Prefetch chunk 3.  Then read ir=5 (chunk 0), then ir=33 (chunk 3).
        // The chunk-0 read must trigger a sync fetch (not consume the
        // stale prefetch). The chunk-3 read should then hit the prefetch.
        r.start_prefetch(3);
        Eigen::MatrixXd m5  = r.get(5);
        Eigen::MatrixXd m33 = r.get(33);
        check(eq_bytes(m5,  a[5]),  "interleaved access at ir=5  (chunk 0)");
        check(eq_bytes(m33, a[33]), "interleaved access at ir=33 (chunk 3, prefetch hit)");
    }

    fs::remove_all(dir);
    std::printf("\n  Total failures: %d\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
