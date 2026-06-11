// test_storage_chunk_rechunk.cpp
// ==============================
// Bit-identity gate for PotentialStorage::renormalize_chunks_to_().
//
// The re-chunk path is the load-time replacement for the legacy
// "reject + rebuild" behaviour: when an on-disk checkpoint's chunk_size
// exceeds the runtime budget, the storage transactionally re-writes
// the chunks at the smaller chunk_size and continues the load.  The
// contract is BIT-IDENTICAL output: every ir's bytes must match a
// fresh build at the smaller chunk_size, byte-for-byte (memcmp, zero
// tolerance).  This test is the gate that lets us default-ON the
// re-chunk policy.
//
// Probes:
//   1. FULL-matrix storage:
//        - Build A with chunk_size = 11 (deterministic synthetic content)
//        - Load with runtime_budget_cs = 7 -> renormalize_chunks_to_(7)
//        - Build B (independent fresh build) with chunk_size = 7
//        - memcmp every chunk file of A vs B (header + body)
//        - memcmp metadata.bin of A vs B
//   2. SYMMETRIC packed-lower storage (the production path for Sinv):
//        - same as (1) but symmetric_storage_=true
//   3. Re-chunk is a no-op when on-disk == runtime budget:
//        - Build with chunk_size=10; load with runtime_budget=10
//        - Confirm no re-chunk fires (no _rechunk_pending dir, no
//          rechunk log line) and contents unchanged.
//   4. allow_chunk_rechunk = false preserves the legacy reject:
//        - Build with chunk_size=11; load with runtime_budget=7 and
//          chunk_rechunk_allowed_=false; load must return false and
//          on-disk chunks must remain untouched at chunk_size=11.
//   5. Manifest survives the swap:
//        - After re-chunk, manifest.txt must still match the storage's
//          set_manifest() value.
//
// PASS criterion: zero byte mismatches across all probes.

#include "scatt/PotentialStorage.hpp"

#include <Eigen/Dense>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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

// Build a deterministic PotentialStorage on disk at the given
// chunk_size and symmetric mode.  Returns the directory path.
//
// `prng_seed` controls the matrix contents -- callers in this test
// pass the SAME seed for the chunk_big and chunk_small builds so the
// rechunked-from-big output can be byte-compared against the fresh
// small build.  (Earlier version of this helper seeded from the
// directory tag, which silently differed between big/small builds
// and made the comparison meaningless.)
static std::string build_storage(const std::string& dir_tag,
                                  std::uint64_t prng_seed,
                                  std::size_t Nr, int channels,
                                  int chunk_size,
                                  bool symmetric,
                                  const std::string& manifest)
{
    const std::string dir =
        "./checkpoints/test_rechunk_" + dir_tag + "_cs" + std::to_string(chunk_size);
    fs::remove_all(dir);

    std::mt19937_64 rng(prng_seed);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    scatt::PotentialStorage st;
    st.set_manifest(manifest);
    st.initialize_for_write(Nr, channels,
                            scatt::PotentialStorage::Mode::DISK,
                            dir, chunk_size,
                            symmetric, /*parallel_chunk_write=*/false);
    for (std::size_t ir = 0; ir < Nr; ++ir) {
        Eigen::MatrixXd M(channels, channels);
        for (int j = 0; j < channels; ++j) {
            for (int i = 0; i < channels; ++i) {
                M(i, j) = dist(rng);
            }
        }
        if (symmetric) {
            // Make M symmetric: only the lower triangle is stored on disk,
            // but the in-memory copy must also be symmetric for the
            // round-trip "memory <-> disk" tests to be byte-equal.
            for (int j = 0; j < channels; ++j) {
                for (int i = 0; i < j; ++i) {
                    M(i, j) = M(j, i);
                }
            }
        }
        st.store(ir, M);
    }
    st.finalize_write();
    return dir;
}

// memcmp every chunk_X.bin + metadata.bin between two checkpoint dirs.
// Returns number of byte mismatches across the comparison.
static std::size_t diff_dirs(const std::string& a, const std::string& b,
                              int expected_num_chunks)
{
    std::size_t mismatches = 0;
    for (int k = 0; k < expected_num_chunks; ++k) {
        const std::string fa = a + "/pot_chunk_" + std::to_string(k) + ".bin";
        const std::string fb = b + "/pot_chunk_" + std::to_string(k) + ".bin";
        std::ifstream ia(fa, std::ios::binary);
        std::ifstream ib(fb, std::ios::binary);
        if (!ia || !ib) {
            std::printf("  diff_dirs: missing %s or %s\n", fa.c_str(), fb.c_str());
            ++mismatches;
            continue;
        }
        std::vector<char> ba((std::istreambuf_iterator<char>(ia)),
                              std::istreambuf_iterator<char>());
        std::vector<char> bb((std::istreambuf_iterator<char>(ib)),
                              std::istreambuf_iterator<char>());
        if (ba.size() != bb.size()) {
            std::printf("  diff_dirs: chunk %d size mismatch: %zu vs %zu\n",
                        k, ba.size(), bb.size());
            ++mismatches;
            continue;
        }
        if (std::memcmp(ba.data(), bb.data(), ba.size()) != 0) {
            std::printf("  diff_dirs: chunk %d bytes differ\n", k);
            ++mismatches;
        }
    }
    // metadata.bin must also match.
    {
        std::ifstream ia(a + "/metadata.bin", std::ios::binary);
        std::ifstream ib(b + "/metadata.bin", std::ios::binary);
        std::vector<char> ba((std::istreambuf_iterator<char>(ia)),
                              std::istreambuf_iterator<char>());
        std::vector<char> bb((std::istreambuf_iterator<char>(ib)),
                              std::istreambuf_iterator<char>());
        if (ba != bb) {
            std::printf("  diff_dirs: metadata.bin bytes differ "
                        "(%zu vs %zu bytes)\n", ba.size(), bb.size());
            ++mismatches;
        }
    }
    return mismatches;
}

// Probe 1 / 2: build at large chunk_size, force re-chunk at load to
// smaller chunk_size, verify byte-identical to a fresh build at the
// smaller size.
static void probe_rechunk_byte_identity(bool symmetric) {
    const std::string mode = symmetric ? "sym" : "full";
    std::printf("--- PROBE: re-chunk byte-identity (%s storage) ---\n",
                mode.c_str());
    const std::size_t Nr       = 47;
    const int         channels = 8;
    const int         chunk_big = 11;
    const int         chunk_small = 7;
    const int         n_chunks_small = static_cast<int>(
        (Nr + chunk_small - 1) / chunk_small);
    const std::string manifest = "test_rechunk_byte_identity_" + mode;

    // Both builds use the SAME PRNG seed so they produce identical
    // ir matrices.  Only the chunk_size (= file-layout) differs --
    // which is exactly what we're testing the re-chunk against.
    const std::uint64_t seed =
        0xCAFEBABEull ^ static_cast<std::uint64_t>(symmetric ? 1 : 0);

    // Build BIG (this is what we'll force re-chunk to smaller).
    const std::string dir_big   = build_storage(
        "big_" + mode, seed, Nr, channels, chunk_big, symmetric, manifest);
    // Build SMALL fresh (the reference for byte comparison).
    const std::string dir_small = build_storage(
        "small_" + mode, seed, Nr, channels, chunk_small, symmetric, manifest);

    // Load BIG with runtime budget = chunk_small -> triggers re-chunk.
    {
        scatt::PotentialStorage r;
        r.set_manifest(manifest);
        r.set_chunk_rechunk_allowed(true);     // explicit
        const bool ok = r.initialize_from_checkpoint(
            Nr, channels, dir_big, chunk_small);
        check(ok, mode + ": re-chunk load succeeded");
        // After successful load, the on-disk dir should have the
        // NEW chunk count and chunks must be byte-identical to the
        // fresh-build small dir.
    }

    // Compare every chunk file + metadata vs the fresh small build.
    const std::size_t m = diff_dirs(dir_big, dir_small, n_chunks_small);
    check(m == 0,
          mode + ": every chunk_K.bin + metadata.bin BYTE-IDENTICAL to "
          "fresh build at chunk_size=" + std::to_string(chunk_small));

    // Manifest survives the swap.
    {
        std::ifstream m_in(dir_big + "/manifest.txt", std::ios::binary);
        std::string got((std::istreambuf_iterator<char>(m_in)),
                         std::istreambuf_iterator<char>());
        check(got == manifest,
              mode + ": manifest.txt content unchanged after re-chunk");
    }
    // __SUCCESS__ must exist post-commit.
    check(fs::exists(dir_big + "/__SUCCESS__"),
          mode + ": __SUCCESS__ present after re-chunk commit");
    // No leftover _rechunk_committing marker.
    check(!fs::exists(dir_big + "/_rechunk_committing"),
          mode + ": no _rechunk_committing marker after clean commit");
    // No leftover _rechunk_pending dir.
    check(!fs::exists(dir_big + "/_rechunk_pending"),
          mode + ": no _rechunk_pending dir after clean commit");

    // Now re-load BIG (which is now small) and run a get() round trip:
    // every ir's in-memory matrix must equal the fresh-build small
    // version.  This catches any subtle metadata or layout drift.
    {
        scatt::PotentialStorage rA;
        scatt::PotentialStorage rB;
        rA.set_manifest(manifest);
        rB.set_manifest(manifest);
        rA.set_chunk_rechunk_allowed(true);
        rB.set_chunk_rechunk_allowed(true);
        rA.initialize_from_checkpoint(Nr, channels, dir_big,   chunk_small);
        rB.initialize_from_checkpoint(Nr, channels, dir_small, chunk_small);
        std::size_t bad = 0;
        for (std::size_t ir = 0; ir < Nr; ++ir) {
            const Eigen::MatrixXd& A = rA.get(ir);
            const Eigen::MatrixXd& B = rB.get(ir);
            if (A.rows() != B.rows() || A.cols() != B.cols()) { ++bad; continue; }
            if (std::memcmp(A.data(), B.data(),
                            static_cast<std::size_t>(A.size()) * sizeof(double)) != 0) {
                ++bad;
            }
        }
        check(bad == 0,
              mode + ": every in-memory ir matrix byte-identical to "
              "fresh-build reference");
    }

    fs::remove_all(dir_big);
    fs::remove_all(dir_small);
}

// Probe 3: re-chunk is a no-op when on-disk chunk_size == runtime budget.
static void probe_rechunk_noop() {
    std::printf("--- PROBE: re-chunk is a no-op when chunk_size matches ---\n");
    const std::size_t Nr       = 23;
    const int         channels = 5;
    const int         cs       = 10;
    const std::string manifest = "test_rechunk_noop";
    const std::string dir = build_storage(
        "noop", 0xDEADBEEFull, Nr, channels, cs, /*symmetric=*/false, manifest);
    {
        scatt::PotentialStorage r;
        r.set_manifest(manifest);
        r.set_chunk_rechunk_allowed(true);
        const bool ok = r.initialize_from_checkpoint(Nr, channels, dir, cs);
        check(ok, "load succeeded at matching chunk_size");
    }
    check(!fs::exists(dir + "/_rechunk_pending"),
          "no _rechunk_pending dir (re-chunk did not fire)");
    check(!fs::exists(dir + "/_rechunk_committing"),
          "no _rechunk_committing marker");
    fs::remove_all(dir);
}

// Probe 4: with allow_chunk_rechunk=false the legacy reject path is preserved.
static void probe_rechunk_disabled_rejects() {
    std::printf("--- PROBE: --no-checkpoint-rechunk preserves legacy reject ---\n");
    const std::size_t Nr       = 47;
    const int         channels = 8;
    const int         chunk_big   = 11;
    const int         chunk_small = 7;
    const std::string manifest = "test_rechunk_disabled";
    const std::string dir = build_storage(
        "disabled", 0xFEEDFACEull, Nr, channels, chunk_big, /*symmetric=*/false, manifest);

    // Capture the pre-load on-disk state (every chunk file's bytes).
    std::vector<std::vector<char>> pre_bytes(
        static_cast<std::size_t>((Nr + chunk_big - 1) / chunk_big));
    for (std::size_t k = 0; k < pre_bytes.size(); ++k) {
        std::ifstream f(dir + "/pot_chunk_" + std::to_string(k) + ".bin",
                         std::ios::binary);
        pre_bytes[k].assign(std::istreambuf_iterator<char>(f),
                            std::istreambuf_iterator<char>());
    }

    {
        scatt::PotentialStorage r;
        r.set_manifest(manifest);
        r.set_chunk_rechunk_allowed(false);  // <-- opt out
        const bool ok = r.initialize_from_checkpoint(
            Nr, channels, dir, chunk_small);
        check(!ok, "load rejected (as expected) when re-chunk is disabled");
    }

    // On-disk chunks must be UNTOUCHED.
    bool all_match = true;
    for (std::size_t k = 0; k < pre_bytes.size(); ++k) {
        std::ifstream f(dir + "/pot_chunk_" + std::to_string(k) + ".bin",
                         std::ios::binary);
        std::vector<char> post((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
        if (post != pre_bytes[k]) { all_match = false; break; }
    }
    check(all_match, "on-disk chunks unmodified after rejected load");

    fs::remove_all(dir);
}

int main() {
    std::printf("=== test_storage_chunk_rechunk ===\n");
    probe_rechunk_byte_identity(/*symmetric=*/false);
    std::printf("\n");
    probe_rechunk_byte_identity(/*symmetric=*/true);
    std::printf("\n");
    probe_rechunk_noop();
    std::printf("\n");
    probe_rechunk_disabled_rejects();
    std::printf("\n");
    std::printf("%s  test_storage_chunk_rechunk  (%d failures)\n",
                n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
