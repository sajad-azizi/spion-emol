// test_potential_storage.cpp -- correctness + thread-safety checks on the
// MEMORY/DISK-backed PotentialStorage.
//
// This test is deliberately detailed because the class has three things
// that historically break silently:
//   1. parallel MEMORY writes from many threads;
//   2. serial DISK write order invariants;
//   3. checkpoint round-trip (file format shared with version_0).
//
// Each probe is independent: we create and destroy the storage object.

#include "scatt/PotentialStorage.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
#include <omp.h>
#include <algorithm>
#endif

using scatt::PotentialStorage;

static int fails = 0;

static void check(bool cond, const std::string& label) {
    std::cout << (cond ? "  [ok  ] " : "  [FAIL] ") << label << "\n";
    if (!cond) ++fails;
}

// Deterministic test matrix: each (ir, i, j) has a unique double so any
// swap / off-by-one is caught.
static Eigen::MatrixXd make_M(std::size_t ir, int channels) {
    Eigen::MatrixXd M(channels, channels);
    for (int i = 0; i < channels; ++i)
        for (int j = 0; j < channels; ++j)
            M(i, j) = 1000.0 * static_cast<double>(ir)
                    +   10.0 * static_cast<double>(i)
                    +         static_cast<double>(j);
    return M;
}

// ============================================================================
int main() {
    const std::size_t Nr       = 237;     // odd, not a multiple of chunk_size
    const int         channels = 17;
    const int         chunk    = 50;      // 5 chunks (50,50,50,50,37)

    // -------------------------------------------------------------------
    std::cout << "--- MEMORY roundtrip ---\n";
    {
        PotentialStorage s;
        s.initialize_for_write(Nr, channels, PotentialStorage::Mode::MEMORY);
        for (std::size_t ir = 0; ir < Nr; ++ir) s.store(ir, make_M(ir, channels));
        s.finalize_write();
        bool ok = true;
        for (std::size_t ir = 0; ir < Nr; ++ir)
            ok = ok && (s.get(ir) - make_M(ir, channels)).cwiseAbs().maxCoeff() == 0.0;
        check(ok, "MEMORY: all stored matrices read back exactly");
        check(s.mode() == PotentialStorage::Mode::MEMORY, "MEMORY: mode()");
    }

    // -------------------------------------------------------------------
    std::cout << "--- DISK roundtrip (sequential, single-thread) ---\n";
    std::string dir = std::filesystem::temp_directory_path().string()
                    + "/scatt_storage_test_" + std::to_string(static_cast<long>(::getpid()));
    std::filesystem::remove_all(dir);
    {
        PotentialStorage s;
        s.initialize_for_write(Nr, channels, PotentialStorage::Mode::DISK, dir, chunk);
        for (std::size_t ir = 0; ir < Nr; ++ir) s.store(ir, make_M(ir, channels));
        s.finalize_write();
        bool ok = true;
        // Read in forward order (streaming).
        for (std::size_t ir = 0; ir < Nr; ++ir)
            ok = ok && (s.get(ir) - make_M(ir, channels)).cwiseAbs().maxCoeff() == 0.0;
        check(ok, "DISK: forward streaming read matches stored data");

        // Read in REVERSE order. Exercises the chunk cache eviction path.
        bool ok2 = true;
        for (std::size_t ir_plus = Nr; ir_plus-- > 0; )
            ok2 = ok2 && (s.get(ir_plus) - make_M(ir_plus, channels)).cwiseAbs().maxCoeff() == 0.0;
        check(ok2, "DISK: reverse read matches (chunk cache eviction)");

        // Read random ir values.
        std::vector<std::size_t> order = {17, 45, 99, 230, 3, 102, 55, 200};
        bool ok3 = true;
        for (std::size_t ir : order)
            ok3 = ok3 && (s.get(ir) - make_M(ir, channels)).cwiseAbs().maxCoeff() == 0.0;
        check(ok3, "DISK: random-order read matches");
    }

    // -------------------------------------------------------------------
    std::cout << "--- DISK checkpoint reload (new object reads files) ---\n";
    {
        PotentialStorage s2;
        bool loaded = s2.initialize_from_checkpoint(Nr, channels, dir, chunk);
        check(loaded, "DISK: initialize_from_checkpoint returns true");
        bool ok = true;
        for (std::size_t ir = 0; ir < Nr; ++ir)
            ok = ok && (s2.get(ir) - make_M(ir, channels)).cwiseAbs().maxCoeff() == 0.0;
        check(ok, "DISK: checkpoint-loaded data exact match");
    }
    // Fail the reload path on mismatched dims.
    {
        PotentialStorage s3;
        bool bad = s3.initialize_from_checkpoint(Nr + 1, channels, dir, chunk);
        check(!bad, "DISK: reload with wrong Nr returns false");
    }
    std::filesystem::remove_all(dir);

    // -------------------------------------------------------------------
    std::cout << "--- DISK out-of-order write aborts (thread-safety guard) ---\n";
    {
        std::string d2 = dir + "_ooo";
        std::filesystem::remove_all(d2);
        PotentialStorage s;
        s.initialize_for_write(16, 4, PotentialStorage::Mode::DISK, d2, 8);
        s.store(0, make_M(0, 4));
        s.store(1, make_M(1, 4));
        bool threw = false;
        try {
            s.store(5, make_M(5, 4));   // illegal: skips 2, 3, 4
        } catch (const std::exception& e) {
            threw = true;
        }
        check(threw, "DISK: store() with non-sequential ir throws");
        std::filesystem::remove_all(d2);
    }

    // -------------------------------------------------------------------
    std::cout << "--- MEMORY parallel write (distinct ir per thread) ---\n";
#if defined(SCATT_HAS_OPENMP) && SCATT_HAS_OPENMP
    {
        PotentialStorage s;
        s.initialize_for_write(Nr, channels, PotentialStorage::Mode::MEMORY);

        // Parallel write: each ir is assigned to exactly one thread via
        // the default static schedule. Also test dynamic schedule since
        // that's what Potentials::build uses.
        #pragma omp parallel for schedule(dynamic, 7)
        for (std::size_t ir = 0; ir < Nr; ++ir) {
            s.store(ir, make_M(ir, channels));
        }

        bool ok = true;
        for (std::size_t ir = 0; ir < Nr; ++ir)
            ok = ok && (s.get(ir) - make_M(ir, channels)).cwiseAbs().maxCoeff() == 0.0;
        check(ok, "MEMORY parallel write: all matrices consistent");
    }
#else
    check(true, "MEMORY parallel write: OpenMP not compiled in, skipped");
#endif

    // -------------------------------------------------------------------
    std::cout << "--- std::thread shredder: hammer MEMORY writes concurrently ---\n";
    {
        PotentialStorage s;
        s.initialize_for_write(Nr, channels, PotentialStorage::Mode::MEMORY);

        const int nthreads = 8;
        std::vector<std::thread> threads;
        // Partition ir space into `nthreads` blocks.
        const std::size_t block = (Nr + nthreads - 1) / nthreads;
        for (int t = 0; t < nthreads; ++t) {
            threads.emplace_back([&, t]() {
                const std::size_t lo = t * block;
                const std::size_t hi = std::min(lo + block, Nr);
                for (std::size_t ir = lo; ir < hi; ++ir) {
                    s.store(ir, make_M(ir, channels));
                }
            });
        }
        for (auto& th : threads) th.join();

        bool ok = true;
        for (std::size_t ir = 0; ir < Nr; ++ir)
            ok = ok && (s.get(ir) - make_M(ir, channels)).cwiseAbs().maxCoeff() == 0.0;
        check(ok, "MEMORY 8-thread partitioned write: all matrices correct");
    }

    std::cout << "\n==> " << (fails == 0 ? "PASS" : "FAIL: " + std::to_string(fails)) << "\n";
    return fails ? 1 : 0;
}
